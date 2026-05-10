from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import numpy as np
import pandas as pd

from stock_predictor.data import fetch_ohlcv
from stock_predictor.model import apply_context_to_prediction, predict_many, predict_symbol
from stock_predictor.recommender import recommend_taiwan_stock
from stock_predictor.research import collect_research_context, search_symbols


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
MAX_JOBS = 80
ANALYSIS_JOBS: dict[str, dict[str, Any]] = {}
JOB_LOCK = threading.Lock()


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class StockAppHandler(SimpleHTTPRequestHandler):
    server_version = "StockPredictorHTTP/1.2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/search":
            self._handle_search(parsed.query)
            return
        if parsed.path == "/api/analyze":
            self._handle_analyze(parsed.query)
            return
        if parsed.path == "/api/analyze/start":
            self._handle_analyze_start(parsed.query)
            return
        if parsed.path == "/api/analyze/status":
            self._handle_analyze_status(parsed.query)
            return
        if parsed.path == "/api/recommend":
            self._handle_recommend(parsed.query)
            return
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/batch":
            self._handle_batch()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _handle_search(self, query_string: str) -> None:
        params = parse_qs(query_string)
        query = _first(params, "q", "")
        market = _first(params, "market", "auto")
        limit = _int(_first(params, "limit", "12"), 12, 1, 25)
        suggestions = [asdict(item) for item in search_symbols(query, market=market, limit=limit)]
        self._send_json({"items": suggestions})

    def _handle_analyze(self, query_string: str) -> None:
        try:
            args = _parse_analysis_args(query_string)
            payload = _cached_analysis(*args)
            self._send_json(payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_analyze_start(self, query_string: str) -> None:
        try:
            args = _parse_analysis_args(query_string)
            _cleanup_jobs()
            job_id = uuid4().hex
            now = time.time()
            with JOB_LOCK:
                ANALYSIS_JOBS[job_id] = {
                    "id": job_id,
                    "status": "queued",
                    "progress": 0,
                    "message": "已建立分析任務，準備開始。",
                    "created_at": now,
                    "updated_at": now,
                    "result": None,
                    "error": None,
                }
            thread = threading.Thread(target=_run_analysis_job, args=(job_id, args), daemon=True)
            thread.start()
            self._send_json({"job_id": job_id, "status": "queued", "progress": 0, "message": "已建立分析任務，準備開始。"})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_analyze_status(self, query_string: str) -> None:
        params = parse_qs(query_string)
        job_id = _first(params, "id", "")
        with JOB_LOCK:
            job = ANALYSIS_JOBS.get(job_id)
            payload = dict(job) if job else None
        if payload is None:
            self._send_json({"error": "找不到分析任務，請重新按分析。"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json(payload)

    def _handle_batch(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            symbols = [str(item).strip() for item in payload.get("symbols", []) if str(item).strip()][:12]
            frame, errors = predict_many(
                symbols=symbols,
                years=int(payload.get("years", 5)),
                market=str(payload.get("market", "auto")),
                threshold=float(payload.get("threshold", 0)),
                backtest_days=min(int(payload.get("backtest_days", 252)), 252),
                retrain_every=int(payload.get("retrain_every", 20)),
                target_mode=str(payload.get("target_mode", "next_open")),
                use_external_context=bool(payload.get("external", True)),
            )
            rows = [] if frame.empty else _clean_json(frame.to_dict(orient="records"))
            self._send_json({"rows": rows, "errors": errors})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_recommend(self, query_string: str) -> None:
        try:
            params = parse_qs(query_string)
            limit = _int(_first(params, "limit", "16"), 16, 5, 36)
            years = _int(_first(params, "years", "2"), 2, 2, 5)
            best, ranked, errors = recommend_taiwan_stock(limit=limit, years=years)
            self._send_json({"best": best.to_row() if best else None, "ranked": ranked, "errors": errors})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(_clean_json(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _parse_analysis_args(query_string: str) -> tuple[str, str, str, int, int, int, int, float, str, bool]:
    params = parse_qs(query_string)
    symbol = _first(params, "symbol", "2408.TW").strip()
    company_name = _first(params, "name", "")
    market = _first(params, "market", "auto")
    years = _int(_first(params, "years", "5"), 5, 2, 10)
    chart_days = _int(_first(params, "chart_days", "180"), 180, 40, 520)
    backtest_days = _int(_first(params, "backtest_days", "252"), 252, 60, 500)
    retrain_every = _int(_first(params, "retrain_every", "20"), 20, 5, 120)
    threshold = _float(_first(params, "threshold", "0"), 0.0, -0.2, 0.2)
    target_mode = _first(params, "target_mode", "next_open")
    use_external = _bool(_first(params, "external", "true"))

    if not symbol:
        raise ValueError("請輸入股票代號")

    symbol, company_name = _resolve_symbol_input(symbol, company_name, market)
    return (symbol, company_name, market, years, chart_days, backtest_days, retrain_every, round(threshold, 5), target_mode, use_external)


@lru_cache(maxsize=64)
def _cached_analysis(
    symbol: str,
    company_name: str,
    market: str,
    years: int,
    chart_days: int,
    backtest_days: int,
    retrain_every: int,
    threshold: float,
    target_mode: str,
    use_external: bool,
) -> dict[str, Any]:
    return _build_analysis_payload(
        symbol,
        company_name,
        market,
        years,
        chart_days,
        backtest_days,
        retrain_every,
        threshold,
        target_mode,
        use_external,
    )


def _build_analysis_payload(
    symbol: str,
    company_name: str,
    market: str,
    years: int,
    chart_days: int,
    backtest_days: int,
    retrain_every: int,
    threshold: float,
    target_mode: str,
    use_external: bool,
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    notify = progress or (lambda _percent, _message: None)
    notify(8, "抓取歷史股價與成交量。")
    prices = fetch_ohlcv(symbol, years=years, market=market)

    notify(30, "建立技術指標並訓練預測模型。")
    prediction = predict_symbol(
        symbol=symbol,
        years=years,
        market=market,
        company_name=company_name,
        threshold=threshold,
        backtest_days=backtest_days,
        retrain_every=retrain_every,
        target_mode=target_mode,
        use_external_context=False,
        ohlcv=prices,
    ).to_row()

    notify(58, "完成模型回測，整理 K 線資料。")
    context = None
    if use_external:
        notify(68, "抓取新聞、法人籌碼、同業與美股關聯。")
        context = collect_research_context(
            symbol,
            company_name=company_name,
            ohlcv=prices,
            news_limit=12,
            institutional_days=8,
            related_years=2,
        )
        notify(88, "整合十因子矩陣並修正機率。")
        prediction = apply_context_to_prediction(prediction, context)
    else:
        notify(88, "快速模式完成，未啟用外部資料修正。")

    return {
        "symbol": prediction["symbol"],
        "name": company_name,
        "prediction": prediction,
        "prices": _price_records(prices.tail(chart_days)),
        "context": _context_payload(context),
    }


def _run_analysis_job(job_id: str, args: tuple[str, str, str, int, int, int, int, float, str, bool]) -> None:
    def progress(percent: int, message: str) -> None:
        _update_job(job_id, status="running", progress=percent, message=message)

    try:
        progress(3, "分析任務開始。")
        result = _build_analysis_payload(*args, progress=progress)
        _update_job(job_id, status="done", progress=100, message="分析完成。", result=result)
    except Exception as exc:
        _update_job(job_id, status="error", progress=100, message="分析失敗。", error=str(exc))


def _update_job(job_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def _cleanup_jobs() -> None:
    cutoff = time.time() - 60 * 60
    with JOB_LOCK:
        for job_id in [job_id for job_id, job in ANALYSIS_JOBS.items() if job.get("updated_at", 0) < cutoff]:
            ANALYSIS_JOBS.pop(job_id, None)
        if len(ANALYSIS_JOBS) > MAX_JOBS:
            oldest = sorted(ANALYSIS_JOBS.items(), key=lambda item: item[1].get("updated_at", 0))
            for job_id, _job in oldest[: len(ANALYSIS_JOBS) - MAX_JOBS]:
                ANALYSIS_JOBS.pop(job_id, None)


def _price_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "date": str(row.Date),
                "open": float(row.Open),
                "high": float(row.High),
                "low": float(row.Low),
                "close": float(row.Close),
                "adjClose": float(getattr(row, "_6", row.Close)),
                "volume": float(row.Volume),
            }
        )
    return records


def _context_payload(context: Any | None) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        "news": {"score": context.news.score, "label": context.news.label, "count": context.news.count, "items": [asdict(item) for item in context.news.items]},
        "institutional": {
            "score": context.institutional.score,
            "label": context.institutional.label,
            "days": context.institutional.days,
            "foreignNet5d": context.institutional.foreign_net_5d,
            "totalNet5d": context.institutional.total_net_5d,
            "rows": [asdict(item) for item in context.institutional.rows],
            "message": context.institutional.message,
        },
        "related": {"score": context.related.score, "label": context.related.label, "symbols": context.related.symbols, "moves": [asdict(item) for item in context.related.moves], "message": context.related.message},
        "market": {"score": context.market.score, "label": context.market.label, "moves": [asdict(item) for item in context.market.moves], "message": context.market.message},
        "events": asdict(context.events),
        "expectations": asdict(context.expectations),
        "valuation": asdict(context.valuation),
        "technical": asdict(context.technical),
        "liquidity": asdict(context.liquidity),
        "derivatives": asdict(context.derivatives),
        "macro": asdict(context.macro),
        "factors": [asdict(item) for item in context.factors],
        "adjustment": context.adjustment,
        "notes": context.notes,
    }


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items() if key not in {"created_at", "updated_at"}}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value) if value is not None and not isinstance(value, (str, bytes)) else False:
        return None
    return value


def _resolve_symbol_input(symbol: str, company_name: str, market: str) -> tuple[str, str]:
    if _looks_like_ticker(symbol):
        return symbol, company_name
    suggestions = search_symbols(symbol, market=market, limit=1)
    if not suggestions:
        return symbol, company_name
    item = suggestions[0]
    return item.symbol, company_name or item.name


def _looks_like_ticker(symbol: str) -> bool:
    allowed = set("ABCDEFFHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.^-=")
    return bool(symbol) and all(char in allowed for char in symbol)


def _first(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    return values[0] if values else default


def _int(value: str, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _float(value: str, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _bool(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off", "否"}


def main() -> int:
    parser = argparse.ArgumentParser(description="啟動股票預測網頁系統")
    default_host = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    default_port = int(os.environ.get("PORT", "8000"))
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()

    if not WEB_ROOT.exists():
        raise SystemExit("找不到 web 目錄，請確認前端檔案存在")

    with ReusableThreadingHTTPServer((args.host, args.port), StockAppHandler) as httpd:
        print(f"股票預測系統已啟動：http://{args.host}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n伺服器已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
