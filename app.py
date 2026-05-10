from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from stock_predictor.data import fetch_ohlcv
from stock_predictor.model import apply_context_to_prediction, predict_many, predict_symbol
from stock_predictor.recommender import recommend_taiwan_stock
from stock_predictor.research import collect_research_context, search_symbols


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class StockAppHandler(SimpleHTTPRequestHandler):
    server_version = "StockPredictorHTTP/1.1"

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
            params = parse_qs(query_string)
            symbol = _first(params, "symbol", "2408.TW").strip()
            company_name = _first(params, "name", "")
            market = _first(params, "market", "auto")
            years = _int(_first(params, "years", "2"), 2, 2, 5)
            chart_days = _int(_first(params, "chart_days", "120"), 120, 40, 260)
            backtest_days = _int(_first(params, "backtest_days", "90"), 90, 40, 180)
            retrain_every = _int(_first(params, "retrain_every", "40"), 40, 20, 120)
            threshold = _float(_first(params, "threshold", "0"), 0.0, -0.2, 0.2)
            target_mode = _first(params, "target_mode", "next_open")
            use_external = _bool(_first(params, "external", "false"))

            if not symbol:
                raise ValueError("請輸入股票代號")

            symbol, company_name = _resolve_symbol_input(symbol, company_name, market)
            payload = _cached_analysis(
                symbol,
                company_name,
                market,
                years,
                chart_days,
                backtest_days,
                retrain_every,
                round(threshold, 5),
                target_mode,
                use_external,
            )
            self._send_json(payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_batch(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            symbols = [str(item).strip() for item in payload.get("symbols", []) if str(item).strip()][:8]
            frame, errors = predict_many(
                symbols=symbols,
                years=int(payload.get("years", 2)),
                market=str(payload.get("market", "auto")),
                threshold=float(payload.get("threshold", 0)),
                backtest_days=min(int(payload.get("backtest_days", 90)), 180),
                retrain_every=int(payload.get("retrain_every", 40)),
                target_mode=str(payload.get("target_mode", "next_open")),
                use_external_context=bool(payload.get("external", False)),
            )
            rows = [] if frame.empty else _clean_json(frame.to_dict(orient="records"))
            self._send_json({"rows": rows, "errors": errors})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_recommend(self, query_string: str) -> None:
        try:
            params = parse_qs(query_string)
            limit = _int(_first(params, "limit", "8"), 8, 5, 16)
            years = _int(_first(params, "years", "2"), 2, 2, 3)
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
    prices = fetch_ohlcv(symbol, years=years, market=market)
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
    context = None
    if use_external:
        context = collect_research_context(
            symbol,
            company_name=company_name,
            ohlcv=prices,
            news_limit=5,
            institutional_days=5,
            related_years=1,
        )
        prediction = apply_context_to_prediction(prediction, context)

    return {
        "symbol": prediction["symbol"],
        "name": company_name,
        "prediction": prediction,
        "prices": _price_records(prices.tail(chart_days)),
        "context": _context_payload(context),
    }


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
        return {key: _clean_json(item) for key, item in value.items()}
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
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.^-=")
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
