from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .model import predict_many
from .universe import fetch_twse_symbols


def _read_symbols(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []
    for item in args.symbols:
        symbols.extend([part.strip() for part in item.split(",") if part.strip()])
    if args.twse_all:
        symbols.extend(fetch_twse_symbols(include_etfs=args.include_etfs))
    if args.symbols_file:
        file_path = Path(args.symbols_file)
        if file_path.suffix.lower() == ".csv":
            frame = pd.read_csv(file_path)
            if "symbol" not in frame.columns:
                raise ValueError("CSV 必須包含 symbol 欄位")
            symbols.extend(frame["symbol"].dropna().astype(str).tolist())
        else:
            symbols.extend(line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#"))
    seen: set[str] = set()
    unique_symbols: list[str] = []
    for symbol in symbols:
        key = symbol.upper()
        if key not in seen:
            seen.add(key)
            unique_symbols.append(symbol)
    if args.limit:
        unique_symbols = unique_symbols[: args.limit]
    return unique_symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="預測股票下一次開盤或下一日收盤上漲/下跌機率")
    parser.add_argument("symbols", nargs="*", help="股票代號，例如 2330.TW AAPL 或 2330")
    parser.add_argument("--symbols-file", help="CSV 或文字檔；CSV 需有 symbol 欄位")
    parser.add_argument("--twse-all", action="store_true", help="自動抓 TWSE 全部上市普通股")
    parser.add_argument("--include-etfs", action="store_true", help="搭配 --twse-all，包含 ETF/ETN 等代號")
    parser.add_argument("--limit", type=int, help="限制最多分析幾檔，方便先測試")
    parser.add_argument("--market", default="auto", help="auto, twse, tpex, us")
    parser.add_argument("--years", type=int, default=5, help="歷史資料年數")
    parser.add_argument("--threshold", type=float, default=0.0, help="漲幅門檻，0.002 代表至少漲 0.2%% 才算上漲")
    parser.add_argument("--target-mode", choices=["next_open", "next_close"], default="next_open", help="預測目標")
    parser.add_argument("--no-external-context", action="store_true", help="只使用技術面，不加入新聞/法人/關聯市場修正")
    parser.add_argument("--backtest-days", type=int, default=252, help="最近幾個交易日做 walk-forward 回測")
    parser.add_argument("--retrain-every", type=int, default=20, help="回測時每幾天重新訓練一次")
    parser.add_argument("--output", help="輸出 CSV 檔案路徑")
    args = parser.parse_args()
    symbols = _read_symbols(args)
    if not symbols:
        parser.error("請提供至少一個股票代號，或使用 --symbols-file")
    print(f"準備分析 {len(symbols)} 檔股票")
    frame, errors = predict_many(
        symbols=symbols,
        years=args.years,
        market=args.market,
        threshold=args.threshold,
        backtest_days=args.backtest_days,
        retrain_every=args.retrain_every,
        target_mode=args.target_mode,
        use_external_context=not args.no_external_context,
        progress_callback=lambda i, total, symbol: print(f"[{i}/{total}] {symbol}"),
    )
    if not frame.empty:
        print(frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if args.output:
            frame.to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"\n已輸出: {args.output}")
    if errors:
        print("\n以下股票無法完成：")
        for error in errors:
            print(f"- {error}")
    return 0 if not frame.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
