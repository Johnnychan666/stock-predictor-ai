from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import urllib3


class MarketDataError(RuntimeError):
    """Raised when market data cannot be fetched or parsed."""


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


@dataclass(frozen=True)
class MarketConfig:
    symbol: str
    query_symbol: str


def resolve_symbol(symbol: str, market: str = "auto") -> MarketConfig:
    raw = symbol.strip().upper()
    if not raw:
        raise ValueError("請輸入股票代號")

    if "." in raw or "-" in raw:
        return MarketConfig(symbol=raw, query_symbol=raw)

    market = market.lower()
    if market in {"tw", "twse", "taiwan"}:
        return MarketConfig(symbol=f"{raw}.TW", query_symbol=f"{raw}.TW")
    if market in {"tpex", "otc"}:
        return MarketConfig(symbol=f"{raw}.TWO", query_symbol=f"{raw}.TWO")
    if market in {"us", "usa"}:
        return MarketConfig(symbol=raw, query_symbol=raw)

    if raw.isdigit():
        return MarketConfig(symbol=f"{raw}.TW", query_symbol=f"{raw}.TW")
    return MarketConfig(symbol=raw, query_symbol=raw)


def fetch_ohlcv(symbol: str, years: int = 5, market: str = "auto") -> pd.DataFrame:
    cache_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    frame = _fetch_ohlcv_cached(symbol.strip().upper(), int(years), market.lower(), cache_day)
    return frame.copy()


@lru_cache(maxsize=256)
def _fetch_ohlcv_cached(symbol: str, years: int, market: str, cache_day: str) -> pd.DataFrame:
    del cache_day
    config = resolve_symbol(symbol, market=market)
    try:
        return _fetch_yahoo_ohlcv(config, years=years)
    except MarketDataError as yahoo_error:
        if _can_use_twse_fallback(config):
            try:
                return _fetch_twse_ohlcv(config, years=years)
            except MarketDataError as twse_error:
                raise MarketDataError(
                    f"{config.symbol}: Yahoo 與 TWSE 都無法取得資料。"
                    f"Yahoo: {yahoo_error}; TWSE: {twse_error}"
                ) from twse_error
        raise yahoo_error


def _fetch_yahoo_ohlcv(config: MarketConfig, years: int) -> pd.DataFrame:
    if years < 1:
        raise ValueError("years 至少要 1")

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(config.query_symbol)}"
    params = {
        "range": f"{years}y",
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }

    try:
        response = requests.get(url, params=params, timeout=10, headers=REQUEST_HEADERS)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise MarketDataError(f"{config.symbol}: 無法取得行情資料: {exc}") from exc
    except ValueError as exc:
        raise MarketDataError(f"{config.symbol}: 行情資料不是有效 JSON") from exc

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = error.get("description") or error.get("code") or "未知錯誤"
        raise MarketDataError(f"{config.symbol}: Yahoo Finance 錯誤: {description}")

    result = (chart.get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise MarketDataError(f"{config.symbol}: 查無可用行情資料")

    timestamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("Asia/Taipei")
    quote_data = (result.get("indicators", {}).get("quote") or [None])[0]
    adjclose_data = (result.get("indicators", {}).get("adjclose") or [None])[0]
    if not quote_data:
        raise MarketDataError(f"{config.symbol}: 行情欄位缺漏")

    frame = pd.DataFrame(
        {
            "Date": timestamps.date,
            "Open": quote_data.get("open"),
            "High": quote_data.get("high"),
            "Low": quote_data.get("low"),
            "Close": quote_data.get("close"),
            "Volume": quote_data.get("volume"),
        }
    )
    if adjclose_data and adjclose_data.get("adjclose") is not None:
        frame["Adj Close"] = adjclose_data.get("adjclose")
    else:
        frame["Adj Close"] = frame["Close"]

    frame["Symbol"] = config.symbol
    frame = frame.dropna(subset=["Date", "Open", "High", "Low", "Close", "Adj Close"])
    frame = frame.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame["Volume"] = frame["Volume"].fillna(0)

    if len(frame) < 220:
        raise MarketDataError(f"{config.symbol}: 歷史資料太少，至少需要約 1 年交易日")

    return frame


def _can_use_twse_fallback(config: MarketConfig) -> bool:
    return config.query_symbol.endswith(".TW") and config.query_symbol.removesuffix(".TW").isalnum()


def _fetch_twse_ohlcv(config: MarketConfig, years: int) -> pd.DataFrame:
    stock_no = config.query_symbol.removesuffix(".TW")
    today = pd.Timestamp.now(tz="Asia/Taipei").normalize()
    start = today - pd.DateOffset(years=years)
    months = pd.period_range(start=start.to_period("M"), end=today.to_period("M"), freq="M")
    frames: list[pd.DataFrame] = []

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    for month in months:
        month_date = f"{month.year}{month.month:02d}01"
        url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
        params = {"date": month_date, "stockNo": stock_no, "response": "json"}
        try:
            response = requests.get(url, params=params, timeout=10, headers=REQUEST_HEADERS, verify=False)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise MarketDataError(f"TWSE 無法取得資料: {exc}") from exc
        except ValueError as exc:
            raise MarketDataError("TWSE 回傳不是有效 JSON") from exc

        if payload.get("stat") != "OK" or not payload.get("data"):
            continue
        frames.append(_parse_twse_month(payload["data"], stock_no=config.symbol))

    if not frames:
        raise MarketDataError("TWSE 查無可用行情資料")

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    frame = frame.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    cutoff = (today - pd.DateOffset(years=years)).date()
    frame = frame[frame["Date"] >= cutoff].reset_index(drop=True)

    if len(frame) < 220:
        raise MarketDataError("TWSE 歷史資料太少，至少需要約 1 年交易日")
    return frame


def _parse_twse_month(rows: list[list[str]], stock_no: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        try:
            date = _parse_roc_date(row[0])
            volume = _parse_twse_number(row[1])
            open_ = _parse_twse_number(row[3])
            high = _parse_twse_number(row[4])
            low = _parse_twse_number(row[5])
            close = _parse_twse_number(row[6])
        except (IndexError, ValueError):
            continue

        records.append(
            {
                "Date": date,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Adj Close": close,
                "Volume": volume,
                "Symbol": stock_no,
            }
        )
    return pd.DataFrame(records)


def _parse_roc_date(value: str) -> object:
    year, month, day = [int(part) for part in value.split("/")]
    return datetime(year + 1911, month, day).date()


def _parse_twse_number(value: str) -> float:
    cleaned = value.strip().replace(",", "")
    if cleaned in {"", "--", "X0.00"}:
        return np.nan
    return float(cleaned)


def last_market_date() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
