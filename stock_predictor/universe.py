from __future__ import annotations

import re

import requests
import urllib3

from .data import REQUEST_HEADERS, MarketDataError


def fetch_twse_symbols(include_etfs: bool = False) -> list[str]:
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        response = requests.get(url, timeout=20, headers=REQUEST_HEADERS, verify=False)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise MarketDataError(f"無法取得 TWSE 上市股票清單: {exc}") from exc
    except ValueError as exc:
        raise MarketDataError("TWSE 上市股票清單不是有效 JSON") from exc
    symbols: list[str] = []
    for item in payload:
        code = str(item.get("Code", "")).strip().upper()
        if not code:
            continue
        if include_etfs:
            if not re.fullmatch(r"[0-9A-Z]+", code):
                continue
        else:
            if not re.fullmatch(r"\d{4}", code) or code.startswith("00"):
                continue
        symbols.append(f"{code}.TW")
    return sorted(set(symbols))
