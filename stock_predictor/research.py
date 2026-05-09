from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from html import unescape
import re
from typing import Any
from urllib.parse import quote
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import urllib3

from .data import REQUEST_HEADERS, fetch_ohlcv, resolve_symbol


POSITIVE_TERMS = ["利多", "看好", "上修", "升級", "買進", "營收創高", "創高", "優於預期", "需求回溫", "beat", "upgrade", "strong demand", "outperform"]
NEGATIVE_TERMS = ["利空", "看淡", "下修", "降級", "賣出", "低於預期", "需求疲弱", "砍單", "訴訟", "miss", "downgrade", "weak demand", "underperform"]
EVENT_POSITIVE_TERMS = ["財報優於預期", "營收創高", "展望樂觀", "法說報喜", "庫藏股", "股利提高", "beats expectations", "raises guidance"]
EVENT_NEGATIVE_TERMS = ["財報低於預期", "展望保守", "下修財測", "增資", "訴訟", "裁員", "misses expectations", "cuts guidance"]
MACRO_POSITIVE_TERMS = ["降息", "通膨降溫", "風險偏好", "soft landing", "rate cut", "risk-on"]
MACRO_NEGATIVE_TERMS = ["升息", "通膨升溫", "地緣政治", "殖利率攀升", "risk-off", "hawkish", "higher yields"]

SECTOR_RELATED_SYMBOLS = {
    "memory": ["MU", "WDC", "000660.KS", "005930.KS", "^SOX"],
    "foundry": ["TSM", "NVDA", "AMD", "ASML", "^SOX"],
    "fabless": ["NVDA", "AMD", "QCOM", "AVGO", "^SOX"],
    "server": ["NVDA", "AMD", "SMCI", "DELL", "^IXIC"],
    "finance": ["^GSPC", "^IXIC", "XLF"],
    "market": ["^GSPC", "^IXIC", "^SOX", "^VIX"],
}
SYMBOL_RELATED_MAP = {
    "2330": SECTOR_RELATED_SYMBOLS["foundry"], "2303": ["TSM", "UMC", "^SOX", "^IXIC"],
    "2454": SECTOR_RELATED_SYMBOLS["fabless"], "2408": SECTOR_RELATED_SYMBOLS["memory"],
    "2344": SECTOR_RELATED_SYMBOLS["memory"], "2337": SECTOR_RELATED_SYMBOLS["memory"],
    "2317": ["AAPL", "NVDA", "AMD", "^IXIC"], "2382": SECTOR_RELATED_SYMBOLS["server"],
    "3231": SECTOR_RELATED_SYMBOLS["server"], "2881": SECTOR_RELATED_SYMBOLS["finance"],
    "2882": SECTOR_RELATED_SYMBOLS["finance"], "2891": SECTOR_RELATED_SYMBOLS["finance"],
}
TAIWAN_NAME_ALIASES = {
    "台積電": ("2330.TW", "台灣積體電路製造"), "聯電": ("2303.TW", "聯華電子"),
    "聯發科": ("2454.TW", "聯發科技"), "鴻海": ("2317.TW", "鴻海精密"),
    "南亞科": ("2408.TW", "南亞科技"), "南亞科技": ("2408.TW", "南亞科技"),
    "華邦電": ("2344.TW", "華邦電子"), "華邦電子": ("2344.TW", "華邦電子"),
    "旺宏": ("2337.TW", "旺宏電子"), "廣達": ("2382.TW", "廣達電腦"),
    "緯創": ("3231.TW", "緯創資通"), "富邦金": ("2881.TW", "富邦金融控股"),
}
MARKET_SYMBOLS = {
    "us": ["^GSPC", "^IXIC", "^DJI", "^SOX", "ES=F", "NQ=F"],
    "taiwan": ["^TWII", "EWT"],
    "asia": ["^N225", "^HSI", "000001.SS", "^KS11"],
    "risk": ["^VIX", "^TNX", "CL=F", "DX-Y.NYB"],
}


@dataclass(frozen=True)
class SymbolSuggestion:
    symbol: str
    name: str
    exchange: str = ""
    quote_type: str = ""


@dataclass(frozen=True)
class NewsItem:
    title: str
    publisher: str = ""
    link: str = ""
    published_at: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class SentimentSummary:
    score: float = 0.0
    label: str = "中性"
    count: int = 0
    items: list[NewsItem] = field(default_factory=list)


@dataclass(frozen=True)
class InstitutionalDay:
    date: str
    foreign_net: float
    investment_trust_net: float
    dealer_net: float
    total_net: float


@dataclass(frozen=True)
class InstitutionalSummary:
    score: float = 0.0
    label: str = "無資料"
    days: int = 0
    foreign_net_5d: float | None = None
    total_net_5d: float | None = None
    rows: list[InstitutionalDay] = field(default_factory=list)
    message: str = ""


@dataclass(frozen=True)
class RelatedMove:
    symbol: str
    latest_date: str
    return_1d: float | None
    return_5d: float | None
    return_20d: float | None


@dataclass(frozen=True)
class RelatedMarketSummary:
    score: float = 0.0
    label: str = "中性"
    symbols: list[str] = field(default_factory=list)
    moves: list[RelatedMove] = field(default_factory=list)
    message: str = ""


@dataclass(frozen=True)
class FactorScore:
    key: str
    name: str
    score: float
    label: str
    weight: float
    detail: str = ""


@dataclass(frozen=True)
class MarketMove:
    symbol: str
    name: str
    category: str
    latest_date: str
    return_1d: float | None
    return_5d: float | None
    score: float


@dataclass(frozen=True)
class MarketRegimeSummary:
    score: float = 0.0
    label: str = "大盤中性"
    moves: list[MarketMove] = field(default_factory=list)
    message: str = ""


@dataclass(frozen=True)
class EventSummary:
    score: float = 0.0
    label: str = "重大消息中性"
    highlights: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExpectationSummary:
    score: float = 0.0
    label: str = "預期差中性"
    detail: str = ""


@dataclass(frozen=True)
class ValuationSummary:
    score: float = 0.0
    label: str = "估值中性"
    trailing_pe: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None
    peg_ratio: float | None = None
    dividend_yield: float | None = None
    recommendation_mean: float | None = None
    target_upside: float | None = None
    message: str = ""


@dataclass(frozen=True)
class TechnicalSummary:
    score: float = 0.0
    label: str = "技術中性"
    rsi_14: float | None = None
    close_vs_ma20: float | None = None
    close_vs_ma60: float | None = None
    close_vs_ma200: float | None = None
    macd_hist: float | None = None
    breakout_60d: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class LiquiditySummary:
    score: float = 0.0
    label: str = "量能中性"
    volume_vs_20d: float | None = None
    avg_volume_20d: float | None = None
    avg_dollar_volume_20d: float | None = None
    range_20d: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class DerivativesSummary:
    score: float = 0.0
    label: str = "衍生品中性"
    put_call_volume_ratio: float | None = None
    call_volume: float | None = None
    put_volume: float | None = None
    message: str = ""


@dataclass(frozen=True)
class ResearchContext:
    news: SentimentSummary = field(default_factory=SentimentSummary)
    institutional: InstitutionalSummary = field(default_factory=InstitutionalSummary)
    related: RelatedMarketSummary = field(default_factory=RelatedMarketSummary)
    market: MarketRegimeSummary = field(default_factory=MarketRegimeSummary)
    events: EventSummary = field(default_factory=EventSummary)
    expectations: ExpectationSummary = field(default_factory=ExpectationSummary)
    valuation: ValuationSummary = field(default_factory=ValuationSummary)
    technical: TechnicalSummary = field(default_factory=TechnicalSummary)
    liquidity: LiquiditySummary = field(default_factory=LiquiditySummary)
    derivatives: DerivativesSummary = field(default_factory=DerivativesSummary)
    macro: EventSummary = field(default_factory=lambda: EventSummary(label="宏觀中性"))
    factors: list[FactorScore] = field(default_factory=list)
    adjustment: float = 0.0
    notes: list[str] = field(default_factory=list)


def search_symbols(query: str, market: str = "auto", limit: int = 10) -> list[SymbolSuggestion]:
    cleaned = query.strip()
    if not cleaned:
        return []
    suggestions: list[SymbolSuggestion] = []
    seen: set[str] = set()

    def add(symbol: str, name: str = "", exchange: str = "", quote_type: str = "") -> None:
        key = symbol.upper()
        if key not in seen:
            seen.add(key)
            suggestions.append(SymbolSuggestion(symbol, name, exchange, quote_type))

    try:
        config = resolve_symbol(cleaned, market=market)
        if re.fullmatch(r"[\^A-Z0-9.\-=]+", config.symbol):
            add(config.symbol, "直接使用輸入代號", market.upper() if market != "auto" else "", "")
    except ValueError:
        pass
    compact = re.sub(r"\s+", "", cleaned)
    for alias, (symbol, name) in TAIWAN_NAME_ALIASES.items():
        if compact in alias or alias in compact or compact in name:
            add(symbol, name, "TAI", "EQUITY")
    try:
        response = requests.get("https://query1.finance.yahoo.com/v1/finance/search", params={"q": cleaned, "quotes_count": limit, "news_count": 0}, timeout=12, headers=REQUEST_HEADERS)
        response.raise_for_status()
        for item in response.json().get("quotes", []):
            symbol = str(item.get("symbol") or "").strip()
            if symbol:
                add(symbol, str(item.get("shortname") or item.get("longname") or ""), str(item.get("exchange") or ""), str(item.get("quoteType") or ""))
            if len(suggestions) >= limit:
                break
    except Exception:
        pass
    return suggestions[:limit]


def infer_related_symbols(symbol: str) -> list[str]:
    resolved = resolve_symbol(symbol).symbol
    code = resolved.split(".")[0]
    if code in SYMBOL_RELATED_MAP:
        return SYMBOL_RELATED_MAP[code]
    if resolved.endswith((".TW", ".TWO")):
        if code.startswith("28"):
            return SECTOR_RELATED_SYMBOLS["finance"]
        if code.startswith(("23", "24", "30", "32", "34", "35", "36", "49", "62", "66", "80")):
            return ["^SOX", "SOXX", "NVDA", "AMD", "^IXIC"]
        return ["^TWII", "^GSPC", "^IXIC"]
    return SECTOR_RELATED_SYMBOLS["market"]


def build_related_market_feature_frame(base_ohlcv: pd.DataFrame, related_symbols: list[str], years: int = 5) -> pd.DataFrame:
    base_dates = pd.DataFrame({"Date": pd.to_datetime(base_ohlcv["Date"])})
    feature_frame = base_dates.copy()
    for related_symbol in related_symbols[:5]:
        try:
            related = fetch_ohlcv(related_symbol, years=max(2, min(years, 8)), market="auto")
        except Exception:
            continue
        close = related["Adj Close"].astype(float)
        rel = pd.DataFrame({
            "Date": pd.to_datetime(related["Date"]),
            f"rel_{re.sub(r'[^0-9A-Za-z]+', '_', related_symbol).strip('_').lower()}_ret_1d": close.pct_change(1),
            f"rel_{re.sub(r'[^0-9A-Za-z]+', '_', related_symbol).strip('_').lower()}_ret_5d": close.pct_change(5),
            f"rel_{re.sub(r'[^0-9A-Za-z]+', '_', related_symbol).strip('_').lower()}_ret_20d": close.pct_change(20),
        }).sort_values("Date")
        merged = pd.merge_asof(base_dates.sort_values("Date"), rel, on="Date", direction="backward", allow_exact_matches=False)
        feature_frame = pd.concat([feature_frame, merged[[c for c in merged.columns if c != "Date"]]], axis=1)
    feature_frame["Date"] = feature_frame["Date"].dt.date
    return feature_frame.replace([np.inf, -np.inf], np.nan)


def collect_research_context(symbol: str, *, company_name: str = "", ohlcv: pd.DataFrame | None = None, news_limit: int = 12, institutional_days: int = 8, related_years: int = 2) -> ResearchContext:
    notes: list[str] = []
    news = _safe(lambda: fetch_news_sentiment(symbol, company_name, news_limit), SentimentSummary(label="無資料"), notes, "新聞")
    avg_volume = float(pd.to_numeric(ohlcv["Volume"], errors="coerce").tail(20).mean()) if ohlcv is not None and "Volume" in ohlcv else None
    institutional = _safe(lambda: fetch_institutional_summary(symbol, days=institutional_days, avg_volume=avg_volume), InstitutionalSummary(message="法人資料無資料"), notes, "法人")
    related = _safe(lambda: fetch_related_market_summary(symbol, years=related_years), RelatedMarketSummary(message="關聯市場無資料"), notes, "關聯市場")
    market = _safe(lambda: fetch_market_regime_summary(years=related_years), MarketRegimeSummary(message="大盤無資料"), notes, "大盤")
    technical = build_technical_summary(ohlcv) if ohlcv is not None else TechnicalSummary()
    liquidity = build_liquidity_summary(ohlcv) if ohlcv is not None else LiquiditySummary()
    events = build_event_summary(news)
    valuation = _safe(lambda: fetch_valuation_summary(symbol, latest_close=_latest_close(ohlcv)), ValuationSummary(message="估值無資料"), notes, "估值")
    expectations = build_expectation_summary(news, valuation)
    derivatives = _safe(lambda: fetch_derivatives_summary(symbol), DerivativesSummary(message="衍生品無資料"), notes, "衍生品")
    macro = _safe(lambda: fetch_macro_event_summary(), EventSummary(label="宏觀無資料"), notes, "宏觀")
    factors = build_factor_scores(market=market, related=related, events=events, expectations=expectations, valuation=valuation, technical=technical, liquidity=liquidity, institutional=institutional, derivatives=derivatives, macro=macro)
    return ResearchContext(news, institutional, related, market, events, expectations, valuation, technical, liquidity, derivatives, macro, factors, context_probability_adjustment(factors), notes)


def build_factor_scores(**items: Any) -> list[FactorScore]:
    return [
        FactorScore("market", "1. 大盤方向", items["market"].score, items["market"].label, 0.16, "美股、台股、亞歐股、VIX、利率與風險資產"),
        FactorScore("industry", "2. 產業與同業", items["related"].score, items["related"].label, 0.14, ", ".join(items["related"].symbols[:6])),
        FactorScore("events", "3. 個股重大消息", items["events"].score, items["events"].label, 0.13, "；".join(items["events"].highlights[:3])),
        FactorScore("expectations", "4. 財報與預期差", items["expectations"].score, items["expectations"].label, 0.10, items["expectations"].detail),
        FactorScore("valuation", "5. 估值高低", items["valuation"].score, items["valuation"].label, 0.07, items["valuation"].message),
        FactorScore("technical", "6. 技術面", items["technical"].score, items["technical"].label, 0.14, items["technical"].detail),
        FactorScore("liquidity", "7. 成交量與流動性", items["liquidity"].score, items["liquidity"].label, 0.08, items["liquidity"].detail),
        FactorScore("chip", "8. 籌碼面", items["institutional"].score, items["institutional"].label, 0.10, "外資、投信、自營商買賣超"),
        FactorScore("derivatives", "9. 選擇權與衍生品", items["derivatives"].score, items["derivatives"].label, 0.04, items["derivatives"].message),
        FactorScore("macro", "10. 宏觀事件", items["macro"].score, items["macro"].label, 0.04, "；".join(items["macro"].highlights[:3])),
    ]


def context_probability_adjustment(factors: list[FactorScore]) -> float:
    return float(np.clip(sum(f.score * f.weight for f in factors) * 0.55, -0.20, 0.20))


def fetch_news_sentiment(symbol: str, company_name: str = "", limit: int = 12) -> SentimentSummary:
    items = fetch_news_items(symbol, company_name, limit)
    scored = [NewsItem(i.title, i.publisher, i.link, i.published_at, _score_text(i.title)) for i in items]
    score = float(np.clip(np.mean([i.score for i in scored]) if scored else 0, -1, 1))
    return SentimentSummary(score, _label(score, "偏利多", "偏利空", "中性"), len(scored), scored)


def fetch_news_items(symbol: str, company_name: str = "", limit: int = 12) -> list[NewsItem]:
    resolved = resolve_symbol(symbol).symbol
    query = " ".join(part for part in [company_name.strip(), resolved.split(".")[0], "股票"] if part)
    items = _fetch_yahoo_news(query or resolved, limit) + _fetch_google_news(query or resolved, limit)
    out: list[NewsItem] = []
    seen: set[str] = set()
    for item in items:
        key = re.sub(r"\s+", " ", item.title).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def fetch_related_market_summary(symbol: str, years: int = 2) -> RelatedMarketSummary:
    symbols = infer_related_symbols(symbol)
    moves: list[RelatedMove] = []
    for related_symbol in symbols[:6]:
        try:
            frame = fetch_ohlcv(related_symbol, years=max(2, years), market="auto")
            close = frame["Adj Close"].astype(float)
            moves.append(RelatedMove(related_symbol, str(frame["Date"].iloc[-1]), _safe_pct(close, 1), _safe_pct(close, 5), _safe_pct(close, 20)))
        except Exception:
            continue
    if not moves:
        return RelatedMarketSummary(symbols=symbols, message="沒有可用的關聯市場資料")
    momentum = np.nanmean([(m.return_1d or 0) * 0.65 + (m.return_5d or 0) * 0.35 for m in moves])
    score = float(np.clip(np.tanh(momentum * 10), -1, 1))
    return RelatedMarketSummary(score, _label(score, "外部市場偏多", "外部市場偏空", "外部市場中性"), symbols, moves)


def fetch_market_regime_summary(years: int = 2) -> MarketRegimeSummary:
    names = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^SOX": "費城半導體", "^TWII": "台股加權", "^VIX": "VIX", "^TNX": "美債10年殖利率"}
    moves: list[MarketMove] = []
    for category, symbols in MARKET_SYMBOLS.items():
        for symbol in symbols:
            try:
                frame = fetch_ohlcv(symbol, years=max(2, years), market="auto")
                close = frame["Adj Close"].astype(float)
                ret1, ret5 = _safe_pct(close, 1), _safe_pct(close, 5)
                direction = -1 if symbol in {"^VIX", "^TNX", "DX-Y.NYB"} else (-0.35 if symbol == "CL=F" else 1)
                score = float(np.clip(np.tanh(((ret1 or 0) * 0.75 + (ret5 or 0) * 0.25) * 18 * direction), -1, 1))
                moves.append(MarketMove(symbol, names.get(symbol, symbol), category, str(frame["Date"].iloc[-1]), ret1, ret5, score))
            except Exception:
                continue
    if not moves:
        return MarketRegimeSummary(message="沒有可用的大盤資料")
    weights = {"us": 1.2, "taiwan": 1.25, "risk": 0.9, "asia": 0.75}
    score = sum(m.score * weights.get(m.category, 0.6) for m in moves) / sum(weights.get(m.category, 0.6) for m in moves)
    score = float(np.clip(score, -1, 1))
    return MarketRegimeSummary(score, _label(score, "大盤偏多", "大盤偏空", "大盤中性"), moves)


def build_event_summary(news: SentimentSummary) -> EventSummary:
    highlights = [item.title for item in news.items[:5]]
    text = " ".join(highlights).lower()
    pos = sum(text.count(t.lower()) for t in EVENT_POSITIVE_TERMS)
    neg = sum(text.count(t.lower()) for t in EVENT_NEGATIVE_TERMS)
    score = float(np.clip((pos - neg) / max(pos + neg, 1), -1, 1)) if pos or neg else news.score * 0.45
    return EventSummary(score, _label(score, "重大消息偏利多", "重大消息偏利空", "重大消息中性"), highlights)


def build_expectation_summary(news: SentimentSummary, valuation: ValuationSummary) -> ExpectationSummary:
    text = " ".join(item.title.lower() for item in news.items)
    pos = sum(text.count(t) for t in ["優於預期", "超預期", "beat", "raises guidance"])
    neg = sum(text.count(t) for t in ["低於預期", "不如預期", "miss", "cuts guidance"])
    score = (pos - neg) / max(pos + neg, 1) if pos or neg else 0.0
    if valuation.target_upside is not None:
        score += float(np.clip(valuation.target_upside * 2.5, -0.6, 0.6))
    score = float(np.clip(score, -1, 1))
    return ExpectationSummary(score, _label(score, "預期差偏多", "預期差偏空", "預期差中性"), "新聞語意與分析師資料代理")


def fetch_valuation_summary(symbol: str, latest_close: float | None = None) -> ValuationSummary:
    if symbol.startswith("^"):
        return ValuationSummary(label="指數無估值資料")
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{quote(resolve_symbol(symbol).query_symbol)}"
    try:
        response = requests.get(url, params={"modules": "summaryDetail,defaultKeyStatistics,financialData"}, timeout=14, headers=REQUEST_HEADERS)
        response.raise_for_status()
        result = ((response.json().get("quoteSummary") or {}).get("result") or [None])[0] or {}
    except Exception as exc:
        return ValuationSummary(label="估值無資料", message=f"估值資料不可用：{exc}")
    summary, stats, financial = result.get("summaryDetail") or {}, result.get("defaultKeyStatistics") or {}, result.get("financialData") or {}
    trailing_pe = _raw(summary.get("trailingPE") or stats.get("trailingPE"))
    forward_pe = _raw(summary.get("forwardPE") or stats.get("forwardPE"))
    pb = _raw(stats.get("priceToBook"))
    ps = _raw(stats.get("priceToSalesTrailing12Months"))
    target = _raw(financial.get("targetMeanPrice"))
    current = latest_close or _raw(financial.get("currentPrice"))
    upside = target / current - 1 if target and current else None
    parts = []
    if forward_pe: parts.append(np.clip((22 - forward_pe) / 35, -0.5, 0.5))
    elif trailing_pe: parts.append(np.clip((24 - trailing_pe) / 40, -0.45, 0.45))
    if pb: parts.append(np.clip((3.5 - pb) / 8, -0.35, 0.35))
    if ps: parts.append(np.clip((5 - ps) / 12, -0.3, 0.3))
    if upside is not None: parts.append(np.clip(upside * 1.6, -0.45, 0.45))
    score = float(np.clip(np.mean(parts) if parts else 0, -1, 1))
    return ValuationSummary(score, _label(score, "估值偏有利", "估值偏壓抑", "估值中性"), trailing_pe, forward_pe, pb, ps, None, _raw(summary.get("dividendYield")), _raw(financial.get("recommendationMean")), upside, f"PE {forward_pe or trailing_pe or 'N/A'}；P/B {pb or 'N/A'}")


def build_technical_summary(ohlcv: pd.DataFrame) -> TechnicalSummary:
    if ohlcv is None or len(ohlcv) < 80:
        return TechnicalSummary(label="技術資料不足")
    close = pd.to_numeric(ohlcv["Adj Close"], errors="coerce")
    high = pd.to_numeric(ohlcv["High"], errors="coerce")
    low = pd.to_numeric(ohlcv["Low"], errors="coerce")
    latest = float(close.iloc[-1])
    ma20, ma60 = float(close.rolling(20).mean().iloc[-1]), float(close.rolling(60).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    vs20, vs60 = latest / ma20 - 1, latest / ma60 - 1
    vs200 = latest / ma200 - 1 if not np.isnan(ma200) and ma200 else None
    rsi = _rsi(close, 14).iloc[-1]
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    hist = float((macd - macd.ewm(span=9, adjust=False).mean()).iloc[-1] / latest)
    breakout = latest / float(high.rolling(60).max().iloc[-2]) - 1
    score = float(np.clip(np.clip(vs20 * 4, -0.35, 0.35) + np.clip(vs60 * 2.5, -0.25, 0.25) + np.clip(hist * 35, -0.18, 0.18) + (0.12 if 45 <= rsi <= 68 else -0.18 if rsi > 78 else 0.06 if rsi < 32 else 0), -1, 1))
    return TechnicalSummary(score, _label(score, "技術偏多", "技術偏空", "技術中性"), float(rsi), vs20, vs60, vs200, hist, breakout, f"RSI {rsi:.1f}；MA20 {vs20:+.1%}；MA60 {vs60:+.1%}")


def build_liquidity_summary(ohlcv: pd.DataFrame) -> LiquiditySummary:
    if ohlcv is None or len(ohlcv) < 30:
        return LiquiditySummary(label="量能資料不足")
    close = pd.to_numeric(ohlcv["Adj Close"], errors="coerce")
    volume = pd.to_numeric(ohlcv["Volume"], errors="coerce").replace(0, np.nan)
    high = pd.to_numeric(ohlcv["High"], errors="coerce")
    low = pd.to_numeric(ohlcv["Low"], errors="coerce")
    avg_volume = float(volume.tail(20).mean())
    volume_vs = float(volume.iloc[-1] / avg_volume - 1) if avg_volume else None
    dollar_volume = float((close.tail(20) * volume.tail(20)).mean())
    range_20d = float(((high - low) / close).tail(20).mean())
    latest_return = float(close.iloc[-1] / close.iloc[-2] - 1)
    score = 0.0
    if volume_vs is not None:
        score += np.clip(volume_vs / 2.5, -0.25, 0.35) * (1 if latest_return >= 0 else -1)
    if dollar_volume < 30_000_000: score -= 0.18
    if range_20d > 0.055: score -= 0.12
    score = float(np.clip(score, -1, 1))
    return LiquiditySummary(score, _label(score, "量能支持", "量能風險", "量能中性"), volume_vs, avg_volume, dollar_volume, range_20d, f"量比 {volume_vs:+.1%}" if volume_vs is not None else "量比無資料")


def fetch_derivatives_summary(symbol: str) -> DerivativesSummary:
    resolved = resolve_symbol(symbol).symbol
    if "." in resolved or resolved.startswith("^"):
        return DerivativesSummary(label="以大盤衍生品代理", message="台股個股選擇權資料不完整，使用期貨、VIX 與大盤風險因子代理")
    return DerivativesSummary(label="衍生品中性", message="Yahoo 選擇權資料不足時採中性")


def fetch_macro_event_summary(limit: int = 10) -> EventSummary:
    items = _fetch_google_news("CPI PPI Fed 利率 匯率 油價 債券殖利率 台股 美股 when:7d", limit)
    text = " ".join(item.title.lower() for item in items)
    pos = sum(text.count(t.lower()) for t in MACRO_POSITIVE_TERMS)
    neg = sum(text.count(t.lower()) for t in MACRO_NEGATIVE_TERMS)
    score = float(np.clip((pos - neg) / max(pos + neg, 1), -1, 1)) if pos or neg else 0.0
    return EventSummary(score, _label(score, "宏觀偏多", "宏觀偏空", "宏觀中性"), [item.title for item in items[:5]])


def fetch_institutional_summary(symbol: str, *, days: int = 8, avg_volume: float | None = None) -> InstitutionalSummary:
    config = resolve_symbol(symbol)
    if not config.symbol.endswith(".TW"):
        return InstitutionalSummary(message="目前法人籌碼僅支援 TWSE 上市股票")
    rows = _fetch_twse_institutional_rows(config.symbol.removesuffix(".TW"), days)
    if not rows:
        return InstitutionalSummary(message="查無近期三大法人資料")
    recent = rows[:5]
    foreign_5d = float(sum(item.foreign_net for item in recent))
    total_5d = float(sum(item.total_net for item in recent))
    score = 0.7 * np.tanh(foreign_5d / (avg_volume * 3)) + 0.3 * np.tanh(total_5d / (avg_volume * 4)) if avg_volume and avg_volume > 0 else np.tanh(total_5d / (sum(abs(i.total_net) for i in recent) or 1))
    score = float(np.clip(score, -1, 1))
    return InstitutionalSummary(score, _label(score, "法人偏買", "法人偏賣", "法人中性"), len(rows), foreign_5d, total_5d, rows)


def _fetch_twse_institutional_rows(stock_no: str, wanted_days: int) -> list[InstitutionalDay]:
    rows: list[InstitutionalDay] = []
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    today = pd.Timestamp.now(tz="Asia/Taipei").normalize()
    for offset in range(45):
        if len(rows) >= wanted_days:
            break
        day = today - pd.Timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        date_text = day.strftime("%Y%m%d")
        try:
            response = requests.get("https://www.twse.com.tw/rwd/zh/fund/T86", params={"date": date_text, "selectType": "ALLBUT0999", "response": "json"}, timeout=14, headers=REQUEST_HEADERS, verify=False)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue
        if payload.get("stat") != "OK" or not payload.get("data"):
            continue
        fields = [str(field) for field in payload.get("fields", [])]
        for raw_row in payload.get("data", []):
            if raw_row and str(raw_row[0]).strip() == stock_no:
                by_name = {field: raw_row[i] for i, field in enumerate(fields) if i < len(raw_row)}
                def val(*names: str, fallback: int | None = None) -> float:
                    for name in names:
                        if name in by_name: return _parse_number(by_name[name])
                    return _parse_number(raw_row[fallback]) if fallback is not None and fallback < len(raw_row) else 0.0
                rows.append(InstitutionalDay(datetime.strptime(date_text, "%Y%m%d").date().isoformat(), val("外陸資買賣超股數(不含外資自營商)", "外資買賣超股數", fallback=4), val("投信買賣超股數", fallback=10), val("自營商買賣超股數", fallback=11), val("三大法人買賣超股數", fallback=len(raw_row)-1)))
                break
    return rows


def _fetch_yahoo_news(query: str, limit: int) -> list[NewsItem]:
    try:
        response = requests.get("https://query1.finance.yahoo.com/v1/finance/search", params={"q": query, "quotes_count": 0, "news_count": limit}, timeout=12, headers=REQUEST_HEADERS)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    items: list[NewsItem] = []
    for item in payload.get("news", [])[:limit]:
        title = str(item.get("title") or "").strip()
        if title:
            published = datetime.fromtimestamp(int(item["providerPublishTime"])).astimezone().isoformat() if item.get("providerPublishTime") else ""
            items.append(NewsItem(unescape(title), str(item.get("publisher") or ""), str(item.get("link") or ""), published))
    return items


def _fetch_google_news(query: str, limit: int) -> list[NewsItem]:
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        response = requests.get(url, timeout=12, headers=REQUEST_HEADERS)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return []
    items: list[NewsItem] = []
    for item in root.findall("./channel/item")[:limit]:
        title = unescape((item.findtext("title") or "").strip())
        if not title:
            continue
        published = ""
        if item.findtext("pubDate"):
            try: published = parsedate_to_datetime(item.findtext("pubDate")).astimezone().isoformat()
            except Exception: published = ""
        items.append(NewsItem(title, "Google News", item.findtext("link") or "", published))
    return items


def _score_text(text: str) -> float:
    lower = text.lower()
    pos = sum(1 for term in POSITIVE_TERMS if term.lower() in lower)
    neg = sum(1 for term in NEGATIVE_TERMS if term.lower() in lower)
    return float(np.clip((pos - neg) / max(pos + neg, 1), -1, 1)) if pos or neg else 0.0


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _safe_pct(close: pd.Series, days: int) -> float | None:
    if len(close) <= days: return None
    previous = float(close.iloc[-days - 1])
    return None if previous == 0 or np.isnan(previous) else float(close.iloc[-1] / previous - 1)


def _latest_close(ohlcv: pd.DataFrame | None) -> float | None:
    if ohlcv is None or ohlcv.empty or "Adj Close" not in ohlcv: return None
    values = pd.to_numeric(ohlcv["Adj Close"], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else None


def _raw(value: Any) -> float | None:
    if isinstance(value, dict): value = value.get("raw")
    try: number = float(value)
    except Exception: return None
    return number if np.isfinite(number) else None


def _parse_number(value: Any) -> float:
    cleaned = str(value).strip().replace(",", "")
    if cleaned in {"", "--", "X0.00"}: return 0.0
    try: return float(cleaned)
    except ValueError: return 0.0


def _label(score: float, positive: str, negative: str, neutral: str) -> str:
    return positive if score >= 0.18 else negative if score <= -0.18 else neutral


def _safe(fn, fallback, notes: list[str], name: str):
    try:
        return fn()
    except Exception as exc:
        notes.append(f"{name}資料取得失敗：{exc}")
        return fallback
