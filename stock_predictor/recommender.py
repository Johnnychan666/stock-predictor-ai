from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .data import fetch_ohlcv
from .research import (
    FactorScore,
    build_event_summary,
    build_expectation_summary,
    build_factor_scores,
    build_liquidity_summary,
    build_technical_summary,
    fetch_institutional_summary,
    fetch_macro_event_summary,
    fetch_market_regime_summary,
    fetch_news_sentiment,
    fetch_related_market_summary,
    fetch_valuation_summary,
)


DEFAULT_TAIWAN_UNIVERSE = [
    "2330.TW", "2317.TW", "2454.TW", "2303.TW", "2408.TW", "2344.TW", "2337.TW", "2382.TW",
    "3231.TW", "6669.TW", "3661.TW", "3443.TW", "2379.TW", "3034.TW", "2357.TW", "2308.TW",
    "2412.TW", "2881.TW", "2882.TW", "2891.TW", "2886.TW", "2603.TW", "2609.TW", "2615.TW",
]


@dataclass(frozen=True)
class TaiwanRecommendation:
    symbol: str
    latest_close: float
    latest_date: str
    score: float
    estimated_up_probability: float
    label: str
    reasons: list[str]
    factors: list[FactorScore]

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["factors"] = [asdict(factor) for factor in self.factors]
        return row


def recommend_taiwan_stock(*, limit: int = 16, years: int = 2, universe: list[str] | None = None) -> tuple[TaiwanRecommendation | None, list[dict[str, Any]], list[str]]:
    symbols = (universe or DEFAULT_TAIWAN_UNIVERSE)[: max(1, limit)]
    rows: list[TaiwanRecommendation] = []
    errors: list[str] = []
    try:
        market = fetch_market_regime_summary(years=years)
    except Exception:
        market = None
    try:
        macro = fetch_macro_event_summary(limit=6)
    except Exception:
        macro = None
    for symbol in symbols:
        try:
            frame = fetch_ohlcv(symbol, years=years, market="auto")
            latest_close = float(frame["Adj Close"].iloc[-1])
            avg_volume = float(frame["Volume"].tail(20).mean())
            news = fetch_news_sentiment(symbol, limit=5)
            related = fetch_related_market_summary(symbol, years=years)
            institutional = fetch_institutional_summary(symbol, days=5, avg_volume=avg_volume)
            technical = build_technical_summary(frame)
            liquidity = build_liquidity_summary(frame)
            valuation = fetch_valuation_summary(symbol, latest_close=latest_close)
            events = build_event_summary(news)
            expectations = build_expectation_summary(news, valuation)
            if market is None:
                from .research import MarketRegimeSummary
                market_item = MarketRegimeSummary(label="大盤無資料")
            else:
                market_item = market
            if macro is None:
                from .research import EventSummary
                macro_item = EventSummary(label="宏觀無資料")
            else:
                macro_item = macro
            from .research import DerivativesSummary
            factors = build_factor_scores(
                market=market_item,
                related=related,
                events=events,
                expectations=expectations,
                valuation=valuation,
                technical=technical,
                liquidity=liquidity,
                institutional=institutional,
                derivatives=DerivativesSummary(label="衍生品無資料"),
                macro=macro_item,
            )
            score = float(np.clip(sum(f.score * f.weight for f in factors) + _momentum_boost(frame), -1, 1))
            probability = float(np.clip(0.5 + score * 0.22, 0.05, 0.95))
            label = "高動能候選" if score >= 0.25 else "觀察候選" if score >= 0.05 else "保守候選"
            rows.append(TaiwanRecommendation(symbol, latest_close, str(frame["Date"].iloc[-1]), score, probability, label, _top_reasons(factors), factors))
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    rows = sorted(rows, key=lambda item: (item.score, item.estimated_up_probability), reverse=True)
    return (rows[0] if rows else None), [item.to_row() for item in rows[:10]], errors


def _momentum_boost(frame) -> float:
    close = frame["Adj Close"].astype(float)
    high = frame["High"].astype(float)
    volume = frame["Volume"].astype(float).replace(0, np.nan)
    latest_return = float(close.iloc[-1] / close.iloc[-2] - 1)
    high_60 = float(high.rolling(60).max().iloc[-2]) if len(frame) >= 61 else float(high.max())
    volume_ratio = float(volume.iloc[-1] / volume.tail(20).mean()) if volume.tail(20).mean() else 1.0
    boost = 0.0
    if close.iloc[-1] >= high_60 * 0.995 and latest_return > 0:
        boost += min(0.08, 0.04 * volume_ratio)
    if latest_return > 0.04:
        boost -= 0.03
    return float(np.clip(boost, -0.08, 0.08))


def _top_reasons(factors: list[FactorScore]) -> list[str]:
    ranked = sorted(factors, key=lambda item: abs(item.score * item.weight), reverse=True)
    return [f"{factor.name} {'加分' if factor.score >= 0 else '扣分'}：{factor.label}" for factor in ranked[:4]]
