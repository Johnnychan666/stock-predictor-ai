from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import fetch_ohlcv
from .features import build_features, latest_feature_row, training_matrix
from .research import collect_research_context


@dataclass
class BacktestMetrics:
    samples: int
    accuracy: float
    balanced_accuracy: float
    baseline_accuracy: float
    edge: float
    precision_up: float
    recall_up: float
    roc_auc: float | None
    brier: float | None


@dataclass
class StockPrediction:
    symbol: str
    latest_date: str
    latest_close: float
    direction: str
    probability_up: float
    probability_down: float
    technical_probability_up: float
    technical_probability_down: float
    external_adjustment: float
    confidence: float
    confidence_label: str
    backtest: BacktestMetrics
    target_mode: str = "next_open"
    news_score: float = 0.0
    news_label: str = "無資料"
    news_count: int = 0
    related_score: float = 0.0
    related_label: str = "無資料"
    related_symbols: str = ""
    market_score: float = 0.0
    market_label: str = "無資料"
    event_score: float = 0.0
    event_label: str = "無資料"
    expectation_score: float = 0.0
    expectation_label: str = "無資料"
    valuation_score: float = 0.0
    valuation_label: str = "無資料"
    technical_score: float = 0.0
    technical_label: str = "無資料"
    liquidity_score: float = 0.0
    liquidity_label: str = "無資料"
    institutional_score: float = 0.0
    institutional_label: str = "無資料"
    derivatives_score: float = 0.0
    derivatives_label: str = "無資料"
    macro_score: float = 0.0
    macro_label: str = "無資料"
    foreign_net_buy_5d: float | None = None
    institutional_net_buy_5d: float | None = None
    message: str = ""

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        backtest = row.pop("backtest")
        row.update({f"backtest_{key}": value for key, value in backtest.items()})
        return row


def build_model(random_state: int = 42) -> VotingClassifier:
    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=random_state)),
        ]
    )
    forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=90,
                    max_depth=6,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    gradient = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(n_estimators=70, learning_rate=0.045, max_depth=2, random_state=random_state)),
        ]
    )
    return VotingClassifier(estimators=[("logistic", logistic), ("forest", forest), ("gradient", gradient)], voting="soft", weights=[1.0, 1.1, 0.9])


def _metrics(y_true: np.ndarray, probability_up: np.ndarray) -> BacktestMetrics:
    y_pred = (probability_up >= 0.5).astype(int)
    up_rate = float(np.mean(y_true))
    baseline = max(up_rate, 1 - up_rate)
    roc_auc = float(roc_auc_score(y_true, probability_up)) if len(np.unique(y_true)) == 2 else None
    brier = float(brier_score_loss(y_true, probability_up)) if len(np.unique(y_true)) == 2 else None
    accuracy = float(accuracy_score(y_true, y_pred))
    return BacktestMetrics(
        samples=int(len(y_true)),
        accuracy=accuracy,
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        baseline_accuracy=float(baseline),
        edge=float(accuracy - baseline),
        precision_up=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_up=float(recall_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        brier=brier,
    )


def walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    backtest_days: int = 90,
    min_train_days: int = 260,
    retrain_every: int = 40,
    random_state: int = 42,
) -> BacktestMetrics:
    del retrain_every
    if len(X) < min_train_days + 20:
        min_train_days = max(120, int(len(X) * 0.65))
    start = max(min_train_days, len(X) - min(backtest_days, 160))
    if start >= len(X) - 10:
        start = max(60, int(len(X) * 0.7))

    model = build_model(random_state=random_state)
    model.fit(X.iloc[:start], y.iloc[:start])
    probabilities = model.predict_proba(X.iloc[start:])[:, 1]
    actuals = y.iloc[start:].to_numpy(dtype=int)
    return _metrics(actuals, probabilities)


def confidence_label(confidence: float, backtest: BacktestMetrics) -> str:
    if backtest.samples < 40:
        return "資料不足"
    if confidence >= 0.62 and backtest.edge > 0 and backtest.accuracy >= 0.53:
        return "高"
    if confidence >= 0.56 and backtest.accuracy >= 0.50:
        return "中"
    return "低"


def apply_context_to_prediction(row: dict[str, Any], context: Any | None) -> dict[str, Any]:
    if context is None:
        return row
    technical_probability_up = float(row["technical_probability_up"])
    external_adjustment = float(context.adjustment)
    probability_up = float(np.clip(technical_probability_up + external_adjustment, 0.03, 0.97))
    probability_down = 1 - probability_up
    row.update(
        {
            "direction": "上漲" if probability_up >= 0.5 else "下跌",
            "probability_up": probability_up,
            "probability_down": probability_down,
            "external_adjustment": external_adjustment,
            "confidence": max(probability_up, probability_down),
            "news_score": context.news.score,
            "news_label": context.news.label,
            "news_count": context.news.count,
            "related_score": context.related.score,
            "related_label": context.related.label,
            "related_symbols": ", ".join(context.related.symbols),
            "market_score": context.market.score,
            "market_label": context.market.label,
            "event_score": context.events.score,
            "event_label": context.events.label,
            "expectation_score": context.expectations.score,
            "expectation_label": context.expectations.label,
            "valuation_score": context.valuation.score,
            "valuation_label": context.valuation.label,
            "technical_score": context.technical.score,
            "technical_label": context.technical.label,
            "liquidity_score": context.liquidity.score,
            "liquidity_label": context.liquidity.label,
            "institutional_score": context.institutional.score,
            "institutional_label": context.institutional.label,
            "derivatives_score": context.derivatives.score,
            "derivatives_label": context.derivatives.label,
            "macro_score": context.macro.score,
            "macro_label": context.macro.label,
            "foreign_net_buy_5d": context.institutional.foreign_net_5d,
            "institutional_net_buy_5d": context.institutional.total_net_5d,
        }
    )
    if context.notes:
        row["message"] = "；".join(context.notes[:2])
    return row


def predict_symbol(
    symbol: str,
    years: int = 3,
    market: str = "auto",
    company_name: str = "",
    threshold: float = 0.0,
    backtest_days: int = 90,
    retrain_every: int = 40,
    random_state: int = 42,
    target_mode: str = "next_open",
    use_external_context: bool = True,
    ohlcv: pd.DataFrame | None = None,
) -> StockPrediction:
    ohlcv = fetch_ohlcv(symbol, years=years, market=market) if ohlcv is None else ohlcv.copy()
    feature_frame = build_features(ohlcv, threshold=threshold, target_mode=target_mode)
    X, y, _ = training_matrix(feature_frame)

    if len(X) < 150:
        raise ValueError(f"{ohlcv['Symbol'].iloc[-1]}: 訓練資料太少，請增加歷史年數")

    backtest = walk_forward_backtest(X, y, backtest_days=backtest_days, retrain_every=retrain_every, random_state=random_state)

    final_model = build_model(random_state=random_state)
    final_model.fit(X, y)
    latest_X = latest_feature_row(feature_frame)
    technical_probability_up = float(final_model.predict_proba(latest_X)[0][1])

    latest = feature_frame.dropna(subset=["Adj Close"]).tail(1).iloc[0]
    probability_up = technical_probability_up
    context = collect_research_context(str(latest["Symbol"]), company_name=company_name, ohlcv=ohlcv, news_limit=5, institutional_days=5, related_years=1) if use_external_context else None
    external_adjustment = float(context.adjustment) if context is not None else 0.0
    probability_up = float(np.clip(probability_up + external_adjustment, 0.03, 0.97))
    probability_down = 1 - probability_up
    confidence = max(probability_up, probability_down)
    label = confidence_label(confidence, backtest)

    prediction = StockPrediction(
        symbol=str(latest["Symbol"]),
        latest_date=str(latest["Date"]),
        latest_close=float(latest["Adj Close"]),
        direction="上漲" if probability_up >= 0.5 else "下跌",
        probability_up=probability_up,
        probability_down=probability_down,
        technical_probability_up=technical_probability_up,
        technical_probability_down=1 - technical_probability_up,
        external_adjustment=external_adjustment,
        confidence=confidence,
        confidence_label=label,
        backtest=backtest,
        target_mode=target_mode,
        message="回測沒有打敗基準，請降低信任度" if backtest.edge <= 0 else "",
    )
    row = apply_context_to_prediction(prediction.to_row(), context)
    return StockPrediction(
        symbol=row["symbol"],
        latest_date=row["latest_date"],
        latest_close=row["latest_close"],
        direction=row["direction"],
        probability_up=row["probability_up"],
        probability_down=row["probability_down"],
        technical_probability_up=row["technical_probability_up"],
        technical_probability_down=row["technical_probability_down"],
        external_adjustment=row["external_adjustment"],
        confidence=row["confidence"],
        confidence_label=row["confidence_label"],
        backtest=backtest,
        target_mode=row["target_mode"],
        news_score=row["news_score"],
        news_label=row["news_label"],
        news_count=row["news_count"],
        related_score=row["related_score"],
        related_label=row["related_label"],
        related_symbols=row["related_symbols"],
        market_score=row["market_score"],
        market_label=row["market_label"],
        event_score=row["event_score"],
        event_label=row["event_label"],
        expectation_score=row["expectation_score"],
        expectation_label=row["expectation_label"],
        valuation_score=row["valuation_score"],
        valuation_label=row["valuation_label"],
        technical_score=row["technical_score"],
        technical_label=row["technical_label"],
        liquidity_score=row["liquidity_score"],
        liquidity_label=row["liquidity_label"],
        institutional_score=row["institutional_score"],
        institutional_label=row["institutional_label"],
        derivatives_score=row["derivatives_score"],
        derivatives_label=row["derivatives_label"],
        macro_score=row["macro_score"],
        macro_label=row["macro_label"],
        foreign_net_buy_5d=row["foreign_net_buy_5d"],
        institutional_net_buy_5d=row["institutional_net_buy_5d"],
        message=row["message"],
    )


def predict_many(
    symbols: list[str],
    years: int = 3,
    market: str = "auto",
    threshold: float = 0.0,
    backtest_days: int = 90,
    retrain_every: int = 40,
    random_state: int = 42,
    target_mode: str = "next_open",
    use_external_context: bool = False,
    progress_callback: Any | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, symbol in enumerate(symbols, start=1):
        if progress_callback:
            progress_callback(index, len(symbols), symbol)
        try:
            prediction = predict_symbol(
                symbol=symbol,
                years=years,
                market=market,
                threshold=threshold,
                backtest_days=backtest_days,
                retrain_every=retrain_every,
                random_state=random_state,
                target_mode=target_mode,
                use_external_context=use_external_context,
            )
            rows.append(prediction.to_row())
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(by=["confidence", "backtest_edge"], ascending=[False, False]).reset_index(drop=True)
    return frame, errors
