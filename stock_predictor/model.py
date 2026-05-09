from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import fetch_ohlcv
from .features import build_features, latest_feature_row, training_matrix
from .research import build_related_market_feature_frame, collect_research_context, infer_related_symbols


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
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state)),
        ]
    )
    forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=7,
                    min_samples_leaf=12,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    gradient = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(n_estimators=160, learning_rate=0.035, max_depth=2, random_state=random_state)),
        ]
    )
    return VotingClassifier(estimators=[("logistic", logistic), ("forest", forest), ("gradient", gradient)], voting="soft", weights=[1.0, 1.2, 1.0])


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


def walk_forward_backtest(X: pd.DataFrame, y: pd.Series, backtest_days: int = 252, min_train_days: int = 504, retrain_every: int = 20, random_state: int = 42) -> BacktestMetrics:
    if len(X) < min_train_days + 30:
        min_train_days = max(120, int(len(X) * 0.65))
    start = max(min_train_days, len(X) - backtest_days)
    if start >= len(X) - 5:
        start = max(30, int(len(X) * 0.7))
    base_model = build_model(random_state=random_state)
    current_model = None
    probabilities: list[float] = []
    actuals: list[int] = []
    for offset, idx in enumerate(range(start, len(X))):
        if current_model is None or offset % retrain_every == 0:
            current_model = clone(base_model)
            current_model.fit(X.iloc[:idx], y.iloc[:idx])
        probabilities.append(float(current_model.predict_proba(X.iloc[[idx]])[0][1]))
        actuals.append(int(y.iloc[idx]))
    return _metrics(np.asarray(actuals), np.asarray(probabilities))


def confidence_label(confidence: float, backtest: BacktestMetrics) -> str:
    if backtest.samples < 60:
        return "資料不足"
    if confidence >= 0.62 and backtest.edge > 0 and backtest.accuracy >= 0.53:
        return "高"
    if confidence >= 0.56 and backtest.accuracy >= 0.50:
        return "中"
    return "低"


def predict_symbol(
    symbol: str,
    years: int = 5,
    market: str = "auto",
    company_name: str = "",
    threshold: float = 0.0,
    backtest_days: int = 252,
    retrain_every: int = 20,
    random_state: int = 42,
    target_mode: str = "next_open",
    use_external_context: bool = True,
) -> StockPrediction:
    ohlcv = fetch_ohlcv(symbol, years=years, market=market)
    feature_frame = build_features(ohlcv, threshold=threshold, target_mode=target_mode)
    context_notes: list[str] = []
    if use_external_context:
        related_symbols = infer_related_symbols(str(ohlcv["Symbol"].iloc[-1]))
        related_features = build_related_market_feature_frame(ohlcv, related_symbols=related_symbols, years=years)
        if len(related_features.columns) > 1:
            feature_frame = feature_frame.merge(related_features, on="Date", how="left")
        else:
            context_notes.append("關聯市場特徵無可用資料")
    X, y, _ = training_matrix(feature_frame)
    if len(X) < 180:
        raise ValueError(f"{ohlcv['Symbol'].iloc[-1]}: 訓練資料太少，請增加歷史年數")
    backtest = walk_forward_backtest(X, y, backtest_days=backtest_days, retrain_every=retrain_every, random_state=random_state)
    final_model = build_model(random_state=random_state)
    final_model.fit(X, y)
    latest_X = latest_feature_row(feature_frame)
    technical_probability_up = float(final_model.predict_proba(latest_X)[0][1])

    context_values = {
        "external_adjustment": 0.0,
        "news_score": 0.0,
        "news_label": "無資料",
        "news_count": 0,
        "related_score": 0.0,
        "related_label": "無資料",
        "related_symbols": "",
        "market_score": 0.0,
        "market_label": "無資料",
        "event_score": 0.0,
        "event_label": "無資料",
        "expectation_score": 0.0,
        "expectation_label": "無資料",
        "valuation_score": 0.0,
        "valuation_label": "無資料",
        "technical_score": 0.0,
        "technical_label": "無資料",
        "liquidity_score": 0.0,
        "liquidity_label": "無資料",
        "institutional_score": 0.0,
        "institutional_label": "無資料",
        "derivatives_score": 0.0,
        "derivatives_label": "無資料",
        "macro_score": 0.0,
        "macro_label": "無資料",
        "foreign_net_buy_5d": None,
        "institutional_net_buy_5d": None,
    }
    if use_external_context:
        context = collect_research_context(str(ohlcv["Symbol"].iloc[-1]), company_name=company_name, ohlcv=ohlcv)
        context_values.update(
            {
                "external_adjustment": context.adjustment,
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
        context_notes.extend(context.notes)

    probability_up = float(np.clip(technical_probability_up + context_values["external_adjustment"], 0.03, 0.97))
    probability_down = 1 - probability_up
    confidence = max(probability_up, probability_down)
    direction = "上漲" if probability_up >= 0.5 else "下跌"
    latest = feature_frame.dropna(subset=["Adj Close"]).tail(1).iloc[0]
    label = confidence_label(confidence, backtest)
    message = ""
    if backtest.edge <= 0:
        message = "回測沒有打敗基準，請降低信任度"
    elif label == "低":
        message = "模型信心偏低，建議只作觀察"
    if context_notes:
        note_text = "；".join(context_notes[:2])
        message = f"{message}；{note_text}" if message else note_text
    return StockPrediction(
        symbol=str(latest["Symbol"]),
        latest_date=str(latest["Date"]),
        latest_close=float(latest["Adj Close"]),
        direction=direction,
        probability_up=probability_up,
        probability_down=probability_down,
        technical_probability_up=technical_probability_up,
        technical_probability_down=1 - technical_probability_up,
        confidence=confidence,
        confidence_label=label,
        backtest=backtest,
        target_mode=target_mode,
        message=message,
        **context_values,
    )


def predict_many(
    symbols: list[str],
    years: int = 5,
    market: str = "auto",
    threshold: float = 0.0,
    backtest_days: int = 252,
    retrain_every: int = 20,
    random_state: int = 42,
    target_mode: str = "next_open",
    use_external_context: bool = True,
    progress_callback: Any | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, symbol in enumerate(symbols, start=1):
        if progress_callback:
            progress_callback(index, len(symbols), symbol)
        try:
            prediction = predict_symbol(symbol, years, market, "", threshold, backtest_days, retrain_every, random_state, target_mode, use_external_context)
            rows.append(prediction.to_row())
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(by=["confidence_label", "confidence", "backtest_edge"], ascending=[True, False, False]).reset_index(drop=True)
    return frame, errors
