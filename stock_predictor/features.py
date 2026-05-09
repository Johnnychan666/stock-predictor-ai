from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_COLUMNS = {"Date", "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
TARGET_COLUMNS = {"Target", "NextReturn"}


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def build_features(ohlcv: pd.DataFrame, threshold: float = 0.0, target_mode: str = "next_open") -> pd.DataFrame:
    df = ohlcv.copy().sort_values("Date").reset_index(drop=True)
    close = df["Adj Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    volume = df["Volume"].replace(0, np.nan).astype(float)
    out = df[["Date", "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    for window in [1, 2, 3, 5, 10, 20, 60]:
        out[f"ret_{window}d"] = close.pct_change(window)
    for window in [5, 10, 20, 60]:
        out[f"volatility_{window}d"] = close.pct_change().rolling(window).std()
        out[f"range_{window}d"] = ((high - low) / close).rolling(window).mean()
    for window in [5, 10, 20, 50, 100, 200]:
        sma = close.rolling(window).mean()
        out[f"close_vs_sma_{window}"] = close / sma - 1
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    out["macd"] = macd / close
    out["macd_signal"] = signal / close
    out["macd_hist"] = (macd - signal) / close
    out["rsi_14"] = _rsi(close, 14) / 100
    lowest_14 = low.rolling(14).min()
    highest_14 = high.rolling(14).max()
    out["stoch_k_14"] = (close - lowest_14) / (highest_14 - lowest_14)
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr_14"] = true_range.rolling(14).mean() / close
    out["gap"] = open_ / prev_close - 1
    out["intraday_return"] = close / open_ - 1
    out["high_low_range"] = (high - low) / close
    for window in [5, 20, 60]:
        out[f"volume_vs_avg_{window}"] = volume / volume.rolling(window).mean() - 1
    dates = pd.to_datetime(out["Date"])
    out["day_of_week"] = dates.dt.dayofweek / 4
    out["month"] = (dates.dt.month - 1) / 11
    if target_mode == "next_close":
        next_return = close.shift(-1) / close - 1
    elif target_mode == "next_open":
        next_return = open_.shift(-1) / close - 1
    else:
        raise ValueError("target_mode 必須是 next_open 或 next_close")
    out["NextReturn"] = next_return
    out["Target"] = (next_return > threshold).astype(float)
    out.loc[next_return.isna(), "Target"] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    return [column for column in feature_frame.columns if column not in PRICE_COLUMNS and column not in TARGET_COLUMNS]


def training_matrix(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    columns = feature_columns(feature_frame)
    train = feature_frame.dropna(subset=["Target"]).copy()
    train = train.dropna(subset=columns, how="all")
    return train[columns], train["Target"].astype(int), columns


def latest_feature_row(feature_frame: pd.DataFrame) -> pd.DataFrame:
    columns = feature_columns(feature_frame)
    latest = feature_frame.dropna(subset=columns, how="all").tail(1)
    if latest.empty:
        raise ValueError("找不到可用的最新特徵列")
    return latest[columns]
