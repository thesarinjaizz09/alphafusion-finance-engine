#!/usr/bin/env python3
"""
StockForecaster - single-file production CLI app (LSTM patch + epoch logging)

Patched v5: sanitizes datetime indices and timezone-aware datetimes to avoid
- statsmodels FutureWarning about unsupported index
- Prophet tz-aware -> tz-naive error

Usage:
  python stock_forecaster.py predict --ticker AAPL --timeframe 1d --lstm

Notes:
- LSTM training prints epoch logs to the CLI via Rich.
- TensorFlow, XGBoost, Prophet are optional; app runs without them.
- Updated it with strategies
"""

from __future__ import annotations
import os
import gc
import json
import math
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Rich UI
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.traceback import install as rich_install
    rich_install()
    console = Console()
except Exception:
    console = None
    def rprint(*args, **kwargs): print(*args, **kwargs)

# CLI: Typer fallback to argparse
USE_TYPER = True
try:
    import typer
    app = typer.Typer(add_completion=False)
except Exception:
    USE_TYPER = False
    import argparse

# External optional libs
try:
    import yfinance as yf
except Exception:
    raise RuntimeError("yfinance is required. Install: pip install yfinance")

# Statsmodels SARIMAX
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

# Prophet (optional)
try:
    from prophet import Prophet
except Exception:
    Prophet = None

# XGBoost (optional)
try:
    import xgboost as xgb
except Exception:
    xgb = None

# TensorFlow/Keras (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, BatchNormalization, Attention
    from tensorflow.keras.callbacks import EarlyStopping, Callback
except Exception:
    tf = None
    
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import optuna
except ImportError:
    optuna = None

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("StockForecaster")

import warnings

# statsmodels convergence exception (optional — only import if available)
try:
    # newer statsmodels exposes ConvergenceWarning here
    from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning
except Exception:
    SMConvergenceWarning = None

# Quiet noisy warnings from statsmodels during SARIMAX parameter search;
# keep other warnings visible. Adjust if you want to see them.
warnings.filterwarnings("ignore", message="A date index has been provided, but it has no associated frequency information")
if SMConvergenceWarning is not None:
    warnings.filterwarnings("ignore", category=SMConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.simplefilter(action='ignore', category=FutureWarning)




# -------------------------
# Config & helpers
# -------------------------
@dataclass
class CLIConfig:
    ticker: str
    timeframe: str = "1d"
    candles: int = 360
    val_horizon: int = 36
    forecast_horizon: int = 4
    use_prophet: bool = True
    use_xgboost: bool = True
    use_lstm: bool = True
    use_cnn_lstm: bool = True
    use_attention_lstm: bool = True
    use_random_forest: bool = True
    use_lightgbm: bool = True
    lstm_epochs: int = 20
    lstm_batch: int = 32
    output_dir: str = "outputs"
    quiet: bool = False

PERIOD_FOR_INTERVAL = {
    "1m": ("7d", "1m"),
    "5m": ("60d", "5m"),
    "15m": ("60d", "15m"),
    "30m": ("60d", "30m"),
    "1h": ("2y", "1h"),
    "1d": ("5y", "1d"),
}


# -------------------------
# Datetime sanitizers
# -------------------------
def _ensure_datetime_index_tz_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a tz-naive DatetimeIndex. If idx is tz-aware, convert to UTC then drop tz.
    If idx is already naive, return as DatetimeIndex.
    """
    idx = pd.to_datetime(idx)
    tz = getattr(idx, 'tz', None)
    if tz is not None:
        # convert to UTC then drop tz info (safe canonicalization)
        try:
            idx = idx.tz_convert('UTC').tz_localize(None)
        except Exception:
            # fallback: remove tz info directly if tz_convert fails
            try:
                idx = idx.tz_localize(None)
            except Exception:
                idx = pd.to_datetime(idx).tz_localize(None)
    return pd.DatetimeIndex(idx)

def _ensure_series_has_datetime_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    try:
        s.index = _ensure_datetime_index_tz_naive(s.index)
    except Exception:
        # last-resort: coerce to range index converted to datetime by position
        s.index = pd.DatetimeIndex(pd.to_datetime(range(len(s))))
    return s


# -------------------------
# Indicators (pandas/numpy)
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, window: int =20, n_std: int = 2):
    ma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0)
    upper = ma + n_std * std
    lower = ma - n_std * std
    return upper, lower

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window:int=14):
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_v = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr_v

def obv(close: pd.Series, volume: pd.Series):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["EMA_12"] = ema(close, 12)
    df["EMA_26"] = ema(close, 26)
    df["SMA_20"] = sma(close, 20)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(close)
    df["RSI_14"] = rsi(close, 14)
    df["BB_UPPER"], df["BB_LOWER"] = bollinger_bands(close, 20, 2)
    df["BB_PCT"] = (close - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"] + 1e-12)
    df["ATR_14"] = atr(high, low, close, 14)
    df["OBV"] = obv(close, vol)
    df["RETURNS"] = close.pct_change().fillna(0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return df


# -------------------------
# Strategies Calculator
# -------------------------
def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume weighted average price for each bar."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    return pv.cumsum() / df["Volume"].cumsum()

def zscore(series: pd.Series, window: int = 20):
    """Rolling z-score relative to moving average and std."""
    m = series.rolling(window=window, min_periods=1).mean()
    s = series.rolling(window=window, min_periods=1).std(ddof=0).replace(0, np.nan)
    return (series - m) / (s + 1e-12)

def volume_spike(volume: pd.Series, window: int = 20, mult: float = 2.0):
    """Return boolean series where volume > mult * rolling_mean(volume)."""
    mv = volume.rolling(window=window, min_periods=1).mean()
    return volume > (mv * mult)

def fibonacci_levels(df: pd.DataFrame) -> Dict[str, float]:
    """Compute Fibonacci retracement levels for the last major swing in df['Close'].
       Returns dictionary with levels (0.0 top -> 1.0 bottom style).
       This is a simple heuristic using last local max/min over lookback.
    """
    lookback = min(len(df), 200)
    s = df["Close"].iloc[-lookback:]
    hi = s.max()
    lo = s.min()
    diff = hi - lo
    if diff == 0:
        return {}
    levels = {
        "0%": hi,
        "23.6%": hi - 0.236 * diff,
        "38.2%": hi - 0.382 * diff,
        "50%": hi - 0.5 * diff,
        "61.8%": hi - 0.618 * diff,
        "100%": lo
    }
    return levels

def is_bullish_engulfing(df: pd.DataFrame, idx: int) -> bool:
    """Detect basic bullish engulfing at index idx (requires idx>=1)."""
    if idx < 1 or idx >= len(df):
        return False
    prev_open = df["Open"].iat[idx-1]
    prev_close = df["Close"].iat[idx-1]
    cur_open = df["Open"].iat[idx]
    cur_close = df["Close"].iat[idx]
    return (prev_close < prev_open) and (cur_close > cur_open) and (cur_close > prev_open) and (cur_open < prev_close)

def is_bearish_engulfing(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1 or idx >= len(df):
        return False
    prev_open = df["Open"].iat[idx-1]
    prev_close = df["Close"].iat[idx-1]
    cur_open = df["Open"].iat[idx]
    cur_close = df["Close"].iat[idx]
    return (prev_close > prev_open) and (cur_close < cur_open) and (cur_close < prev_open) and (cur_open > prev_close)

def is_pin_bar(df: pd.DataFrame, idx: int, tail_ratio: float = 2.0) -> bool:
    """Basic pin bar: long wick relative to body."""
    if idx < 0 or idx >= len(df):
        return False
    high = df["High"].iat[idx]
    low = df["Low"].iat[idx]
    open_ = df["Open"].iat[idx]
    close = df["Close"].iat[idx]
    body = abs(close - open_)
    tail_top = high - max(open_, close)
    tail_bot = min(open_, close) - low
    # consider tail as the larger of the two wicks
    tail = max(tail_top, tail_bot)
    if body <= 0:
        return False
    return tail / (body + 1e-12) >= tail_ratio

def detect_market_regime(df: pd.DataFrame, window: int = 20) -> str:
    """Return 'low', 'medium', or 'high' volatility regime based on rolling std of returns and ATR."""
    ret = df["Close"].pct_change().rolling(window=window, min_periods=1).std()
    atr_v = atr(df["High"], df["Low"], df["Close"], window=window)
    # normalize values
    vol = ret.fillna(0).iloc[-1]
    atr_val = atr_v.fillna(0).iloc[-1]
    # thresholds heuristic; tweak as needed
    if vol < 0.005 and atr_val < (0.01 * df["Close"].iloc[-1]):
        return "low"
    elif vol < 0.02:
        return "medium"
    else:
        return "high"


# -------------------------
# Data fetcher
# -------------------------
def fetch_last_candles(ticker: str, timeframe: str, candles: int = 360) -> pd.DataFrame:
    if timeframe not in PERIOD_FOR_INTERVAL:
        raise ValueError(f"timeframe must be one of {list(PERIOD_FOR_INTERVAL.keys())}")
    period, interval = PERIOD_FOR_INTERVAL[timeframe]
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned from yfinance for {ticker} ({timeframe})")
    df = df[["Open","High","Low","Close","Volume"]].dropna()

    # --- SANITIZE INDEX: ensure tz-naive DatetimeIndex to avoid statsmodels/prophet issues ---
    try:
        df.index = _ensure_datetime_index_tz_naive(df.index)
    except Exception:
        # fallback: attempt to coerce and continue
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            logger.warning("Failed to fully sanitize index timezone; proceeding with available index.")

    if len(df) < candles:
        logger.warning(f"Only {len(df)} candles available for {timeframe}. Requested {candles}. Using available.")
    return df.tail(candles)


# -------------------------
# Utilities: metrics & weights
# -------------------------
def safe_mape(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-8
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0

def inverse_error_weights(y_true: np.ndarray, preds_dict: Dict[str, np.ndarray]):
    errs = {}
    for k, v in preds_dict.items():
        try:
            errs[k] = safe_mape(y_true, v)
        except Exception:
            errs[k] = float("inf")
    eps = 1e-8
    weights = {}
    denom = 0.0
    for k, e in errs.items():
        w = 0.0 if not np.isfinite(e) else 1.0 / (e + eps)
        weights[k] = w
        denom += w
    if denom <= 0:
        # all errors infinite / invalid -> fallback to equal weights
        n = len(preds_dict)
        return {k: 1.0 / n for k in preds_dict.keys()}
    return {k: float(v / denom) for k, v in weights.items()}


# -------------------------
# Models
# -------------------------
def ema_baseline(series: pd.Series, span:int, val_horizon: int):
    s = series.dropna().astype(float)
    ema_s = ema(s, span)
    preds = ema_s.shift(1).values[-val_horizon:]
    next_step = float(ema_s.iloc[-1])
    return np.array(preds, dtype=float), float(next_step)

def sarimax_small_grid_forecast(train_series: pd.Series, val_horizon: int, exog: Optional[pd.DataFrame]=None,
                                max_p=2, max_d=1, max_q=2, seasonal=False, m=1, max_fits=20, maxiter=100):
    """
    Improved SARIMAX grid search with:
    - freq inference to avoid date/freq warnings where possible
    - limited number of candidate fits (max_fits)
    - bounded optimizer iterations (maxiter)
    - per-fit suppression of ConvergenceWarning so logs are not flooded
    """
    if SARIMAX is None:
        raise RuntimeError("statsmodels SARIMAX not available")

    # Ensure datetime index tz-naive
    train_series = _ensure_series_has_datetime_index(train_series)

    # Try to infer and set frequency if it looks sensible (helps statsmodels)
    try:
        freq = pd.infer_freq(train_series.index)
        if freq is not None:
            # make a copy with explicit freq to help forecasting internals
            try:
                train_series = train_series.asfreq(freq)
            except Exception:
                # ignore if asfreq fails
                pass
    except Exception:
        pass

    # Reindex exog to train index if provided
    if exog is not None:
        try:
            exog = exog.reindex(train_series.index).fillna(method='ffill').fillna(method='bfill')
        except Exception:
            exog = exog.loc[train_series.index] if all(i in exog.index for i in train_series.index) else exog

    best_aic = np.inf
    best_model = None
    tried = 0

    # Candidate order list (small grid)
    for p in range(0, max_p+1):
        for d in range(0, max_d+1):
            for q in range(0, max_q+1):
                if p == 0 and d == 0 and q == 0:
                    continue
                if tried >= max_fits:
                    break
                try:
                    # build model
                    if seasonal:
                        model = SARIMAX(train_series, exog=exog, order=(p,d,q), seasonal_order=(1,0,0,m),
                                        enforce_stationarity=False, enforce_invertibility=False)
                    else:
                        model = SARIMAX(train_series, exog=exog, order=(p,d,q),
                                        enforce_stationarity=False, enforce_invertibility=False)

                    # Fit with bounded iterations and silence per-fit convergence warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # pass maxiter to fit; older statsmodels accept 'maxiter' kwarg
                        try:
                            res = model.fit(disp=False, maxiter=maxiter)
                        except TypeError:
                            # some older versions expect method and maxiter inside method args
                            res = model.fit(disp=False)
                    tried += 1

                    # candidate evaluation
                    if hasattr(res, "aic") and res.aic < best_aic and np.isfinite(res.aic):
                        best_aic = res.aic
                        best_model = res
                except Exception:
                    # ignore this candidate and continue
                    continue
            if tried >= max_fits:
                break
        if tried >= max_fits:
            break

    if best_model is None:
        # fallback: persistence forecast (last value repeated)
        preds = np.repeat(float(train_series.iloc[-1]), val_horizon)
        return np.array(preds, dtype=float), None

    # Prepare exog for forecasting (repeat last row) if exog exists
    try:
        if exog is not None and exog.shape[0] >= 1:
            last_exog = exog.iloc[-1:].values
            exog_fore = np.repeat(last_exog, val_horizon, axis=0)
            fc = best_model.get_forecast(steps=val_horizon, exog=exog_fore).predicted_mean
        else:
            fc = best_model.get_forecast(steps=val_horizon).predicted_mean
    except Exception:
        # last-resort predict using model's dynamic or fallback to last value
        try:
            fc = best_model.get_forecast(steps=val_horizon).predicted_mean
        except Exception:
            fc = np.repeat(float(train_series.iloc[-1]), val_horizon)

    return np.array(fc, dtype=float), best_model.params if hasattr(best_model, "params") else None

def prophet_forecast(train_df: pd.Series, val_horizon: int, freq: str = 'D'):
    if Prophet is None:
        raise RuntimeError("Prophet not installed")

    # Ensure tz-naive datetime index
    idx = pd.to_datetime(train_df.index)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_convert('UTC').tz_localize(None) if idx.tz else idx.tz_localize(None)

    dfp = pd.DataFrame({"ds": idx, "y": train_df.values.astype(float)})
    dfp = dfp.sort_values('ds').reset_index(drop=True)

    # Optionally interpolate missing dates
    dfp = dfp.set_index('ds').asfreq(freq).interpolate().reset_index()

    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(dfp)

    future = m.make_future_dataframe(periods=val_horizon, freq=freq)
    fc = m.predict(future)
    preds = fc["yhat"].values[-val_horizon:]
    return np.array(preds, dtype=float), m

def xgboost_forecast(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_future: Optional[np.ndarray] = None,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Train XGBoost model and return validation + forecast predictions.
    
    Args:
        X_train, y_train: Training data
        X_val: Optional validation set for backtesting
        X_future: Optional future features for forward forecast horizon
    
    Returns:
        (val_preds, forecast_preds, model)
    """
    if xgb is None:
        raise RuntimeError("xgboost not installed")

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        n_jobs=n_jobs,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    val_preds, forecast_preds = None, None

    if X_val is not None:
        val_preds = model.predict(X_val)

    if X_future is not None:
        forecast_preds = model.predict(X_future)

    return (
        np.array(val_preds, dtype=float) if val_preds is not None else None,
        np.array(forecast_preds, dtype=float) if forecast_preds is not None else None,
        model,
    )

def build_small_lstm(input_shape, units=64, dropout_rate=0.2, recurrent_dropout=0.1):
    """
    Build a multi-feature LSTM model for price prediction.

    Args:
        input_shape: tuple, (timesteps, features)
        units: int, LSTM units for first layer
        dropout_rate: float, dropout after LSTM layers
        recurrent_dropout: float, recurrent dropout in LSTM layers

    Returns:
        Compiled Keras model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # First LSTM layer (returns sequences for stacking)
    model.add(LSTM(units, activation="tanh", return_sequences=True, recurrent_dropout=recurrent_dropout))

    # Second LSTM layer (captures longer dependencies)
    model.add(LSTM(units//2, activation="tanh", recurrent_dropout=recurrent_dropout))

    # Dropout for regularization
    model.add(Dropout(dropout_rate))

    # Output layer predicting next Close price
    model.add(Dense(1, activation='linear'))

    # Compile
    model.compile(optimizer='adam', loss='mse')

    return model

def build_cnn_lstm(input_shape, conv_filters=32, conv_kernel=3, lstm_units=64, dropout_rate=0.2):
    """
    Build CNN-LSTM model for multi-feature price prediction.
    """
    inp = Input(shape=input_shape)
    x = Conv1D(filters=conv_filters, kernel_size=conv_kernel, activation="relu", padding="causal")(inp)
    x = BatchNormalization()(x)
    x = Conv1D(filters=conv_filters*2, kernel_size=conv_kernel, activation="relu", padding="causal")(x)
    x = BatchNormalization()(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = LSTM(lstm_units//2)(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

def build_attention_lstm(input_shape, lstm_units=64, dropout_rate=0.2):
    """
    Build Attention-LSTM model for multi-feature price prediction.
    """
    inp = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=True)(inp)
    x = Attention()([x, x])  # self-attention
    x = LSTM(lstm_units//2)(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


# -------------------------
# Walk-forward CV helpers
# -------------------------
def walk_forward_splits(n: int, initial_train: int, step: int, val_horizon: int):
    """Yield train/val index splits for walk-forward CV."""
    train_end = initial_train
    while train_end + val_horizon <= n:
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, train_end + val_horizon)
        yield train_idx, val_idx
        train_end += step

# -------------------------
# RandomForest & LightGBM
# -------------------------
def train_random_forest_v51(X_train, y_train, X_val=None, params: dict = None):
    params = params or {}
    model = RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=params.get("max_depth", None),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds_val = model.predict(X_val) if X_val is not None else None
    return preds_val, model

def train_lightgbm_v51(X_train, y_train, X_val=None, params: dict = None, quantile=None):
    if lgb is None:
        raise RuntimeError("LightGBM not installed")
    params = params or {}
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_train[-len(X_val):]) if X_val is not None else None
    params = {**params, "objective": "quantile" if quantile else "regression"}
    if quantile: params["alpha"] = quantile

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=int(params.get("n_estimators", 500)),
        valid_sets=[val_data] if val_data else None,
        verbose_eval=False,
    )
    preds_val = booster.predict(X_val) if X_val is not None else None
    return preds_val, booster

# -------------------------
# CNN-LSTM & Attention-LSTM
# -------------------------
def build_cnn_lstm_v51(input_shape, conv_filters=32, conv_kernel=3, lstm_units=64, dropout=0.2):
    if tf is None:
        raise RuntimeError("TensorFlow not available")
    inp = Input(shape=input_shape)
    x = Conv1D(filters=conv_filters, kernel_size=conv_kernel, activation="relu", padding="causal")(inp)
    x = BatchNormalization()(x)
    x = Conv1D(filters=conv_filters * 2, kernel_size=conv_kernel, activation="relu", padding="causal")(x)
    x = BatchNormalization()(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Attention()([x, x])  # self-attention
    x = LSTM(lstm_units // 2)(x)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

def build_attention_lstm_v51(input_shape, lstm_units=64, dropout=0.2):
    if tf is None:
        raise RuntimeError("TensorFlow not available")
    inp = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=True)(inp)
    x = Attention()([x, x])
    x = LSTM(lstm_units // 2)(x)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------
# Stacking (meta-learner)
# -------------------------
def generate_oof_predictions_v51(models, X, y, initial_train, step, val_horizon):
    n = len(y)
    oof = {i: np.full(n, np.nan) for i in range(len(models))}
    for tr, va in walk_forward_splits(n, initial_train, step, val_horizon):
        for i, fn in enumerate(models):
            try:
                preds, _ = fn(X[tr], y[tr], X[va])
            except Exception:
                preds = np.repeat(y[tr][-1], len(va))
            oof[i][va] = preds
    return oof

def train_meta_learner_v51(oof_preds: Dict[int, np.ndarray], y: np.ndarray):
    stack = np.vstack([oof_preds[k] for k in sorted(oof_preds.keys())]).T
    mask = ~np.isnan(stack).any(axis=1)
    X_meta, y_meta = stack[mask], y[mask]
    meta = Ridge(alpha=1.0)
    meta.fit(X_meta, y_meta)
    return meta

# -------------------------
# Simple backtest
# -------------------------
def simple_backtest_v51(signals: pd.Series, prices: pd.Series,
                        init_capital=10000.0, fee=0.0005, slippage=0.0005):
    df = pd.DataFrame({"price": prices, "signal": signals}).copy()
    df["position"] = df["signal"].shift(1).fillna(0)
    df["returns"] = df["price"].pct_change().fillna(0)
    df["strategy_ret"] = df["position"] * df["returns"]
    trades = df["position"].diff().abs() > 0
    df["strategy_ret_adj"] = df["strategy_ret"] - trades * (fee + slippage)
    df["equity"] = init_capital * (1 + df["strategy_ret_adj"]).cumprod()

    total_return = df["equity"].iloc[-1] / init_capital - 1
    sharpe = df["strategy_ret_adj"].mean() / (df["strategy_ret_adj"].std() + 1e-9) * np.sqrt(252)
    max_dd = (df["equity"].cummax() - df["equity"]).max()

    summary = {"final_equity": df["equity"].iloc[-1],
               "total_return": total_return,
               "sharpe": sharpe,
               "max_drawdown": max_dd}
    return df, summary



# -------------------------
# Strategy detectors (NEW) - these are additive and do not change model logic
# -------------------------
def detect_breakout(df: pd.DataFrame, target_col: str = "Close", vol_mult: float = 2.0):
    """Detect breakout on target_col using Bollinger Bands + volume spike + recent higher highs."""
    out = {"signal": None, "reason": None, "vol_spike": False}
    try:
        upper, lower = bollinger_bands(df[target_col], window=20, n_std=2)
        latest = df[target_col].iat[-1]
        prev = df[target_col].iat[-2]
        # volume spike
        vol_flag = volume_spike(df["Volume"], window=20, mult=vol_mult).iat[-1]
        out["vol_spike"] = bool(vol_flag)
        # breakout above upper band
        if prev <= upper.iat[-2] and latest > upper.iat[-1]:
            out["signal"] = "bullish"
            out["reason"] = "price above upper Bollinger + vol spike" if vol_flag else "price above upper Bollinger"
        elif prev >= lower.iat[-2] and latest < lower.iat[-1]:
            out["signal"] = "bearish"
            out["reason"] = "price below lower Bollinger + vol spike" if vol_flag else "price below lower Bollinger"
        else:
            out["signal"] = "none"
            out["reason"] = "no band breakout"
    except Exception as e:
        out["signal"] = None
        out["reason"] = f"error: {e}"
    return out

def detect_mean_reversion(df: pd.DataFrame, target_col: str = "Close"):
    """Mean reversion signals: RSI extremes, BB touches, z-score extremes."""
    out = {"signal": None, "reasons": []}
    try:
        r = rsi(df[target_col], 14).iat[-1]
        z = zscore(df[target_col], window=20).iat[-1]
        upper, lower = bollinger_bands(df[target_col], window=20, n_std=2)
        last = df[target_col].iat[-1]
        if r is not None:
            if r > 70:
                out["reasons"].append("RSI_overbought")
            elif r < 30:
                out["reasons"].append("RSI_oversold")
        if last >= upper.iat[-1]:
            out["reasons"].append("BB_upper_touch")
        if last <= lower.iat[-1]:
            out["reasons"].append("BB_lower_touch")
        if z is not None:
            if z > 2:
                out["reasons"].append("Zscore_high")
            elif z < -2:
                out["reasons"].append("Zscore_low")
        if len(out["reasons"]) == 0:
            out["signal"] = "none"
        else:
            # heuristic: if RSI oversold or zscore low or lower band -> buy (mean revert)
            if any(x in out["reasons"] for x in ["RSI_oversold", "BB_lower_touch", "Zscore_low"]):
                out["signal"] = "buy_revert"
            elif any(x in out["reasons"] for x in ["RSI_overbought", "BB_upper_touch", "Zscore_high"]):
                out["signal"] = "sell_revert"
            else:
                out["signal"] = "watch"
    except Exception as e:
        out["signal"] = None
        out["reasons"].append(f"error:{e}")
    return out

def detect_fibonacci_pullback(df: pd.DataFrame, target_col: str = "Close"):
    """Compute fib levels and check if price is near a fib retracement (38.2/50/61.8)."""
    out = {"levels": {}, "near_level": None, "distance": None}
    try:
        levels = fibonacci_levels(df)
        out["levels"] = levels
        if not levels:
            return out
        last = df[target_col].iat[-1]
        # compute nearest level
        diffs = {k: abs(last - v) for k, v in levels.items()}
        nearest = min(diffs.items(), key=lambda x: x[1])
        out["near_level"] = nearest[0]
        out["distance"] = float(nearest[1])
    except Exception as e:
        out["levels"] = {}
        out["near_level"] = None
        out["distance"] = None
        logger.debug(f"Fibonacci detection error: {e}")
    return out

def detect_price_action(df: pd.DataFrame, idx_offset: int = 0):
    """Scan recent bars for simple price-action patterns; returns latest findings."""
    out = {"bullish_engulfing": False, "bearish_engulfing": False, "pin_bar": False}
    try:
        n = len(df)
        idx = n - 1 - idx_offset
        if idx >= 1:
            out["bullish_engulfing"] = is_bullish_engulfing(df, idx)
            out["bearish_engulfing"] = is_bearish_engulfing(df, idx)
            out["pin_bar"] = is_pin_bar(df, idx)
    except Exception:
        pass
    return out

def detect_swing_trade(df: pd.DataFrame, target_col: str = "Close"):
    """Basic swing trade signal: EMA cross + MACD direction + RSI confirmation."""
    out = {"signal": None, "reasons": []}
    try:
        ema12 = ema(df[target_col], 12)
        ema26 = ema(df[target_col], 26)
        macd_line, macd_signal, _ = macd(df[target_col])
        r = rsi(df[target_col], 14)
        # last
        if ema12.iat[-1] > ema26.iat[-1] and ema12.iat[-2] <= ema26.iat[-2]:
            # bullish crossover
            if macd_line.iat[-1] > macd_signal.iat[-1] and r.iat[-1] < 70:
                out["signal"] = "bullish"
                out["reasons"].append("EMA_cross + MACD > signal")
        elif ema12.iat[-1] < ema26.iat[-1] and ema12.iat[-2] >= ema26.iat[-2]:
            if macd_line.iat[-1] < macd_signal.iat[-1] and r.iat[-1] > 30:
                out["signal"] = "bearish"
                out["reasons"].append("EMA_cross_down + MACD < signal")
        else:
            out["signal"] = "none"
    except Exception:
        out["signal"] = None
    return out

def detect_scalping_opportunity(df: pd.DataFrame, target_col: str = "Close"):
    """Scalping helper: uses VWAP and short-term band width. Best on intraday data."""
    out = {"signal": None, "reason": None}
    try:
        if "VWAP" not in df.columns:
            df["VWAP"] = vwap(df)
        last = df[target_col].iat[-1]
        vwap_last = df["VWAP"].iat[-1]
        # micro scalping heuristic: price deviates from VWAP slightly with volume
        if last > vwap_last and df["Volume"].iat[-1] > df["Volume"].rolling(20, min_periods=1).mean().iat[-1]:
            out["signal"] = "long_momentum"
            out["reason"] = "price_above_vwap_with_volume"
        elif last < vwap_last and df["Volume"].iat[-1] > df["Volume"].rolling(20, min_periods=1).mean().iat[-1]:
            out["signal"] = "short_momentum"
            out["reason"] = "price_below_vwap_with_volume"
    except Exception as e:
        out["signal"] = None
    return out

def pairs_trade_signals(base_ticker: str, alt_ticker: str, timeframe: str = "1d", candles: int = 360):
    """
    Pairs trading helper: fetches both tickers, computes spread, correlation, z-score of spread.
    Returns dict with correlation, zscore, signal if zscore beyond thresholds.
    NOTE: this is optional and requires network calls (yfinance).
    """
    out = {"correlation": None, "zscore": None, "signal": None}
    try:
        df1 = fetch_last_candles(base_ticker, timeframe, candles)
        df2 = fetch_last_candles(alt_ticker, timeframe, candles)
        # align
        series1, series2 = df1["Close"].align(df2["Close"], join="inner")
        if len(series1) < 30:
            return out
        corr = series1.corr(series2)
        spread = series1 - series2
        zs = (spread - spread.mean()) / (spread.std() + 1e-12)
        z_latest = zs.iloc[-1]
        out["correlation"] = float(corr)
        out["zscore"] = float(z_latest)
        if z_latest > 2:
            out["signal"] = "short_spread"
        elif z_latest < -2:
            out["signal"] = "long_spread"
        else:
            out["signal"] = "none"
    except Exception as e:
        out["signal"] = None
    return out

def fetch_news_sentiment_stub(ticker: str):
    """Stub: if you add a news API (e.g., NewsAPI, Finnhub, AlphaVantage), plug it here.
       Returns basic structure with 'sentiment' in {'positive','neutral','negative'} or None.
       For now it returns None and logs info.
    """
    logger.debug(f"No news API configured; skipping sentiment for {ticker}")
    return None

def fetch_options_flow_stub(ticker: str):
    """Stub: show how to fetch options summary from yfinance; not real 'flow' analysis.
       Returns dict with available expiries and a simple OI summary if present.
    """
    out = {}
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        out["expiries"] = list(exps) if exps is not None else []
        # quick sample: pick nearest expiry and compute total call/put OI from chain (if available)
        if out["expiries"]:
            chain = tk.option_chain(out["expiries"][0])
            calls = chain.calls
            puts = chain.puts
            out["calls_oi_sum"] = int(calls["openInterest"].sum()) if "openInterest" in calls else None
            out["puts_oi_sum"] = int(puts["openInterest"].sum()) if "openInterest" in puts else None
    except Exception as e:
        logger.debug(f"Options scan failed (yfinance): {e}")
    return out


# -------------------------
# Rich Keras callback for epoch logging & ml training
# -------------------------
if tf is not None:
    class RichKerasLogger(Callback):
        def __init__(self, total_epochs:int, task_label:str="LSTM training"):
            super().__init__()
            self.total = total_epochs
            self.task_label = task_label
            self.epoch = 0
            self.pbar = None

        def on_train_begin(self, logs=None):
            self.epoch = 0
            if console:
                self.pbar = Progress(
                    SpinnerColumn(),
                    BarColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TextColumn("[green]{task.completed}/{task.total} epochs"),
                    console=console
                )
                self._task = self.pbar.add_task(self.task_label, total=self.total)
                self.pbar.start()

        def on_epoch_end(self, epoch, logs=None):
            self.epoch += 1
            if console:
                loss = logs.get("loss")
                val_loss = logs.get("val_loss")

                loss_str = f"{loss:.6f}" if loss is not None else "NA"
                val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "NA"

                console.log(
                    f"[yellow]{self.task_label}[/yellow] epoch {self.epoch}/{self.total} "
                    f"— loss={loss_str} val_loss={val_loss_str}"
                )
                self.pbar.update(self._task, advance=1)

        def on_train_end(self, logs=None):
            if console and self.pbar:
                self.pbar.stop()

def get_future_timestamps(df: pd.DataFrame, horizon: int) -> list:
    """
    Generate future timestamps for the next 'horizon' steps
    based on df's index frequency.
    """
    last_time = df.index[-1]
    freq = pd.infer_freq(df.index)  # e.g., 'D', 'H', '15T'
    if freq is None:
        # fallback: use median diff
        freq = pd.to_timedelta(np.median(np.diff(df.index)))
    
    # Use pd.date_range directly with start=last_time + freq
    return pd.date_range(start=last_time + pd.tseries.frequencies.to_offset(freq),
                         periods=horizon,
                         freq=freq).tolist()

def timedelta_from_freq(freq: str) -> pd.Timedelta:
    """
    Convert pandas frequency string to Timedelta-compatible arguments.
    Example: '1D' -> pd.Timedelta(days=1), '5h' -> pd.Timedelta(hours=5)
    """
    import re
    match = re.match(r"(\d+)([A-Za-z]+)", freq)
    if not match:
        raise ValueError(f"Invalid frequency string: {freq}")
    val, unit = match.groups()
    val = int(val)
    unit_map = {
        'D': 'days',
        'd': 'days',
        'H': 'hours',
        'h': 'hours',
        'T': 'minutes',  # T is pandas alias for min
        'min': 'minutes',
        'S': 'seconds',
        's': 'seconds'
    }
    if unit not in unit_map:
        raise ValueError(f"Unsupported freq unit: {unit}")
    return pd.Timedelta(**{unit_map[unit]: val})

def train_lstm_model(X_seq_train, y_seq_train_scaled, lookback, n_features, cfg, scaler_y, X_seq_val=None, y_seq_val_scaled=None):
    model = build_small_lstm((lookback, n_features), units=128)

    callbacks = []
    if 'EarlyStopping' in globals():
        callbacks.append(EarlyStopping(monitor='loss', patience=5, restore_best_weights=True))
    callbacks.append(RichKerasLogger(total_epochs=cfg.lstm_epochs, task_label="LSTM"))

    model.fit(
        X_seq_train, y_seq_train_scaled,
        validation_data=(X_seq_val, y_seq_val_scaled) if X_seq_val is not None else None,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch,
        callbacks=callbacks,
        verbose=0
    )

    return model

def recursive_lstm_forecast(model, last_window, forecast_h, scaler_y):
    preds = []
    current_window = last_window.copy()

    for _ in range(forecast_h):
        next_scaled = model.predict(current_window, verbose=0).reshape(-1)[0]
        next_val = scaler_y.inverse_transform([[next_scaled]])[0, 0]
        preds.append(float(next_val))

        # update window with new prediction (append to y-channel or features if univariate)
        # if multivariate, we may need synthetic features; fallback is persistence
        current_window = np.roll(current_window, -1, axis=1)
        current_window[0, -1, 0] = next_scaled  # assumes target is first feature

    return np.array(preds, dtype=float)

def train_cnn_lstm_model(X_seq_train, y_seq_train_scaled, lookback, n_features, cfg, scaler_y, X_seq_val=None, y_seq_val_scaled=None):
    model = build_cnn_lstm((lookback, n_features), lstm_units=128, dropout_rate=0.2)

    callbacks = []
    if 'EarlyStopping' in globals():
        callbacks.append(EarlyStopping(monitor='loss', patience=5, restore_best_weights=True))
    callbacks.append(RichKerasLogger(total_epochs=cfg.lstm_epochs, task_label="CNN-LSTM"))

    model.fit(
        X_seq_train, y_seq_train_scaled,
        validation_data=(X_seq_val, y_seq_val_scaled) if X_seq_val is not None else None,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch,
        callbacks=callbacks,
        verbose=0
    )

    return model

def recursive_cnn_lstm_forecast(model, last_window, forecast_h, scaler_y):
    preds = []
    current_window = last_window.copy()

    for _ in range(forecast_h):
        next_scaled = model.predict(current_window, verbose=0).reshape(-1)[0]
        next_val = scaler_y.inverse_transform([[next_scaled]])[0, 0]
        preds.append(float(next_val))
        current_window = np.roll(current_window, -1, axis=1)
        current_window[0, -1, 0] = next_scaled  # target assumed first feature

    return np.array(preds, dtype=float)

def train_attention_lstm_model(X_seq_train, y_seq_train_scaled, lookback, n_features, cfg, scaler_y, X_seq_val=None, y_seq_val_scaled=None):
    model = build_attention_lstm((lookback, n_features), lstm_units=128, dropout_rate=0.2)

    callbacks = []
    if 'EarlyStopping' in globals():
        callbacks.append(EarlyStopping(monitor='loss', patience=5, restore_best_weights=True))
    callbacks.append(RichKerasLogger(total_epochs=cfg.lstm_epochs, task_label="Attention-LSTM"))

    model.fit(
        X_seq_train, y_seq_train_scaled,
        validation_data=(X_seq_val, y_seq_val_scaled) if X_seq_val is not None else None,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch,
        callbacks=callbacks,
        verbose=0
    )

    return model

def recursive_attention_lstm_forecast(model, last_window, forecast_h, scaler_y):
    preds = []
    current_window = last_window.copy()

    for _ in range(forecast_h):
        next_scaled = model.predict(current_window, verbose=0).reshape(-1)[0]
        next_val = scaler_y.inverse_transform([[next_scaled]])[0, 0]
        preds.append(float(next_val))
        current_window = np.roll(current_window, -1, axis=1)
        current_window[0, -1, 0] = next_scaled  # target assumed first feature

    return np.array(preds, dtype=float)


# -------------------------
# Forecast pipeline per target
# -------------------------
def forecast_univariate(series: pd.Series,
                        features_df: pd.DataFrame,
                        cfg: CLIConfig,
                        target_name: str = "Close"):  # <-- new param
    saved_models = {}  # will store all trained models
    last_seq_dict = {} # for LSTM-family
    val_h = cfg.val_horizon  # use dynamic horizon
    forecast_h = cfg.forecast_horizon

    # Align and sanitize indices
    series = _ensure_series_has_datetime_index(series)
    features_df.index = _ensure_datetime_index_tz_naive(features_df.index)

    features_df, series = features_df.align(series, join="inner", axis=0)
    features_df = features_df.dropna()
    series = series.loc[features_df.index].dropna()
    if len(series) < val_h + 20:
        raise ValueError("Not enough data after indicators for training/validation")

    n_total = len(series)
    n_train = n_total - val_h
    train_series = series.iloc[:n_train]

    # EMA baseline
    ema_val_preds, ema_next = ema_baseline(series, span=max(12, cfg.candles//3), val_horizon=val_h)
    ema_forecast = np.repeat(ema_next, forecast_h)

    # SARIMAX
    sarima_preds, sarima_info = sarimax_small_grid_forecast(train_series, val_h, seasonal=False)
    sarima_next = float(sarima_preds[-1]) if len(sarima_preds) > 0 else float(train_series.iloc[-1])
    sarima_forecast_preds, _ = sarimax_small_grid_forecast(series, forecast_h, seasonal=False)
    
    # Prophet
    prophet_preds = None
    if cfg.use_prophet and Prophet is not None:
        try:
            prophet_preds, prophet_model = prophet_forecast(train_series, val_h)
        except Exception as e:
            logger.warning(f"Prophet failed: {e}")
            prophet_preds = None
    prophet_forecast_preds = None
    if cfg.use_prophet and Prophet is not None:
        try:
            prophet_forecast_preds, prophet_model_full = prophet_forecast(series, forecast_h)
        except Exception as e:
            logger.warning(f"Prophet forward failed: {e}")
            prophet_forecast_preds = None
            
    saved_models["EMA_last"] = float(ema_next)
    if sarima_forecast_preds is not None:
        saved_models["SARIMAX"] = sarima_info['model']  # assuming sarimax_small_grid_forecast returns model
    if prophet_forecast_preds is not None:
        saved_models["Prophet"] = prophet_model_full  # assuming you keep prophet model object

    # Prepare tabular features for XGBoost/LSTM: fit scalers on train only
    features = features_df.copy()
    X_all = features.values.astype(float)
    y_all = series.values.astype(float)

    # Fit scaler on training rows only
    scaler_X = RobustScaler()
    scaler_X.fit(X_all[:n_train])
    X_scaled_full = scaler_X.transform(X_all)
    
     # ----------------- Tree Models: RF, LGBM -----------------
    X_tab_train = X_scaled_full[:n_train]
    y_tab_train = y_all[:n_train]
    X_tab_val   = X_scaled_full[n_train:n_train+val_h]
    
    tree_model_fns = []
    if cfg.use_random_forest:
        def rf_fn(X_tab_train, y_tab_train, X_tab_val):
            params_rf = {
                "n_estimators": getattr(cfg, "rf_n_estimators", 300),
                "max_depth": getattr(cfg, "rf_max_depth", None),
                "min_samples_leaf": getattr(cfg, "rf_min_samples_leaf", 1),
            }
            mdl = train_random_forest_v51(X_tab_train, y_tab_train, X_tab_val, params=params_rf)
            saved_models["Random_Forest"] = mdl  # save in memory
            # optional: save to disk
            # joblib.dump(mdl, "saved_models/random_forest.pkl")
            return mdl
        tree_model_fns.append(rf_fn)

    if cfg.use_lightgbm and lgb:
        def lgb_fn(X_tab_train, y_tab_train, X_tab_val):
            params_lgb = {
                "n_estimators": getattr(cfg, "lgb_n_estimators", 500),
                "learning_rate": getattr(cfg, "lgb_learning_rate", 0.05),
                "max_depth": getattr(cfg, "lgb_max_depth", -1),
                "num_leaves": getattr(cfg, "lgb_num_leaves", 31),
            }
            mdl = train_lightgbm_v51(X_tab_train, y_tab_train, X_tab_val, params=params_lgb)
            saved_models["Lightgbm"] = mdl  # save in memory
            # optional: save to disk
            # joblib.dump(mdl, "saved_models/lightgbm.pkl")
            return mdl
        tree_model_fns.append(lgb_fn)

    # Generate OOF predictions and train meta-learner
    preds_by_tree_model, forecast_by_tree_model, tree_metrics, tree_weights = {}, {}, {}, {}
    if tree_model_fns:
        oof = generate_oof_predictions_v51(tree_model_fns, X_scaled_full, y_all, max(int(n_total*0.5),200), val_h, val_h)
        meta_learner = train_meta_learner_v51(oof, y_all)
        for i, fn in enumerate(tree_model_fns):
            try:
                _, mdl = fn(X_scaled_full, y_all, None)
                saved_models[f"Tree_Model_{i}"] = mdl
                last_feat = np.repeat(X_scaled_full[-1:], forecast_h, axis=0)
                forecast_by_tree_model[f"MODEL_{i}"] = mdl.predict(last_feat)
                preds_by_tree_model[f"MODEL_{i}"] = mdl.predict(X_scaled_full[-val_h:])
            except Exception:
                preds_by_tree_model[f"MODEL_{i}"] = np.repeat(y_all[-1], val_h)
                forecast_by_tree_model[f"MODEL_{i}"] = np.repeat(y_all[-1], forecast_h)

        stack_for_forecast = np.vstack([forecast_by_tree_model[k] for k in sorted(forecast_by_tree_model.keys())]).T
        ensemble_forecast_tree = meta_learner.predict(stack_for_forecast)
        saved_models["Meta_Learner"] = meta_learner
        preds_by_tree_model["ENSEMBLE_META"] = np.repeat(ensemble_forecast_tree[0], val_h)
        forecast_by_tree_model["ENSEMBLE_META"] = ensemble_forecast_tree


    # XGBoost
    xgb_preds, xgb_forecast, xgb_model_obj = None, None, None

    if cfg.use_xgboost and xgb is not None:
        try:
            # Validation phase
            X_tab_train = X_scaled_full[:n_train]
            y_tab_train = y_all[:n_train]
            X_tab_val   = X_scaled_full[n_train:n_train+val_h]

            xgb_preds, _, _ = xgboost_forecast(X_tab_train, y_tab_train, X_val=X_tab_val)

            # Forecast phase (refit on full data)
            X_tab_full = X_scaled_full
            y_tab_full = y_all

            if "features_future" not in locals() or features_future is None:
            # fallback: repeat last row of features
                features_future = np.repeat(X_scaled_full[-1:], forecast_h, axis=0)

            _, xgb_forecast, xgb_model_obj = xgboost_forecast(
                X_tab_full, y_tab_full, X_future=features_future)

            if xgb_forecast is None:
                # fallback if no future features
                xgb_forecast = np.repeat(float(y_tab_full[-1]), forecast_h)

        except Exception as e:
            logger.warning(f"XGBoost failed: {e}")
            xgb_preds, xgb_forecast = None, None


    # LSTM: robust multi-step sequence creation & scaling
    lstm_preds = lstm_forecast = None
    cnn_lstm_preds, cnn_lstm_forecast = None, None
    attn_lstm_preds, attn_lstm_forecast = None, None

    if cfg.use_lstm or cfg.use_cnn_lstm or cfg.use_attention_lstm:
        if tf is None:
            logger.warning("LSTM requested but TensorFlow not installed; skipping LSTM.")
        else:
            try:
                # 1. Scale features
                X_all = features_df.values.astype(float)
                scaler_X = RobustScaler()
                scaler_X.fit(X_all[:n_train])
                X_scaled_full = scaler_X.transform(X_all)

                # 2. Scale target
                y_all_float = series.values.astype(float)
                y_train = y_all_float[:n_train].reshape(-1, 1)
                scaler_y = RobustScaler()
                scaler_y.fit(y_train)

                # 3. Build sequences
                lookback = min(60, max(10, cfg.candles // 3))
                n_seq = X_scaled_full.shape[0] - lookback
                if n_seq <= 0:
                    raise RuntimeError("Not enough rows to build sequences for LSTM")

                X_seq = np.array([X_scaled_full[i:i+lookback] for i in range(n_seq)], dtype=float)
                y_seq = y_all_float[lookback:]  # aligned with X_seq

                # 4. Split sequences into train and validation
                last_train_seq_index = n_train - lookback
                if last_train_seq_index <= 0:
                    raise RuntimeError("Not enough sequence training rows for LSTM after lookback")

                X_seq_train = X_seq[:last_train_seq_index]
                y_seq_train = y_seq[:last_train_seq_index]

                X_seq_val = X_seq[last_train_seq_index:last_train_seq_index + val_h]
                y_seq_val = y_seq[last_train_seq_index:last_train_seq_index + val_h]

                # pad validation if smaller than val_h
                if X_seq_val.shape[0] < val_h:
                    pad_n = val_h - X_seq_val.shape[0]
                    X_seq_val = np.vstack([np.repeat(X_seq_val[-1:], pad_n, axis=0), X_seq_val])
                    y_seq_val = np.concatenate([np.repeat(y_seq_val[-1], pad_n), y_seq_val])

                # 5. Scale y sequences
                y_seq_train_scaled = scaler_y.transform(y_seq_train.reshape(-1, 1)).reshape(-1)
                y_seq_val_scaled = scaler_y.transform(y_seq_val.reshape(-1, 1)).reshape(-1)

                if cfg.use_lstm:
                    model = train_lstm_model(X_seq_train, y_seq_train_scaled, lookback, X_seq.shape[2], cfg, scaler_y,
                                             X_seq_val, y_seq_val_scaled)
                    preds_scaled = model.predict(X_seq_val, verbose=0).reshape(-1)
                    lstm_preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).reshape(-1)
                    model_full = train_lstm_model(X_seq, scaler_y.transform(y_seq.reshape(-1,1)).reshape(-1),
                                                  lookback, X_seq.shape[2], cfg, scaler_y)
                    lstm_forecast = recursive_lstm_forecast(model_full, X_seq[-1].reshape(1,lookback,X_seq.shape[2]), forecast_h, scaler_y)

                if cfg.use_cnn_lstm:
                    model = train_cnn_lstm_model(X_seq_train, y_seq_train_scaled, lookback, X_seq.shape[2], cfg, scaler_y,
                                                 X_seq_val, y_seq_val_scaled)
                    preds_scaled = model.predict(X_seq_val, verbose=0).reshape(-1)
                    cnn_lstm_preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).reshape(-1)
                    model_full = train_cnn_lstm_model(X_seq, scaler_y.transform(y_seq.reshape(-1,1)).reshape(-1),
                                                      lookback, X_seq.shape[2], cfg, scaler_y)
                    cnn_lstm_forecast = recursive_cnn_lstm_forecast(model_full, X_seq[-1].reshape(1,lookback,X_seq.shape[2]), forecast_h, scaler_y)

                if cfg.use_attention_lstm:
                    model = train_attention_lstm_model(X_seq_train, y_seq_train_scaled, lookback, X_seq.shape[2], cfg, scaler_y,
                                                      X_seq_val, y_seq_val_scaled)
                    preds_scaled = model.predict(X_seq_val, verbose=0).reshape(-1)
                    attn_lstm_preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).reshape(-1)
                    model_full = train_attention_lstm_model(X_seq, scaler_y.transform(y_seq.reshape(-1,1)).reshape(-1),
                                                            lookback, X_seq.shape[2], cfg, scaler_y)
                    attn_lstm_forecast = recursive_attention_lstm_forecast(model_full, X_seq[-1].reshape(1,lookback,X_seq.shape[2]), forecast_h, scaler_y)

            except Exception as e:
                logger.warning(f"LSTM failed: {e}")
                lstm_preds = None
                lstm_next = float(series.iloc[-1])


    # Collect predictions
    preds_by_model = {}
    next_preds = {}

    preds_by_model["EMA"] = ema_val_preds[-val_h:]
    preds_by_model["SARIMAX"] = sarima_preds
    if prophet_preds is not None:
        preds_by_model["Prophet"] = prophet_preds
    if xgb_preds is not None:
        preds_by_model["XGBoost"] = np.array(xgb_preds, dtype=float)
    # preds_by_model.update(np.array(preds_by_tree_model, dtype=float) or {})
    preds_by_model.update(preds_by_tree_model or {})
    if lstm_preds is not None:
        preds_by_model["LSTM"] = np.array(lstm_preds, dtype=float)
    if cnn_lstm_preds is not None: 
        preds_by_model["CNN-LSTM"] = np.array(cnn_lstm_preds, dtype=float)
    if attn_lstm_preds is not None: 
        preds_by_model["Attention-LSTM"] = np.array(attn_lstm_preds, dtype=float)

    # Make sure all preds are length val_h
    for k in list(preds_by_model.keys()):
        arr = np.asarray(preds_by_model[k], dtype=float)
        if arr.shape[0] < val_h:
            if arr.size == 0:
                arr = np.full(val_h, float(series.iloc[-1]))
            else:
                arr = np.concatenate([np.full(val_h - arr.shape[0], arr[-1]), arr])
        elif arr.shape[0] > val_h:
            arr = arr[-val_h:]
        preds_by_model[k] = arr
        
    forecast_by_model = {}
    forecast_by_model["EMA"] = ema_forecast
    if sarima_forecast_preds is not None:
        forecast_by_model["SARIMAX"] = np.asarray(sarima_forecast_preds, dtype=float)
    if prophet_forecast_preds is not None:
        forecast_by_model["Prophet"] = np.asarray(prophet_forecast_preds, dtype=float)
    if xgb_forecast is not None:
        forecast_by_model["XGBoost"] = np.asarray(xgb_forecast, dtype=float)
    # forecast_by_model.update(np.array(forecast_by_tree_model, dtype=float) or {})
    forecast_by_model.update(forecast_by_tree_model or {})
    if lstm_forecast is not None:
        forecast_by_model["LSTM"] = np.asarray(lstm_forecast, dtype=float)
    if cnn_lstm_forecast is not None: 
        forecast_by_model["CNN-LSTM"] = np.array(cnn_lstm_forecast, dtype=float)
    if attn_lstm_forecast is not None: 
        forecast_by_model["Attention-LSTM"] = np.array(attn_lstm_forecast, dtype=float)

    # Normalize lengths to forecast_h
    for k in list(forecast_by_model.keys()):
        arr = np.asarray(forecast_by_model[k], dtype=float)
        if arr.shape[0] < forecast_h:
            if arr.size == 0:
                arr = np.full(forecast_h, float(series.iloc[-1]))
            else:
                arr = np.concatenate([arr, np.full(forecast_h - arr.shape[0], arr[-1])])
        elif arr.shape[0] > forecast_h:
            arr = arr[:forecast_h]
        forecast_by_model[k] = arr
        
    # y_val_actual = y_all[n_train:n_train+val_h].astype(float)

    # # -------------------- Compute ATR for VolAdjError --------------------
    # price_changes = np.abs(np.diff(y_val_actual))
    # avg_true_range = price_changes.mean() if len(price_changes) > 0 else 1.0

    # # -------------------- Ensemble validation preds (use last weights if available) --------------------
    # if "last_weights" in locals() and last_weights:   # carry-over from previous run
    #     val_weights = last_weights
    # else:
    #     # fallback: equal weights if first run
    #     equal_weight = 1.0 / len(preds_by_model)
    #     val_weights = {k: equal_weight for k in preds_by_model}

    # ensemble_val_preds = np.zeros_like(y_val_actual, dtype=float)
    # for k, w in val_weights.items():
    #     if k in preds_by_model:
    #         ensemble_val_preds += preds_by_model[k] * w
    # preds_by_model["ENSEMBLE"] = ensemble_val_preds

    # # -------------------- Metrics --------------------
    # metrics = {}
    # for k, arr in preds_by_model.items():
    #     mae = float(mean_absolute_error(y_val_actual, arr))
    #     rmse = float(math.sqrt(mean_squared_error(y_val_actual, arr)))
    #     mape = float(safe_mape(y_val_actual, arr))
    #     vol_adj = mae / (avg_true_range + 1e-6)  # penalize models that ignore volatility
    #     metrics[k] = {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "VolAdjError": vol_adj}

    # # -------------------- VolAdj-based weights (new weights for next round) --------------------
    # eps = 1e-8
    # inv = {k: 1.0 / (metrics[k]["VolAdjError"] + eps) for k in metrics}
    # total = sum(inv.values())
    # weights = {k: v / total for k, v in inv.items()}

    # # bias LSTMs a little if they all agree
    # lstm_models = ["LSTM", "CNN-LSTM", "Attention-LSTM"]
    # lstm_forecasts = [forecast_by_model[m] for m in lstm_models if m in forecast_by_model]

    # if len(lstm_forecasts) >= 2:
    #     slopes = [np.sign(f[-1] - f[0]) for f in lstm_forecasts]
    #     if all(s == 1 for s in slopes):   # bullish bias
    #         for m in lstm_models:
    #             if m in weights:
    #                 weights[m] *= 1.2
    #     elif all(s == -1 for s in slopes):  # bearish bias
    #         for m in lstm_models:
    #             if m in weights:
    #                 weights[m] *= 1.2

    # # -------------------- Final Ensemble forecast --------------------
    # ensemble_forecast = np.zeros(forecast_h, dtype=float)
    # for k, w in weights.items():
    #     if k in forecast_by_model:
    #         ensemble_forecast += forecast_by_model[k] * w
    # forecast_by_model["ENSEMBLE"] = ensemble_forecast

    # # -------------------- Save weights for next iteration --------------------
    # last_weights = weights.copy()
    
    
    y_val_actual = y_all[n_train:n_train+val_h].astype(float) 
    weights = inverse_error_weights(y_val_actual, preds_by_model)
    price_changes = np.abs(np.diff(y_val_actual))
    avg_true_range = price_changes.mean() if len(price_changes) > 0 else 1.0
    
    lstm_models = ["LSTM", "CNN-LSTM", "Attention-LSTM"] 
    lstm_forecasts = [forecast_by_model[m] for m in lstm_models if m in forecast_by_model]
    
    if len(lstm_forecasts) >= 2:
        slopes = [np.sign(f[-1] - f[0]) for f in lstm_forecasts]
        if all(s == 1 for s in slopes):
            for m in lstm_models:
                if m in weights: weights[m] *= 1.35
        elif all(s == -1 for s in slopes):
            for m in lstm_models:
                if m in weights: weights[m] *= 1.35
    
    ensemble_val_preds = np.zeros_like(y_val_actual, dtype=float)
    for k,w in weights.items():
        if k in preds_by_model:
            ensemble_val_preds += preds_by_model[k] * w
    preds_by_model["ENSEMBLE"] = ensemble_val_preds
    
    ensemble_forecast = np.zeros(forecast_h, dtype=float) 
    for k,w in weights.items(): 
        if k in forecast_by_model: 
            ensemble_forecast += forecast_by_model[k] * w 
    forecast_by_model["ENSEMBLE"] = ensemble_forecast   
    
    metrics = {}
    for k, arr in preds_by_model.items():
        mae = float(mean_absolute_error(y_val_actual, arr))
        rmse = float(math.sqrt(mean_squared_error(y_val_actual, arr)))
        mape = float(safe_mape(y_val_actual, arr))
        vol_adj = mae / (avg_true_range + 1e-6)  # penalize models that ignore volatility
        metrics[k] = {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "VolAdjError": vol_adj}         

  
    # -------------------- Future timestamps --------------------
    last_time = series.index[-1]
    freq = pd.infer_freq(series.index) or '1D'
    future_timestamps = pd.date_range(
        start=last_time + timedelta_from_freq(freq),
        periods=forecast_h, freq=freq).tolist()
    next_preds = {model: {str(ts): float(pred) for ts, pred in zip(future_timestamps, arr)}
                  for model, arr in forecast_by_model.items()}
    
     # --- NEW: run strategy detectors for this target, add to results ---
    strategies = {}
    try:
        # df for the target: we operate on features_df aligned earlier
        target_df = features_df.copy()
        # Breakout
        strategies["breakout"] = detect_breakout(target_df, target_col=target_name)
        # Mean reversion
        strategies["mean_reversion"] = detect_mean_reversion(target_df, target_col=target_name)
        # Fibonacci
        strategies["fibonacci"] = detect_fibonacci_pullback(target_df, target_col=target_name)
        # Price action
        strategies["price_action"] = detect_price_action(target_df)
        # Swing trade
        strategies["swing"] = detect_swing_trade(target_df, target_col=target_name)
        # Scalping helper (advisory)
        strategies["scalping_helper"] = detect_scalping_opportunity(target_df, target_col=target_name)
        # Market regime
        strategies["regime"] = detect_market_regime(target_df)
        # Options summary (light)
        strategies["options_summary"] = fetch_options_flow_stub(cfg.ticker) if hasattr(cfg, "ticker") else {}
        # News sentiment (stub)
        strategies["news_sentiment"] = fetch_news_sentiment_stub(cfg.ticker) if hasattr(cfg, "ticker") else None
    except Exception as e:
        logger.debug(f"Strategy detection failed: {e}")
        strategies = {}
        
    if strategies:
        ensemble_forecast_arr = forecast_by_model.get("ENSEMBLE", [])
        if len(ensemble_forecast_arr) > 1:
            slope = ensemble_forecast_arr[-1] - ensemble_forecast_arr[0]
            if all(
            (v is None) or 
            (str(v).lower() in ["none", "null"]) or 
            pd.isna(v)
            for v in strategies.values()
        ):
                if slope > 0:
                    strategies["soft_bias"] = "bullish_bias"
                elif slope < 0:
                    strategies["soft_bias"] = "bearish_bias"
                else:
                    strategies["soft_bias"] = "neutral"

    # attempt to free memory
    try:
        del X_all, X_scaled_full
        gc.collect()
    except Exception:
        pass

    return preds_by_model, next_preds, metrics, weights, strategies

def forecast_incremental(new_series: pd.Series,
                         new_features_df: pd.DataFrame,
                         cfg: CLIConfig,
                         saved_models: dict,
                         last_seq_dict: dict,
                         target_name: str = "Close"):
    """
    Incremental forecasting using previously trained models.
    Returns:
        preds_by_model: in-sample predictions for new data
        next_preds: forecast for future timestamps
        metrics: error metrics on validation
        weights: ensemble weights
        strategies: detected strategies
        incremental_preds: same as next_preds but using only incremental logic
    """
    val_h = cfg.val_horizon
    forecast_h = cfg.forecast_horizon

    # Align and sanitize indices
    new_series = _ensure_series_has_datetime_index(new_series)
    new_features_df.index = _ensure_datetime_index_tz_naive(new_features_df.index)
    new_features_df, new_series = new_features_df.align(new_series, join="inner", axis=0)
    new_features_df = new_features_df.dropna()
    new_series = new_series.loc[new_features_df.index].dropna()
    if len(new_series) < val_h + 1:
        raise ValueError("Not enough incremental data after alignment")

    X_all = new_features_df.values.astype(float)
    y_all = new_series.values.astype(float)

    # -------------------- EMA / SARIMAX / Prophet --------------------
    ema_last = saved_models.get("EMA_last", float(y_all[-1]))
    ema_val_preds = np.repeat(ema_last, len(new_series))
    ema_forecast = np.repeat(ema_last, forecast_h)

    sarima_preds = None
    if saved_models.get("SARIMAX") is not None:
        sarima_model = saved_models["SARIMAX"]
        sarima_preds = sarima_model.predict(start=0, end=len(new_series)-1)
        sarima_forecast_preds = sarima_model.get_forecast(steps=forecast_h).predicted_mean.values
    else:
        sarima_forecast_preds = np.repeat(float(y_all[-1]), forecast_h)

    prophet_preds = None
    prophet_forecast_preds = None
    if saved_models.get("Prophet") is not None:
        try:
            prophet_model = saved_models["Prophet"]
            prophet_preds = prophet_model.predict(pd.DataFrame({"ds": new_series.index}))["yhat"].values
            future_df = pd.DataFrame({"ds": pd.date_range(start=new_series.index[-1] + timedelta_from_freq(pd.infer_freq(new_series.index) or '1D'),
                                                          periods=forecast_h, freq=pd.infer_freq(new_series.index) or '1D')})
            prophet_forecast_preds = prophet_model.predict(future_df)["yhat"].values
        except Exception:
            prophet_forecast_preds = np.repeat(float(y_all[-1]), forecast_h)

    # -------------------- Tree / XGBoost / LGBM --------------------
    scaler_X = RobustScaler()
    scaler_X.fit(X_all)
    X_scaled_full = scaler_X.transform(X_all)
    features_future = np.repeat(X_scaled_full[-1:], forecast_h, axis=0)

    tree_models = ["RF", "LGBM", "XGBoost"]
    preds_by_model = {}
    forecast_by_model = {}
    incremental_preds = {}

    for model_name in tree_models:
        model = saved_models.get(model_name)
        if model is not None:
            preds_by_model[model_name] = model.predict(X_scaled_full)
            forecast_by_model[model_name] = model.predict(features_future)
            incremental_preds[model_name] = forecast_by_model[model_name]

    # -------------------- LSTM-family --------------------
    lstm_models = ["LSTM", "CNN-LSTM", "Attention-LSTM"]
    for model_name in lstm_models:
        model = saved_models.get(model_name)
        last_seq = last_seq_dict.get(model_name)
        if model is not None and last_seq is not None:
            incremental_preds[model_name] = recursive_lstm_forecast(model, last_seq, forecast_h, cfg.scaler_y)

    # -------------------- EMA / SARIMAX / Prophet incremental --------------------
    incremental_preds["EMA"] = ema_forecast
    incremental_preds["SARIMAX"] = sarima_forecast_preds
    if prophet_forecast_preds is not None:
        incremental_preds["Prophet"] = prophet_forecast_preds

    # -------------------- In-sample preds --------------------
    preds_by_model["EMA"] = ema_val_preds
    preds_by_model["SARIMAX"] = sarima_preds if sarima_preds is not None else ema_val_preds
    if prophet_preds is not None:
        preds_by_model["Prophet"] = prophet_preds

    # -------------------- Ensemble weights --------------------
    y_val_actual = y_all[-val_h:].astype(float)
    weights = inverse_error_weights(y_val_actual, preds_by_model)

    # LSTM slope bias
    lstm_forecasts = [incremental_preds[m] for m in lstm_models if m in incremental_preds]
    if len(lstm_forecasts) >= 2:
        slopes = [np.sign(f[-1]-f[0]) for f in lstm_forecasts]
        if all(s==1 for s in slopes):
            for m in lstm_models: weights[m] = weights.get(m, 0) * 1.35
        elif all(s==-1 for s in slopes):
            for m in lstm_models: weights[m] = weights.get(m, 0) * 1.35

    # Ensemble incremental
    ensemble_forecast = np.zeros(forecast_h, dtype=float)
    for k, w in weights.items():
        if k in incremental_preds:
            ensemble_forecast += incremental_preds[k] * w
    incremental_preds["ENSEMBLE"] = ensemble_forecast

    # -------------------- Metrics --------------------
    price_changes = np.abs(np.diff(y_val_actual))
    avg_true_range = price_changes.mean() if len(price_changes) > 0 else 1.0
    metrics = {}
    for k, arr in preds_by_model.items():
        mae = float(mean_absolute_error(y_val_actual, arr))
        rmse = float(math.sqrt(mean_squared_error(y_val_actual, arr)))
        mape = float(safe_mape(y_val_actual, arr))
        vol_adj = mae / (avg_true_range + 1e-6)
        metrics[k] = {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "VolAdjError": vol_adj}

    # -------------------- Future timestamps --------------------
    last_time = new_series.index[-1]
    freq = pd.infer_freq(new_series.index) or '1D'
    future_timestamps = pd.date_range(
        start=last_time + timedelta_from_freq(freq),
        periods=forecast_h, freq=freq).tolist()
    next_preds = {model: {str(ts): float(pred) for ts, pred in zip(future_timestamps, arr)}
                  for model, arr in incremental_preds.items()}

    # -------------------- Strategy detection --------------------
    strategies = {}
    try:
        target_df = new_features_df.copy()
        strategies["breakout"] = detect_breakout(target_df, target_col=target_name)
        strategies["mean_reversion"] = detect_mean_reversion(target_df, target_col=target_name)
        strategies["fibonacci"] = detect_fibonacci_pullback(target_df, target_col=target_name)
        strategies["price_action"] = detect_price_action(target_df)
        strategies["swing"] = detect_swing_trade(target_df, target_col=target_name)
        strategies["scalping_helper"] = detect_scalping_opportunity(target_df, target_col=target_name)
        strategies["regime"] = detect_market_regime(target_df)
        strategies["options_summary"] = fetch_options_flow_stub(cfg.ticker) if hasattr(cfg, "ticker") else {}
        strategies["news_sentiment"] = fetch_news_sentiment_stub(cfg.ticker) if hasattr(cfg, "ticker") else None
    except Exception:
        strategies = {}

    return preds_by_model, next_preds, metrics, weights, strategies, incremental_preds



def adjust_predictions(target, next_preds, model_sums, current_values):
    """
    Adjusts predictions for a given target (Open, High, Low, Close) 
    based on error sums and deviation rules.
    """
    adjusted_preds = {}

    # Get the current reference value (e.g., current_open, current_close)
    current_val = current_values[target]

    for model_name, preds in next_preds.items():
        if model_name not in model_sums:
            continue  # skip if no error metrics

        model_sum = model_sums[model_name]
        adjusted_preds[model_name] = {}

        for ts, pred in preds.items():
            diff = pred - current_val

            # Check deviation against error sum
            if abs(diff) >= model_sum:
                if pred > current_val:
                    adjusted_pred = pred - model_sum
                else:
                    adjusted_pred = pred + model_sum
            else:
                adjusted_pred = pred

            adjusted_preds[model_name][ts] = adjusted_pred

    return adjusted_preds


# -------------------------
# Orchestration for OHLC
# -------------------------
def predict_all_ohlcv(df: pd.DataFrame, cfg: CLIConfig):
    results = {}
    try:
        df.index = _ensure_datetime_index_tz_naive(df.index)
    except Exception:
        pass
    
    last_candle = df.iloc[-1]
    current_close = last_candle.get('Close', 0)
    current_high = last_candle.get('High', 0)
    current_low = last_candle.get('Low', 0)
    current_open = last_candle.get('Open', 0)
    current_values = {
        "Open": current_open,
        "High": current_high,
        "Low": current_low,
        "Close": current_close
    }
    
    features = compute_indicators(df)

    # for target in ["Open", "High", "Low", "Close"]:
    for target in ["Close"]:
        try:
            preds_by_model, next_preds, metrics, weights, strategies = forecast_univariate(
                df[target], features, cfg, target_name=target)
            # print('Next Preds -> ', next_preds)
            
            model_sums = {name: vals["MAE"] + vals["RMSE"] + vals["MAPE%"] for name, vals in metrics.items()}
            
            
            adjusted_preds = adjust_predictions(target, next_preds, model_sums, current_values)
            # print(adjusted_preds)    

            # No need to call get_future_timestamps() separately; forecast_univariate already returns mapped timestamps
            results[target] = {
                "next_preds": next_preds,
                "adjusted_preds": adjusted_preds,
                "metrics": metrics,
                "weights": weights,
            }

        except Exception as e:
            logger.exception(f"Failed forecasting {target}: {e}")
            results[target] = {"error": str(e)}

    return results


# -------------------------
# Trading signal analysis
# -------------------------
def analyze_candle(open_, high, low, close):
    """Return OHLC candle confirmation features."""
    body = abs(close - open_)
    upper_wick = high - open_
    lower_wick = close - low
    candle_range = high - low

    signals = []
    if body < candle_range * 0.3:  # small body, big wicks
        signals.append("indecision/doji")
    if lower_wick > body * 2:
        signals.append("long_lower_wick (buying pressure)")
    if upper_wick > body * 2:
        signals.append("long_upper_wick (selling pressure)")
    if body > candle_range * 0.6:
        if close > open_:
            signals.append("strong_bullish_body")
        else:
            signals.append("strong_bearish_body")

    return signals

def mean_ensemble(pred_dict, default_value=0):
    """
    Compute mean of all values in an ENSEMBLE dictionary.
    Returns default_value if dict is empty or None.
    """
    if isinstance(pred_dict, dict) and pred_dict:
        return sum(pred_dict.values()) / len(pred_dict)
    return default_value

def generate_trading_signal(json_data: Dict[str, Any], last_candle: pd.Series) -> Dict[str, Any]:
    results = json_data.get('results', {})
    if not results:
        return {
            'signal': 'HOLD',
            'confidence': '0/0',
            'reasons': ['No prediction data available'],
            'current_price': last_candle.get('Close', None),
            'predicted_price': None,
            'predicted_change': None
        }

    # ====== Extract Predicted Values ======
    # next_open = list(results.get('Open', {}).get('adjusted_preds', {}).get('ENSEMBLE', {}).values())[0]
    # next_high = list(results.get('High', {}).get('adjusted_preds', {}).get('ENSEMBLE', {}).values())[0]
    # The code is trying to access a specific value from a nested dictionary structure. It is
    # attempting to retrieve the value of the key 'ENSEMBLE' from the dictionary
    # results['Open']['adjusted_preds'], and then getting the first value from that dictionary.
    # next_low = list(results.get('Low', {}).get('adjusted_preds', {}).get('ENSEMBLE', {}).values())[0]
    next_close = list(results.get('Close', {}).get('adjusted_preds', {}).get('ENSEMBLE', {}).values())[0]


    current_close = last_candle.get('Close', 0)
    current_high = last_candle.get('High', 0)
    current_low = last_candle.get('Low', 0)
    current_open = last_candle.get('Open', 0)

    # ====== Scoring Setup ======
    bull_score, bear_score = 0, 0
    reasons = []
    strategiesInfluenced = []
    indicatorsCalculated = [
        'EMA_12',
        'EMA_26',
        'SMA_20',
        'MACD',
        'MACD_SIGNAL',
        'MACD_HIST',
        'RSI_14',
        'BB_UPPER',
        'BB_LOWER',
        'BB_PCT',
        'ATR_14',
        'RETURNS',
    ]

    WEIGHTS = {
        "candle": 1,
        "prediction": 4,
        "breakout": 3,
        "mean_reversion": 2,
        "fibonacci": 2,
        "price_action": 2,
        "scalping": 1,
        "volatility": 1,
        "options": 2,
        "direction": 2
    }

    metrics = results["Close"]["metrics"]
    weights = results["Close"]["weights"]

    # --- Weighted prediction ---
    model_preds = {
        m: list(results["Close"]["adjusted_preds"][m].values())[0]  # take first forecasted value
        for m in weights
    }

    weighted_pred = sum(
        model_preds[m] * weights[m] / max(metrics[m]["MAPE%"], 1e-6)
        for m in weights
    ) / sum(weights[m] / max(metrics[m]["MAPE%"], 1e-6) for m in weights)

    # --- Signal direction ---
    change_pct = (weighted_pred - current_close) / current_close * 100
    if change_pct > 1:
        bull_score += WEIGHTS["direction"]
        reasons.append(f"Ensemble expects a rise of {change_pct:.2f}% (from {current_close:.2f} →    {weighted_pred:.2f}) — directional bullish signal (exceeds ±1% threshold).")
        strategiesInfluenced.append("Momentum Analytics")
    elif change_pct < -1:
        # BUGFIX: use += not assignment (your original set bear_score = WEIGHTS["direction"])
        bear_score += WEIGHTS["direction"]
        reasons.append(f"Ensemble expects a fall of {change_pct:.2f}% (from {current_close:.2f} →    {weighted_pred:.2f}) — directional bearish signal (exceeds ±1% threshold).")
        strategiesInfluenced.append("Momentum Analytics")
    else:
        reasons.append(f"Ensemble expects a minor change of {change_pct:.2f}% (from {current_close:.2f} → {weighted_pred:.2f}), inside ±1% — no directional action.")
        strategiesInfluenced.append("Momentum Analytics")

    # ====== 1. Previous Candle ======
    prev_signals = analyze_candle(current_open, current_high, current_low, current_close)

    # Track the positions of lower and upper wick signals
    lower_wick_index = -1
    upper_wick_index = -1

    for i, s in enumerate(prev_signals):
        s_lower = s.lower()

        # Remember the index of the wick signals
        if "long_lower_wick" in s_lower:
            lower_wick_index = i
        elif "long_upper_wick" in s_lower:
            upper_wick_index = i

        # Original scoring logic
        if "long_lower_wick" in s_lower or "buy" in s_lower:
            bull_score += WEIGHTS["candle"]
            reasons.append("Previous candle shows a long lower wick → buyers defended lower prices; short-term bullish pressure.")
            strategiesInfluenced.append("Previous Candle Analytics")
        elif "long_upper_wick" in s_lower or "sell" in s_lower:
            bear_score += WEIGHTS["candle"]
            reasons.append("Previous candle shows a long upper wick → sellers rejected higher prices; short-term bearish pressure.")
            strategiesInfluenced.append("Previous Candle Analytics")
        elif "strong_bullish_body" in s_lower:
            bull_score += WEIGHTS["candle"]
            reasons.append("Previous candle closed strongly higher (large body) — bullish momentum confirmed.")
            strategiesInfluenced.append("Previous Candle Analytics")
        elif "strong_bearish_body" in s_lower:
            bear_score += WEIGHTS["candle"]
            reasons.append("Previous candle closed strongly lower (large body) — bearish momentum confirmed.")
            strategiesInfluenced.append("Previous Candle Analytics")
        elif "indecision" in s_lower or "doji" in s_lower:
            reasons.append("Previous candle is indecisive (small body, large wicks) — market uncertainty; wait for confirmation.")
            strategiesInfluenced.append("Previous Candle Analytics")
        else:
            reasons.append(f"Previous candle pattern: {s}")
            strategiesInfluenced.append("Previous Candle Analytics")

    # ------------------- New logic for sequential wick influence -------------------
    if lower_wick_index != -1 and upper_wick_index != -1:
        if lower_wick_index > upper_wick_index:  # lower wick after upper wick → strong bullish
            bull_score += 1
            reasons.append("Lower wick appeared after upper wick → extra bullish pressure.")
            strategiesInfluenced.append("Sequential Candle Wick Analysis")
        elif lower_wick_index < upper_wick_index:  # lower wick before upper wick → strong bearish
            bear_score += 1
            reasons.append("Lower wick appeared before upper wick → extra bearish pressure.")
            strategiesInfluenced.append("Sequential Candle Wick Analysis")

    # ====== 2. Predicted Candle ======
    # next_signals = analyze_candle(next_open, next_high, next_low, next_close)
    # for s in next_signals:
    #     s_lower = s.lower()
    #     if "long_lower_wick" in s_lower or "buy" in s_lower:
    #         bull_score += WEIGHTS["candle"]
    #         reasons.append("Predicted candle shows a long lower wick → model expects buying pressure    near lows.")
    #         strategiesInfluenced.append("Forecasted Candle Analytics")
    #     elif "long_upper_wick" in s_lower or "sell" in s_lower:
    #         bear_score += WEIGHTS["candle"]
    #         reasons.append("Predicted candle shows a long upper wick → model expects selling pressure   near highs.")
    #         strategiesInfluenced.append("Forecasted Candle Analytics")
    #     elif "strong_bullish_body" in s_lower:
    #         bull_score += WEIGHTS["candle"]
    #         reasons.append("Predicted candle indicates a strong bullish body — expected upward  momentum.")
    #         strategiesInfluenced.append("Forecasted Candle Analytics")
    #     elif "strong_bearish_body" in s_lower:
    #         bear_score += WEIGHTS["candle"]
    #         reasons.append("Predicted candle indicates a strong bearish body — expected downward    momentum.")
    #         strategiesInfluenced.append("Forecasted Candle Analytics")
    #     elif "indecision" in s_lower or "doji" in s_lower:
    #         reasons.append("Predicted candle looks indecisive — model forecasts uncertainty.")
    #         strategiesInfluenced.append("Forecasted Candle Analytics")
    #     else:
    #         reasons.append(f"Predicted candle pattern: {s}")
    #         strategiesInfluenced.append("Forecasted Candle Analytics")

    # ====== 3. Prediction Trend ======
    if next_close > current_close:
        bull_score += WEIGHTS["prediction"]
        reasons.append(f"Model ensemble predicts next close {next_close:.2f} > current {current_close:.2f} (+{(next_close-current_close)/max(current_close,1e-9)*100:.2f}%) — supports bullish bias.")
        strategiesInfluenced.append("Trend Analytics")
    elif next_close < current_close:
        bear_score += WEIGHTS["prediction"]
        reasons.append(f"Model ensemble predicts next close {next_close:.2f} < current {current_close:.2f} ({(next_close-current_close)/max(current_close,1e-9)*100:.2f}%) — supports bearish bias.")
        strategiesInfluenced.append("Trend Analytics")

    # ====== 4. Strategies ======
    strategies = {k: results.get(k, {}).get('strategies', {}) for k in ['Open', 'High', 'Low', 'Close']}

    # Breakout
    breakout_signals = [s.get('breakout', {}).get('signal') for s in strategies.values()]
    if any("bull" in str(sig).lower() for sig in breakout_signals):
        bull_score += WEIGHTS["breakout"]
        reasons.append("Breakout detected above the upper Bollinger band with volume — suggests strong upside momentum; trend-following entry favored.")
        strategiesInfluenced.append("Breakout Analytics")
    if any("bear" in str(sig).lower() for sig in breakout_signals):
        bear_score += WEIGHTS["breakout"]
        reasons.append("Breakdown detected below the lower Bollinger band with volume — suggests strong downside momentum; trend-following short favored.")
        strategiesInfluenced.append("Breakout Analytics")

    # Mean Reversion
    mean_rev = [s.get('mean_reversion', {}).get('signal') for s in strategies.values()]
    if any("long" in str(sig).lower() for sig in mean_rev):
        bull_score += WEIGHTS["mean_reversion"]
        reasons.append("Mean-reversion signals (RSI oversold / lower band / negative z-score) — likely bounce to the mean; buy opportunity.")
        strategiesInfluenced.append("Mean Reversion")
    if any("short" in str(sig).lower() for sig in mean_rev):
        bear_score += WEIGHTS["mean_reversion"]
        reasons.append("Mean-reversion signals (RSI overbought / upper band / high z-score) — likely pullback to the mean; caution for longs.")
        strategiesInfluenced.append("Mean Reversion")

    # Fibonacci
    fib = strategies.get('Close', {}).get('fibonacci', {})
    near = fib.get('near_level')
    dist = fib.get('distance', None)
    levels = fib.get('levels', {})
    if near == '61.8%' and dist is not None and dist < 2:
        bull_score += WEIGHTS["fibonacci"]
        lvl = levels.get('61.8%', None)
        reasons.append(f"Price is within {dist:.2f} of 61.8% Fibonacci ({lvl:.2f}) — common support zone; watch for bounce.")
        strategiesInfluenced.append("Fibonacci Tracement")
    if near == '0%' and dist is not None and dist < 2:
        bear_score += WEIGHTS["fibonacci"]
        lvl = levels.get('0%', None)
        reasons.append(f"Price is within {dist:.2f} of Fibonacci 0% (swing high {lvl:.2f}) — strong resistance; likely rejection.")
        strategiesInfluenced.append("Fibonacci Tracement")

    # Price Action
    bull_engulfing = sum(s.get('price_action', {}).get('bullish_engulfing', 0) for s in strategies.values())
    bear_engulfing = sum(s.get('price_action', {}).get('bearish_engulfing', 0) for s in strategies.values())
    if bull_engulfing > 1:
        bull_score += WEIGHTS["price_action"]
        reasons.append("Multiple bullish engulfing patterns detected — strong buyer conviction.")
        strategiesInfluenced.append("Price Action")
    if bear_engulfing > 1:
        bear_score += WEIGHTS["price_action"]
        reasons.append("Multiple bearish engulfing patterns detected — strong seller conviction.")
        strategiesInfluenced.append("Price Action")
    if any(s.get('price_action', {}).get('pin_bar', 0) for s in strategies.values()):
        reasons.append("Pin bar(s) detected — price rejection at a key level; watch for reversal.")
        strategiesInfluenced.append("Price Action")

    # Scalping
    scalping = [s.get('scalping_helper', {}).get('signal') for s in strategies.values()]
    if any("long" in str(sig).lower() for sig in scalping):
        bull_score += WEIGHTS["scalping"]
        reasons.append("Intraday: price above VWAP + rising volume — short-term buying momentum.")
        strategiesInfluenced.append("Scalping Helper")
    if any("short" in str(sig).lower() for sig in scalping):
        bear_score += WEIGHTS["scalping"]
        reasons.append("Intraday: price below VWAP + rising volume — short-term selling momentum.")
        strategiesInfluenced.append("Scalping Helper")
    
    
        # Swing
    swing = strategies.get('Close', {}).get('swing', {})
    swing_sig = swing.get('signal', 'none')
    if swing_sig and swing_sig.lower() != "none":
        if "long" in swing_sig.lower():
            bull_score += WEIGHTS.get("price_action", 2)  # or give swing its own weight
            reasons.append("Swing strategy indicates long bias — market showing medium-term upward pressure.")
            strategiesInfluenced.append("Swing Analytics")
        elif "short" in swing_sig.lower():
            bear_score += WEIGHTS.get("price_action", 2)
            reasons.append("Swing strategy indicates short bias — market showing medium-term downward pressure.")
            strategiesInfluenced.append("Swing Analytics")

    # Regime
    regime = results.get('Close', {}).get('regime', None)
    if regime:
        if regime.lower() == "high":
            bull_score += 1
            reasons.append("Market regime detected as HIGH volatility/uptrend — favorable for momentum strategies.")
            strategiesInfluenced.append("Regime Detection")
        elif regime.lower() == "low":
            bear_score += 1
            reasons.append("Market regime detected as LOW volatility/downtrend — risk of sideways or falling market.")
            strategiesInfluenced.append("Regime Detection")

    # Options
    opts = strategies.get('Close', {}).get('options_summary', {})
    calls = opts.get('calls_oi_sum', 0)
    puts = opts.get('puts_oi_sum', 0) or 1
    oi_ratio = calls / max(puts, 1)
    if oi_ratio > 1.2:
        bull_score += WEIGHTS["options"]
        reasons.append(f"Options positioning bullish: Calls {calls} vs Puts {puts} (ratio {oi_ratio:.2f}) — sentiment supports bullish bias.")
        strategiesInfluenced.append("Options Positioning")
    elif oi_ratio < 0.8:
        bear_score += WEIGHTS["options"]
        reasons.append(
            f"Options positioning bearish: Calls {calls} vs Puts {puts} (ratio {oi_ratio:.2f}) — sentiment supports bearish bias."
        )
        strategiesInfluenced.append("Options Positioning")

    # ====== Final Signal ======
    if bull_score >= bear_score + 4:
        final_signal = "STRONG_BUY"
        market_trend = "BULLISH MOMENTUM - RALLYING"
    elif bull_score > bear_score:
        final_signal = "BUY"
        market_trend = "BULLISH MOMENTUM - CLIMBING"
    elif bear_score >= bull_score + 4:
        final_signal = "STRONG_SELL"
        market_trend = "BEARISH MOMENTUM - PLUNGING"
    elif bear_score > bull_score:
        final_signal = "SELL"
        market_trend = "BEARISH MOMENTUM - FALLING"
    else:
        final_signal = "HOLD"
        market_trend = "NEUTRAL MOMENTUM - SIDEWAYS"

    prediction_strength = abs(next_close - current_close) / max(current_close, 1e-9) * 100
    reasons.append(f"Ensemble predicts {('bullish' if next_close>current_close else 'bearish')} move of {prediction_strength:.2f}% (from {current_close:.2f} to {next_close:.2f}).")
    # reasons.append(f"Next-candle forecast: O:{next_open:.2f} H:{next_high:.2f} L:{next_low:.2f} C:{next_close:.2f}.")
    reasons.append(f"Next-candle forecast: C:{next_close:.2f}.")
    
     # --------------------
    # Confidence calculation (robust + interpretable)
    # --------------------
    eps = 1e-9
    score_sum = bull_score + bear_score

    if score_sum <= 0:
        conf_raw = 0.0
    else:
        # 1) consensus: how decisive is the vote (0..1)
        consensus = abs(bull_score - bear_score) / (score_sum + eps)

        # 2) magnitude: scale by how big the winning side is relative to 'strong' threshold
        # (you already use +4 to declare STRONG buy/sell; reuse that as scaling reference)
        strong_thresh = 4.0
        magnitude = min(1.0, max(bull_score, bear_score) / strong_thresh)

        # 3) reliability: downweight when ensemble error is large
        ensemble_mape = None
        try:
            ensemble_mape = float(results.get("Close", {}).get("metrics", {}).get("ENSEMBLE", {}).get("MAPE%", None))
        except Exception:
            ensemble_mape = None

        if ensemble_mape is None:
            reliability = 1.0
        else:
            # map MAPE -> reliability in (0,1], larger MAPE -> smaller reliability
            # using 1 / (1 + MAPE) is stable (MAPE expected in percent e.g. 0.1..10)
            reliability = 1.0 / (1.0 + max(0.0, ensemble_mape))

        # raw combined confidence (0..1)
        conf_raw = consensus * magnitude * reliability

    # scale to percent
    conf_pct = float(max(0.0, min(1.0, conf_raw))) * 100.0

    # label
    if conf_pct >= 66.0:
        conf_label = "HIGH"
    elif conf_pct >= 33.0:
        conf_label = "MEDIUM"
    else:
        conf_label = "LOW"

    # compose final confidence string (and also keep numeric)
    confidence_str = f"{conf_label} ({conf_pct:.1f}%)"
    # optionally include numeric confidence_score in returned dict for downstream use
    
    return {
        'signal': final_signal,
        'trend': market_trend,
        'confidence': confidence_str,
        'confidence_score': conf_pct,
        'reasons': reasons,
        'current_price': current_close,
        'predicted_price': next_close,
        'predicted_change': f"{prediction_strength:.2f}%",
        'strategies_influenced': strategiesInfluenced,
        'indicators': indicatorsCalculated
    }


# -------------------------
# CLI & I/O helpers
# -------------------------
def ensure_dirs(base="outputs"):
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base,"plots"), exist_ok=True)
    os.makedirs(os.path.join(base,"json"), exist_ok=True)

def pretty_print_results(ticker: str, timeframe: str, results: dict, cfg: CLIConfig):
    if console:
        console.rule(f"[bold cyan]Forecast Results — {ticker.upper()} ({timeframe})[/bold cyan]")
        table = Table(title="Next-step predictions (ensemble)", show_edge=False)
        table.add_column("Target")
        table.add_column("Prediction", justify="right")
        table.add_column("Top model weight", justify="right")

        # for t in ["Open", "High", "Low", "Close"]:
        for t in ["Close"]:
            if "error" in results[t]:
                table.add_row(t, "[red]Error[/red]", results[t]["error"])
                continue

            next_preds = results[t].get("adjusted_preds", {})
            ens = next_preds.get("ENSEMBLE", None)

            # Format ensemble predictions safely
            if isinstance(ens, dict):
                # Only show first 5 predictions to avoid huge strings
                ens_str = ", ".join(f"{v:.4f}" for v in list(ens.values())[:5])
            elif isinstance(ens, (list, np.ndarray)):
                ens_str = ", ".join(f"{v:.4f}" for v in ens[:5])
            elif isinstance(ens, (float, int, np.floating)):
                ens_str = f"{float(ens):.4f}"
            else:
                ens_str = "N/A"

            # Safely compute top model
            weights = results[t].get("weights", {})
            numeric_weights = {k: float(v) for k, v in weights.items() if isinstance(v, (int, float, np.floating))}
            if numeric_weights:
                top_model = max(numeric_weights.items(), key=lambda x: x[1])[0]
                top_weight = numeric_weights[top_model]
                top_str = f"{top_model} ({top_weight:.2f})"
            else:
                top_str = "N/A"

            table.add_row(t, ens_str, top_str)

        console.print(table)

        # console.print("\n[bold]Strategy highlights per target:[/bold]")
        # for t in ["Open", "High", "Low", "Close"]:
        #     strat = results[t].get("strategies", {})
        #     console.print(f"[cyan]{t}[/cyan] → {json.dumps(strat, default=str)}")
    else:
        print(json.dumps(results, indent=2, default=str))

def save_outputs(ticker: str, timeframe: str, df: pd.DataFrame, results: Dict[str, Any], cfg: CLIConfig):
    ensure_dirs(cfg.output_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Prepare JSON data
    in_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "timeframe": timeframe,
        "cfg": asdict(cfg),
        "results": results,
    }

    # Generate summary
    summary = generate_trading_signal(in_json, df.iloc[-1])

    out_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "timeframe": timeframe,
        "cfg": asdict(cfg),
        "results": results,
        "summary": summary
    }

    # Save JSON safely
    json_path = os.path.join(cfg.output_dir, "json", f"{ticker}_{timeframe}_{ts}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2, default=float)

    # Plotting
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(df.index, df["Close"], label="Close (historical)", linewidth=1)
        ax.scatter(df.index[-1], df["Close"].iloc[-1], color="black", s=30, label="Last")

        close_res = results.get("Close", {})

        # Plot ensemble prediction safely
        next_preds = close_res.get("next_preds", {})
        ens = next_preds.get("ENSEMBLE")
        if isinstance(ens, (int, float)) and len(df.index) > 1:
            delta = df.index[-1] - df.index[-2]
            next_time = df.index[-1] + delta
            ax.scatter([next_time], [ens], color="red", s=50, label=f"Ensemble next: {ens:.4f}")
        elif ens is not None:
            logger.warning(f"Skipping ensemble plot for {ticker}: ENSEMBLE is not numeric ({ens})")

        # Plot strategy annotations safely
        strat = close_res.get("strategies", {})
        breakout_signal = strat.get("breakout", {}).get("signal")
        if breakout_signal == "bullish":
            ax.annotate(
                "Breakout (bull)",
                xy=(df.index[-1], df["Close"].iloc[-1]),
                xytext=(0, 20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="green")
            )

        ax.set_title(f"{ticker.upper()} {timeframe} — Close & forecast")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.autofmt_xdate()

        # Save plot
        plot_path = os.path.join(cfg.output_dir, "plots", f"{ticker}_{timeframe}_{ts}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        logger.warning(f"Plot saving failed for {ticker}: {e}")

    # Always return JSON path
    return json_path


# -------------------------
# CLI runner
# -------------------------
def _run_predict(ticker: str,
                 timeframe: str = "1d",
                 candles: int = 360,
                 val_horizon: int = 36,
                 forecast_horizon: int = 4,
                 use_prophet: bool = True,
                 use_xgboost: bool = True,
                 use_lstm: bool = True,
                 use_cnn_lstm: bool = True,
                 use_attention_lstm: bool = True,
                 use_random_forest: bool = True,
                 use_lightgbm: bool = True,
                 lstm_epochs: int = 20,
                 lstm_batch:int = 32,
                 quiet: bool = False):
    cfg = CLIConfig(
        ticker=ticker,
        timeframe=timeframe,
        candles=candles,
        val_horizon=val_horizon,
        forecast_horizon=forecast_horizon,
        use_prophet=use_prophet,
        use_xgboost=use_xgboost,
        use_lstm=use_lstm,
        use_cnn_lstm=use_cnn_lstm,
        use_attention_lstm=use_attention_lstm,
        use_random_forest=use_random_forest,
        use_lightgbm=use_lightgbm,
        lstm_epochs=lstm_epochs,
        lstm_batch=lstm_batch,
        output_dir="outputs",
        quiet=quiet
    )
    if console and not quiet:
        console.print(Panel.fit(f"[bold green]AlphaFusion Finance[/bold green] — {cfg.ticker.upper()} — {cfg.timeframe}"))
    if console and not quiet:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
            t = prog.add_task("Fetching market data...", total=None)
            try:
                df = fetch_last_candles(ticker, timeframe, candles)
            except Exception as e:
                prog.stop()
                logger.error(f"Failed to fetch data: {e}")
                raise
            prog.update(t, description=f"Fetched {len(df)} rows")
    else:
        df = fetch_last_candles(ticker, timeframe, candles)

    results = predict_all_ohlcv(df, cfg)
    json_path = save_outputs(ticker, timeframe, df, results, cfg)
    if console and not quiet:
        console.print(f"[green]Saved JSON →[/green] {json_path}")
    pretty_print_results(ticker, timeframe, results, cfg)


# -------------------------
# CLI entrypoints
# -------------------------
if USE_TYPER:
    @app.command()
    def predict(
        ticker: str = typer.Option(..., help="Ticker symbol, e.g., AAPL"),
        timeframe: str = typer.Option("1d", help="One of: 1m,5m,15m,30m,1h,1d"),
        candles: int = typer.Option(360, help="Number of historical candles to fetch (default 360)"),
        val_horizon: int = typer.Option(36, help="Validation horizon (bars)"),
        forecast_horizon: int = typer.Option(4, help="Forecast horizon (steps)"),
        use_prophet: bool = typer.Option(False, help="Enable Prophet model if installed"),
        use_xgboost: bool = typer.Option(True, help="Enable XGBoost if installed"),
        use_lstm: bool = typer.Option(True, help="Enable LSTM (requires TensorFlow)"),
        use_cnn_lstm: bool = typer.Option(False, help="Enable CNN-LSTM (requires TensorFlow)"),
        use_attention_lstm: bool = typer.Option(True, help="Enable ATT-LSTM (requires TensorFlow)"),
        use_random_forest: bool = typer.Option(True, help="Enable RandomForest Regressor (requires TensorFlow)"),
        use_lightgbm: bool = typer.Option(True, help="Enable ATT-LSTM (requires TensorFlow)"),
        lstm_epochs: int = typer.Option(20, help="LSTM epochs"),
        lstm_batch: int = typer.Option(32, help="LSTM batch size"),
        quiet: bool = typer.Option(False, help="Quiet mode")
    ):
        _run_predict(ticker, timeframe, candles, val_horizon, forecast_horizon,
                     use_prophet, use_xgboost, use_lstm, use_cnn_lstm, use_attention_lstm, use_random_forest, use_lightgbm, lstm_epochs, lstm_batch, quiet)

    if __name__ == "__main__":
        app()
else:
    def main_argparse():
        parser = argparse.ArgumentParser(description="StockForecaster CLI")
        parser.add_argument("--ticker", required=True)
        parser.add_argument("--timeframe", default="1d", choices=list(PERIOD_FOR_INTERVAL.keys()))
        parser.add_argument("--candles", type=int, default=360)
        parser.add_argument("--val-horizon", type=int, default=36)
        parser.add_argument("--forecast-horizon", type=int, default=4)
        parser.add_argument("--no-prophet", dest="use_prophet", action="store_false")
        parser.add_argument("--no-xgb", dest="use_xgboost", action="store_false")
        parser.add_argument("--lstm", dest="use_lstm", action="store_false")
        parser.add_argument("--cnn-lstm", dest="use_cnn_lstm", action="store_false")
        parser.add_argument("--att-lstm", dest="use_attention_lstm", action="store_false")
        parser.add_argument("--random-forest", dest="use_random_forest", action="store_false")
        parser.add_argument("--lightbgm", dest="use_lightgbm", action="store_false")
        parser.add_argument("--lstm-epochs", type=int, default=20)
        parser.add_argument("--lstm-batch", type=int, default=32)
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()
        _run_predict(args.ticker, args.timeframe, args.candles, args.val_horizon, args.forecast_horizon, args.use_prophet, args.use_xgboost, args.use_lstm, args.use_cnn_lstm, args.use_attention_lstm, args.use_random_forest, args.use_lightgbm, args.lstm_epochs, args.lstm_batch, args.quiet)

    if __name__ == "__main__":
        main_argparse()
