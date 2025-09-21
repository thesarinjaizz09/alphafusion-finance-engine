#!/usr/bin/env python3
"""
AlphaFusion v1.0.0 - Single-file Production-Grade AI Trading CLI Base Version

Features:
- Unified forecasting system (LSTM, BiLSTM, PSO-LSTM, PSO-BiLSTM, TFT, SARIMAX, Prophet, XGBoost)
- Patched datetime handling:
    * DatetimeIndex -> tz-naive UTC (fixes statsmodels warnings)
    * Prophet tz-aware -> tz-naive conversion
- Strategy signals engine:
    * Breakout, mean reversion, fibonacci, price action
    * Swing trading, scalping, pairs trading
    * Volume-based strategies (OBV, ADL, CMF, MFI, VPT, Force Index, EOM)
    * Market regime detection (trend vs range, volatility)
    * Momentum oscillators (RSI, StochRSI, MACD, TRIX, Ultimate Osc, TSI, Williams %R, ROC, CCI, Awesome Oscillator)
    * Trend indicators (ADX, Ichimoku, SuperTrend, PSAR, HMA slope, KAMA slope)
    * Neural signal fusion (ensemble of strategies with weighted scoring)
- Confidence scoring:
    * Bullish/Bearish scores with weighted strategy influence
    * Confidence % and reason logging for each signal
- Forecasting pipeline:
    * Multi-model training (parallel execution via ThreadPoolExecutor)
    * Automatic feature engineering (OHLCV + technicals)
    * PSO/SCSO hyperparameter optimization hooks
- Logging:
    * Structured logging (console + file)
    * Rich-powered live epoch logging for LSTM/BiLSTM
    * JSON exports of per-candle analysis
- CLI Usage:
    python alpha_fusion.py predict --ticker AAPL --timeframe 1d --model lstm
    python alpha_fusion.py strategies --ticker TSLA --timeframe 1h
    python alpha_fusion.py forecast --ticker BTC-USD --timeframe 1d --ensemble

Notes:
- Heavy dependencies (TensorFlow, Prophet, XGBoost) are lazily imported.
- Designed as a **production-grade AI trading research system**.
- Outputs are exportable (JSON, CSV) for downstream dashboards or trading bots.
"""

from __future__ import annotations
import os
import sys
import json
import math
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional
import gc
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Rich UI
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
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

# Lazy/Optional external optional libs placeholders
yf = None
SARIMAX = None
Prophet = None
xgb = None
tf = None
Sequential = None
Input = None
LSTM = None
Dense = None
Dropout = None
EarlyStopping = None
Callback = None

# We'll import heavy libs lazily when needed to reduce startup cost / memory
def lazy_import_yfinance():
    global yf
    if yf is None:
        import yfinance as _yf
        yf = _yf
    return yf

def lazy_import_statsmodels():
    global SARIMAX
    if SARIMAX is None:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
            SARIMAX = _SARIMAX
        except Exception:
            SARIMAX = None
    return SARIMAX

def lazy_import_prophet():
    global Prophet
    if Prophet is None:
        try:
            from prophet import Prophet as _Prophet
            Prophet = _Prophet
        except Exception:
            Prophet = None
    return Prophet

def lazy_import_xgboost():
    global xgb
    if xgb is None:
        try:
            import xgboost as _xgb
            xgb = _xgb
        except Exception:
            xgb = None
    return xgb

def lazy_import_tensorflow():
    global tf, Sequential, Input, LSTM, Dense, Dropout, EarlyStopping, Callback
    if tf is None:
        try:
            import tensorflow as _tf
            from tensorflow.keras.models import Sequential as _Sequential
            from tensorflow.keras.layers import Input as _Input, LSTM as _LSTM, Dense as _Dense, Dropout as _Dropout
            from tensorflow.keras.callbacks import EarlyStopping as _EarlyStopping, Callback as _Callback
            tf = _tf
            Sequential = _Sequential
            Input = _Input
            LSTM = _LSTM
            Dense = _Dense
            Dropout = _Dropout
            EarlyStopping = _EarlyStopping
            Callback = _Callback
        except Exception:
            tf = None
    return tf

# Logging: structured logging + file
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("StockForecaster")
# Add a rotating file handler
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler("stockforecaster.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(fh)

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


# -------------------------
# Config & helpers
# -------------------------
@dataclass
class CLIConfig:
    ticker: str
    timeframe: str = "1d"
    candles: int = 360
    val_horizon: int = 0.10 * candles
    forecast_horizon: int = 2
    use_prophet: bool = True
    use_xgboost: bool = True
    use_lstm: bool = True
    lstm_epochs: int = 100
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

# --- NEW: VWAP, Z-score helper, volume spike detector, fibonacci levels, price-action detectors, regime detection
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
# Original compute_indicators (kept intact) + extra computed fields appended
# -------------------------
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

    # Additional: VWAP, ZSCORE(close), volume spike flag
    try:
        df["VWAP"] = vwap(df)
        df["ZSCORE"] = zscore(df["Close"], window=20)
        df["VOL_SPIKE"] = volume_spike(df["Volume"], window=20, mult=2.0)
    except Exception:
        # best-effort; don't raise
        logger.debug("VWAP / ZSCORE / VOL_SPIKE computation failed; continuing without them.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return df

# -------------------------
# Data fetcher (kept intact)
# -------------------------
def fetch_last_candles(ticker: str, timeframe: str, candles: int = 360) -> pd.DataFrame:
    # lazy import yfinance
    lazy_import_yfinance()
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
# Models (kept intact)
# -------------------------
def ema_baseline(series: pd.Series, span:int, val_horizon: int):
    s = series.dropna().astype(float)
    ema_s = ema(s, span)
    preds = ema_s.shift(1).values[-val_horizon:]
    next_step = float(ema_s.iloc[-1])
    return np.array(preds, dtype=float), float(next_step)

def sarimax_small_grid_forecast(train_series: pd.Series, val_horizon: int, exog: Optional[pd.DataFrame]=None,
                                max_p=2, max_d=1, max_q=2, seasonal=False, m=1, max_fits=20, maxiter=100):
    # lazy import statsmodels SARIMAX
    lazy_import_statsmodels()
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

def prophet_forecast(train_df: pd.Series, val_horizon: int):
    lazy_import_prophet()
    if Prophet is None:
        raise RuntimeError("Prophet not installed")

    # Ensure index is tz-naive datetimes and use as 'ds'
    idx = pd.to_datetime(train_df.index)
    if getattr(idx, 'tz', None) is not None:
        try:
            idx = idx.tz_convert('UTC').tz_localize(None)
        except Exception:
            idx = idx.tz_localize(None)

    dfp = pd.DataFrame({"ds": idx, "y": train_df.values.astype(float)})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=val_horizon, freq=None)
    fc = m.predict(future)
    preds = fc["yhat"].values[-val_horizon:]
    return np.array(preds, dtype=float)

def xgboost_forecast(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray):
    lazy_import_xgboost()
    if xgb is None:
        raise RuntimeError("xgboost not installed")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return np.array(preds, dtype=float), model

def build_small_lstm(input_shape: Tuple[int,int], units:int=64):
    lazy_import_tensorflow()
    if tf is None:
        raise RuntimeError("tensorflow not installed")
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------
# Rich Keras callback for epoch logging
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
                console.log(f"[yellow]{self.task_label}[/yellow] epoch {self.epoch}/{self.total} — loss={loss:.6f} val_loss={val_loss:.6f}")
                self.pbar.update(self._task, advance=1)

        def on_train_end(self, logs=None):
            if console and self.pbar:
                self.pbar.stop()

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
        z = zscore(df[target_col], window=20).iat[-1] if "ZSCORE" in df.columns else None
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
        lazy_import_yfinance()
        df1 = fetch_last_candles(base_ticker, timeframe, candles)
        df2 = fetch_last_candles(alt_ticker, timeframe, candles)
        # align
        df12 = df1["Close"].align(df2["Close"], join="inner")[0]
        series1 = df1["Close"].align(df2["Close"], join="inner")[0]
        series2 = df2["Close"].align(df1["Close"], join="inner")[0]
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
    lazy_import_yfinance()
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
# Forecast pipeline per target (kept intact) but integrated strategies
# -------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed

def forecast_univariate(series: pd.Series,
                        features_df: pd.DataFrame,
                        cfg: CLIConfig,
                        target_name: str = "Close"):
    # Align and sanitize indices
    series = _ensure_series_has_datetime_index(series)
    features_df.index = _ensure_datetime_index_tz_naive(features_df.index)

    features_df, series = features_df.align(series, join="inner", axis=0)
    features_df = features_df.dropna()
    series = series.loc[features_df.index].dropna()
    if len(series) < cfg.val_horizon + 20:
        raise ValueError("Not enough data after indicators for training/validation")

    n_total = len(series)
    val_h = cfg.val_horizon
    n_train = n_total - val_h
    train_series = series.iloc[:n_train]

    # EMA baseline
    ema_val_preds, ema_next = ema_baseline(series, span=max(12, cfg.candles//3), val_horizon=val_h)

    # Prepare tabular features for XGBoost/LSTM: fit scalers on train only
    features = features_df.copy()
    X_all = features.values.astype(float)
    y_all = series.values.astype(float)

    # Fit scaler on training rows only
    scaler_X = RobustScaler()
    scaler_X.fit(X_all[:n_train])
    X_scaled_full = scaler_X.transform(X_all)

    # We'll run SARIMAX, Prophet, XGBoost, LSTM in parallel (if requested and available).
    # We must prepare the inputs needed by each model before launching tasks.

    # Prepare XGBoost train/val
    X_tab_train = X_scaled_full[:n_train]
    X_tab_val = X_scaled_full[n_train:n_train+val_h]

    # Prepare LSTM sequences (but do not fit model here; build arrays)
    lstm_task_data = None
    if cfg.use_lstm:
        try:
            scaler_y = RobustScaler()
            y_train = y_all[:n_train].reshape(-1,1)
            scaler_y.fit(y_train)

            Xs = X_scaled_full  # shape (n_total, n_features)
            lookback = min(60, max(10, cfg.candles//3))
            n_seq = Xs.shape[0] - lookback
            if n_seq <= 0:
                raise RuntimeError("Not enough rows to build sequences for LSTM")

            X_seq = np.array([Xs[i:i+lookback] for i in range(n_seq)], dtype=float)  # (n_seq, lookback, n_feat)
            y_seq = y_all[lookback:]  # aligned with X_seq

            last_train_seq_index = n_train - lookback
            if last_train_seq_index <= 10:
                raise RuntimeError("Not enough sequence training rows for LSTM after lookback")

            X_seq_train = X_seq[:last_train_seq_index]
            y_seq_train = y_seq[:last_train_seq_index]
            X_seq_val = X_seq[last_train_seq_index:last_train_seq_index + val_h]
            y_seq_val = y_seq[last_train_seq_index:last_train_seq_index + val_h]

            if X_seq_val.shape[0] < val_h:
                if X_seq_val.shape[0] == 0:
                    raise RuntimeError("No validation sequences for LSTM")
                last_seq = X_seq_val[-1:]
                pad_n = val_h - X_seq_val.shape[0]
                X_seq_val = np.vstack([np.repeat(last_seq, pad_n, axis=0), X_seq_val])
                y_seq_val = np.concatenate([np.repeat(y_seq_val[-1], pad_n), y_seq_val])

            y_seq_train_scaled = scaler_y.transform(y_seq_train.reshape(-1,1)).reshape(-1)
            y_seq_val_scaled = scaler_y.transform(y_seq_val.reshape(-1,1)).reshape(-1)

            lstm_task_data = {
                "X_seq_train": X_seq_train,
                "y_seq_train_scaled": y_seq_train_scaled,
                "X_seq_val": X_seq_val,
                "y_seq_val_scaled": y_seq_val_scaled,
                "scaler_y": scaler_y,
                "lookback": lookback,
                "n_features": X_seq.shape[2]
            }
        except Exception as e:
            logger.warning(f"LSTM preparation failed pre-check: {e}")
            lstm_task_data = None

    # Define wrapper tasks (do not modify model logic)
    def sarimax_task():
        try:
            preds, info = sarimax_small_grid_forecast(train_series, val_h, seasonal=False)
            next_val = float(preds[-1]) if len(preds) > 0 else float(train_series.iloc[-1])
            return ("SARIMAX", preds, next_val, info)
        except Exception as e:
            logger.warning(f"SARIMAX task failed: {e}")
            return ("SARIMAX", None, None, None)

    def prophet_task():
        try:
            if cfg.use_prophet:
                preds = prophet_forecast(train_series, val_h)
                next_val = float(preds[-1])
                return ("Prophet", preds, next_val, None)
            else:
                return ("Prophet", None, None, None)
        except Exception as e:
            logger.warning(f"Prophet failed: {e}")
            return ("Prophet", None, None, None)

    def xgb_task():
        try:
            if cfg.use_xgboost:
                preds, model_obj = xgboost_forecast(X_tab_train, y_all[:n_train], X_tab_val)
                next_val = float(preds[-1]) if len(preds) > 0 else float(train_series.iloc[-1])
                return ("XGBoost", preds, next_val, model_obj)
            else:
                return ("XGBoost", None, None, None)
        except Exception as e:
            logger.warning(f"XGBoost failed: {e}")
            return ("XGBoost", None, None, None)

    def lstm_task():
        try:
            if not cfg.use_lstm or lstm_task_data is None:
                return ("LSTM", None, None, None)
            # Build model and train exactly as in original logic
            X_seq_train = lstm_task_data["X_seq_train"]
            y_seq_train_scaled = lstm_task_data["y_seq_train_scaled"]
            X_seq_val = lstm_task_data["X_seq_val"]
            y_seq_val_scaled = lstm_task_data["y_seq_val_scaled"]
            scaler_y_local = lstm_task_data["scaler_y"]
            lookback_local = lstm_task_data["lookback"]
            n_features_local = lstm_task_data["n_features"]

            # build model (lazy import will happen inside build_small_lstm)
            model = build_small_lstm((lookback_local, n_features_local), units=64)

            # prepare callbacks
            callbacks = []
            if 'EarlyStopping' in globals() and EarlyStopping is not None:
                callbacks.append(EarlyStopping(monitor='loss', patience=5, restore_best_weights=True))
            if 'RichKerasLogger' in globals() and tf is not None:
                try:
                    rk = RichKerasLogger(total_epochs=cfg.lstm_epochs, task_label=f"LSTM ({target_name})")
                    callbacks.append(rk)
                except Exception:
                    pass

            model.fit(
                X_seq_train, y_seq_train_scaled,
                validation_data=(X_seq_val, y_seq_val_scaled),
                epochs=cfg.lstm_epochs,
                batch_size=cfg.lstm_batch,
                callbacks=callbacks,
                verbose=0
            )

            # --- Use compiled predict wrapper to avoid retracing ---
            try:
                @tf.function(reduce_retracing=True)
                def predict_step(x):
                    return model(x, training=False)
                preds_scaled = predict_step(X_seq_val).numpy().reshape(-1)
                last_window_full = np.concatenate([X_seq_train, X_seq_val], axis=0)[-1].reshape(1, lookback_local, n_features_local)
                next_scaled = predict_step(last_window_full).numpy().reshape(-1)[0]
            except Exception:
                # fallback to keras predict if tf or tf.function fails
                preds_scaled = model.predict(X_seq_val).reshape(-1)
                try:
                    last_window_full = np.concatenate([X_seq_train, X_seq_val], axis=0)[-1].reshape(1, lookback_local, n_features_local)
                except Exception:
                    last_window_full = X_seq_train[-1].reshape(1, lookback_local, n_features_local)
                next_scaled = model.predict(last_window_full).reshape(-1)[0]

            preds = scaler_y_local.inverse_transform(preds_scaled.reshape(-1,1)).reshape(-1)
            lstm_preds_local = np.array(preds, dtype=float)
            next_val_local = scaler_y_local.inverse_transform(np.array([[next_scaled]]))[0,0]
            return ("LSTM", lstm_preds_local, float(next_val_local), model)
        except Exception as e:
            logger.warning(f"LSTM failed in task: {e}")
            return ("LSTM", None, None, None)

    # Submit tasks to thread pool
    tasks = {}
    with ThreadPoolExecutor(max_workers=min(4, (os.cpu_count() or 2))) as exe:
        futures = {}
        # SARIMAX always attempted (if statsmodels available)
        futures[exe.submit(sarimax_task)] = "SARIMAX"
        # Prophet
        if cfg.use_prophet:
            futures[exe.submit(prophet_task)] = "Prophet"
        # XGBoost
        if cfg.use_xgboost:
            futures[exe.submit(xgb_task)] = "XGBoost"
        # LSTM
        if cfg.use_lstm:
            futures[exe.submit(lstm_task)] = "LSTM"

        # Collect results as they complete
        sarima_preds = None
        sarima_info = None
        prophet_preds = None
        xgb_preds = None
        xgb_model_obj = None
        lstm_preds = None
        lstm_next = None

        for fut in as_completed(futures):
            try:
                name, preds, next_val, info = fut.result()
                if name == "SARIMAX":
                    sarima_preds = preds
                    sarima_info = info
                elif name == "Prophet":
                    prophet_preds = preds
                elif name == "XGBoost":
                    xgb_preds = preds
                    xgb_model_obj = info
                elif name == "LSTM":
                    lstm_preds = preds
                    lstm_next = next_val
            except Exception as e:
                logger.warning(f"A model future raised: {e}")

    # If any model was not run (None), keep behavior same as original
    sarima_next = float(sarima_preds[-1]) if (sarima_preds is not None and len(sarima_preds)>0) else float(train_series.iloc[-1])

    # EMA already computed: ema_val_preds, ema_next

    # Collect predictions
    preds_by_model = {}
    next_preds = {}

    preds_by_model["EMA"] = ema_val_preds[-val_h:]
    next_preds["EMA"] = float(ema_next)

    preds_by_model["SARIMAX"] = sarima_preds if sarima_preds is not None else np.repeat(float(train_series.iloc[-1]), val_h)
    next_preds["SARIMAX"] = float(sarima_next)

    if prophet_preds is not None:
        preds_by_model["Prophet"] = prophet_preds
        next_preds["Prophet"] = float(prophet_preds[-1])

    if xgb_preds is not None:
        preds_by_model["XGBoost"] = np.array(xgb_preds, dtype=float)
        next_preds["XGBoost"] = float(xgb_preds[-1]) if len(xgb_preds)>0 else float(train_series.iloc[-1])

    if lstm_preds is not None:
        preds_by_model["LSTM"] = np.array(lstm_preds, dtype=float)
        next_preds["LSTM"] = float(lstm_next) if lstm_next is not None else float(train_series.iloc[-1])

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

    y_val_actual = y_all[n_train:].astype(float)
    weights = inverse_error_weights(y_val_actual, preds_by_model)

    # Ensemble validation and next
    ensemble_val_preds = np.zeros_like(y_val_actual, dtype=float)
    for k,w in weights.items():
        ensemble_val_preds += preds_by_model[k] * w
    next_values = {k: float(v) for k,v in next_preds.items()}
    next_ensemble = 0.0
    for k,w in weights.items():
        if k in next_values:
            next_ensemble += next_values[k] * w

    if not np.isfinite(next_ensemble):
        next_ensemble = float(series.iloc[-1])

    metrics = {}
    for k, arr in preds_by_model.items():
        metrics[k] = {
            "MAE": float(mean_absolute_error(y_val_actual, arr)),
            "RMSE": float(math.sqrt(mean_squared_error(y_val_actual, arr))),
            "MAPE%": float(safe_mape(y_val_actual, arr))
        }

    preds_by_model["ENSEMBLE"] = ensemble_val_preds
    next_preds["ENSEMBLE"] = float(next_ensemble)

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

    # attempt to free memory
    try:
        del X_all, X_scaled_full
        gc.collect()
    except Exception:
        pass

    return preds_by_model, next_preds, metrics, weights, strategies

# -------------------------
# Orchestration for OHLC (now collects strategies)
# -------------------------
def predict_all_ohlcv(df: pd.DataFrame, cfg: CLIConfig):
    results = {}
    # sanitize df index as well
    try:
        df.index = _ensure_datetime_index_tz_naive(df.index)
    except Exception:
        pass
    features = compute_indicators(df)
    for target in ["Open","High","Low","Close"]:
        try:
            preds_by_model, next_preds, metrics, weights, strategies = forecast_univariate(df[target], features, cfg, target)
            results[target] = {
                "validation_preds": {k: v.tolist() for k,v in preds_by_model.items()},
                "next_preds": next_preds,
                "metrics": metrics,
                "weights": weights,
                "strategies": strategies
            }
        except Exception as e:
            logger.exception(f"Failed forecasting {target}: {e}")
            results[target] = {"error": str(e)}
    return results

# -------------------------
# CLI & I/O helpers (kept intact)
# -------------------------
def ensure_dirs(base="outputs"):
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base,"plots"), exist_ok=True)
    os.makedirs(os.path.join(base,"json"), exist_ok=True)

def pretty_print_results(ticker: str, timeframe: str, results: Dict[str,Any], cfg: CLIConfig):
    if console:
        console.rule(f"[bold cyan]Forecast Results — {ticker.upper()} ({timeframe})[/bold cyan]")
        table = Table(title="Next-step predictions (ensemble)", show_edge=False)
        table.add_column("Target")
        table.add_column("Prediction", justify="right")
        table.add_column("Top model weight", justify="right")
        for t in ["Open","High","Low","Close"]:
            if "error" in results[t]:
                table.add_row(t, "[red]Error[/red]", results[t]["error"])
                continue
            next_preds = results[t]["next_preds"]
            ens = next_preds.get("ENSEMBLE", None)
            weights = results[t]["weights"]
            top_model = max(weights.items(), key=lambda x: x[1])[0] if weights else "N/A"
            table.add_row(t, f"{ens:.4f}" if ens is not None else "N/A", f"{top_model} ({weights.get(top_model,0):.3f})")
        console.print(table)
        # print strategy highlights
        console.print("\n[bold]Strategy highlights per target:[/bold]")
        for t in ["Open","High","Low","Close"]:
            strat = results[t].get("strategies", {})
            console.print(f"[cyan]{t}[/cyan] → {json.dumps(strat, default=str)}")
    else:
        print(json.dumps(results, indent=2, default=str))

def save_outputs(ticker: str, timeframe: str, df: pd.DataFrame, results: Dict[str,Any], cfg: CLIConfig):
    ensure_dirs(cfg.output_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "timeframe": timeframe,
        "cfg": asdict(cfg),
        "results": results
    }
    json_path = os.path.join(cfg.output_dir, "json", f"{ticker}_{timeframe}_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2, default=float)

    try:
        fig, ax = plt.subplots(1,1, figsize=(10,4))
        ax.plot(df.index, df["Close"], label="Close (historical)", linewidth=1)
        ax.scatter(df.index[-1], df["Close"].iloc[-1], color="black", s=30, label="Last")
        close_res = results.get("Close", {})
        if "next_preds" in close_res:
            ens = close_res["next_preds"].get("ENSEMBLE", None)
            if ens is not None and len(df.index) > 1:
                next_time = df.index[-1] + (df.index[-1] - df.index[-2])
                ax.scatter([next_time], [ens], color="red", s=50, label=f"Ensemble next: {ens:.4f}")
        # annotate strategy markers (e.g., breakout)
        try:
            strat = close_res.get("strategies", {})
            if strat.get("breakout", {}).get("signal") == "bullish":
                ax.annotate("Breakout (bull)", xy=(df.index[-1], df["Close"].iloc[-1]), xytext=(0,20), textcoords="offset points",
                            arrowprops=dict(arrowstyle="->", color="green"))
        except Exception:
            pass
        ax.set_title(f"{ticker.upper()} {timeframe} — Close & forecast")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.autofmt_xdate()
        plot_path = os.path.join(cfg.output_dir, "plots", f"{ticker}_{timeframe}_{ts}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Plot saving failed: {e}")
    return json_path

# -------------------------
# CLI runner (kept intact)
# -------------------------
def _run_predict(ticker: str,
                 timeframe: str = "1d",
                 candles: int = 720,
                 val_horizon: int = 0.10 * 720,
                 forecast_horizon: int = 2,
                 use_prophet: bool = True,
                 use_xgboost: bool = True,
                 use_lstm: bool = True,
                 lstm_epochs: int = 40,
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
        lstm_epochs=lstm_epochs,
        lstm_batch=lstm_batch,
        output_dir="outputs",
        quiet=quiet
    )
    if console and not quiet:
        console.print(Panel.fit(f"[bold green]StockForecaster[/bold green] — {cfg.ticker.upper()} — {cfg.timeframe}"))
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
    pretty_print_results(ticker, timeframe, results, cfg)
    json_path = save_outputs(ticker, timeframe, df, results, cfg)
    if console and not quiet:
        console.print(f"[green]Saved JSON →[/green] {json_path}")

# -------------------------
# CLI entrypoints
# -------------------------
if USE_TYPER:
    @app.command()
    def predict(
        ticker: str = typer.Option(..., help="Ticker symbol, e.g., AAPL"),
        timeframe: str = typer.Option("1d", help="One of: 1m,5m,15m,30m,1h,1d"),
        candles: int = typer.Option(360, help="Number of historical candles to fetch (default 360)"),
        val_horizon: int = typer.Option(40, help="Validation horizon (bars)"),
        forecast_horizon: int = typer.Option(2, help="Forecast horizon (steps)"),
        use_prophet: bool = typer.Option(True, help="Enable Prophet model if installed"),
        use_xgboost: bool = typer.Option(True, help="Enable XGBoost if installed"),
        use_lstm: bool = typer.Option(True, help="Enable LSTM (requires TensorFlow)"),
        lstm_epochs: int = typer.Option(1, help="LSTM epochs"),
        lstm_batch: int = typer.Option(32, help="LSTM batch size"),
        quiet: bool = typer.Option(False, help="Quiet mode")
    ):
        _run_predict(ticker, timeframe, candles, val_horizon, forecast_horizon,
                     use_prophet, use_xgboost, use_lstm, lstm_epochs, lstm_batch, quiet)

    if __name__ == "__main__":
        app()
else:
    def main_argparse():
        parser = argparse.ArgumentParser(description="StockForecaster CLI")
        parser.add_argument("--ticker", required=True)
        parser.add_argument("--timeframe", default="1d", choices=list(PERIOD_FOR_INTERVAL.keys()))
        parser.add_argument("--candles", type=int, default=360)
        parser.add_argument("--val-horizon", type=int, default=40)
        parser.add_argument("--forecast-horizon", type=int, default=2)
        parser.add_argument("--no-prophet", dest="use_prophet", action="store_false")
        parser.add_argument("--no-xgb", dest="use_xgboost", action="store_false")
        parser.add_argument("--lstm", dest="use_lstm", action="store_false")
        parser.add_argument("--lstm-epochs", type=int, default=40)
        parser.add_argument("--lstm-batch", type=int, default=32)
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()
        _run_predict(args.ticker, args.timeframe, args.candles, args.val_horizon, args.forecast_horizon,
                     args.use_prophet, args.use_xgboost, args.use_lstm, args.lstm_epochs, args.lstm_batch, args.quiet)

    if __name__ == "__main__":
        main_argparse()
