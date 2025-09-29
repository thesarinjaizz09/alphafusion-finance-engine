#!/usr/bin/env python3
"""
AlphaFusion Finance v1.2.0-beta - Single-file AI Trading CLI (Under Development)
Copyright (c) 2025 Alphafusion. All Rights Reserved.

NOTICE:
This software is the confidential and proprietary property of Alphafusion.
Unauthorized copying, distribution, modification, or use of this software
for commercial purposes is strictly prohibited and may result in civil
and criminal penalties under the Copyright Act of India and international laws.

By using this file, you acknowledge that you have read, understood, and
agreed to these terms.

Status:
- Beta release, under active development
- Combines features from previous versions
- Core focus: HMM + SCSO-LSTM + 80+ indicators + 30+ strategy calculations
- ML model integration (multi-model ensemble, PSO, TFT, etc.) planned for upcoming releases

Features (Current):
- Technical Analysis Engine:
    * Calculates **80+ indicators** from OHLCV data
    * Categories include:
        - Trend indicators (ADX, Ichimoku, SuperTrend, PSAR, HMA slope, KAMA slope)
        - Momentum oscillators (RSI, StochRSI, MACD, TRIX, Ultimate Osc, TSI, Williams %R, ROC, CCI, Awesome Oscillator)
        - Volume indicators (OBV, ADL, CMF, MFI, VPT, Force Index, EOM)
        - Price action & volatility (Bollinger Bands, Donchian, Keltner, Fibonacci levels)
- Strategy Signals:
    * Evaluates **30+ strategies** in real-time
    * Includes breakout, mean reversion, swing trading, scalping, volume analysis, market regime detection
    * Prints weighted signals, reasons, and bullish/bearish confidence scores in the console
- Crypto & Multi-Asset Support:
    * Works with stocks, ETFs, and cryptocurrencies
    * Handles OHLCV data from Yahoo Finance, Binance, or other exchange APIs
    * Supports multiple timeframes (1m → 1d)
- Logging & Output:
    * Rich-powered console output for signals and strategy breakdown
    * JSON export for each candle with detailed indicator & strategy values
    * Production-ready folder structure for outputs and plots

Planned Features (Upcoming):
- Multi-model ML integration: XGBoost, RandomForest, SCSO-LSTM, HiddenMarkovModel
- Ensemble-based weighted prediction outputs
- Enhanced confidence scoring and reasoning system
- Full production-grade optimization (multi-threaded execution, parallel pipelines)

CLI Usage (Beta):
    python alpha_fusion.py predict --ticker AAPL --timeframe 1d
    python alpha_fusion.py strategies --ticker TSLA --timeframe 1h
    python alpha_fusion.py forecast --ticker BTC-USD --timeframe 1d

Notes:
- Designed as a **research-grade, production-ready AI trading system**, even in beta
- Current release focuses on indicator calculation and strategy signal printing
- ML forecasting and ensemble predictions to be added in future updates
"""

#--------------------
# REQUIRED PACKAGES & LIBRARIES
#--------------------
from __future__ import annotations
import re
import ta
import os
import gc
import sys
import json
import math
import time
import ccxt
import joblib
import logging
import warnings
import threading
import contextlib
import numpy as np
import talib as tlb
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from multiprocessing import Manager, Queue
from typing import Dict, Any, Tuple, List, Optional
from mplfinance.original_flavor import candlestick_ohlc
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


#--------------------
# RICH UI
#--------------------
try:
    from rich.text import Text
    from rich.panel import Panel
    from rich.table import Table
    from rich.spinner import Spinner
    from rich.console import Console, RenderableType
    from rich.traceback import install as rich_install
    from rich.progress import Progress, SpinnerColumn, Task, TextColumn, ProgressColumn
    rich_install()
    console = Console()
except Exception:
    console = None
    def rprint(*args, **kwargs): print(*args, **kwargs)


#--------------------
# CLI TYPER FALLBACK TO ARGPARSER
#--------------------
USE_TYPER = True
try:
    import typer
    app = typer.Typer(add_completion=False)
except Exception:
    USE_TYPER = False
    import argparse


#--------------------
# YFINANCE LIBRARY
#--------------------
try:
    import yfinance as yf
except Exception:
    raise RuntimeError("yfinance is required. Install: pip install yfinance")


#--------------------
# XGBoost REGRESSOR
#--------------------
try:
    from xgboost import XGBClassifier
except Exception:
    raise RuntimeError("XGBoost is required. Install: pip install xgboost")


#--------------------
# TENSORFLOW
#--------------------
with contextlib.redirect_stdout(open(os.devnull, 'w')), \
     contextlib.redirect_stderr(open(os.devnull, 'w')):
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        from tensorflow.keras import backend as K
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.callbacks import EarlyStopping, Callback
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    except Exception:
        tf = None


#--------------------
# STATSMODEL
#--------------------
try:
    from statsmodels.tsa.stattools import acf
except Exception:
    acf = None
    

#--------------------
# SCIKIT-LEARN
#--------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report


#--------------------
# LOGGING CONFIGS
#--------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("AlphaFusion-Finance")


#--------------------
# UTILITIES
#--------------------
def _ensure_datetime_index_tz_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a tz-naive DatetimeIndex. If idx is tz-aware, convert to UTC then drop tz.
    If idx is already naive, return as DatetimeIndex.
    """
    idx = pd.to_datetime(idx)
    tz = getattr(idx, 'tz', None)
    if tz is not None:
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

def normalize_freq(freq: str) -> str:
    """Map yfinance/pandas shorthand to safe pandas offsets."""
    mapping = {
        "m": "min",    # lowercase m = minutes
        "h": "H",      # hours
        "d": "D",      # days
        "wk": "W",     # weeks
        "mo": "MS",    # month start (use 'M' for month end)
    }
    
    import re
    match = re.match(r"(\d+)([A-Za-z]+)", freq)
    if not match:
        return freq
    val, unit = match.groups()
    unit = unit.lower()
    if unit in mapping:
        return f"{val}{mapping[unit]}"
    return freq

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-8
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0

def safe_filename(s: str) -> str:
    # replace invalid chars (:, /, \, etc.) with "-"
    return re.sub(r'[<>:"/\\|?*]', '-', s)

class SpinnerOrTickColumn(ProgressColumn):
    """Show spinner while running, tick when finished."""

    def render(self, task: Task) -> RenderableType:
        if task.finished:
            return Text("✔", style="green")
        return Spinner("dots", style="cyan")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif obj is np.False_:
            return False
        elif obj is np.True_:
            return True
        else:
            return super().default(obj)
@dataclass
class CLIConfig:
    ticker: str
    timeframe: str = "1d"
    candles: int = 180
    val_horizon: int = 18
    forecast_horizon: int = 5
    use_lstm: bool = True
    use_random_forest: bool = True
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


#--------------------
# CORE CLASSED
#--------------------
class FinancialDataFetcher:
    """Class to fetch stock and cryptocurrency data"""
    
    def __init__(self):
        """
        Fetch financial data from Yahoo Finance for stocks or CCXT for cryptocurrencies
        
        Parameters:
        ticker (str): Stock symbol or crypto pair (e.g., 'AAPL' or 'BTC/USDT')
        timeframe (str): Timeframe for data (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        limit (int): Number of candles to fetch
        
        Returns:
        pd.DataFrame: OHLCV data
        """
        self.candles = 200
        self.crypto_exchange = ccxt.binance()
        self.stocks_exchange = yf
        self.PERIOD_FOR_INTERVAL = PERIOD_FOR_INTERVAL
        
    def fetch_data(self, ticker: str, timeframe: str, candles: int = 360):
        self.candles = candles + self.candles
        if timeframe not in self.PERIOD_FOR_INTERVAL:
            raise ValueError(f"timeframe must be one of {list(self.PERIOD_FOR_INTERVAL.keys())}")
        period, interval = self.PERIOD_FOR_INTERVAL[timeframe]
        try:
            # Check if it's a crypto pair (contains '/')
            if '/' in ticker:
                # Fetch crypto data
                ohlcv = self.crypto_exchange.fetch_ohlcv(ticker, timeframe, limit=self.candles)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            else:
                # Fetch stock data
                tk = yf.Ticker(ticker)
                df = tk.history(period=period, interval=interval, auto_adjust=False)
    
                
            if df is None or df.empty:
                raise RuntimeError(f"No data returned from the exchange for {ticker} ({timeframe})")
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            
            try:
                df.index = _ensure_datetime_index_tz_naive(df.index)
            except Exception:
                # fallback: attempt to coerce and continue
                try:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                except Exception:
                    print("Failed to fully sanitize index timezone; proceeding with available index.")

            # ✅ Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)


            # # ✅ Ensure each OHLCV column is 1D Series
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            if len(df) < self.candles:
                print(f"Only {len(df)} candles available for {timeframe}. Requested {self.candles}. Using available.")
                
            return df.tail(self.candles)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

class FeatureEngineer:
    """Class to compute technical indicators and features"""
    
    def __init__(self):
        self.scalers = {}
        
    def add_all_indicators(self, df, cfg: CLIConfig):
        """
        Add all technical indicators to the DataFrame
        
        Parameters:
        df (pd.DataFrame): OHLCV data
        
        Returns:
        pd.DataFrame: Data with all technical indicators
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close', 'Volume' columns")
        
        if console and not cfg.quiet:
            with Progress(SpinnerOrTickColumn(), TextColumn("[progress.description]{task.description}")) as prog:
                t = prog.add_task("Initiating indicators engineer...", total=None)
                try:
                    # Add all technical indicators
                    df = self._add_trend_indicators(df)
                    df = self._add_momentum_indicators(df)
                    df = self._add_volatility_indicators(df)
                    df = self._add_volume_indicators(df)
                    
                    # Add other features
                    df = self._add_time_features(df)
                    df = self._add_price_features(df)
                    prog.update(t, description=f"[green]✔ Engineered 80+ indicators.") 
                except Exception as e:
                    prog.stop()
                    logger.error(f"Failed to detect indicators: {e}")
                    raise
        else:
            # Add all technical indicators
            df = self._add_trend_indicators(df)
            df = self._add_momentum_indicators(df)
            df = self._add_volatility_indicators(df)
            df = self._add_volume_indicators(df)
            
            # Add other features
            df = self._add_time_features(df)
            df = self._add_price_features(df)
    
        
        # Drop rows with NaN values created by indicators
        df.dropna(inplace=True)
        
        return df
    
    def _add_trend_indicators(self, df):
        """Add trend indicators"""
        # SMA
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = ta.trend.SMAIndicator(df['Close'], window=period).sma_indicator()
        
        # EMA
        for period in [5, 10, 20, 50, 100]:
            df[f'EMA_{period}'] = ta.trend.EMAIndicator(df['Close'], window=period).ema_indicator()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        
        # ADX
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        df["ATR_14"] = self._calculate_atr(df["High"], df['Low'], df['Close'], 14)
        df["OBV"] = self._calculate_obv(df['Close'], df['Volume'])
        df["RETURNS"] = df['Close'].pct_change().fillna(0)
        
        # Vortex
        vortex = ta.trend.VortexIndicator(df['High'], df['Low'], df['Close'])
        df['Vortex_pos'] = vortex.vortex_indicator_pos()
        df['Vortex_neg'] = vortex.vortex_indicator_neg()
        df['Vortex_diff'] = df['Vortex_pos'] - df['Vortex_neg']
        
        return df
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window:int=14):
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_v = tr.ewm(alpha=1/window, adjust=False).mean()
        return atr_v
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series):
        direction = np.sign(close.diff().fillna(0))
        return (direction * volume).fillna(0).cumsum()
    
    def _add_momentum_indicators(self, df):
        """Add momentum indicators"""
        # RSI
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = ta.momentum.RSIIndicator(df['Close'], window=period).rsi()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_%K'] = stoch.stoch()
        df['Stoch_%D'] = stoch.stoch_signal()
        
        # Williams %R
        df['Williams_%R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        
        # Awesome Oscillator
        df['AO'] = ta.momentum.AwesomeOscillatorIndicator(df['High'], df['Low']).awesome_oscillator()
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = ta.momentum.ROCIndicator(df['Close'], window=period).roc()
        
        return df
    
    def _add_volatility_indicators(self, df):
        """Add volatility indicators"""
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_width'] = bollinger.bollinger_wband()
        df['BB_%B'] = bollinger.bollinger_pband()
        df["BB_PCT"] = (df['Close'] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-12)
        
        # ATR
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['Keltner_upper'] = keltner.keltner_channel_hband()
        df['Keltner_middle'] = keltner.keltner_channel_mband()
        df['Keltner_lower'] = keltner.keltner_channel_lband()
        
        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
        df['Donchian_upper'] = donchian.donchian_channel_hband()
        df['Donchian_middle'] = donchian.donchian_channel_mband()
        df['Donchian_lower'] = donchian.donchian_channel_lband()
        
        return df
    
    def _add_volume_indicators(self, df):
        """Add volume indicators"""
        # OBV
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            df['Close'], df['Volume']).on_balance_volume()
        
        # CMF
        df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()
        
        # Force Index
        for period in [5, 13, 50]:
            df[f'Force_Index_{period}'] = ta.volume.ForceIndexIndicator(
                df['Close'], df['Volume'], window=period).force_index()
        
        # Volume SMA
        for period in [5, 10, 20]:
            df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
        
        # ✅ Replace Volume RSI (not in ta) with VWAP
        vwap = ta.volume.VolumeWeightedAveragePrice(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']
        )
        df['VWAP'] = vwap.volume_weighted_average_price()
        
        return df
    
    def _add_time_features(self, df):
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['week'] = df.index.isocalendar().week.astype(int)   # ✅ fixed
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df
    
    def _add_price_features(self, df):
        """Add price-based features"""
        # Returns
        df['return_1'] = df['Close'].pct_change(1)
        df['return_5'] = df['Close'].pct_change(5)
        df['return_10'] = df['Close'].pct_change(10)
        
        # Volatility
        df['volatility_5'] = df['return_1'].rolling(window=5).std()
        df['volatility_10'] = df['return_1'].rolling(window=10).std()
        df['volatility_20'] = df['return_1'].rolling(window=20).std()
        
        # High/Low ratios
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Price position in daily range
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return df

class StrategiesEngineer:
    """Compute advanced trading strategies using TA indicators + custom rules."""

    def __init__(self):
        self.indicators: Dict[str, pd.Series] = {}
        self.strategies: Dict[str, dict] = {}
        
    # ----------------------
    # Precompute Indicators
    # ----------------------
    def _precompute_indicators(self, df: pd.DataFrame, target_col: str = "Close"):
        """Precompute heavy indicators once for efficiency."""
        c = df[target_col]
        h, l = df["High"], df["Low"]
        v = df["Volume"] if "Volume" in df.columns else None

        try:
            self.indicators["ema20"] = tlb.EMA(c, timeperiod=20)
            self.indicators["ema50"] = tlb.EMA(c, timeperiod=50)
            self.indicators["ema12"] = tlb.EMA(c, timeperiod=12)
            self.indicators["ema26"] = tlb.EMA(c, timeperiod=26)

            macd, macd_signal, _ = tlb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
            self.indicators["macd"] = macd
            self.indicators["macd_signal"] = macd_signal

            self.indicators["rsi14"] = tlb.RSI(c, timeperiod=14)
            self.indicators["atr14"] = tlb.ATR(h, l, c, timeperiod=14)

            self.indicators["bb_up"], self.indicators["bb_mid"], self.indicators["bb_low"] = tlb.BBANDS(
                c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )

            self.indicators["donchian_high"] = h.rolling(window=20).max()
            self.indicators["donchian_low"] = l.rolling(window=20).min()

            self.indicators["zscore20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
            self.indicators["volume_ma20"] = v.rolling(20).mean() if v is not None else None
        except Exception as e:
            raise RuntimeError(f"Error precomputing indicators: {e}")

    # ----------------------
    # Helper: fetch indicator
    # ----------------------
    def _get(self, name: str) -> Optional[pd.Series]:
        return self.indicators.get(name)

    # ----------------------
    # Standardized output
    # ----------------------
    def _format(self, strategy: str, signal: str, confidence: float = 0.0,
            reasons: Optional[List[str]] = None, entry: Optional[float] = None,
            stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
            trend: Optional[str] = None, volume_spike: Optional[bool] = None) -> dict:
        return {
            "strategy": strategy,
            "signal": signal,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "reasons": reasons if isinstance(reasons, list) else ([reasons] if reasons else []),
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trend": trend,
            "volume_spike": volume_spike
        }

    # ----------------------
    # Market Regime Strategy
    # ----------------------
    def detect_market_regime(self, df: pd.DataFrame, window: int = 20, target_col: str = "Close") -> dict:
        """Detects volatility regime using ATR, EMA trend, and volume."""
        if len(df) < window:
            return self._format("market_regime", "none", reasons="Not enough data")

        try:
            atr = self._get("atr14")
            ema20 = self._get("ema20")
            ema50 = self._get("ema50")
            returns_vol = df[target_col].pct_change().rolling(window).std()
            avg_vol = df["Volume"].rolling(window).mean()

            # Trend detection
            diff = (ema20.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]
            if abs(diff) < 0.002:
                trend = "sideways"
            elif ema20.iloc[-1] > ema50.iloc[-1]:
                trend = "bullish"
            else:
                trend = "bearish"

            vol_spike = df["Volume"].iloc[-1] > 1.5 * avg_vol.iloc[-1]

            # Volatility regime
            if returns_vol.iloc[-1] < 0.005 and atr.iloc[-1] < 0.01 * df[target_col].iloc[-1]:
                signal = "low_volatility"
                reason = "Sideways regime / accumulation"
            elif returns_vol.iloc[-1] < 0.02:
                signal = "medium_volatility"
                reason = "Trending regime with controlled volatility"
            else:
                signal = "high_volatility"
                reason = "Explosive volatility regime"

            confidence = max(0.0, min(1.0, 1 - (returns_vol.iloc[-1] / 0.05)))

            stop_loss = df[target_col].iloc[-1] - 1.5 * atr.iloc[-1] if trend == "bullish" else df[target_col].iloc[-1] + 1.5 * atr.iloc[-1]
            take_profit = df[target_col].iloc[-1] + 3 * atr.iloc[-1] if trend == "bullish" else df[target_col].iloc[-1] - 3 * atr.iloc[-1]

            return self._format(
                strategy="market_regime",
                signal=signal,
                confidence=confidence,
                reasons=[reason],
                entry=df[target_col].iloc[-1],
                stop_loss=stop_loss,
                take_profit=take_profit,
                trend=trend,
                volume_spike=vol_spike
            )
        except Exception as e:
            return self._format("market_regime", "error", reasons=str(e), trend=None, volume_spike=None)

    # ----------------------
    # Breakout Strategy
    # ----------------------
    def detect_breakout(self, df: pd.DataFrame, vol_mult: float = 2.0, target_col: str = "Close") -> dict:
        """
        Detect breakout using Bollinger Bands, Donchian Channels, ATR, volume, and EMA trend.
        Returns standardized output including trend, volume spike, confidence, and risk levels.
        """
        if len(df) < 50:
            return self._format(
                strategy="breakout",
                signal="none",
                reasons="Not enough data",
                trend=None,
                volume_spike=None
            )

        try:
            c = df[target_col]
            h, l, v = df["High"], df["Low"], df["Volume"]

            # Use precomputed indicators if available
            upper_bb = self._get("bb_up") if self._get("bb_up") is not None else tlb.BBANDS(c, timeperiod=20)[0]
            lower_bb = self._get("bb_low") if self._get("bb_low") is not None else tlb.BBANDS(c, timeperiod=20)[2]
            ema20 = self._get("ema20") if self._get("ema20") is not None else tlb.EMA(c, timeperiod=20)
            atr = self._get("atr14") if self._get("atr14") is not None else tlb.ATR(h, l, c, timeperiod=14)
            upper_dc = h.rolling(20).max()
            lower_dc = l.rolling(20).min()

            # Last scalar values
            last = c.iat[-1]
            last_upper_bb, last_lower_bb = upper_bb.iat[-1], lower_bb.iat[-1]
            last_ema20, last_atr = ema20.iat[-1], atr.iat[-1]
            last_upper_dc, last_lower_dc = upper_dc.iat[-1], lower_dc.iat[-1]

            # Trend detection
            trend = "bullish" if last > last_ema20 else "bearish" if last < last_ema20 else "sideways"

            # Volume spike
            avg_vol = v.rolling(20).mean().iat[-1]
            volume_spike = v.iat[-1] > vol_mult * avg_vol

            # Volatility baseline
            vol_std = c.pct_change().rolling(14).std().iat[-1]

            # Initialize output
            signal, reasons, confidence = "none", [], 0.0

            # Bullish breakout
            if last > last_upper_bb and last > last_upper_dc and last > last_ema20 and last_atr > vol_std:
                signal = "bullish"
                reasons = ["Price > BB upper", "Price > Donchian high", "Above EMA20", "ATR confirms volatility"]
                confidence = min(1.0, (last - last_ema20) / last_atr + (0.2 if volume_spike else 0))

            # Bearish breakout
            elif last < last_lower_bb and last < last_lower_dc and last < last_ema20 and last_atr > vol_std:
                signal = "bearish"
                reasons = ["Price < BB lower", "Price < Donchian low", "Below EMA20", "ATR confirms volatility"]
                confidence = min(1.0, (last_ema20 - last) / last_atr + (0.2 if volume_spike else 0))

            # Risk management
            stop_loss, take_profit = None, None
            if signal == "bullish":
                stop_loss = last - 2 * last_atr
                take_profit = last + 3 * last_atr
            elif signal == "bearish":
                stop_loss = last + 2 * last_atr
                take_profit = last - 3 * last_atr

            # Return standardized output
            return self._format(
                strategy="breakout",
                signal=signal,
                confidence=round(confidence, 2),
                reasons=reasons if reasons else ["No breakout detected"],
                entry=last,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trend=trend,
                volume_spike=volume_spike
            )
        except Exception as e:
            return self._format(
                strategy="breakout",
                signal="error",
                reasons=str(e),
                trend=None,
                volume_spike=None
            )

    # ----------------------
    # Mean Reversion Strategy
    # ----------------------
    def detect_mean_reversion(self, df: pd.DataFrame, target_col: str = "Close") -> dict:
        """Detect mean reversion with RSI, Bollinger Bands, Z-score, and volume (TA-Lib version) with uniform output."""
        try:
            if len(df) < 20:
                return self._format(
                    strategy="mean_reversion",
                    signal="none",
                    confidence=0.0,
                    reasons=["not enough data"],
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    volume_spike=False,
                    trend="sideways"
                )

            # Precomputed or calculate indicators
            rsi_series = self._get("rsi14") if self._get("rsi14") is not None else tlb.RSI(df[target_col], timeperiod=14)
            rsi = rsi_series.iloc[-1]  # always the last value
            upper_bb, middle_bb, lower_bb = tlb.BBANDS(df[target_col], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            mean = df[target_col].rolling(20).mean().iloc[-1]
            std = df[target_col].rolling(20).std().iloc[-1]
            last = df[target_col].iloc[-1]
            z = ((df[target_col] - df[target_col].rolling(20).mean()) / df[target_col].rolling(20).std()).iloc[-1]

            vol_spike = df['Volume'].iloc[-1] > 1.5 * df['Volume'].rolling(20).mean().iloc[-1]

            # Trend detection (bullish / bearish / sideways)
            ema20 = self._get("ema20") if self._get("ema20") is not None else tlb.EMA(df[target_col], timeperiod=20)
            ema50 = self._get("ema50") if self._get("ema50") is not None else tlb.EMA(df[target_col], timeperiod=50)
            if ema20.iloc[-1] > ema50.iloc[-1]:
                trend = "bullish"
            elif ema20.iloc[-1] < ema50.iloc[-1]:
                trend = "bearish"
            else:
                trend = "sideways"

            # Conditions
            reasons, signal, confidence = [], "none", 0.0
            if rsi > 70: reasons.append("RSI_overbought")
            elif rsi < 30: reasons.append("RSI_oversold")
            if last >= upper_bb.iloc[-1]: reasons.append("BB_upper_touch")
            if last <= lower_bb.iloc[-1]: reasons.append("BB_lower_touch")
            if z > 2: reasons.append("Zscore_high")
            elif z < -2: reasons.append("Zscore_low")

            if any(x in reasons for x in ["RSI_oversold", "BB_lower_touch", "Zscore_low"]):
                signal = "buy_revert"
                confidence = min(1.0, abs(z) / 3 + 0.2)
            elif any(x in reasons for x in ["RSI_overbought", "BB_upper_touch", "Zscore_high"]):
                signal = "sell_revert"
                confidence = min(1.0, z / 3 + 0.2)

            # Risk management
            stop_loss = last + 2 * std if signal == "buy_revert" else last - 2 * std if signal == "sell_revert" else None
            take_profit = mean if signal in ["buy_revert", "sell_revert"] else None

            return self._format(
                strategy="mean_reversion",
                signal=signal,
                confidence=round(confidence, 2),
                reasons=reasons,
                entry=last,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume_spike=vol_spike,
                trend=trend
            )

        except Exception as e:
            return self._format(
                strategy="mean_reversion",
                signal="none",
                confidence=0.0,
                reasons=[f"error: {str(e)}"],
                entry=None,
                stop_loss=None,
                take_profit=None,
                volume_spike=False,
                trend="sideways"
            )

    # ----------------------
    # Price Action Strategy
    # ----------------------
    def detect_price_action(self, df: pd.DataFrame, idx_offset: int = 0) -> dict:
        """Detect candlestick patterns with volume confirmation and uniform output."""
        try:
            if len(df) < 2:
                return self._format(
                    strategy="price_action",
                    signal="none",
                    confidence=0.0,
                    reasons=["not enough data"],
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    volume_spike=False,
                    trend="sideways"
                )

            idx = len(df) - 1 - idx_offset
            last = df.iloc[idx]
            prev = df.iloc[idx - 1]

            vol_spike = df["Volume"].iloc[-1] > 1.5 * df["Volume"].rolling(20).mean().iloc[-1]

            bullish_engulfing = last["Close"] > last["Open"] and prev["Close"] < prev["Open"] and last["Close"] > prev["Open"]
            bearish_engulfing = last["Close"] < last["Open"] and prev["Close"] > prev["Open"] and last["Close"] < prev["Open"]
            pinbar = abs(last["Close"] - last["Open"]) < (last["High"] - last["Low"]) * 0.25

            # Confidence based on candle body vs range
            body_size = abs(last['Close'] - last['Open'])
            range_size = last['High'] - last['Low']
            confidence = round(body_size / range_size if range_size != 0 else 0, 2)

            # Determine main signal
            if bullish_engulfing:
                signal = "bullish"
                reasons = ["bullish_engulfing"]
                entry = last["Close"]
                stop_loss = last["Low"]
                take_profit = last["Close"] + (last["Close"] - last["Low"]) * 2  # 2:1 RR
            elif bearish_engulfing:
                signal = "bearish"
                reasons = ["bearish_engulfing"]
                entry = last["Close"]
                stop_loss = last["High"]
                take_profit = last["Close"] - (last["High"] - last["Close"]) * 2  # 2:1 RR
            elif pinbar:
                signal = "pin_bar"
                reasons = ["pin_bar"]
                entry = last["Close"]
                stop_loss = last["Low"] if last["Close"] > last["Open"] else last["High"]
                take_profit = last["Close"] + (last["Close"] - stop_loss) * 2
            else:
                signal = "none"
                reasons = ["no clear pattern"]
                entry = stop_loss = take_profit = None

            # Trend detection
            ema20 = self._get("ema20") if self._get("ema20") is not None else tlb.EMA(df["Close"], timeperiod=20)
            ema50 = self._get("ema50") if self._get("ema50") is not None else tlb.EMA(df["Close"], timeperiod=50)
            if ema20.iloc[-1] > ema50.iloc[-1]:
                trend = "bullish"
            elif ema20.iloc[-1] < ema50.iloc[-1]:
                trend = "bearish"
            else:
                trend = "sideways"

            return self._format(
                strategy="price_action",
                signal=signal,
                confidence=confidence,
                reasons=reasons,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume_spike=vol_spike,
                trend=trend
            )

        except Exception as e:
            return self._format(
                strategy="price_action",
                signal="none",
                confidence=0.0,
                reasons=[f"error: {str(e)}"],
                entry=None,
                stop_loss=None,
                take_profit=None,
                volume_spike=False,
                trend="sideways"
            )

    # ----------------------
    # Swing Trade Strategy
    # ----------------------
    def detect_swing_trade(self, df: pd.DataFrame, target_col: str = "Close") -> dict:
        """Detect swing trades with EMA cross, MACD, RSI, ATR, volume, and standardized output."""
        try:
            if len(df) < 50:
                return self._format(
                    strategy="swing_trade",
                    signal="none",
                    confidence=0.0,
                    reasons=["not enough data"],
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    volume_spike=False,
                    trend="sideways"
                )

            ema12 = tlb.EMA(df[target_col], timeperiod=12)
            ema26 = tlb.EMA(df[target_col], timeperiod=26)
            macd, macd_signal, macd_hist = tlb.MACD(df[target_col], fastperiod=12, slowperiod=26, signalperiod=9)
            rsi = tlb.RSI(df[target_col], timeperiod=14)
            atr = tlb.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
            avg_vol = df["Volume"].rolling(20).mean()

            last_ema12 = ema12.iloc[-1]
            last_ema26 = ema26.iloc[-1]
            last_macd_hist = macd_hist.iloc[-1]
            last_rsi = rsi.iloc[-1]
            last_atr = atr.iloc[-1]
            last_vol = df["Volume"].iloc[-1]
            avg_vol_last = avg_vol.iloc[-1]

            # Volume spike
            vol_spike = last_vol > 1.5 * avg_vol_last

            reasons, signal = [], "none"

            # Bullish swing
            if (
                last_ema12 > last_ema26
                and last_macd_hist > 0
                and last_rsi < 70
                and last_atr > 0
                and last_vol > avg_vol_last
            ):
                signal = "bullish"
                reasons = ["EMA cross up", "MACD positive", "RSI < 70", "ATR valid", "Volume confirmed"]

            # Bearish swing
            elif (
                last_ema12 < last_ema26
                and last_macd_hist < 0
                and last_rsi > 30
                and last_atr > 0
                and last_vol > avg_vol_last
            ):
                signal = "bearish"
                reasons = ["EMA cross down", "MACD negative", "RSI > 30", "ATR valid", "Volume confirmed"]

            # Confidence: proportion of conditions met
            conditions = [
                last_ema12 > last_ema26 if signal == "bullish" else last_ema12 < last_ema26,
                last_macd_hist > 0 if signal == "bullish" else last_macd_hist < 0,
                (last_rsi < 70 if signal == "bullish" else last_rsi > 30),
                last_atr > 0,
                last_vol > avg_vol_last
            ]
            confidence = sum(conditions) / len(conditions) if signal != "none" else 0.0

            # Entry, stop loss, take profit
            entry = df[target_col].iloc[-1] if signal != "none" else None
            stop_loss = entry - 1.5 * last_atr if signal == "bullish" else (
                        entry + 1.5 * last_atr if signal == "bearish" else None)
            take_profit = entry + 3 * last_atr if signal == "bullish" else (
                        entry - 3 * last_atr if signal == "bearish" else None)

            # Trend detection
            ema20 = self._get("ema20") if self._get("ema20") is not None else tlb.EMA(df["Close"], timeperiod=20)
            ema50 = self._get("ema50") if self._get("ema50") is not None else tlb.EMA(df["Close"], timeperiod=50)
            if ema20.iloc[-1] > ema50.iloc[-1]:
                trend = "bullish"
            elif ema20.iloc[-1] < ema50.iloc[-1]:
                trend = "bearish"
            else:
                trend = "sideways"

            return self._format(
                strategy="swing_trade",
                signal=signal,
                confidence=round(confidence, 2),
                reasons=reasons,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume_spike=vol_spike,
                trend=trend
            )

        except Exception as e:
            return self._format(
                strategy="swing_trade",
                signal="none",
                confidence=0.0,
                reasons=[f"error: {str(e)}"],
                entry=None,
                stop_loss=None,
                take_profit=None,
                volume_spike=False,
                trend="sideways"
            )

    # ----------------------
    # Scalping Strategy
    # ----------------------
    def detect_scalping_opportunity(self, df: pd.DataFrame, target_col: str = "Close") -> dict:
        """Detect scalping opportunities using VWAP, RSI, StochRSI, volume, and standardized output."""
        try:
            if "VWAP" not in df.columns:
                df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

            last = df[target_col].iloc[-1]
            vwap_last = df["VWAP"].iloc[-1]
            sma5 = tlb.SMA(df[target_col], timeperiod=5)[-1]
            sma20 = tlb.SMA(df[target_col], timeperiod=20)[-1]
            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
            rsi = tlb.RSI(df[target_col], timeperiod=14)[-1]
            fastk, fastd = tlb.STOCHRSI(df[target_col], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
            stochrsi = fastk[-1]

            # Volume spike
            vol_spike = df["Volume"].iloc[-1] > 1.5 * avg_vol

            # Trend detection
            ema20 = self._get("ema20") if self._get("ema20") is not None else tlb.EMA(df["Close"], timeperiod=20)
            ema50 = self._get("ema50") if self._get("ema50") is not None else tlb.EMA(df["Close"], timeperiod=50)
            if ema20.iloc[-1] > ema50.iloc[-1]:
                trend = "bullish"
            elif ema20.iloc[-1] < ema50.iloc[-1]:
                trend = "bearish"
            else:
                trend = "sideways"

            risk_pct = 0.003  # 0.3%
            reward_multiple = 2.0
            reasons, signal, confidence, entry, stop_loss, take_profit = [], "none", 0.0, None, None, None

            # LONG SETUP
            if last > vwap_last and df["Volume"].iloc[-1] > avg_vol and rsi < 70 and stochrsi > 0.5:
                signal = "long_momentum"
                reasons = ["VWAP above price", "Volume high", "RSI < 70", "StochRSI rising"]
                confidence += 0.25 * sum([last > vwap_last, df["Volume"].iloc[-1] > avg_vol, rsi < 70, stochrsi > 0.5])
                entry = round(last, 2)
                stop_loss = round(last * (1 - risk_pct), 2)
                take_profit = round(last * (1 + risk_pct * reward_multiple), 2)

            # SHORT SETUP
            elif last < vwap_last and df["Volume"].iloc[-1] > avg_vol and rsi > 30 and stochrsi < 0.5:
                signal = "short_momentum"
                reasons = ["VWAP below price", "Volume high", "RSI > 30", "StochRSI falling"]
                confidence += 0.25 * sum([last < vwap_last, df["Volume"].iloc[-1] > avg_vol, rsi > 30, stochrsi < 0.5])
                entry = round(last, 2)
                stop_loss = round(last * (1 + risk_pct), 2)
                take_profit = round(last * (1 - risk_pct * reward_multiple), 2)

            return self._format(
                strategy="scalping_opportunity",
                signal=signal,
                confidence=round(confidence, 2),
                reasons=reasons if reasons else "No valid setup",
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume_spike=vol_spike,
                trend=trend
            )

        except Exception as e:
            return self._format(
                strategy="scalping_opportunity",
                signal="none",
                confidence=0.0,
                reasons=[f"error: {str(e)}"],
                entry=None,
                stop_loss=None,
                take_profit=None,
                volume_spike=False,
                trend="sideways"
            )

    # ----------------------
    # Options Strategy
    # ----------------------
    def fetch_options_flow_stub(self, ticker: str) -> dict:
        """Fetch basic options flow data (open interest sums) using yfinance."""
        out = {
            "expiries": [],
            "calls_oi_sum": None,
            "puts_oi_sum": None
        }
        try:
            tk = yf.Ticker(ticker)
            exps = tk.options
            if exps:
                out["expiries"] = list(exps)
                chain = tk.option_chain(out["expiries"][0])
                calls, puts = chain.calls, chain.puts
                if "openInterest" in calls.columns:
                    out["calls_oi_sum"] = int(calls["openInterest"].sum())
                if "openInterest" in puts.columns:
                    out["puts_oi_sum"] = int(puts["openInterest"].sum())
        except Exception as e:
            logger.debug(f"Options scan failed for {ticker} (yfinance): {e}")
        return out

    # ----------------------
    # Trend Strategy
    # ----------------------
    def detect_trend_strength(self, df: pd.DataFrame, target_col: str = "Close") -> dict:
        """
        Detect market trend strength using EMA10 vs EMA50,
        with volume spike and market regime detection.
        """
        try:
            if len(df) < 50:
                return self._format(
                    strategy="trend_strength",
                    signal="none",
                    confidence=0.0,
                    reasons=["not enough data"],
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    trend=None,
                    volume_spike=None,
                )

            # Use precomputed indicators if available
            ema10 = self._get("ema10") if "ema10" in self.indicators else tlb.EMA(df[target_col], timeperiod=10)
            ema50 = self._get("ema50") if "ema50" in self.indicators else tlb.EMA(df[target_col], timeperiod=50)
            last_ema10 = ema10.iloc[-1]
            last_ema50 = ema50.iloc[-1]

            # Trend calculation
            diff = last_ema10 - last_ema50
            if diff > 0:
                signal = "bull_trend"
                trend = "bullish"
            elif diff < 0:
                signal = "bear_trend"
                trend = "bearish"
            else:
                signal = "sideways"
                trend = "sideways"

            # Confidence: normalized EMA difference
            confidence = float(np.clip(abs(diff) / last_ema50, 0.0, 1.0))

            # Volume spike detection
            vol_spike = df["Volume"].iloc[-1] > 1.5 * df["Volume"].rolling(20).mean().iloc[-1] if "Volume" in df.columns else False

            # Entry, stop loss, take profit estimates (realistic)
            atr = self._get("atr14") if "atr14" in self.indicators else tlb.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
            last_atr = atr.iloc[-1]
            last_price = df[target_col].iloc[-1]

            entry = last_price
            stop_loss = last_price - 1.5 * last_atr if signal == "bull_trend" else (last_price + 1.5 * last_atr if signal == "bear_trend" else None)
            take_profit = last_price + 3 * last_atr if signal == "bull_trend" else (last_price - 3 * last_atr if signal == "bear_trend" else None)

            return self._format(
                strategy="trend_strength",
                signal=signal,
                confidence=confidence,
                reasons=[f"EMA10 {'>' if diff>0 else '<' if diff<0 else '='} EMA50"],
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trend=trend,
                volume_spike=vol_spike,
            )

        except Exception as e:
            return self._format(
                strategy="trend_strength",
                signal="error",
                confidence=0.0,
                reasons=[f"Error in trend detection: {e}"],
                entry=None,
                stop_loss=None,
                take_profit=None,
                trend=None,
                volume_spike=None,
            )

    # ----------------------
    # Fibonacci Strategy
    # ----------------------
    def detect_fibonacci_pullback(self, df: pd.DataFrame, target_col: str = "Close") -> dict:
        """
        Detect Fibonacci pullbacks with ATR, trend, and EMA confirmation,
        including precomputed indicators, uniform output, confidence,
        realistic SL/TP, volume spike, and error handling.
        """
        try:
            if len(df) < 50:
                return self._format(
                    strategy="fibonacci_pullback",
                    signal="none",
                    confidence=0.0,
                    reasons=["not enough data"],
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    trend=None,
                    volume_spike=None,
                )

            # Price high/low range
            high, low = df[target_col].max(), df[target_col].min()
            diff = high - low

            # Fibonacci retracement levels
            levels = {
                "0.236": high - 0.236 * diff,
                "0.382": high - 0.382 * diff,
                "0.5": high - 0.5 * diff,
                "0.618": high - 0.618 * diff,
                "0.786": high - 0.786 * diff,
            }

            last_price = df[target_col].iloc[-1]

            # Nearest Fibonacci level
            nearest_level_key, nearest_level_value = min(levels.items(), key=lambda x: abs(last_price - x[1]))

            # ATR & EMA using precomputed if available
            atr = self._get("atr14").iloc[-1] if "atr14" in self.indicators else tlb.ATR(df["High"], df["Low"], df["Close"], timeperiod=14).iloc[-1]
            ema50 = self._get("ema50").iloc[-1] if "ema50" in self.indicators else tlb.EMA(df[target_col], timeperiod=50).iloc[-1]

            # Trend confirmation
            trend = "bullish" if last_price > ema50 else "bearish"
            signal = "pullback"  # uniform naming

            # Volume spike
            vol_spike = df["Volume"].iloc[-1] > 1.5 * df["Volume"].rolling(20).mean().iloc[-1] if "Volume" in df.columns else False

            # Confidence: inverse distance to nearest level normalized by ATR
            confidence = float(np.clip(1 - abs(last_price - nearest_level_value) / atr, 0.0, 1.0))

            # Risk management: realistic SL/TP
            stop_loss = last_price - 1.5 * atr if trend == "bullish" else last_price + 1.5 * atr
            take_profit = last_price + 3 * atr if trend == "bullish" else last_price - 3 * atr

            return self._format(
                strategy="fibonacci_pullback",
                signal=signal,
                confidence=confidence,
                reasons=[f"Nearest Fibonacci: {nearest_level_key}"],
                entry=last_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trend=trend,
                volume_spike=vol_spike,
            )

        except Exception as e:
            return self._format(
                strategy="fibonacci_pullback",
                signal="error",
                confidence=0.0,
                reasons=[f"Error in Fibonacci pullback detection: {e}"],
                entry=None,
                stop_loss=None,
                take_profit=None,
                trend=None,
                volume_spike=None,
            )

    # ----------------------
    # Volume Spike Strategy
    # ----------------------
    def detect_volume_spike(self, df: pd.DataFrame):
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        last_vol = df['Volume'].iloc[-1]
        last_price = df['Close'].iloc[-1]

        signal = 'none'
        confidence = 0.0
        stop_loss = take_profit = None

        vol_spike = last_vol > 1.5 * avg_vol
        if vol_spike:
            signal = 'volume_spike'
            confidence = round((last_vol - avg_vol)/avg_vol, 2)
            atr = self._get('atr14').iloc[-1] if self._get('atr14') is not None else (last_price*0.01)
            stop_loss = last_price - atr
            take_profit = last_price + 2*atr

        trend = 'bullish' if self._get('ema20').iloc[-1] > self._get('ema50').iloc[-1] else 'bearish'

        return self._format("volume_spike", signal, confidence, reasons=["Volume spike detected"] if vol_spike else [], 
                            entry=last_price, stop_loss=stop_loss, take_profit=take_profit, trend=trend, volume_spike=vol_spike)

    # ----------------------
    # RSI Strategy
    # ----------------------
    def detect_rsi_strategy(self, df: pd.DataFrame):
        last_price = df['Close'].iloc[-1]
        rsi = tlb.RSI(df['Close'], timeperiod=14).iloc[-1]
        atr = self._get('atr14').iloc[-1] if self._get('atr14') is not None else (last_price*0.01)

        signal = 'none'
        confidence = 0.0
        stop_loss = take_profit = None
        reasons = []

        if rsi > 70:
            signal = 'overbought'
            confidence = round((rsi-70)/30, 2)
            stop_loss = last_price + atr
            take_profit = last_price - 2*atr
            reasons.append("RSI overbought")
        elif rsi < 30:
            signal = 'oversold'
            confidence = round((30-rsi)/30, 2)
            stop_loss = last_price - atr
            take_profit = last_price + 2*atr
            reasons.append("RSI oversold")

        trend = 'bullish' if self._get('ema20').iloc[-1] > self._get('ema50').iloc[-1] else 'bearish'
        vol_spike = df['Volume'].iloc[-1] > 1.5 * df['Volume'].rolling(20).mean().iloc[-1]

        return self._format("rsi", signal, confidence, reasons, entry=last_price, stop_loss=stop_loss, 
                            take_profit=take_profit, trend=trend, volume_spike=vol_spike)

    # ----------------------
    # MACD Strategy
    # ----------------------
    def detect_macd_strategy(self, df: pd.DataFrame):
        last_price = df['Close'].iloc[-1]
        macd, signal_line, _ = tlb.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        macd_last = macd.iloc[-1]
        signal_last = signal_line.iloc[-1]
        atr = self._get('atr14').iloc[-1] if self._get('atr14') is not None else (last_price*0.01)

        signal = 'none'
        confidence = 0.0
        stop_loss = take_profit = None
        reasons = []

        if macd_last > signal_last:
            signal = 'bull_macd'
            confidence = round((macd_last - signal_last)/abs(signal_last+1e-12), 2)
            stop_loss = last_price - atr
            take_profit = last_price + 2*atr
            reasons.append("MACD bullish crossover")
        elif macd_last < signal_last:
            signal = 'bear_macd'
            confidence = round((signal_last - macd_last)/abs(signal_last+1e-12), 2)
            stop_loss = last_price + atr
            take_profit = last_price - 2*atr
            reasons.append("MACD bearish crossover")

        trend = 'bullish' if self._get('ema20').iloc[-1] > self._get('ema50').iloc[-1] else 'bearish'
        vol_spike = df['Volume'].iloc[-1] > 1.5 * df['Volume'].rolling(20).mean().iloc[-1]

        return self._format("macd", signal, confidence, reasons, entry=last_price, stop_loss=stop_loss, 
                            take_profit=take_profit, trend=trend, volume_spike=vol_spike)

    # ----------------------
    # Bollinger Strategy
    # ----------------------
    def detect_bollinger_strategy(self, df: pd.DataFrame):
        last_price = df['Close'].iloc[-1]
        bb_upper, bb_middle, bb_lower = tlb.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        atr = self._get('atr14').iloc[-1] if self._get('atr14') is not None else (last_price*0.01)

        signal = 'none'
        confidence = 0.0
        stop_loss = take_profit = None
        reasons = []

        if last_price > bb_upper.iloc[-1]:
            signal = 'sell_bb'
            confidence = round((last_price - bb_upper.iloc[-1])/bb_upper.iloc[-1], 2)
            stop_loss = last_price + atr
            take_profit = last_price - 2*atr
            reasons.append("Price above BB upper")
        elif last_price < bb_lower.iloc[-1]:
            signal = 'buy_bb'
            confidence = round((bb_lower.iloc[-1] - last_price)/bb_lower.iloc[-1], 2)
            stop_loss = last_price - atr
            take_profit = last_price + 2*atr
            reasons.append("Price below BB lower")

        trend = 'bullish' if self._get('ema20').iloc[-1] > self._get('ema50').iloc[-1] else 'bearish'
        vol_spike = df['Volume'].iloc[-1] > 1.5 * df['Volume'].rolling(20).mean().iloc[-1]

        return self._format("bollinger", signal, confidence, reasons, entry=last_price, stop_loss=stop_loss,
                            take_profit=take_profit, trend=trend, volume_spike=vol_spike)

    # ----------------------
    # Pro Ad-Ons
    # ----------------------
    def detect_extra_strategies(self, df: pd.DataFrame):
        """Calculates all pro strategies with full multi-condition analysis."""
        close_last = df["Close"].iat[-1]
        last_vol = df["Volume"].iat[-1]
        avg_vol = df["Volume"].rolling(20).mean().iat[-1]
        atr = self._get('atr14').iloc[-1] if self._get('atr14') is not None else (close_last*0.01)
        trend = 'bullish' if self._get('ema20').iloc[-1] > self._get('ema50').iloc[-1] else 'bearish'
        vol_spike = last_vol > 1.5 * avg_vol

        # ---------------- TREND STRATEGIES ---------------- #
        try:
            ichi = ta.trend.IchimokuIndicator(df["High"], df["Low"], window1=9, window2=26, window3=52)
            ichi_a = ichi.ichimoku_a().iat[-1]
            ichi_b = ichi.ichimoku_b().iat[-1]
            if close_last > ichi_a and close_last > ichi_b:
                signal, reasons = "bullish", ["above ichimoku cloud"]
            elif close_last < ichi_a and close_last < ichi_b:
                signal, reasons = "bearish", ["below ichimoku cloud"]
            else:
                signal, reasons = "neutral", ["inside ichimoku cloud"]

            self.strategies["ichimoku"] = self._format("ichimoku", signal, confidence=0.8 if signal!="neutral" else 0.0,
                                                    reasons=reasons, entry=close_last,
                                                    stop_loss=close_last-atr if signal=="bullish" else close_last+atr if signal=="bearish" else None,
                                                    take_profit=close_last+2*atr if signal=="bullish" else close_last-2*atr if signal=="bearish" else None,
                                                    trend=trend, volume_spike=vol_spike)
        except Exception:
            self.strategies["ichimoku"] = None

        try:
            psar = ta.trend.PSARIndicator(df["High"], df["Low"], df["Close"])
            psar_val = psar.psar().iat[-1]
            signal = "bullish" if close_last > psar_val else "bearish"
            reasons = ["PSAR trend"]
            self.strategies["psar_trend"] = self._format("psar_trend", signal, confidence=0.8,
                                                        reasons=reasons, entry=close_last,
                                                        stop_loss=close_last-atr if signal=="bullish" else close_last+atr,
                                                        take_profit=close_last+2*atr if signal=="bullish" else close_last-2*atr,
                                                        trend=trend, volume_spike=vol_spike)
        except Exception:
            self.strategies["psar_trend"] = None

        try:
            kama = ta.momentum.KAMAIndicator(df["Close"]).kama()
            slope = np.sign(kama.diff().iat[-1])
            signal = "bullish" if slope>0 else "bearish" if slope<0 else "neutral"
            reasons = ["KAMA slope"]
            self.strategies["kama_slope"] = self._format("kama_slope", signal, confidence=0.7 if signal!="neutral" else 0.0,
                                                        reasons=reasons, entry=close_last,
                                                        stop_loss=close_last-atr if signal=="bullish" else close_last+atr if signal=="bearish" else None,
                                                        take_profit=close_last+2*atr if signal=="bullish" else close_last-2*atr if signal=="bearish" else None,
                                                        trend=trend, volume_spike=vol_spike)
        except Exception:
            self.strategies["kama_slope"] = None

        try:
            trix = ta.trend.TRIXIndicator(df["Close"]).trix()
            slope = np.sign(trix.diff().iat[-1])
            signal = "bullish" if slope>0 else "bearish" if slope<0 else "neutral"
            reasons = ["TRIX momentum"]
            self.strategies["trix_momentum"] = self._format("trix_momentum", signal, confidence=0.7 if signal!="neutral" else 0.0,
                                                            reasons=reasons, entry=close_last,
                                                            stop_loss=close_last-atr if signal=="bullish" else close_last+atr if signal=="bearish" else None,
                                                            take_profit=close_last+2*atr if signal=="bullish" else close_last-2*atr if signal=="bearish" else None,
                                                            trend=trend, volume_spike=vol_spike)
        except Exception:
            self.strategies["trix_momentum"] = None

        # ---------------- VOLUME STRATEGIES ---------------- #
        volume_indicators = {
            "obv": ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume(),
            "cmf": ta.volume.ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).chaikin_money_flow(),
            "mfi": ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).money_flow_index(),
            "eom": ta.volume.EaseOfMovementIndicator(df["High"], df["Low"], df["Volume"]).ease_of_movement(),
            "force_index": ta.volume.ForceIndexIndicator(df["Close"], df["Volume"]).force_index(),
            "adl": ta.volume.AccDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).acc_dist_index(),
            "vpt": ta.volume.VolumePriceTrendIndicator(df["Close"], df["Volume"]).volume_price_trend()
        }

        for key, series in volume_indicators.items():
            try:
                val = series.iat[-1]
                slope = np.sign(series.diff().iat[-1]) if key not in ["cmf","mfi","eom","force_index"] else 0
                signal = "bullish" if slope>0 or val>0 else "bearish" if slope<0 or val<0 else "neutral"
                reasons = [f"{key} signal"]
                self.strategies[key] = self._format(key, signal, confidence=0.7 if signal!="neutral" else 0.0,
                                                    reasons=reasons, entry=close_last,
                                                    stop_loss=close_last-atr if signal=="bullish" else close_last+atr if signal=="bearish" else None,
                                                    take_profit=close_last+2*atr if signal=="bullish" else close_last-2*atr if signal=="bearish" else None,
                                                    trend=trend, volume_spike=vol_spike)
            except Exception:
                self.strategies[key] = None

        # ---------------- VOLATILITY STRATEGIES ---------------- #
        volatility_indicators = {
            "keltner": ta.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"]),
            "donchian": ta.volatility.DonchianChannel(df["High"], df["Low"], df["Close"])
        }

        for key, indicator in volatility_indicators.items():
            try:
                upper = indicator.keltner_channel_hband().iat[-1] if key=="keltner" else indicator.donchian_channel_hband().iat[-1]
                lower = indicator.keltner_channel_lband().iat[-1] if key=="keltner" else indicator.donchian_channel_lband().iat[-1]
                signal = "bullish" if close_last>upper else "bearish" if close_last<lower else "neutral"
                reasons = [f"{key} breakout"]
                self.strategies[key] = self._format(key, signal, confidence=0.8 if signal!="neutral" else 0.0,
                                                    reasons=reasons, entry=close_last,
                                                    stop_loss=close_last-atr if signal=="bullish" else close_last+atr if signal=="bearish" else None,
                                                    take_profit=close_last+2*atr if signal=="bullish" else close_last-2*atr if signal=="bearish" else None,
                                                    trend=trend, volume_spike=vol_spike)
            except Exception:
                self.strategies[key] = None

        # ---------------- SUPER TREND ---------------- #
        try:
            atr_series = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
            factor = 3.0
            hl2 = (df["High"] + df["Low"])/2
            supertrend = hl2 - factor * atr_series
            signal = "bullish" if close_last > supertrend.iat[-1] else "bearish"
            reasons = ["supertrend breakout"]
            self.strategies["supertrend"] = self._format("supertrend", signal, confidence=0.8,
                                                        reasons=reasons, entry=close_last,
                                                        stop_loss=close_last-atr if signal=="bullish" else close_last+atr,
                                                        take_profit=close_last+2*atr if signal=="bullish" else close_last-2*atr,
                                                        trend=trend, volume_spike=vol_spike)
        except Exception:
            self.strategies["supertrend"] = None

        # ---------------- HULL MA ---------------- #
        try:
            def hull_moving_average(series, window=14):
                half_length = int(window/2)
                sqrt_length = int(np.sqrt(window))
                wma = series.rolling(window, min_periods=1).mean()
                wma_half = series.rolling(half_length, min_periods=1).mean()
                raw_hma = (2*wma_half - wma).rolling(sqrt_length, min_periods=1).mean()
                return raw_hma

            hma = hull_moving_average(df["Close"], 20)
            slope = np.sign(hma.diff().iat[-1])
            signal = "bullish" if slope>0 else "bearish" if slope<0 else "neutral"
            reasons = ["HMA slope"]
            self.strategies["hma"] = self._format("hma", signal, confidence=0.7 if signal!="neutral" else 0.0,
                                                reasons=reasons, entry=close_last,
                                                stop_loss=close_last-atr if signal=="bullish" else close_last+atr if signal=="bearish" else None,
                                                take_profit=close_last+2*atr if signal=="bullish" else close_last-2*atr if signal=="bearish" else None,
                                                trend=trend, volume_spike=vol_spike)
        except Exception:
            self.strategies["hma"] = None


    def strategies_generator(self, df: pd.DataFrame, cfg: CLIConfig, target_name: str = 'Close'):
        target_df = df.copy()
        self._precompute_indicators(df, target_name)
        
        self.strategies["breakout"] = self.detect_breakout(target_df, target_col=target_name)
        # Mean reversion
        self.strategies["mean_reversion"] = self.detect_mean_reversion(target_df, target_col=target_name)
        # Fibonacci
        self.strategies["fibonacci"] = self.detect_fibonacci_pullback(target_df, target_col=target_name)
        # Price action
        self.strategies["price_action"] = self.detect_price_action(target_df)
        # Swing trade
        self.strategies["swing"] = self.detect_swing_trade(target_df, target_col=target_name)
        # Scalping helper (advisory)
        self.strategies["scalping_helper"] = self.detect_scalping_opportunity(target_df, target_col=target_name)
        # Market regime
        self.strategies["regime"] = self.detect_market_regime(target_df)
        # Options summary (light)
        self.strategies["options_summary"] = self.fetch_options_flow_stub(cfg.ticker) if hasattr(cfg, 'ticker') else {}
        # Trend strength
        self.strategies["trend_strength"] = self.detect_trend_strength(target_df) if hasattr(cfg, 'ticker') else None
        # Volume spike
        self.strategies["volume_spike"] = self.detect_volume_spike(target_df) if hasattr(cfg, 'ticker') else None
        # MACD strategy
        self.strategies["macd_strategy"] = self.detect_macd_strategy(target_df) if hasattr(cfg, 'ticker') else None
        # Bollinger strategy
        self.strategies["bollinger_strategy"] = self.detect_bollinger_strategy(target_df) if hasattr(cfg, 'ticker') else None
        # Extra pro indicators
        self.detect_extra_strategies(target_df)
    
    # ----------------------
    # Function Call
    # ----------------------
    def detect_all_strategies(self, df: pd.DataFrame, cfg: CLIConfig, target_name: str = 'Close'):
        if console and not cfg.quiet:
            with Progress(SpinnerOrTickColumn(), TextColumn("[progress.description]{task.description}")) as prog:
                t = prog.add_task("Initiating strategies engineer...", total=None)
                try:
                    self.strategies_generator(df, cfg, target_name)
                    prog.stop_task(t)
                    prog.update(t, description=f"Engineered 30+ strategies.")
                    return self.strategies
                except Exception as e:
                    prog.stop()
                    logger.error(f"Failed to detect strategies: {e}")
                    raise
        else:
            self.strategies_generator(df, cfg, target_name)
            return self.strategies

class TradingSignalEngineer:
    """Production-grade trading signal generator using multiple strategies."""

    def __init__(self, weights: Dict[str, int] = None):
        # Default weights if none provided
        self.WEIGHTS = weights or {
            "candle": 1,
            "breakout": 2,
            "mean_reversion": 1,
            "fibonacci": 1,
            "price_action": 1,
            "swing": 1,
            "scalping_helper": 0.5,
            "regime": 1,
            "options": 1,
            "ichimoku_bull": 0.5,
            "psar_trend": 0.5,
            "kama_slope": 0.5,
            "trix_momentum": 0.5,
            "obv_slope": 0.5,
            "cmf": 0.5,
            "mfi": 0.5,
            "eom": 0.5,
            "force_index": 0.5,
            "adl_trend": 0.5,
            "vpt_trend": 0.5,
            "keltner_breakout": 2,
            "donchian_breakout": 2,
            "stoch_rsi": 0.5,
            "ultimate_osc": 0.5,
            "awesome_osc": 0.5,
            "tsi": 0.5,
            "cci": 0.5,
            "williams_r": 0.5,
            "roc": 0.5,
            "adx": 0.5,
            "adx_trend": 0.5,
            "supertrend_signal": 1,
            "hma_slope": 1
        }

    @staticmethod
    def analyze_candle(open_, high, low, close):
        """Return OHLC candle confirmation features."""
        body = abs(close - open_)
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        candle_range = high - low + 1e-9

        signals = []
        if body < candle_range * 0.3:
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

    def weighted_signal(self, name, value, bull_cond=None, bear_cond=None):
        """Helper for calculating weighted bull/bear scores and handling neutral/missing."""
        b_score, br_score, r_score = 0, 0, 0
        weight = self.WEIGHTS.get(name, 1)
        if value is None or str(value).lower() in ["none", "neutral"]:
            return b_score, br_score, f"{name} missing/neutral → ignored", "neutral"
        val_str = str(value).lower() if isinstance(value, str) else value
        if bull_cond and bull_cond(val_str):
            b_score = weight
            return b_score, br_score, f"{name} bullish signal detected.", "bull"
        elif bear_cond and bear_cond(val_str):
            br_score = weight
            return b_score, br_score, f"{name} bearish signal detected.", "bear"
        else:
            return b_score, br_score, f"{name} neutral → no scoring", "neutral"

    def generate_signal(self, json_data: Dict[str, Any], last_candle: pd.Series, cfg: CLIConfig, next_candle: pd.Series = None):
        """Generate trading signal with full reasoning using all strategies."""
        bull_score, bear_score, r_score = 0, 0, 0
        reasons, strategies_influenced, neutral_or_missing = [], [], []
        
        if console and not cfg.quiet:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
                t = prog.add_task("Initiating trading signal engineer...", total=None)
                try:
                    # --- Candle analysis ---
                    candle_signals = self.analyze_candle(
                        last_candle.get("Open", 0),
                        last_candle.get("High", 0),
                        last_candle.get("Low", 0),
                        last_candle.get("Close", 0)
                    )
                    lower_wick_index, upper_wick_index = -1, -1
                    for i, s in enumerate(candle_signals):
                        s_lower = s.lower()
                        if "long_lower_wick" in s_lower:
                            bull_score += self.WEIGHTS["candle"]
                            lower_wick_index = i
                            reasons.append("Previous candle long lower wick → buying pressure.")
                            strategies_influenced.append("candle")
                        elif "long_upper_wick" in s_lower:
                            bear_score += self.WEIGHTS["candle"]
                            upper_wick_index = i
                            reasons.append("Previous candle long upper wick → selling pressure.")
                            strategies_influenced.append("candle")
                        elif "strong_bullish_body" in s_lower:
                            bull_score += self.WEIGHTS["candle"]
                            reasons.append("Strong bullish body → momentum up.")
                            strategies_influenced.append("candle")
                        elif "strong_bearish_body" in s_lower:
                            bear_score += self.WEIGHTS["candle"]
                            reasons.append("Strong bearish body → momentum down.")
                            strategies_influenced.append("candle")
                        elif "indecision" in s_lower or "doji" in s_lower:
                            neutral_or_missing.append("candle")
                            reasons.append("Indecisive candle → neutral signal.")
                            strategies_influenced.append("candle")

                    if lower_wick_index != -1 and upper_wick_index != -1:
                        if lower_wick_index > upper_wick_index:
                            bull_score += 1
                            reasons.append("Lower wick after upper wick → extra bullish")
                            strategies_influenced.append("candle")
                        elif lower_wick_index < upper_wick_index:
                            bear_score += 1
                            reasons.append("Lower wick before upper wick → extra bearish")
                            strategies_influenced.append("candle")

                    # --- Apply all strategies ---
                    breakout_signals = json_data.get('breakout', {}).get('signal')
                    if any("bull" in str(sig).lower() for sig in breakout_signals):
                        bull_score += self.WEIGHTS["breakout"]
                        reasons.append("Breakout detected above the upper Bollinger band with volume — suggests strong upside momentum; trend-following entry favored.")
                        strategies_influenced.append("Breakout Analytics")
                    elif any("bear" in str(sig).lower() for sig in breakout_signals):
                        bear_score += self.WEIGHTS["breakout"]
                        reasons.append("Breakdown detected below the lower Bollinger band with volume — suggests strong downside momentum; trend-following short favored.")
                        strategies_influenced.append("Breakout Analytics")
                    else:
                        reasons.append("No breakout or breakdown detected — suggests sideways momentum")
                        neutral_or_missing.append("Breakout Analytics")

                    # Mean Reversion
                    mean_rev = json_data.get('mean_reversion', {}).get('signal')
                    if any(str(sig) == "buy_revert" for sig in mean_rev):
                        bull_score += self.WEIGHTS["mean_reversion"]
                        reasons.append(
                            "Mean-reversion signals (RSI oversold / lower band / negative z-score) — likely bounce to the mean; buy opportunity."
                        )
                        strategies_influenced.append("Mean Reversion")
                    elif any(str(sig) == "sell_revert" for sig in mean_rev):
                        bear_score += self.WEIGHTS["mean_reversion"]
                        reasons.append(
                            "Mean-reversion signals (RSI overbought / upper band / high z-score) — likely pullback to the mean; caution for longs."
                        )
                        strategies_influenced.append("Mean Reversion")
                    else:
                        reasons.append(
                            "No mean-reversion signals  — likely sideways momentum"
                        )
                        neutral_or_missing.append("Mean Reversion")

                    # Fibonacci
                    fib = json_data.get('fibonacci', {})
                    near = fib.get('near_level')
                    dist = fib.get('distance', None)
                    levels = fib.get('levels', {})
                    if near in ["61.8%", "50%", "38.2%"] and dist is not None and dist < 2:
                        bull_score += self.WEIGHTS["fibonacci"]
                        lvl_val = levels.get(near)
                        reasons.append(
                            f"Price is within {dist:.2f} of {near} Fibonacci ({lvl_val:.2f}) — common support zone; watch for bounce."
                        )
                        strategies_influenced.append("Fibonacci Tracement")
                    elif near == "0%" and dist is not None and dist < 2:
                        bear_score += self.WEIGHTS["fibonacci"]
                        lvl_val = levels.get("0%")
                        reasons.append(
                            f"Price is within {dist:.2f} of Fibonacci 0% (swing high {lvl_val:.2f}) — strong resistance; likely rejection."
                        )
                        strategies_influenced.append("Fibonacci Tracement")
                    else:
                        reasons.append(
                            f"Price is within {dist:.2f} of Fibonacci 0% — likely sideways."
                        )
                        neutral_or_missing.append("Fibonacci Tracement")

                    # Price Action
                    bull_engulfing = json_data.get('price_action', {}).get('bullish_engulfing', 0)
                    bear_engulfing = json_data.get('price_action', {}).get('bearish_engulfing', 0)
                    if bull_engulfing > 1:
                        bull_score += self.WEIGHTS["price_action"]
                        reasons.append("Multiple bullish engulfing patterns detected — strong buyer conviction.")
                        strategies_influenced.append("Price Action")
                    elif bear_engulfing > 1:
                        bear_score += self.WEIGHTS["price_action"]
                        reasons.append("Multiple bearish engulfing patterns detected — strong seller conviction.")
                        strategies_influenced.append("Price Action")
                    if json_data.get('price_action', {}).get('pin_bar', 0):
                        r_score += self.WEIGHTS['Price Action']
                        reasons.append("Pin bar(s) detected — price rejection at a key level; watch for reversal.")
                        neutral_or_missing.append("Price Action")

                    # Scalping
                    scalping = json_data.get('scalping_helper', {}).get('signal')
                    if any("long" in str(sig).lower() for sig in scalping):
                        bull_score += self.WEIGHTS["scalping"]
                        reasons.append("Intraday: price above VWAP + rising volume — short-term buying momentum.")
                        strategies_influenced.append("Scalping Helper")
                    elif any("short" in str(sig).lower() for sig in scalping):
                        bear_score += self.WEIGHTS["scalping"]
                        reasons.append("Intraday: price below VWAP + rising volume — short-term selling momentum.")
                        strategies_influenced.append("Scalping Helper")
                    else:
                        reasons.append("Intraday: no signal detected — sideways momentum.")
                        neutral_or_missing.append("Scalping Helper")
                    
                    
                        # Swing
                    swing = json_data.get('swing', {})
                    swing_sig = swing.get('signal', 'none')
                    if swing_sig and swing_sig.lower() != "none":
                        if "bullish" in swing_sig.lower():
                            bull_score += self.WEIGHTS.get("price_action", 2)  # or give swing its own weight
                            reasons.append("Swing strategy indicates long bias — market showing medium-term upward pressure.")
                            strategies_influenced.append("Swing Analytics")
                        elif "bearish" in swing_sig.lower():
                            bear_score += self.WEIGHTS.get("price_action", 2)
                            reasons.append("Swing strategy indicates short bias — market showing medium-term downward pressure.")
                            strategies_influenced.append("Swing Analytics")
                    else:
                        reasons.append("Swing strategy indicates no bias — market showing sideways pressure.")
                        neutral_or_missing.append("Swing Analytics")

                    # Regime
                    regime = json_data.get('regime', None)
                    if regime:
                        regime_lower = regime.lower()
                        if regime_lower == "high":
                            bull_score += self.WEIGHTS['regime']
                            reasons.append("Market regime detected as HIGH volatility/uptrend — favorable for momentum strategies.")
                            strategies_influenced.append("Regime Detection")
                        elif regime_lower == "low":
                            bear_score += self.WEIGHTS['regime']
                            reasons.append("Market regime detected as LOW volatility/downtrend — risk of sideways or falling market.")
                            strategies_influenced.append("Regime Detection")
                        elif regime_lower == "medium":
                            r_score += self.WEIGHTS['regime']
                            reasons.append("Market regime detected as medium volatility/sidetrend — risk of sideways or market reversal.")
                            neutral_or_missing.append("Regime Detection")
                        

                    # Options
                    opts = json_data.get('options_summary', {})
                    calls = opts.get('calls_oi_sum', 0)
                    puts = opts.get('puts_oi_sum', 0) or 1
                    oi_ratio = calls / max(puts, 1)
                    if oi_ratio > 1.2:
                        bull_score += self.WEIGHTS["options"]
                        reasons.append(f"Options positioning bullish: Calls {calls} vs Puts {puts} (ratio {oi_ratio:.2f}) — sentiment supports bullish bias.")
                        strategies_influenced.append("Options Positioning")
                    elif oi_ratio < 0.8:
                        bear_score += self.WEIGHTS["options"]
                        reasons.append(
                            f"Options positioning bearish: Calls {calls} vs Puts {puts} (ratio {oi_ratio:.2f}) — sentiment supports bearish bias."
                        )
                        strategies_influenced.append("Options Positioning")
                    
                    # Strategy list for weighted signals
                    strategy_conditions = [
                        ("ichimoku_bull", lambda x: str(x).lower() in ["true","bullish"], lambda x: str(x).lower() in ["false","bearish"]),
                        ("psar_trend", lambda x: str(x).lower() in ["bullish"], lambda x: str(x).lower() in ["bearish"]),
                        ("adx_trend", lambda x: str(x).lower() in ["bullish"], lambda x: str(x).lower() in ["bearish"]),
                        ("supertrend_signal", lambda x: str(x).lower() in ["bullish"], lambda x: str(x).lower() in ["bearish"]),
                        ("adl_trend", lambda x: float(x)>0, lambda x: float(x)<0),
                        ("vpt_trend", lambda x: float(x)>0, lambda x: float(x)<0)
                    ]

                    for name, bull_cond, bear_cond in strategy_conditions:
                        val = json_data.get(name)
                        b, r, reason, status = self.weighted_signal(name, val, bull_cond, bear_cond)
                        bull_score += b; bear_score += r
                        reasons.append(reason)
                        if status=="neutral": neutral_or_missing.append(name)
                        strategies_influenced.append(name)

                    # --- Numeric indicators ---
                    numeric_strategies = [
                        "kama_slope","trix_momentum","obv_slope","cmf","mfi","eom",
                        "force_index","keltner_breakout","donchian_breakout","stoch_rsi",
                        "ultimate_osc","awesome_osc","tsi","cci","williams_r","roc","adx","hma_slope"
                    ]
                    for s in numeric_strategies:
                        val = json_data.get(s)
                        if val is None:
                            neutral_or_missing.append(s)
                            reasons.append(f"{s} missing → neutral")
                        else:
                            if isinstance(val,(int,float)):
                                if val>0: bull_score += self.WEIGHTS.get(s,1); reasons.append(f"{s} positive → bullish"); strategies_influenced.append(s)
                                elif val<0: bear_score += self.WEIGHTS.get(s,1); reasons.append(f"{s} negative → bearish"); strategies_influenced.append(s)
                                else: neutral_or_missing.append(s); reasons.append(f"{s} zero → neutral")
                            elif isinstance(val,str):
                                if "bull" in val.lower(): bull_score += self.WEIGHTS.get(s,1)
                                elif "bear" in val.lower(): bear_score += self.WEIGHTS.get(s,1)
                                else: neutral_or_missing.append(s)
                                
                    prog.update(t, description=f"Engineered trading signal.")
                except Exception as e:
                    prog.stop()
                    logger.error(f"Failed to detect strategies: {e}")
                    raise
        else:
            # --- Candle analysis ---
            candle_signals = self.analyze_candle(
                last_candle.get("Open", 0),
                last_candle.get("High", 0),
                last_candle.get("Low", 0),
                last_candle.get("Close", 0)
            )
            lower_wick_index, upper_wick_index = -1, -1
            for i, s in enumerate(candle_signals):
                s_lower = s.lower()
                if "long_lower_wick" in s_lower:
                    bull_score += self.WEIGHTS["candle"]
                    lower_wick_index = i
                    reasons.append("Previous candle long lower wick → buying pressure.")
                    strategies_influenced.append("candle")
                elif "long_upper_wick" in s_lower:
                    bear_score += self.WEIGHTS["candle"]
                    upper_wick_index = i
                    reasons.append("Previous candle long upper wick → selling pressure.")
                    strategies_influenced.append("candle")
                elif "strong_bullish_body" in s_lower:
                    bull_score += self.WEIGHTS["candle"]
                    reasons.append("Strong bullish body → momentum up.")
                    strategies_influenced.append("candle")
                elif "strong_bearish_body" in s_lower:
                    bear_score += self.WEIGHTS["candle"]
                    reasons.append("Strong bearish body → momentum down.")
                    strategies_influenced.append("candle")
                elif "indecision" in s_lower or "doji" in s_lower:
                    neutral_or_missing.append("candle")
                    reasons.append("Indecisive candle → neutral signal.")
                    strategies_influenced.append("candle")

            if lower_wick_index != -1 and upper_wick_index != -1:
                if lower_wick_index > upper_wick_index:
                    bull_score += 1
                    reasons.append("Lower wick after upper wick → extra bullish")
                    strategies_influenced.append("candle")
                elif lower_wick_index < upper_wick_index:
                    bear_score += 1
                    reasons.append("Lower wick before upper wick → extra bearish")
                    strategies_influenced.append("candle")

            # --- Apply all strategies ---
            breakout_signals = json_data.get('breakout', {}).get('signal')
            if any("bull" in str(sig).lower() for sig in breakout_signals):
                bull_score += self.WEIGHTS["breakout"]
                reasons.append("Breakout detected above the upper Bollinger band with volume — suggests strong upside momentum; trend-following entry favored.")
                strategies_influenced.append("Breakout Analytics")
            elif any("bear" in str(sig).lower() for sig in breakout_signals):
                bear_score += self.WEIGHTS["breakout"]
                reasons.append("Breakdown detected below the lower Bollinger band witvolume — suggests strong downside momentum; trend-following shorfavored.")
                strategies_influenced.append("Breakout Analytics")
            else:
                reasons.append("No breakout or breakdown detected — suggests sideway momentum")
                neutral_or_missing.append("Breakout Analytics")

            # Mean Reversion
            mean_rev = json_data.get('mean_reversion', {}).get('signal')
            if any(str(sig) == "buy_revert" for sig in mean_rev):
                bull_score += self.WEIGHTS["mean_reversion"]
                reasons.append(
                            "Mean-reversion signals (RSI oversold / lower band / negative z-score) — likely bounce to the mean; buy opportunity."
                        )
                strategies_influenced.append("Mean Reversion")
            elif any(str(sig) == "sell_revert" for sig in mean_rev):
                bear_score += self.WEIGHTS["mean_reversion"]
                reasons.append(
                            "Mean-reversion signals (RSI overbought / upper band / high z-score) — likely pullback to the mean; caution for longs."
                        )
                strategies_influenced.append("Mean Reversion")
            else:
                reasons.append(
                            "No mean-reversion signals  — likely sideways momentum"
                        )
                neutral_or_missing.append("Mean Reversion")

                    # Fibonacci
            fib = json_data.get('fibonacci', {})
            near = fib.get('near_level')
            dist = fib.get('distance', None)
            levels = fib.get('levels', {})
            if near in ["61.8%", "50%", "38.2%"] and dist is not None and dist < 2:
                bull_score += self.WEIGHTS["fibonacci"]
                lvl_val = levels.get(near)
                reasons.append(
                    f"Price is within {dist:.2f} of {near} Fibonacci ({lvl_val:.2f}) — common support zone; watch for bounce."
                )
                strategies_influenced.append("Fibonacci Tracement")
            elif near == "0%" and dist is not None and dist < 2:
                bear_score += self.WEIGHTS["fibonacci"]
                lvl_val = levels.get("0%")
                reasons.append(
                    f"Price is within {dist:.2f} of Fibonacci 0% (swing high {lvl_val:.2f}) — strong resistance; likely rejection."
                )
                strategies_influenced.append("Fibonacci Tracement")
            else:
                reasons.append(
                    f"Price is within {dist:.2f} of Fibonacci 0% (swing {lvl_val:.2f}) — strong momentum; likely sideways."
                )
                neutral_or_missing.append("Fibonacci Tracement")

                    # Price Action
            bull_engulfing = json_data.get('price_action', {}).get('bullish_engulfing', 0)
            bear_engulfing = json_data.get('price_action', {}).get('bearish_engulfing', 0)
            if bull_engulfing > 1:
                bull_score += self.WEIGHTS["price_action"]
                reasons.append("Multiple bullish engulfing patterns detected — strong buyer conviction.")
                strategies_influenced.append("Price Action")
            elif bear_engulfing > 1:
                bear_score += self.WEIGHTS["price_action"]
                reasons.append("Multiple bearish engulfing patterns detected — strong seller conviction.")
                strategies_influenced.append("Price Action")
            if any(s.get('price_action', {}).get('pin_bar', 0) for s in json_data.values()):
                r_score += self.WEIGHTS['Price Action']
                reasons.append("Pin bar(s) detected — price rejection at a key level; watch for reversal.")
                neutral_or_missing.append("Price Action")

            # Scalping
            scalping = json_data.get('scalping_helper', {}).get('signal')
            if any("long" in str(sig).lower() for sig in scalping):
                bull_score += self.WEIGHTS["scalping"]
                reasons.append("Intraday: price above VWAP + rising volume — short-term buying momentum.")
                strategies_influenced.append("Scalping Helper")
            elif any("short" in str(sig).lower() for sig in scalping):
                bear_score += self.WEIGHTS["scalping"]
                reasons.append("Intraday: price below VWAP + rising volume — short-term selling momentum.")
                strategies_influenced.append("Scalping Helper")
            else:
                reasons.append("Intraday: no signal detected — sideways momentum.")
                neutral_or_missing.append("Scalping Helper")
                    
                    
                # Swing
            swing = json_data.get('swing', {})
            swing_sig = swing.get('signal', 'none')
            if swing_sig and swing_sig.lower() != "none":
                if "bullish" in swing_sig.lower():
                    bull_score += self.WEIGHTS.get("price_action", 2)  # or give swing its own weight
                    reasons.append("Swing strategy indicates long bias — market showing medium-term upward pressure.")
                    strategies_influenced.append("Swing Analytics")
                elif "bearish" in swing_sig.lower():
                    bear_score += self.WEIGHTS.get("price_action", 2)
                    reasons.append("Swing strategy indicates short bias — market showing medium-term downward pressure.")
                    strategies_influenced.append("Swing Analytics")
            else:
                reasons.append("Swing strategy indicates no bias — market showing sideways pressure.")
                neutral_or_missing.append("Swing Analytics")

            # Regime
            regime = json_data.get('regime', None)
            if regime:
                regime_lower = regime.lower()
                if regime_lower == "high":
                    bull_score += self.WEIGHTS['regime']
                    reasons.append("Market regime detected as HIGH volatility/uptrend — favorable for momentum strategies.")
                    strategies_influenced.append("Regime Detection")
                elif regime_lower == "low":
                    bear_score += self.WEIGHTS['regime']
                    reasons.append("Market regime detected as LOW volatility/downtrend — risk of sideways or falling market.")
                    strategies_influenced.append("Regime Detection")
                elif regime_lower == "medium":
                    r_score += self.WEIGHTS['regime']
                    reasons.append("Market regime detected as medium volatility/sidetrend — risk of sideways or market reversal.")
                    neutral_or_missing.append("Regime Detection")
                        

            # Options
            opts = json_data.get('options_summary', {})
            calls = opts.get('calls_oi_sum', 0)
            puts = opts.get('puts_oi_sum', 0) or 1
            oi_ratio = calls / max(puts, 1)
            if oi_ratio > 1.2:
                bull_score += self.WEIGHTS["options"]
                reasons.append(f"Options positioning bullish: Calls {calls} vs Puts {puts} (ratio {oi_ratio:.2f}) — sentiment supports bullish bias.")
                strategies_influenced.append("Options Positioning")
            elif oi_ratio < 0.8:
                bear_score += self.WEIGHTS["options"]
                reasons.append(
                    f"Options positioning bearish: Calls {calls} vs Puts {puts} (ratio {oi_ratio:.2f}) — sentiment supports bearish bias."
                )
                strategies_influenced.append("Options Positioning")
                    
            # Strategy list for weighted signals
            strategy_conditions = [
                ("ichimoku_bull", lambda x: str(x).lower() in ["true","bullish"], lambda x: str(x).lower() in ["false","bearish"]),
                ("psar_trend", lambda x: str(x).lower() in ["bullish"], lambda x: str(x).lower() in ["bearish"]),
                ("adx_trend", lambda x: str(x).lower() in ["bullish"], lambda x: str(x).lower() in ["bearish"]),
                ("supertrend_signal", lambda x: str(x).lower() in ["bullish"], lambda x: str(x).lower() in ["bearish"]),
                ("adl_trend", lambda x: float(x)>0, lambda x: float(x)<0),
                ("vpt_trend", lambda x: float(x)>0, lambda x: float(x)<0)
            ]

            for name, bull_cond, bear_cond in strategy_conditions:
                val = json_data.get(name)
                b, r, reason, status = self.weighted_signal(name, val, bull_cond, bear_cond)
                bull_score += b; bear_score += r
                reasons.append(reason)
                if status=="neutral": neutral_or_missing.append(name)
                strategies_influenced.append(name)

            # --- Numeric indicators ---
            numeric_strategies = [
                "kama_slope","trix_momentum","obv_slope","cmf","mfi","eom",
                "force_index","keltner_breakout","donchian_breakout","stoch_rsi",
                "ultimate_osc","awesome_osc","tsi","cci","williams_r","roc","adx","hma_slope"
            ]
            for s in numeric_strategies:
                val = json_data.get(s)
                if val is None:
                    neutral_or_missing.append(s)
                    reasons.append(f"{s} missing → neutral")
                else:
                    if isinstance(val,(int,float)):
                        if val>0: bull_score += self.WEIGHTS.get(s,1); reasons.append(f"{s} positive → bullish"); strategies_influenced.append(s)
                        elif val<0: bear_score += self.WEIGHTS.get(s,1); reasons.append(f"{s} negative → bearish"); strategies_influenced.append(s)
                        else: neutral_or_missing.append(s); reasons.append(f"{s} zero → neutral")
                    elif isinstance(val,str):
                        if "bull" in val.lower(): bull_score += self.WEIGHTS.get(s,1)
                        elif "bear" in val.lower(): bear_score += self.WEIGHTS.get(s,1)
                        else: neutral_or_missing.append(s)
        
        # --- Final Signal ---
        if bull_score >= bear_score + 4: final_signal="STRONG_BUY"; trend="BULLISH"
        elif bull_score > bear_score: final_signal="BUY"; trend="BULLISH"
        elif bear_score >= bull_score + 4: final_signal="STRONG_SELL"; trend="BEARISH"
        elif bear_score > bull_score: final_signal="SELL"; trend="BEARISH"
        else: final_signal="HOLD"; trend="NEUTRAL"
        
        total_score = bull_score + bear_score + 1e-9
        conf_pct = abs(bull_score - bear_score)/total_score*100
        conf_label = "HIGH" if conf_pct>66 else "MEDIUM" if conf_pct>33 else "LOW"
        confidence_str = f"{conf_label} ({conf_pct:.1f}%)"
        
        return {
            "signal": final_signal,
            "trend": trend,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "confidence": confidence_str,
            "confidence_score": conf_pct,
            "reasons": reasons,
            "neutral_or_missing": neutral_or_missing,
            "strategies_influenced": strategies_influenced
        }

class MLSignalAggregator:
    """
    ML-based trading signal aggregator with dynamic weighting and confidence calibration.
    """

    def __init__(self, model_path: str = None):
        self.signal_map = {
            "bullish": 1, "buy": 1, "bull_macd": 1, "bull_trend": 1,
            "bearish": -1, "sell": -1, "bear_macd": -1,
            "neutral": 0, "none": 0, "hold": 0, "medium_volatility": 0, "high_volatility": -1, "low_volatility": 1, "buy_revert": -1, "sell_revert": 1, 
        }
        self.reverse_signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        self.model = joblib.load(model_path) if model_path else None
        self.scaler = StandardScaler()

    def flatten_json(self, data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        for key, val in data.items():
            if isinstance(val, dict) and "signal" in val:
                sig = val.get("signal", "none")
                conf = val.get("confidence", 0)
                features[f"{key}_sig"] = self.signal_map.get(sig.lower(), 0)
                features[f"{key}_conf"] = float(conf)
            if isinstance(val, dict) and "value" in val:
                features[f"{key}_value"] = float(val["value"])
            if key == "fibonacci" and "levels" in val:
                for level, lvl_val in val["levels"].items():
                    features[f"fibonacci_{level}"] = float(lvl_val)
            if key in ["volume_spike", "keltner", "donchian", "supertrend", "psar_trend", "hma"]:
                for subkey, subval in val.items():
                    if isinstance(subval, (int, float)):
                        features[f"{key}_{subkey}"] = float(subval)
        return features

    def calibrate_confidence(self, raw_conf: float, history_accuracy: float) -> float:
        """
        Calibrate confidence based on historical accuracy of the strategies.
        """
        return np.clip(raw_conf * history_accuracy, 0, 1)

    def predict_signal(self, flat_features: Dict[str, float], history_acc: float = 0.8) -> Dict[str, Any]:
        """
        Predict final signal using ML model, dynamic weights, and confidence calibration.
        """
        df = pd.DataFrame([flat_features])
        if self.model is None:
            raise ValueError("No ML model loaded. Train or provide a model_path.")

        # Scale features
        df_scaled = self.scaler.fit_transform(df)

        # ML Prediction
        pred = self.model.predict(df_scaled)[0]
        pred_proba = self.model.predict_proba(df_scaled).max()

        # Calibrated confidence
        confidence = self.calibrate_confidence(pred_proba, history_acc)

        # Market trend (weighted by top trend indicators)
        trend_indicators = ["trend_strength_sig", "psar_trend_sig", "supertrend_sig", "hma_sig"]
        trend_score = np.mean([flat_features.get(ti, 0) for ti in trend_indicators])
        if trend_score > 0.2:
            trend = "Bullish"
        elif trend_score < -0.2:
            trend = "Bearish"
        else:
            trend = "Neutral"

        # Reasons: top 3 contributing indicators
        sig_indicators = {k: v for k, v in flat_features.items() if k.endswith("_sig")}
        top_contributors = sorted(sig_indicators.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        reasons = [f"{k.replace('_sig','').upper()} -> {'Bullish' if v>0 else 'Bearish' if v<0 else 'Neutral'}" for k,v in top_contributors]

        return {
            "final_signal": self.reverse_signal_map[pred],
            "market_trend": trend,
            "confidence": float(confidence),
            "reasons": reasons,
            "ml_raw_signal": int(pred),
            "ml_raw_proba": float(pred_proba)
        }

    def train_model(self, X: pd.DataFrame, y: pd.Series):
        """Train ML model from historical data"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
        self.model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("ML Model Training Complete")
        print(classification_report(y_test, y_pred))
        # Save model
        joblib.dump(self.model, "ml_signal_aggregator.pkl")

    def generate(self, strategy_json: Dict[str, Any], history_acc: float = 0.8) -> Dict[str, Any]:
        flat = self.flatten_json(strategy_json)
        return flat
        # return self.predict_signal(flat, history_acc)

class OutputManager:
    def __init__(self):
        self.logger = logger
        self.strategies_engineer = StrategiesEngineerPro()
        self.trading_signal_engineer = TradingSignalEngineer()

    @staticmethod
    def ensure_dirs(base="outputs"):
        os.makedirs(base, exist_ok=True)
        os.makedirs(os.path.join(base, "plots"), exist_ok=True)
        os.makedirs(os.path.join(base, "json"), exist_ok=True)
        
        
    def _pretty_print_results(self, results: dict):
        if console:
            # collect all timestamps from the ensemble forecasts (dict keys)
            all_times = []
            for t in ["Open", "High", "Low", "Close"]:
                ens = results.get(t, {}).get("next_preds", {}).get("ENSEMBLE", None)
                if isinstance(ens, dict):
                    times = list(ens.keys())
                    if len(times) > len(all_times):  # use the longest set of times
                        all_times = times

            # create table with dynamic columns
            table = Table(title="Next-step predictions (ensemble)", show_edge=False)
            table.add_column("Target")
            for ts in all_times:
                table.add_column(ts, justify="right")
            table.add_column("Top model weight", justify="right")

            # add rows
            for t in ["Open", "High", "Low", "Close"]:
                if "error" in results.get(t, {}):
                    row = [t] + ["Error"] * len(all_times) + [results[t]["error"]]
                    table.add_row(*row)
                    continue

                next_preds = results[t].get("next_preds", {})
                ens = next_preds.get("ENSEMBLE", {})

                # values aligned with timestamps
                if isinstance(ens, dict):
                    values = [f"{ens.get(ts, float('nan')):.4f}" for ts in all_times]
                elif isinstance(ens, (list, np.ndarray)):
                    values = [f"{v:.4f}" for v in ens]
                    # pad if shorter
                    if len(values) < len(all_times):
                        values += ["N/A"] * (len(all_times) - len(values))
                else:
                    values = ["N/A"] * len(all_times)

                # compute top model
                weights = results[t].get("weights", {})
                numeric_weights = {
                    k: float(v)
                    for k, v in weights.items()
                    if isinstance(v, (int, float, np.floating))
                }
                if numeric_weights:
                    top_model, top_weight = max(numeric_weights.items(), key=lambda x: x[1])
                    top_str = f"{top_model} ({top_weight:.2f})"
                else:
                    top_str = "N/A"

                table.add_row(t, *values, top_str)

            panel = Panel(
                table, title="📊 Forecast Summary", border_style="cyan", padding=(1, 2)
            )
            console.print(panel)
        else:
            print(json.dumps(results, indent=2, default=str))

    def save_outputs(
        self, df: pd.DataFrame, results: Dict[str, Any], cfg: CLIConfig
    ):
        self.ensure_dirs(cfg.output_dir)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        last_candle = df.iloc[-1]
        timestamp = last_candle.name

        # Convert to dict with timestamp for each field
        last_candle_dict = {
            "open": {"TimeStamp": str(timestamp), "value": float(last_candle["Open"])},
            "high": {"TimeStamp": str(timestamp), "value": float(last_candle["High"])},
            "low": {"TimeStamp": str(timestamp), "value": float(last_candle["Low"])},
            "close": {"TimeStamp": str(timestamp), "value": float(last_candle["Close"])},
            "volume": {"TimeStamp": str(timestamp), "value": int(last_candle["Volume"])}
        }
        strategies = self.strategies_engineer.detect_all_strategy(df, cfg)
        trading_signal = self.trading_signal_engineer.generate_signal(strategies, last_candle, cfg)

        # Prepare JSON data
        in_json = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ticker": cfg.ticker,
            "timeframe": cfg.timeframe,
            "cfg": asdict(cfg),
            "last_candle": last_candle_dict,
            "results": results,
            "strategies": strategies,
            "trading_signal": trading_signal,
            "disclaimer": "Forecasts ≠ financial advice. Any losses are your responsibility—trade responsibly."
        }

        # Save JSON safely
        json_path = os.path.join(
            cfg.output_dir, "json", f"{cfg.ticker}_{cfg.timeframe}_{ts}.json"
        )
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(in_json, f, indent=2, default=float)

        # Save plot
        self._pretty_print_results(results)
        plot_path = self._save_plot(cfg.ticker, cfg.timeframe, df, ts, cfg, results)

        return json_path, plot_path

    def _save_plot(self, ticker: str, timeframe: str, df: pd.DataFrame, ts, cfg, results):
        try:
            # Take last 100 candles
            hist_df = df.tail(len(df) - 200).copy()
            hist_df = hist_df.reset_index()
            hist_df.rename(columns={hist_df.columns[0]: "Date"}, inplace=True)

            # Extract ensemble predictions for O/H/L/C
            pred_dicts = {
                "Open": results.get("Open", {}).get("next_preds", {}).get("ENSEMBLE", {}),
                "High": results.get("High", {}).get("next_preds", {}).get("ENSEMBLE", {}),
                "Low": results.get("Low", {}).get("next_preds", {}).get("ENSEMBLE", {}),
                "Close": results.get("Close", {}).get("next_preds", {}).get("ENSEMBLE", {}),
            }

            # Build prediction DataFrame
            pred_times = list(pred_dicts["Close"].keys())
            pred_times = pd.to_datetime(pred_times)

            pred_df = pd.DataFrame({
                "Date": pred_times,
                "Open": list(pred_dicts["Open"].values()),
                "High": list(pred_dicts["High"].values()),
                "Low": list(pred_dicts["Low"].values()),
                "Close": list(pred_dicts["Close"].values()),
            })
            
            hist_df["DateNum"] = mdates.date2num(hist_df["Date"])
            pred_df["DateNum"] = mdates.date2num(pred_df["Date"])

            hist_ohlc = hist_df[["DateNum", "Open", "High", "Low", "Close"]].values
            pred_ohlc = pred_df[["DateNum", "Open", "High", "Low", "Close"]].values
            
            # Prepare figure
            fig, ax = plt.subplots(figsize=(12,6))
            candlestick_ohlc(
                ax,
                hist_ohlc,
                width=0.0007 if timeframe.endswith("m") else 0.4,  # adapt width
                colorup="green", colordown="red", alpha=0.8
            )
            candlestick_ohlc(
                ax,
                pred_ohlc,
                width=0.0007 if timeframe.endswith("m") else 0.4,  # adapt width
                colorup="blue", colordown="orange", alpha=0.8
            )

            # Formatting
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            ax.set_title(f"{ticker.upper()} {timeframe} — Last 100 + Ensemble Forecast", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.grid(alpha=0.3)
            ax.set_xlim(hist_ohlc[0,0] - 0.001, pred_ohlc[-1,0] + 0.005)

            fig.autofmt_xdate()
            fig.tight_layout()

            plot_path = os.path.join(cfg.output_dir, "plots", f"{ticker}_{timeframe}_{ts}.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return plot_path

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Plot saving failed for {ticker}: {e}")

class ForecastUnivariate:
    def __init__(self, series, features_df, cfg, target_name="Close", progress_queue=None):
        # --- Align indexes ---
        self.series = _ensure_series_has_datetime_index(series)
        features_df.index = _ensure_datetime_index_tz_naive(features_df.index)
        self.features_df, self.series = features_df.align(self.series, join="inner", axis=0)
        self.features_df = self.features_df.dropna()
        self.series = self.series.loc[self.features_df.index].dropna()
        

        # --- Config ---
        self.cfg = cfg
        self.target_name = target_name
        self.progress_queue = progress_queue

        self.val_h = cfg.val_horizon
        self.forecast_h = cfg.forecast_horizon
        self.n_total = len(self.series)
        self.n_train = self.n_total - self.val_h
        if len(self.series) < self.val_h + 20:
            raise ValueError("Not enough data after indicators for training/validation")

        # --- Data arrays ---
        self.y_all = self.series.values.astype(float)
        self.X_all = self.features_df.values.astype(float)

        # --- Placeholders ---
        self.pred_models = {}
        self.forecast_models = {}
        self.weights = {}
        self.metrics = {}
        self.strategies = {}

    # ---------- Utilities ----------
    def _notify(self, msg):
        if self.progress_queue:
            self.progress_queue.put({"target": self.target_name, "status": msg})

    def _create_lag_features(self, y, n_lags=5):
        X_lags = [pd.Series(y).shift(i) for i in range(n_lags, 0, -1)]
        X_lags = pd.concat(X_lags, axis=1)
        X_lags.columns = [f"lag_{i}" for i in range(n_lags, 0, -1)]
        X_lags = X_lags[n_lags:]
        y_trimmed = y[n_lags:]
        return X_lags.values.astype(float), y_trimmed.astype(float)
    
    def _pick_n_lags(self, series, max_lags=30, threshold=0.2):
        """
        series: pd.Series of your target
        max_lags: maximum number of lags to consider
        threshold: minimum autocorrelation to consider significant
        """
        if not acf: return 5
        
        acf_vals = acf(series, nlags=max_lags, fft=False)
        
        # Find the last lag where autocorrelation is above the threshold
        significant_lags = np.where(np.abs(acf_vals[1:]) > threshold)[0]  # exclude lag 0
        if len(significant_lags) == 0:
            return 1  # fallback to 1 lag if none significant
        return significant_lags[-1] + 1  # add 1 because we skipped lag 0
    
    # ---------- Models ----------
    def _build_random_forest(self, X_train, y_train, X_val=None, params: dict = None):
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
    
    def _build_lstm(self, input_shape, units=64, dropout_rate=0.2, recurrent_dropout=0.1):
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
    
    # ---------- LSTM utilities ----------
    def _train_lstm_model(self, X_seq_train, y_seq_train_scaled, lookback, n_features, X_seq_val=None, y_seq_val_scaled=None):
        model = self._build_lstm((lookback, n_features), units=128)

        class QueueLogger(Callback):
            def __init__(self, total_epochs, target_name, progress_queue):
                super().__init__()
                self.total_epochs = total_epochs
                self.target_name = target_name
                self.progress_queue = progress_queue

            def on_epoch_end(self, epoch, logs=None):
                if self.progress_queue:
                    loss = logs.get("loss", 0.0)
                    val_loss = logs.get("val_loss", 0.0)
                    self.progress_queue.put({
                        "target": self.target_name,
                        "status": f"LSTM epoch {epoch+1}/{self.total_epochs} "
                                f"loss={loss:.4f} val_loss={val_loss:.4f}"
                    })

        callbacks = []
        if 'EarlyStopping' in globals():
            callbacks.append(EarlyStopping(monitor='loss', patience=5, restore_best_weights=True))
        callbacks.append(QueueLogger(self.cfg.lstm_epochs, self.target_name, self.progress_queue))

        model.fit(
            X_seq_train, y_seq_train_scaled,
            validation_data=(X_seq_val, y_seq_val_scaled) if X_seq_val is not None else None,
            epochs=self.cfg.lstm_epochs,
            batch_size=self.cfg.lstm_batch,
            callbacks=callbacks,
            verbose=0
        )

        return model

    def _recursive_lstm_forecast(self, model, last_window, forecast_h, scaler_y):
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
    
    def _recursive_rf_forecast(self, mdl, last_known_lags, last_features, forecast_h):
        """
        mdl: trained RandomForest
        last_known_lags: np.array of last n_lags target values
        last_features: np.array of last original features (same length as X_all columns)
        forecast_h: int, how many steps to forecast
        """
        forecasts = []
        lags = last_known_lags.copy()

        for _ in range(forecast_h):
            X_input = np.hstack([lags, last_features]).reshape(1, -1)
            y_pred = mdl.predict(X_input)[0]
            forecasts.append(y_pred)
            # update lag vector
            lags[:-1] = lags[1:]
            lags[-1] = y_pred

        return np.array(forecasts)
    
    # ---------- Run Models ----------
    def _run_tree_models(self):
        if self.cfg.use_random_forest:
            try:
                self._notify("Running Tree Model Boosters...")
                n_train = self.n_train
                X_all = self.features_df.values.astype(float)
                y_all = self.series.values.astype(float)


                # Fit scaler on training rows only
                scaler_X = RobustScaler()
                scaler_X.fit(X_all[:n_train])
                X_scaled_full = scaler_X.transform(X_all)
                
                # ----------------- Tree Models: RF, LGBM -----------------
                X_tab_train = X_scaled_full[:n_train]
                y_tab_train = y_all[:n_train]
                X_tab_val   = X_scaled_full[n_train:n_train + self.val_h]
                
                rf_preds, rf_forecast, rf_model_obj = None, None, None
                if self.cfg.use_random_forest:
                    self._notify("Running RandomForest Regressor...")
                    try:
                        rf_params = {
                            "n_estimators": 300,
                            "max_depth": None,
                            "min_samples_leaf": 1,
                        }
                        rf_preds, rf_model_obj = self._build_random_forest(
                        X_tab_train, y_tab_train, X_tab_val, params=rf_params
                        )
                        
                        last_feat = np.repeat(X_scaled_full[-1:], self.forecast_h, axis=0)
                        rf_forecast = rf_model_obj.predict(last_feat)
                        rf_preds = rf_model_obj.predict(X_scaled_full[-self.val_h:])

                        if rf_preds is not None:
                            self.pred_models["RF"] = np.array(rf_preds, dtype=float)
                        if rf_forecast is not None:
                            self.forecast_models["RF"] = np.array(rf_forecast, dtype=float)
                    except Exception as e:
                        logger.warning(f"RandomForest failed: {e}")
                    self._notify("Running XGBoost Regressor...")
            except Exception as e:
                logger.warning(f"{self.target_name} Tree models failed: {e}")

    def _run_lstms(self):
        if self.cfg.use_lstm:
            try:
                self._notify("Running Vanilla LSTM...")
                lstm_preds = lstm_forecast = None
                
                if tf is None:
                    logger.warning("LSTM requested but TensorFlow not installed; skipping LSTM.")
                else:
                    # 1. Scale features
                    X_all = self.features_df.values.astype(float)
                    scaler_X = RobustScaler()
                    scaler_X.fit(X_all[:self.n_train])
                    X_scaled_full = scaler_X.transform(X_all)

                    # 2. Scale target
                    y_all_float = self.series.values.astype(float)
                    y_train = y_all_float[:self.n_train].reshape(-1, 1)
                    scaler_y = RobustScaler()
                    scaler_y.fit(y_train)

                    # 3. Build sequences
                    lookback = min(60, max(10, self.cfg.candles // 3))
                    n_seq = X_scaled_full.shape[0] - lookback
                    if n_seq <= 0:
                        raise RuntimeError("Not enough rows to build sequences for LSTM")

                    X_seq = np.array([X_scaled_full[i:i+lookback] for i in range(n_seq)], dtype=float)
                    y_seq = y_all_float[lookback:]  # aligned with X_seq

                    # 4. Split sequences into train and validation
                    last_train_seq_index = self.n_train - lookback
                    if last_train_seq_index <= 0:
                        raise RuntimeError("Not enough sequence training rows for LSTM after lookback")

                    X_seq_train = X_seq[:last_train_seq_index]
                    y_seq_train = y_seq[:last_train_seq_index]

                    X_seq_val = X_seq[last_train_seq_index:last_train_seq_index + self.val_h]
                    y_seq_val = y_seq[last_train_seq_index:last_train_seq_index + self.val_h]

                    # pad validation if smaller than val_h
                    if X_seq_val.shape[0] < self.val_h:
                        pad_n = self.val_h - X_seq_val.shape[0]
                        X_seq_val = np.vstack([np.repeat(X_seq_val[-1:], pad_n, axis=0), X_seq_val])
                        y_seq_val = np.concatenate([np.repeat(y_seq_val[-1], pad_n), y_seq_val])

                    # 5. Scale y sequences
                    y_seq_train_scaled = scaler_y.transform(y_seq_train.reshape(-1, 1)).reshape(-1)
                    y_seq_val_scaled = scaler_y.transform(y_seq_val.reshape(-1, 1)).reshape(-1)

                    if self.cfg.use_lstm:
                        try:
                            model = self._train_lstm_model(X_seq_train, y_seq_train_scaled, lookback, X_seq.shape[2], X_seq_val, y_seq_val_scaled)
                            preds_scaled = model.predict(X_seq_val, verbose=0).reshape(-1)
                            lstm_preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).reshape(-1)
                            model.fit(
                                X_seq,
                                scaler_y.transform(y_seq.reshape(-1, 1)).reshape(-1),
                                epochs=max(1, self.cfg.lstm_epochs // 2),
                                batch_size=self.cfg.lstm_batch,
                                verbose=0
                            )
                            lstm_forecast = self._recursive_lstm_forecast(model, X_seq[-1].reshape(1,lookback,X_seq.shape[2]), self.forecast_h, scaler_y)
                            
                            self.pred_models["LSTM"] = np.array(lstm_preds, dtype=float)
                            self.forecast_models["LSTM"] = np.array(lstm_forecast, dtype=float)
                        except Exception as e:
                            logger.warning(f"Vanilla LSTM failed: {e}")
                        finally:
                            K.clear_session()
                            gc.collect()
            except Exception as e:
                logger.warning(f"{self.target_name} LSTM failed: {e}")

    def _calculate_metrics(self):
        self._notify("Calculating Weights & Metrics...")

        # --- Align predictions with validation / forecast horizons ---
        for k in list(self.pred_models.keys()):
            arr = np.asarray(self.pred_models[k], dtype=float)
            if arr.shape[0] < self.val_h:
                if arr.size == 0:
                    arr = np.full(self.val_h, float(self.series.iloc[-1]))
                else:
                    arr = np.concatenate([np.full(self.val_h - arr.shape[0], arr[-1]), arr])
            elif arr.shape[0] > self.val_h:
                arr = arr[-self.val_h:]
            self.pred_models[k] = arr

        for k in list(self.forecast_models.keys()):
            arr = np.asarray(self.forecast_models[k], dtype=float)
            if arr.shape[0] < self.forecast_h:
                if arr.size == 0:
                    arr = np.full(self.forecast_h, float(self.series.iloc[-1]))
                else:
                    arr = np.concatenate([arr, np.full(self.forecast_h - arr.shape[0], arr[-1])])
            elif arr.shape[0] > self.forecast_h:
                arr = arr[:self.forecast_h]
            self.forecast_models[k] = arr

        # --- Actual validation values ---
        y_val_actual = self.y_all[self.n_train : self.n_train + self.val_h].astype(float)

        # --- Calculate errors for all models ---
        model_errors = {}
        for k, arr in self.pred_models.items():
            mae = float(mean_absolute_error(y_val_actual, arr))
            model_errors[k] = mae

        # --- Separate baseline vs trend models ---
        baseline_models = ["RF"]  # treat RF as baseline
        trend_models = [k for k in self.pred_models.keys() if k not in baseline_models]

        # --- Dynamic baseline weight based on error ---
        baseline_mae = np.mean([model_errors[k] for k in baseline_models])
        trend_mae = np.mean([model_errors[k] for k in trend_models])
        
        # baseline weight proportional to inverse error ratio
        baseline_weight = baseline_mae / (baseline_mae + trend_mae)  
        # ensure baseline weight is not too small or too large
        baseline_weight = np.clip(baseline_weight, 0.4, 0.6)

        # --- Trend model weights (inverse-error, normalized to remaining weight) ---
        trend_weights = {}
        inv_errors = np.array([1.0 / (model_errors[k] + 1e-6) for k in trend_models])
        if inv_errors.sum() > 0:
            inv_errors /= inv_errors.sum()  # normalize
            for i, k in enumerate(trend_models):
                trend_weights[k] = inv_errors[i] * (1 - baseline_weight)
        else:
            for k in trend_models:
                trend_weights[k] = (1 - baseline_weight) / len(trend_models)

        # --- Ensemble predictions for validation ---
        ensemble_val_preds = np.zeros_like(y_val_actual, dtype=float)
        for k in baseline_models:
            ensemble_val_preds += self.pred_models[k] * baseline_weight
        for k, w in trend_weights.items():
            ensemble_val_preds += self.pred_models[k] * w
        self.pred_models["ENSEMBLE"] = ensemble_val_preds

        # --- Ensemble forecasts ---
        ensemble_forecast = np.zeros(self.forecast_h, dtype=float)
        for k in baseline_models:
            ensemble_forecast += self.forecast_models[k] * baseline_weight
        for k, w in trend_weights.items():
            ensemble_forecast += self.forecast_models[k] * w
        self.forecast_models["ENSEMBLE"] = ensemble_forecast

        # --- Metrics ---
        price_changes = np.abs(np.diff(y_val_actual))
        avg_true_range = price_changes.mean() if len(price_changes) > 0 else 1.0

        self.weights = {**{k: baseline_weight for k in baseline_models}, **trend_weights}

        for k, arr in self.pred_models.items():
            mae = float(mean_absolute_error(y_val_actual, arr))
            rmse = float(math.sqrt(mean_squared_error(y_val_actual, arr)))
            mape = float(safe_mape(y_val_actual, arr))
            vol_adj = mae / (avg_true_range + 1e-6)
            self.metrics[k] = {
                "MAE": mae,
                "RMSE": rmse,
                "MAPE%": mape,
                "VolAdjError": vol_adj,
            }


    # ---------- Main ----------
    def run(self):
        self._run_tree_models()
        self._run_lstms()
        self._calculate_metrics()

        # Future timestamps
        last_time = self.series.index[-1]
        freq = normalize_freq(self.cfg.timeframe)  # e.g. '1h', '5min', '1d'

        # Generate future timestamps correctly aligned with the timeframe
        future_timestamps = pd.date_range(
            start=last_time,
            periods=self.forecast_h + 1,   # include the last_time + forecast_h steps
            freq=freq
        ).tolist()[1:]  # skip the first one because it's just last_time

        # Map forecasts to timestamps
        next_preds = {
            model: {
                str(ts): float(pred)
                for ts, pred in zip(future_timestamps, arr)
            }
            for model, arr in self.forecast_models.items()
        }

        return next_preds, self.metrics, self.weights

class OHLCVPredictor:
    def __init__(self, df: pd.DataFrame, cfg, features: pd.DataFrame):    
        console.print()
        console.rule(f"[bold cyan]Forecast Progress", align='left')    
        self.df = df
        self.cfg = cfg
        self.targets = ["Open", "High", "Low", "Close"]
        self.manager = Manager()
        self.progress_queue = self.manager.Queue()
        self.results = {}
        self.features = features


        # Count total steps per target (roughly # of models per target)
        self.total_steps = 3 + cfg.lstm_epochs  # EMA, SARIMA, Prophet, XGBoost, RF, LGBM, LSTM, CNN-LSTM, Attn-LSTM, Meta

        # Initialize tqdm bars
        self.progress_bars = {
            target: tqdm(
                total=self.total_steps,
                desc=target,
                position=i,
                leave=True,
                ncols=int(os.get_terminal_size().columns * 0.6),
                bar_format="{desc:<8} {percentage:3.0f}%|{bar}| {postfix} [{elapsed}]",
            )
            for i, target in enumerate(self.targets)
        }

    def _queue_reader(self):
        completed_bars = {t: 0 for t in self.targets}
        while True:
            msg = self.progress_queue.get()
            if msg is None:  # Sentinel to stop
                break
            target, status = msg.get("target"), msg.get("status")
            if target in self.progress_bars:
                bar = self.progress_bars[target]
                bar.update(1)
                completed_bars[target] += 1
                bar.set_postfix_str(status)
                bar.refresh()

    def run_forecasts(self):
        # Start queue reader thread
        reader_thread = threading.Thread(target=self._queue_reader, daemon=True)
        reader_thread.start()

        # Run forecasting in parallel
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    ForecastUnivariate(
                        self.df[target],
                        self.features,
                        self.cfg,
                        target_name=target,
                        progress_queue=self.progress_queue
                    ).run
                ): target
                for target in self.targets
            }

            for future in as_completed(futures):
                target = futures[future]
                try:
                    next_preds, metrics, weights = future.result()
                    self.results[target] = {
                        "next_preds": next_preds,
                        "metrics": metrics,
                        "weights": weights,
                    }
                except Exception as e:
                    self.results[target] = {"error": str(e)}

        # Stop the reader thread
        self.progress_queue.put(None)
        reader_thread.join()

        # Close all bars
        for bar in self.progress_bars.values():
            bar.n = self.total_steps
            bar.set_postfix_str("Done")
            bar.close()
            bar.clear()

        console.rule(style="cyan")
        return self.results
    
class StockForecasterCLI:
    def __init__(self):
        self.app = typer.Typer() if USE_TYPER else None
        self.data_fetcher = FinancialDataFetcher()
        self.feature_engieer = FeatureEngineer()
        self.strategies_engineer = StrategiesEngineer()
        self.PERIOD_FOR_INTERVAL = PERIOD_FOR_INTERVAL
        self.output_engineer = OutputManager()
        self.strategies_aggregator = MLSignalAggregator()

    def run_predict(self,
                    ticker: str,
                    timeframe: str = "1d",
                    candles: int = 360,
                    val_horizon: int = 36,
                    forecast_horizon: int = 4,
                    use_lstm: bool = True,
                    use_random_forest: bool = True,
                    lstm_epochs: int = 20,
                    lstm_batch: int = 32,
                    quiet: bool = False):
        """Core runner for prediction pipeline"""
        cfg = CLIConfig(
            ticker=ticker,
            timeframe=timeframe,
            candles=candles,
            val_horizon=val_horizon,
            forecast_horizon=forecast_horizon,
            use_lstm=use_lstm,
            use_random_forest=use_random_forest,
            lstm_epochs=lstm_epochs,
            lstm_batch=lstm_batch,
            output_dir="outputs",
            quiet=quiet
        )

        start_time = time.perf_counter()
        if console and not quiet:
            console.print(Panel.fit(f"[bold green]AlphaFusion Finance[/bold green] — {cfg.ticker.upper()} — {cfg.timeframe}"))

        # Fetch candles
        if console and not quiet:
            with Progress(SpinnerOrTickColumn(), TextColumn("[progress.description]{task.description}")) as prog:
                t = prog.add_task("Fetching market data...", total=None)
                try:
                    df = self.data_fetcher.fetch_data(ticker, timeframe, candles)
                except Exception as e:
                    prog.stop()
                    logger.error(f"Failed to fetch data: {e}")
                    raise
                prog.stop_task(t)
                prog.update(t, description=f"[green]✔ Fetched {len(df)} rows")
        else:
            df = self.data_fetcher.fetch_data(ticker, timeframe, candles)

        # Optimized snippet
        # features = self.feature_engieer.add_all_indicators(df, cfg)
        # results = OHLCVPredictor(df, cfg, features).run_forecasts()
        # json_path, plot_path = self.output_engineer.save_outputs(df, results, cfg)
        # ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # if console and not quiet:
        #     console.print(f"[green]Saved JSON →[/green] {json_path}")
        #     console.print(f"[green]Saved PLOT →[/green] {plot_path}")
        strategies = self.strategies_engineer.detect_all_strategies(df, cfg)
        with open('dump.json', "w") as f:
            json.dump(strategies, f, indent=4, cls=NumpyEncoder)
        flattend_json = self.strategies_aggregator.generate(strategies, )
        with open('flatten_dump.json', "w") as f:
            json.dump(flattend_json, f, indent=4, cls=NumpyEncoder)
        

        # Timing
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        console.print(f"[bold green]✅ Finished Prediction![/bold green] "
                      f"(took [yellow]{elapsed:.2f}[/yellow]s)\n")

    def register_typer(self):
        """Register typer commands if USE_TYPER"""
        if not self.app:
            return

        @self.app.command()
        def predict(
            ticker: str = typer.Option(..., help="Ticker symbol, e.g., AAPL"),
            timeframe: str = typer.Option("1d", help="One of: 1m,5m,15m,30m,1h,1d"),
            candles: int = typer.Option(180, help="Number of historical candles to fetch (default 360)"),
            val_horizon: int = typer.Option(18, help="Validation horizon (bars)"),
            forecast_horizon: int = typer.Option(5, help="Forecast horizon (steps)"),
            use_lstm: bool = typer.Option(True, help="Enable LSTM (requires TensorFlow)"),
            use_random_forest: bool = typer.Option(True, help="Enable RandomForest"),
            lstm_epochs: int = typer.Option(100, help="LSTM epochs"),
            lstm_batch: int = typer.Option(32, help="LSTM batch size"),
            quiet: bool = typer.Option(False, help="Quiet mode")
        ):
            self.run_predict(ticker, timeframe, candles, val_horizon, forecast_horizon, use_lstm, use_random_forest, lstm_epochs, lstm_batch, quiet)

    def main_argparse(self):
        """Argparse fallback if Typer is not used"""
        parser = argparse.ArgumentParser(description="StockForecaster CLI")
        parser.add_argument("--ticker", required=True)
        parser.add_argument("--timeframe", default="1d", choices=list(self.PERIOD_FOR_INTERVAL.keys()))
        parser.add_argument("--candles", type=int, default=180)
        parser.add_argument("--val-horizon", type=int, default=18)
        parser.add_argument("--forecast-horizon", type=int, default=5)
        parser.add_argument("--no-lstm", dest="use_lstm", action="store_false")
        parser.add_argument("--no-random-forest", dest="use_random_forest", action="store_false")
        parser.add_argument("--lstm-epochs", type=int, default=100)
        parser.add_argument("--lstm-batch", type=int, default=32)
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()

        self.run_predict(args.ticker, args.timeframe, args.candles,
                         args.val_horizon, args.forecast_horizon, args.use_lstm,
                         args.use_random_forest, args.lstm_epochs, args.lstm_batch, args.quiet)

    def run(self):
        """Entry point for the CLI"""
        if USE_TYPER:
            self.register_typer()
            self.app()
        else:
            self.main_argparse()


if __name__ == "__main__":
    cli = StockForecasterCLI()
    cli.run()