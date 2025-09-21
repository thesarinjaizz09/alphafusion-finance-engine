#!/usr/bin/env python3
"""
AlphaFusion v1.2.0-beta - Single-file AI Trading CLI (Under Development)

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


from __future__ import annotations
import ta
import os
import gc
import sys
import json
import math
import time
import ccxt
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue
from tqdm import tqdm
import threading
import talib as tlb

import warnings

# Quiet noisy warnings from statsmodels during SARIMAX parameter search;
# keep other warnings visible. Adjust if you want to see them.
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Rich UI
try:
    from rich import print as rprint
    from rich.live import Live
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

# XGBoost (optional)
try:
    import xgboost as xgb
except Exception:
    xgb = None

# TensorFlow/Keras (optional)
with contextlib.redirect_stdout(open(os.devnull, 'w')), \
     contextlib.redirect_stderr(open(os.devnull, 'w')):
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        from tensorflow.keras import backend as K
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.callbacks import EarlyStopping, Callback
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, BatchNormalization, Attention
    except Exception:
        tf = None

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("StockForecaster")

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
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
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
                    prog.update(t, description=f"Engineered 80+ indicators.") 
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
    """Compute advanced trading strategies using TA indicators + custom rules"""

    def __init__(self):
        self.strategies = {}

    def detect_all_strategies_per_candle(self, df: pd.DataFrame, cfg: CLIConfig, target_name: str = 'Close'):
        """Compute all strategies for each candle without changing any logic"""
        n = len(df)
        all_strategies = []
        
        if console and not cfg.quiet:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
                t = prog.add_task("Initiating strategies engineer...", total=None)
                try:
                    for i in range(n):
                        sub_df = df.iloc[:i+1]  # Include all candles up to current index
                        self.strategies = {}  # Reset strategies for this candle

                        # Use your existing method exactly as-is
                        self.detect_all_strategies(sub_df, cfg, target_name)

                        # Store a copy for this candle
                        all_strategies.append(self.strategies.copy())
                        prog.update(t, description=f"Engineered strategy for {i+1} candle...")
                        
                    prog.update(t, description=f"Engineered 30+ strategies for {len(all_strategies)} candles.")
                except Exception as e:
                    prog.stop()
                    logger.error(f"Failed to detect strategies for multiple candles: {e}")
                    raise
        else:
            for i in range(n):
                sub_df = df.iloc[:i+1]  # Include all candles up to current index
                self.strategies = {}  # Reset strategies for this candle

                # Use your existing method exactly as-is
                self.detect_all_strategies(sub_df, cfg, target_name)

                # Store a copy for this candle
                all_strategies.append(self.strategies.copy())
        
        return all_strategies
        

    def detect_all_strategies(self, df: pd.DataFrame, cfg: CLIConfig, target_name: str = 'Close'):
        
        target_df = df.copy()
        
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
        # News sentiment (stub)
        self.strategies["news_sentiment"] = self.fetch_news_sentiment_stub(cfg.ticker) if hasattr(cfg, 'ticker') else None
        # Extra pro indicators
        self.detect_extra_strategies(target_df)
        
    def detect_all_strategy(self, df: pd.DataFrame, cfg: CLIConfig, target_name: str = 'Close'):
        
        target_df = df.copy()
        
        if console and not cfg.quiet:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
                t = prog.add_task("Initiating strategies engineer...", total=None)
                try:
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
                    # News sentiment (stub)
                    self.strategies["news_sentiment"] = self.fetch_news_sentiment_stub(cfg.ticker) if hasattr(cfg, 'ticker') else None
                    # Extra pro indicators
                    self.detect_extra_strategies(target_df)
                    prog.update(t, description=f"Engineered 30+ strategies.")
                    return self.strategies
                except Exception as e:
                    prog.stop()
                    logger.error(f"Failed to detect strategies: {e}")
                    raise
        else:
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
            # News sentiment (stub)
            self.strategies["news_sentiment"] = self.fetch_news_sentiment_stub(cfg.ticker) if hasattr(cfg, 'ticker') else None
            # Extra pro indicators
            self.detect_extra_strategies(target_df)
            return self.strategies

    # ----------------------------- INDICATORS ----------------------------- #
    def _vwap(self, df: pd.DataFrame) -> pd.Series:
        vwap = ta.volume.VolumeWeightedAveragePrice(
            high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14
        )
        return vwap.vwap()

    def _zscore(self, series: pd.Series, window: int = 20):
        m = series.rolling(window=window, min_periods=1).mean()
        s = series.rolling(window=window, min_periods=1).std(ddof=0).replace(0, np.nan)
        return (series - m) / (s + 1e-12)

    def _volume_spike(self, volume: pd.Series, window: int = 20, mult: float = 2.0):
        if len(volume) < window:
            return pd.Series([False] * len(volume), index=volume.index)
        mv = volume.rolling(window=window, min_periods=1).mean()
        return volume > (mv * mult)

    def _fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        lookback = min(len(df), 200)
        s = df["Close"].iloc[-lookback:]
        hi, lo = s.max(), s.min()
        diff = hi - lo
        if diff == 0:
            return {}
        return {
            "0%": hi,
            "23.6%": hi - 0.236 * diff,
            "38.2%": hi - 0.382 * diff,
            "50%": hi - 0.5 * diff,
            "61.8%": hi - 0.618 * diff,
            "100%": lo,
        }

    # -------------------------- CANDLE PATTERNS --------------------------- #
    def _is_bullish_engulfing(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < 1 or idx >= len(df):
            return False
        return tlb.CDLENGULFING(df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values)[idx] > 0

    def _is_bearish_engulfing(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < 1 or idx >= len(df):
            return False
        return tlb.CDLENGULFING(df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values)[idx] < 0

    def _is_pin_bar(self, df: pd.DataFrame, idx: int, tail_ratio: float = 2.0) -> bool:
        if idx < 0 or idx >= len(df):
            return False
        high, low, open_, close = df["High"].iat[idx], df["Low"].iat[idx], df["Open"].iat[idx], df["Close"].iat[idx]
        body = abs(close - open_)
        tail_top = high - max(open_, close)
        tail_bot = min(open_, close) - low
        tail = max(tail_top, tail_bot)
        if body <= 0:
            return False
        return tail / (body + 1e-12) >= tail_ratio

    # ----------------------------- STRATEGIES ----------------------------- #
    def detect_market_regime(self, df: pd.DataFrame, window: int = 20) -> str:
        if len(df) < window:
            return {"signal": "none", "reason": "not enough data", "vol_spike": False}
        atr = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=window).average_true_range()
        ret_vol = df["Close"].pct_change().rolling(window).std()
        vol, atr_val = ret_vol.iloc[-1], atr.iloc[-1]
        if vol < 0.005 and atr_val < (0.01 * df["Close"].iloc[-1]):
            return "low"
        elif vol < 0.02:
            return "medium"
        else:
            return "high"

    def detect_breakout(self, df: pd.DataFrame, target_col: str = "Close", vol_mult: float = 2.0):
        if len(df) < 2:
            return {"signal": "none", "reason": "not enough data", "vol_spike": False}
        bb = ta.volatility.BollingerBands(df[target_col], window=20, window_dev=2)
        upper, lower = bb.bollinger_hband(), bb.bollinger_lband()
        latest, prev = df[target_col].iat[-1], df[target_col].iat[-2]
        vol_flag = self._volume_spike(df["Volume"], window=20, mult=vol_mult).iat[-1]
        if prev <= upper.iat[-2] and latest > upper.iat[-1]:
            return {"signal": "bullish", "reason": "BB breakout", "vol_spike": bool(vol_flag)}
        elif prev >= lower.iat[-2] and latest < lower.iat[-1]:
            return {"signal": "bearish", "reason": "BB breakdown", "vol_spike": bool(vol_flag)}
        return {"signal": "none", "reason": "no breakout", "vol_spike": bool(vol_flag)}

    def detect_mean_reversion(self, df: pd.DataFrame, target_col: str = "Close"):
        rsi = ta.momentum.RSIIndicator(df[target_col], 14).rsi().iat[-1]
        bb = ta.volatility.BollingerBands(df[target_col], 20, 2)
        upper, lower = bb.bollinger_hband().iat[-1], bb.bollinger_lband().iat[-1]
        last = df[target_col].iat[-1]
        z = self._zscore(df[target_col], window=20).iat[-1]
        reasons, signal = [], "none"
        if rsi > 70:
            reasons.append("RSI_overbought")
        elif rsi < 30:
            reasons.append("RSI_oversold")
        if last >= upper:
            reasons.append("BB_upper_touch")
        if last <= lower:
            reasons.append("BB_lower_touch")
        if z > 2:
            reasons.append("Zscore_high")
        elif z < -2:
            reasons.append("Zscore_low")
        if any(x in reasons for x in ["RSI_oversold", "BB_lower_touch", "Zscore_low"]):
            signal = "buy_revert"
        elif any(x in reasons for x in ["RSI_overbought", "BB_upper_touch", "Zscore_high"]):
            signal = "sell_revert"
        return {"signal": signal, "reasons": reasons}

    def detect_fibonacci_pullback(self, df: pd.DataFrame, target_col: str = "Close"):
        levels = self._fibonacci_levels(df)
        if not levels:
            return {"levels": {}, "near_level": None, "distance": None}
        last = df[target_col].iat[-1]
        diffs = {k: abs(last - v) for k, v in levels.items()}
        near = min(diffs.items(), key=lambda x: x[1])
        return {"levels": levels, "near_level": near[0], "distance": float(near[1])}

    def detect_price_action(self, df: pd.DataFrame, idx_offset: int = 0):
        idx = len(df) - 1 - idx_offset
        return {
            "bullish_engulfing": self._is_bullish_engulfing(df, idx),
            "bearish_engulfing": self._is_bearish_engulfing(df, idx),
            "pin_bar": self._is_pin_bar(df, idx),
        }

    def detect_swing_trade(self, df: pd.DataFrame, target_col: str = "Close"):
        if len(df) < 2:
            return {"signal": "none", "reason": "not enough data", "vol_spike": False}
        ema12 = ta.trend.EMAIndicator(df[target_col], 12).ema_indicator()
        ema26 = ta.trend.EMAIndicator(df[target_col], 26).ema_indicator()
        macd = ta.trend.MACD(df[target_col])
        rsi = ta.momentum.RSIIndicator(df[target_col], 14).rsi()
        if ema12.iat[-1] > ema26.iat[-1] and ema12.iat[-2] <= ema26.iat[-2]:
            if macd.macd_diff().iat[-1] > 0 and rsi.iat[-1] < 70:
                return {"signal": "bullish", "reasons": ["EMA_cross + MACD bullish"]}
        elif ema12.iat[-1] < ema26.iat[-1] and ema12.iat[-2] >= ema26.iat[-2]:
            if macd.macd_diff().iat[-1] < 0 and rsi.iat[-1] > 30:
                return {"signal": "bearish", "reasons": ["EMA_cross_down + MACD bearish"]}
        return {"signal": "none", "reasons": []}

    def detect_scalping_opportunity(self, df: pd.DataFrame, target_col: str = "Close"):
        if "VWAP" not in df.columns:
            df["VWAP"] = self._vwap(df)
        last, vwap_last = df[target_col].iat[-1], df["VWAP"].iat[-1]
        avg_vol = df["Volume"].rolling(20, min_periods=1).mean().iat[-1]
        if last > vwap_last and df["Volume"].iat[-1] > avg_vol:
            return {"signal": "long_momentum", "reason": "above VWAP + vol"}
        elif last < vwap_last and df["Volume"].iat[-1] > avg_vol:
            return {"signal": "short_momentum", "reason": "below VWAP + vol"}
        return {"signal": "none", "reason": None}

    def fetch_news_sentiment_stub(self, ticker: str):
        logger.debug(f"No news API configured; skipping sentiment for {ticker}")
        return None

    def fetch_options_flow_stub(self, ticker: str):
        out = {}
        try:
            tk = yf.Ticker(ticker)
            exps = tk.options
            out["expiries"] = list(exps) if exps is not None else []
            if out["expiries"]:
                chain = tk.option_chain(out["expiries"][0])
                calls = chain.calls
                puts = chain.puts
                out["calls_oi_sum"] = int(calls["openInterest"].sum()) if "openInterest" in calls else None
                out["puts_oi_sum"] = int(puts["openInterest"].sum()) if "openInterest" in puts else None
        except Exception as e:
            logger.debug(f"Options scan failed (yfinance): {e}")
        return out

    # -------------------------- PRO ADD-ONS ------------------------------- #
    def detect_extra_strategies(self, df: pd.DataFrame):
        """Calculates all the pro strategies"""

        # === TREND ===
        try:
            ichi = ta.trend.IchimokuIndicator(df["High"], df["Low"], window1=9, window2=26, window3=52)
            self.strategies["ichimoku_bull"] = ichi.ichimoku_a().iat[-1] < df["Close"].iat[-1]
        except Exception:
            self.strategies["ichimoku_bull"] = None

        try:
            psar = ta.trend.PSARIndicator(df["High"], df["Low"], df["Close"])
            self.strategies["psar_trend"] = "bullish" if df["Close"].iat[-1] > psar.psar().iat[-1] else "bearish"
        except Exception:
            self.strategies["psar_trend"] = None

        try:
            kama = ta.momentum.KAMAIndicator(df["Close"]).kama()
            self.strategies["kama_slope"] = np.sign(kama.diff().iat[-1])
        except Exception:
            self.strategies["kama_slope"] = None

        try:
            trix = ta.trend.TRIXIndicator(df["Close"]).trix()
            self.strategies["trix_momentum"] = np.sign(trix.iat[-1])
        except Exception:
            self.strategies["trix_momentum"] = None

        # === VOLUME ===
        try:
            obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
            self.strategies["obv_slope"] = np.sign(obv.diff().iat[-1])
        except Exception:
            self.strategies["obv_slope"] = None

        try:
            cmf = ta.volume.ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"])
            self.strategies["cmf"] = cmf.chaikin_money_flow().iat[-1]
        except Exception:
            self.strategies["cmf"] = None

        try:
            mfi = ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).money_flow_index()
            self.strategies["mfi"] = mfi.iat[-1]
        except Exception:
            self.strategies["mfi"] = None

        try:
            eom = ta.volume.EaseOfMovementIndicator(df["High"], df["Low"], df["Volume"]).ease_of_movement()
            self.strategies["eom"] = eom.iat[-1]
        except Exception:
            self.strategies["eom"] = None

        try:
            fi = ta.volume.ForceIndexIndicator(df["Close"], df["Volume"]).force_index()
            self.strategies["force_index"] = fi.iat[-1]
        except Exception:
            self.strategies["force_index"] = None

        try:
            adl = ta.volume.AccDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).acc_dist_index()
            self.strategies["adl_trend"] = np.sign(adl.diff().iat[-1])
        except Exception:
            self.strategies["adl_trend"] = None

        try:
            vpt = ta.volume.VolumePriceTrendIndicator(df["Close"], df["Volume"]).volume_price_trend()
            self.strategies["vpt_trend"] = np.sign(vpt.diff().iat[-1])
        except Exception:
            self.strategies["vpt_trend"] = None

        # === VOLATILITY ===
        try:
            kc = ta.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"])
            self.strategies["keltner_breakout"] = (
                "bullish" if df["Close"].iat[-1] > kc.keltner_channel_hband().iat[-1]
                else "bearish" if df["Close"].iat[-1] < kc.keltner_channel_lband().iat[-1]
                else "neutral"
            )
        except Exception:
            self.strategies["keltner_breakout"] = None

        try:
            donchian = ta.volatility.DonchianChannel(df["High"], df["Low"], df["Close"])
            self.strategies["donchian_breakout"] = (
                "bullish" if df["Close"].iat[-1] > donchian.donchian_channel_hband().iat[-1]
                else "bearish" if df["Close"].iat[-1] < donchian.donchian_channel_lband().iat[-1]
                else "neutral"
            )
        except Exception:
            self.strategies["donchian_breakout"] = None

        # === MOMENTUM ===
        try:
            stoch_rsi = ta.momentum.StochRSIIndicator(df["Close"]).stochrsi()
            self.strategies["stoch_rsi"] = stoch_rsi.iat[-1]
        except Exception:
            self.strategies["stoch_rsi"] = None

        try:
            uo = ta.momentum.UltimateOscillator(df["High"], df["Low"], df["Close"])
            self.strategies["ultimate_osc"] = uo.ultimate_oscillator().iat[-1]
        except Exception:
            self.strategies["ultimate_osc"] = None

        try:
            ao = ta.momentum.AwesomeOscillatorIndicator(df["High"], df["Low"]).awesome_oscillator()
            self.strategies["awesome_osc"] = ao.iat[-1]
        except Exception:
            self.strategies["awesome_osc"] = None

        try:
            tsi = ta.momentum.TSIIndicator(df["Close"]).tsi()
            self.strategies["tsi"] = tsi.iat[-1]
        except Exception:
            self.strategies["tsi"] = None

        try:
            cci = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"]).cci()
            self.strategies["cci"] = cci.iat[-1]
        except Exception:
            self.strategies["cci"] = None

        try:
            willr = ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"]).williams_r()
            self.strategies["williams_r"] = willr.iat[-1]
        except Exception:
            self.strategies["williams_r"] = None

        try:
            roc = ta.momentum.ROCIndicator(df["Close"]).roc()
            self.strategies["roc"] = roc.iat[-1]
        except Exception:
            self.strategies["roc"] = None

        try:
            adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
            self.strategies["adx"] = adx.adx().iat[-1]
            self.strategies["adx_trend"] = "bullish" if adx.adx_pos().iat[-1] > adx.adx_neg().iat[-1] else "bearish"
        except Exception:
            self.strategies["adx"] = None
            self.strategies["adx_trend"] = None

        # SuperTrend (ATR-based)
        try:
            atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
            factor, period = 3.0, 10
            hl2 = (df["High"] + df["Low"]) / 2
            supertrend = hl2 - (factor * atr)
            self.strategies["supertrend_signal"] = "bullish" if df["Close"].iat[-1] > supertrend.iat[-1] else "bearish"
        except Exception:
            self.strategies["supertrend_signal"] = None

        # Hull MA (manual)
        try:
            def hull_moving_average(series, window=14):
                half_length = int(window / 2)
                sqrt_length = int(np.sqrt(window))
                wma = series.rolling(window, min_periods=1).mean()
                wma_half = series.rolling(half_length, min_periods=1).mean()
                raw_hma = (2 * wma_half - wma).rolling(sqrt_length, min_periods=1).mean()
                return raw_hma

            hma = hull_moving_average(df["Close"], 20)
            self.strategies["hma_slope"] = np.sign(hma.diff().iat[-1])
        except Exception:
            self.strategies["hma_slope"] = None

class TradingSignalEngineer:
    """Production-grade trading signal generator using multiple strategies."""

    def __init__(self, weights: Dict[str, int] = None):
        # Default weights if none provided
        self.WEIGHTS = weights or {
            "candle": 1,
            "breakout": 3,
            "mean_reversion": 2,
            "fibonacci": 2,
            "price_action": 2,
            "swing": 2,
            "scalping_helper": 1,
            "regime": 1,
            "options_summary": 2,
            "ichimoku_bull": 1,
            "psar_trend": 1,
            "kama_slope": 1,
            "trix_momentum": 1,
            "obv_slope": 1,
            "cmf": 1,
            "mfi": 1,
            "eom": 1,
            "force_index": 1,
            "adl_trend": 1,
            "vpt_trend": 1,
            "keltner_breakout": 1,
            "donchian_breakout": 1,
            "stoch_rsi": 1,
            "ultimate_osc": 1,
            "awesome_osc": 1,
            "tsi": 1,
            "cci": 1,
            "williams_r": 1,
            "roc": 1,
            "adx": 1,
            "adx_trend": 1,
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
        b_score, br_score = 0, 0
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
        bull_score, bear_score = 0, 0
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
                    # Strategy list for weighted signals
                    strategy_conditions = [
                        ("breakout", lambda x: "bull" in x, lambda x: "bear" in x),
                        ("mean_reversion", lambda x: "buy" in x, lambda x: "sell" in x),
                        ("swing", lambda x: "bull" in x, lambda x: "bear" in x),
                        ("scalping_helper", lambda x: "long" in x, lambda x: "short" in x),
                        ("regime", lambda x: "high" in x, lambda x: "low" in x),
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
            # Strategy list for weighted signals
            strategy_conditions = [
                ("breakout", lambda x: "bull" in x, lambda x: "bear" in x),
                ("mean_reversion", lambda x: "buy" in x, lambda x: "sell" in x),
                ("swing", lambda x: "bull" in x, lambda x: "bear" in x),
                ("scalping_helper", lambda x: "long" in x, lambda x: "short" in x),
                ("regime", lambda x: "high" in x, lambda x: "low" in x),
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

class OutputManager:
    def __init__(self):
        self.logger = logger

    @staticmethod
    def ensure_dirs(base="outputs"):
        os.makedirs(base, exist_ok=True)
        # os.makedirs(os.path.join(base, "plots"), exist_ok=True)
        os.makedirs(os.path.join(base, "json"), exist_ok=True)

    def pretty_print_results(self, ticker: str, timeframe: str, results: dict):
        if console:
            table = Table(title="Next-step predictions (ensemble)", show_edge=False)
            table.add_column("Target")
            table.add_column("Prediction", justify="right")
            table.add_column("Top model weight", justify="right")

            for t in ["Open", "High", "Low", "Close"]:
                if "error" in results.get(t, {}):
                    table.add_row(t, "[red]Error[/red]", results[t]["error"])
                    continue

                next_preds = results[t].get("adjusted_preds", {})
                ens = next_preds.get("ENSEMBLE", None)

                # Format ensemble predictions safely
                if isinstance(ens, dict):
                    ens_str = ", ".join(f"{v:.4f}" for v in list(ens.values())[:5])
                elif isinstance(ens, (list, np.ndarray)):
                    ens_str = ", ".join(f"{v:.4f}" for v in ens[:5])
                elif isinstance(ens, (float, int, np.floating)):
                    ens_str = f"{float(ens):.4f}"
                else:
                    ens_str = "N/A"

                # Safely compute top model
                weights = results[t].get("weights", {})
                numeric_weights = {
                    k: float(v)
                    for k, v in weights.items()
                    if isinstance(v, (int, float, np.floating))
                }
                if numeric_weights:
                    top_model, top_weight = max(
                        numeric_weights.items(), key=lambda x: x[1]
                    )
                    top_str = f"{top_model} ({top_weight:.2f})"
                else:
                    top_str = "N/A"

                table.add_row(t, ens_str, top_str)

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

        # Prepare JSON data
        in_json = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ticker": cfg.ticker,
            "timeframe": cfg.timeframe,
            "cfg": asdict(cfg),
            "results": results,
        }

        # Save JSON safely
        json_path = os.path.join(
            cfg.output_dir, "json", f"{cfg.ticker}_{cfg.timeframe}_{ts}.json"
        )
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(in_json, f, indent=2, default=float)

        # Save plot
        # self._save_plot(ticker, timeframe, df, results, ts)

        return json_path

    def _save_plot(self, ticker: str, timeframe: str, df: pd.DataFrame, results, ts):
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(df.index, df["Close"], label="Close (historical)", color="black", linewidth=1.5)
            ax.scatter(df.index[-1], df["Close"].iloc[-1], color="black", s=50, label="Last Close")

            close_res = results.get("Close", {})
            next_preds = close_res.get("adjusted_preds", {})

            model_colors = ["red", "blue", "green", "orange", "purple", "brown", "cyan"]

            for i, (model_name, pred_val) in enumerate(next_preds.items()):
                if isinstance(pred_val, (int, float)) and len(df.index) > 1:
                    delta = df.index[-1] - df.index[-2]
                    next_time = df.index[-1] + delta
                    ax.scatter(
                        [next_time],
                        [pred_val],
                        color=model_colors[i % len(model_colors)],
                        s=60,
                        label=f"{model_name} Prediction: {pred_val:.4f}",
                        marker="X",
                    )

            ens = next_preds.get("ENSEMBLE")
            if isinstance(ens, (int, float)) and len(df.index) > 1:
                delta = df.index[-1] - df.index[-2]
                next_time = df.index[-1] + delta
                ax.scatter(
                    [next_time],
                    [ens],
                    color="magenta",
                    s=80,
                    label=f"Ensemble next: {ens:.4f}",
                    marker="D",
                )

            # Strategy annotation
            strat = close_res.get("strategies", {})
            breakout_signal = strat.get("breakout", {}).get("signal")
            if breakout_signal == "bullish":
                ax.annotate(
                    "Breakout (Bullish)",
                    xy=(df.index[-1], df["Close"].iloc[-1]),
                    xytext=(0, 25),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="green", lw=2),
                    fontsize=10,
                    fontweight="bold",
                )
            elif breakout_signal == "bearish":
                ax.annotate(
                    "Breakout (Bearish)",
                    xy=(df.index[-1], df["Close"].iloc[-1]),
                    xytext=(0, 25),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_title(f"{ticker.upper()} {timeframe} — Close & Forecast", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            fig.autofmt_xdate()
            fig.tight_layout()

            plot_path = os.path.join(self.cfg.output_dir, "plots", f"{ticker}_{timeframe}_{ts}.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Plot saving failed for {ticker}: {e}")


class StockForecasterCLI:
    def __init__(self):
        self.app = typer.Typer() if USE_TYPER else None
        self.data_fetcher = FinancialDataFetcher()
        self.feature_engieer = FeatureEngineer()
        self.strategies_engineer = StrategiesEngineer()
        self.PERIOD_FOR_INTERVAL = PERIOD_FOR_INTERVAL
        self.trading_signal_engineer = TradingSignalEngineer()
        self.output_engineer = OutputManager()

    def run_predict(self,
                    ticker: str,
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
                    lstm_batch: int = 32,
                    quiet: bool = False):
        """Core runner for prediction pipeline"""
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

        start_time = time.perf_counter()
        if console and not quiet:
            console.print(Panel.fit(f"[bold green]AlphaFusion Finance[/bold green] — {cfg.ticker.upper()} — {cfg.timeframe}"))

        # Fetch candles
        if console and not quiet:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
                t = prog.add_task("Fetching market data...", total=None)
                try:
                    df = self.data_fetcher.fetch_data(ticker, timeframe, candles)
                except Exception as e:
                    prog.stop()
                    logger.error(f"Failed to fetch data: {e}")
                    raise
                prog.update(t, description=f"Fetched {len(df)} rows")
        else:
            df = self.data_fetcher.fetch_data(ticker, timeframe, candles)

        # Optimized snippet
        last_candle = df.iloc[-1]

        # Compute features, strategies, and signals
        features_df = self.feature_engieer.add_all_indicators(df, cfg)
        strategies = self.strategies_engineer.detect_all_strategy(features_df, cfg, target_name='Close')
        signals = self.trading_signal_engineer.generate_signal(strategies, last_candle, cfg)

        # Pack results cleanly
        results = {
            "candle": {
                col: float(last_candle.get(col, 0))  # safe conversion to float
                for col in ["Open", "High", "Low", "Close"]
            },
            "strategies": strategies,
            "signals": signals
        }

        self.output_engineer.save_outputs(df, results, cfg)
        

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
            candles: int = typer.Option(360, help="Number of historical candles to fetch (default 360)"),
            val_horizon: int = typer.Option(36, help="Validation horizon (bars)"),
            forecast_horizon: int = typer.Option(4, help="Forecast horizon (steps)"),
            use_prophet: bool = typer.Option(False, help="Enable Prophet model if installed"),
            use_xgboost: bool = typer.Option(True, help="Enable XGBoost if installed"),
            use_lstm: bool = typer.Option(True, help="Enable LSTM (requires TensorFlow)"),
            use_cnn_lstm: bool = typer.Option(False, help="Enable CNN-LSTM (requires TensorFlow)"),
            use_attention_lstm: bool = typer.Option(True, help="Enable ATT-LSTM (requires TensorFlow)"),
            use_random_forest: bool = typer.Option(True, help="Enable RandomForest"),
            use_lightgbm: bool = typer.Option(True, help="Enable LightGBM"),
            lstm_epochs: int = typer.Option(20, help="LSTM epochs"),
            lstm_batch: int = typer.Option(32, help="LSTM batch size"),
            quiet: bool = typer.Option(False, help="Quiet mode")
        ):
            self.run_predict(ticker, timeframe, candles, val_horizon, forecast_horizon,
                             use_prophet, use_xgboost, use_lstm, use_cnn_lstm,
                             use_attention_lstm, use_random_forest, use_lightgbm,
                             lstm_epochs, lstm_batch, quiet)

    def main_argparse(self):
        """Argparse fallback if Typer is not used"""
        parser = argparse.ArgumentParser(description="StockForecaster CLI")
        parser.add_argument("--ticker", required=True)
        parser.add_argument("--timeframe", default="1d", choices=list(self.PERIOD_FOR_INTERVAL.keys()))
        parser.add_argument("--candles", type=int, default=360)
        parser.add_argument("--val-horizon", type=int, default=36)
        parser.add_argument("--forecast-horizon", type=int, default=4)
        parser.add_argument("--no-prophet", dest="use_prophet", action="store_false")
        parser.add_argument("--no-xgb", dest="use_xgboost", action="store_false")
        parser.add_argument("--no-lstm", dest="use_lstm", action="store_false")
        parser.add_argument("--cnn-lstm", dest="use_cnn_lstm", action="store_false")
        parser.add_argument("--att-lstm", dest="use_attention_lstm", action="store_false")
        parser.add_argument("--random-forest", dest="use_random_forest", action="store_false")
        parser.add_argument("--lightgbm", dest="use_lightgbm", action="store_false")
        parser.add_argument("--lstm-epochs", type=int, default=20)
        parser.add_argument("--lstm-batch", type=int, default=32)
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()

        self.run_predict(args.ticker, args.timeframe, args.candles,
                         args.val_horizon, args.forecast_horizon,
                         args.use_prophet, args.use_xgboost, args.use_lstm,
                         args.use_cnn_lstm, args.use_attention_lstm,
                         args.use_random_forest, args.use_lightgbm,
                         args.lstm_epochs, args.lstm_batch, args.quiet)

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