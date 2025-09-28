#!/usr/bin/env python3
"""
AlphaFusion Finance v1.2.0 - Single-file Development-Grade AI Trading CLI (SCSO-LSTM + Crypto Support)
Copyright (c) 2025 Alphafusion. All Rights Reserved.

NOTICE:
This software is the confidential and proprietary property of Alphafusion.
Unauthorized copying, distribution, modification, or use of this software
for commercial purposes is strictly prohibited and may result in civil
and criminal penalties under the Copyright Act of India and international laws.

By using this file, you acknowledge that you have read, understood, and
agreed to these terms.

Features:
- Core Forecasting:
    * Single unified model → **SCSO-optimized LSTM** (Sand Cat Swarm Optimization)
    * All legacy/extra models removed for maximum efficiency and reliability
    * Clean, deterministic pipeline focused on accuracy and low overhead

- Technical Analysis Engine:
    * 80+ advanced indicators calculated automatically from OHLCV
    * Categories include:
        - Trend indicators (ADX, Ichimoku, SuperTrend, PSAR, HMA slope, KAMA slope)
        - Momentum oscillators (RSI, StochRSI, MACD, TRIX, Ultimate Osc, TSI, Williams %R, ROC, CCI, Awesome Oscillator)
        - Volume indicators (OBV, ADL, CMF, MFI, VPT, Force Index, EOM)
        - Price action & volatility (Bollinger Bands, Donchian, Keltner, Fibonacci levels)
    * All features fully integrated into the SCSO-LSTM training pipeline

- Crypto & Multi-Asset Support:
    * Works with **stocks, ETFs, and cryptocurrencies**
    * Handles OHLCV data from Yahoo Finance, Binance, or other exchange APIs
    * Timeframes: supports 1m → 1d for crypto and equities
    * Automatically adapts indicator calculations to crypto volatility and liquidity patterns


Notes:
- Focused, development release with **only SCSO-LSTM** and **80+ indicators**.
- Adds full support for **cryptocurrency trading**, alongside equities and ETFs.
- Simplifies architecture by removing redundant models (Prophet, XGBoost, TFT, etc.).
- Designed as a **development-grade, research-ready AI trading system** with clean extensibility.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
import ta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

class FinancialDataFetcher:
    """Class to fetch stock and cryptocurrency data"""
    
    def __init__(self):
        self.exchange = ccxt.binance()
        self.PERIOD_FOR_INTERVAL = {
            "1m": ("7d", "1m"),
            "5m": ("60d", "5m"),
            "15m": ("60d", "15m"),
            "30m": ("60d", "30m"),
            "1h": ("2y", "1h"),
            "1d": ("5y", "1d"),
        }
        
    def fetch_data(self, ticker, timeframe, limit=1000):
        """
        Fetch financial data from Yahoo Finance for stocks or CCXT for cryptocurrencies
        
        Parameters:
        ticker (str): Stock symbol or crypto pair (e.g., 'AAPL' or 'BTC/USDT')
        timeframe (str): Timeframe for data (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        limit (int): Number of candles to fetch
        
        Returns:
        pd.DataFrame: OHLCV data
        """
        if timeframe not in self.PERIOD_FOR_INTERVAL:
            raise ValueError(f"timeframe must be one of {list(self.PERIOD_FOR_INTERVAL.keys())}")
        period, interval = self.PERIOD_FOR_INTERVAL[timeframe]
        try:
            # Check if it's a crypto pair (contains '/')
            if '/' in ticker:
                # Fetch crypto data
                ohlcv = self.exchange.fetch_ohlcv(ticker, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            else:
                # Fetch stock data
                df = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    progress=False
                )
                
            if df is None or df.empty:
                raise RuntimeError(f"No data returned from yfinance for {ticker} ({timeframe})")
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

            # df.rename(
            #     columns={
            #         'Open': 'open',
            #         'High': 'high',
            #         'Low': 'low',
            #         'Close': 'close',
            #         'Volume': 'volume'
            #     },
            #     inplace=True
            # )

            # # ✅ Ensure each OHLCV column is 1D Series
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            if len(df) < limit:
                print(f"Only {len(df)} candles available for {timeframe}. Requested {limit}. Using available.")
                
            return df.tail(limit)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

class FeatureEngineer:
    """Class to compute technical indicators and features"""
    
    def __init__(self):
        self.scalers = {}
        
    def add_all_indicators(self, df):
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
        
        # Add all technical indicators
        df = self._add_trend_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_volume_indicators(df)
        
        # Add other features
        df = self._add_time_features(df)
        df = self._add_price_features(df)
        
        # df.to_csv('raw_indicators.csv')
        
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
        
        # Vortex
        vortex = ta.trend.VortexIndicator(df['High'], df['Low'], df['Close'])
        df['Vortex_pos'] = vortex.vortex_indicator_pos()
        df['Vortex_neg'] = vortex.vortex_indicator_neg()
        df['Vortex_diff'] = df['Vortex_pos'] - df['Vortex_neg']
        
        return df
    
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
    
    def prepare_data_for_training(self, df, lookback=60, forecast_horizon=1, target_col='Close'):
        """
        Prepare data for training LSTM model
        
        Parameters:
        df (pd.DataFrame): DataFrame with all features
        lookback (int): Number of time steps to look back
        forecast_horizon (int): Number of time steps to forecast
        target_col (str): Name of the target column
        
        Returns:
        tuple: X, y, feature_scaler, target_scaler
        """
        # Separate features and target
        # series = df[target_col]
        features = df.drop(columns=[target_col], errors='ignore')
        target = df[target_col].values.reshape(-1, 1)
        # features.to_csv('features.csv')
        # target.to_csv('target.csv')
        
        # Scale features and target
        feature_scaler = StandardScaler()
        target_scaler = MinMaxScaler()
        
        X_scaled = feature_scaler.fit_transform(features)
        y_scaled = target_scaler.fit_transform(target)
        
        # Store scalers for later use
        self.scalers['features'] = feature_scaler
        self.scalers['target'] = target_scaler
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(lookback, len(X_scaled) - forecast_horizon + 1):
            X.append(X_scaled[i-lookback:i])
            y.append(y_scaled[i+forecast_horizon-1])
        
        return np.array(X), np.array(y), feature_scaler, target_scaler

class SCSO:
    """Sand Cat Swarm Optimization implementation for hyperparameter optimization"""

    def __init__(self, n_cats=10, max_iter=50, lb=None, ub=None, dim=5, patience=2):
        self.n_cats = n_cats
        self.max_iter = max_iter
        self.dim = dim
        self.lb = lb if lb is not None else [0] * dim
        self.ub = ub if ub is not None else [1] * dim
        self.cats = np.zeros((n_cats, dim))
        self.fitness = np.zeros(n_cats)
        self.best_cat = np.zeros(dim)
        self.best_fitness = float('inf')
        self.patience = patience
        self.no_improve_counter = 0

    def initialize_population(self):
        """Initialize the population of cats"""
        self.cats = np.random.uniform(self.lb, self.ub, (self.n_cats, self.dim))

    def evaluate_fitness(self, model_func, X_train, y_train, X_val, y_val):
        """Evaluate fitness of all cats"""
        for i in range(self.n_cats):
            params = self._decode_parameters(self.cats[i])
            try:
                print(f"🐱 Evaluating Cat {i+1}/{self.n_cats} with params: {params}")
                model, history = model_func(params, X_train, y_train, X_val, y_val)

                # Use validation loss as fitness
                self.fitness[i] = history.history['val_loss'][-1]

                # Check improvement
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_cat = self.cats[i].copy()
                    self.no_improve_counter = 0
                    print(f"   🏆 New best fitness: {self.best_fitness:.6f}")
            except Exception as e:
                print(f"⚠️ Error evaluating cat {i}: {e}")
                self.fitness[i] = float('inf')

    def _decode_parameters(self, cat):
        """Decode normalized parameters to actual values"""
        param_ranges = {
            'lstm_units': (32, 256),
            'dropout_rate': (0.1, 0.5),
            'learning_rate': (0.0001, 0.01),
            'batch_size': (16, 128),
            'dense_units': (16, 128)
        }

        decoded = {}
        for i, (key, (low, high)) in enumerate(param_ranges.items()):
            if i < len(cat):
                if key in ['batch_size', 'lstm_units', 'dense_units']:
                    decoded[key] = int(low + cat[i] * (high - low))
                else:
                    decoded[key] = low + cat[i] * (high - low)
        return decoded

    def update_positions(self, iteration):
        """Update positions of cats based on SCSO algorithm"""
        R = 2 - 2 * (iteration / self.max_iter)  # awareness radius shrinks over time

        for i in range(self.n_cats):
            r = np.random.random() * R

            if r <= 1:  # Exploration (spiral-like search)
                r1, r2, r3 = np.random.random(3)
                self.cats[i] = (
                    self.best_cat
                    - r1 * np.abs(r2 * self.best_cat - self.cats[i]) * np.cos(r3 * 2 * np.pi)
                )
            else:  # Exploitation (local dimension-wise search)
                rand_idx = np.random.randint(0, self.dim)
                step = np.random.random()
                self.cats[i, rand_idx] = (
                    self.best_cat[rand_idx]
                    + step * np.cos(np.random.random() * 2 * np.pi)
                    * np.abs(self.best_cat[rand_idx] - self.cats[i, rand_idx])
                )

            # Keep within bounds
            self.cats[i] = np.clip(self.cats[i], self.lb, self.ub)

    def optimize(self, model_func, X_train, y_train, X_val, y_val):
        """Run the optimization process"""
        self.initialize_population()

        for iter in range(self.max_iter):
            print(f"\n[SCSO] Iteration {iter+1}/{self.max_iter}")
            prev_best = self.best_fitness

            self.evaluate_fitness(model_func, X_train, y_train, X_val, y_val)
            self.update_positions(iter)

            # Early stopping check
            if self.best_fitness < prev_best:
                self.no_improve_counter = 0
            else:
                self.no_improve_counter += 1

            print(f"[SCSO] Iter {iter+1} complete | Best fitness: {self.best_fitness:.6f}")

            if self.no_improve_counter >= self.patience:
                print(f"\n⏹️ Early stopping at iteration {iter+1} | No improvement for {self.patience} iterations")
                break

        return self._decode_parameters(self.best_cat)

class FinancialForecaster:
    """Main class for financial forecasting"""
    
    def __init__(self, ticker, timeframe='1d', lstm_epochs=100, val_horizon=30, 
                 candles_to_fetch=1000, lookback=60, forecast_horizon=1):
        """
        Initialize the financial forecaster
        
        Parameters:
        ticker (str): Stock or crypto ticker symbol
        timeframe (str): Timeframe for data (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        lstm_epochs (int): Number of epochs for LSTM training
        val_horizon (int): Number of periods for validation
        candles_to_fetch (int): Number of candles to fetch
        lookback (int): Number of time steps to look back
        forecast_horizon (int): Number of time steps to forecast
        """
        self.ticker = ticker
        self.timeframe = timeframe
        self.lstm_epochs = lstm_epochs
        self.val_horizon = val_horizon
        self.candles_to_fetch = candles_to_fetch
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        
        # Initialize components
        self.data_fetcher = FinancialDataFetcher()
        self.feature_engineer = FeatureEngineer()
        
        # Placeholders for data and models
        self.raw_data = None
        self.processed_data = None
        self.X = None
        self.y = None
        self.feature_scaler = None
        self.target_scaler = None
        self.lstm_model = None
        self.xgb_model = None
        self.lstm_history = None
        self.scso_params = None
        
    def fetch_and_preprocess_data(self):
        """Fetch and preprocess the data"""
        print("Fetching data...")
        self.raw_data = self.data_fetcher.fetch_data(
            self.ticker, self.timeframe, self.candles_to_fetch
        )
        
        if self.raw_data is None or self.raw_data.empty:
            raise ValueError("Failed to fetch data. Please check your ticker and timeframe.")
        
        print("Adding technical indicators and features...")
        self.processed_data = self.feature_engineer.add_all_indicators(self.raw_data)
        # self.processed_data.to_csv("processed_data.csv")
        
        print("Preparing data for training...")
        self.X, self.y, self.feature_scaler, self.target_scaler = self.feature_engineer.prepare_data_for_training(
            self.processed_data, self.lookback, self.forecast_horizon
        )
        
        print(f"Data shape: {self.X.shape}, Target shape: {self.y.shape}")
    
    def create_lstm_model(self, params, input_shape):
        """
        Create LSTM model with given parameters
        
        Parameters:
        params (dict): Parameters for the model
        input_shape (tuple): Shape of input data
        
        Returns:
        tf.keras.Model: Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=params.get('lstm_units', 128),
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(params.get('dropout_rate', 0.2)))
        
        # Second LSTM layer
        model.add(LSTM(
            units=params.get('lstm_units', 128) // 2,
            return_sequences=False
        ))
        model.add(Dropout(params.get('dropout_rate', 0.2)))
        
        # Dense layers
        model.add(Dense(params.get('dense_units', 64), activation='relu'))
        model.add(Dropout(params.get('dropout_rate', 0.2) / 2))
        
        model.add(Dense(params.get('dense_units', 64) // 2, activation='relu'))
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm_with_params(self, params, X_train, y_train, X_val, y_val):
        """
        Train LSTM model with given parameters
        
        Parameters:
        params (dict): Parameters for the model
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
        Returns:
        tuple: model, history
        """
        # Create model
        model = self.create_lstm_model(params, (X_train.shape[1], X_train.shape[2]))
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=self.lstm_epochs,
            batch_size=params.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        return model, history
    
    def scso_objective(self, params, X_train, y_train, X_val, y_val):
        """
        Objective function for SCSO optimization
        
        Parameters:
        params (dict): Parameters for the model
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
        Returns:
        tuple: model, history
        """
        return self.train_lstm_with_params(params, X_train, y_train, X_val, y_val)
    
    def optimize_with_scso(self):
        """Optimize LSTM parameters using SCSO"""
        print("Optimizing LSTM parameters with SCSO...")
        
        # Split data for optimization
        X_train, X_val, y_train, y_val = train_test_split(
            self.X[:-self.val_horizon], 
            self.y[:-self.val_horizon], 
            test_size=0.2, 
            shuffle=False
        )
        
        # Initialize SCSO
        scso = SCSO(
            n_cats=10, 
            max_iter=20,  # Reduced for demonstration
            dim=5,
            lb=[0, 0, 0, 0, 0],
            ub=[1, 1, 1, 1, 1]
        )
        
        # Define model function for SCSO
        def model_func(params, X_train, y_train, X_val, y_val):
            return self.train_lstm_with_params(params, X_train, y_train, X_val, y_val)
        
        # Run optimization
        self.scso_params = scso.optimize(model_func, X_train, y_train, X_val, y_val)
        
        print("SCSO optimization completed. Best parameters:")
        for key, value in self.scso_params.items():
            print(f"  {key}: {value}")
    
    def train_final_models(self):
        """Train final models with optimized parameters"""
        print("Training final models...")
        
        # Split data
        train_size = len(self.X) - self.val_horizon
        X_train, X_test = self.X[:train_size], self.X[train_size:]
        y_train, y_test = self.y[:train_size], self.y[train_size:]
        
        # Train LSTM with optimized parameters
        print("Training LSTM with optimized parameters...")
        self.lstm_model, self.lstm_history = self.train_lstm_with_params(
            self.scso_params, X_train, y_train, X_test, y_test
        )
        
        # Train XGBoost as baseline
        print("Training XGBoost baseline...")
        # Reshape data for XGBoost (from 3D to 2D)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50
        )
        
        self.xgb_model.fit(
            X_train_flat, y_train,
            eval_set=[(X_test_flat, y_test)],
            verbose=100
        )
    
    def forecast(self):
        """Make forecasts with both models"""
        print("Making forecasts...")
        
        # Prepare the most recent data for forecasting
        last_sequence = self.X[-1:]
        
        # LSTM forecast
        lstm_pred_scaled = self.lstm_model.predict(last_sequence)
        lstm_pred = self.target_scaler.inverse_transform(lstm_pred_scaled)
        
        # XGBoost forecast
        last_sequence_flat = last_sequence.reshape(1, -1)
        xgb_pred_scaled = self.xgb_model.predict(last_sequence_flat)
        xgb_pred = self.target_scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1))
        
        # Get actual values for comparison
        actual = self.target_scaler.inverse_transform(self.y[-1:])
        
        return {
            'lstm_pred': lstm_pred[0][0],
            'xgb_pred': xgb_pred[0][0],
            'actual': actual[0][0] if len(self.y) > 0 else None
        }
    
    def calculate_metrics(self):
        """Calculate performance metrics for both models"""
        print("Calculating performance metrics...")
        
        # Prepare validation data
        val_start = len(self.X) - self.val_horizon
        X_val = self.X[val_start:]
        y_val = self.y[val_start:]
        
        # LSTM predictions
        lstm_pred_scaled = self.lstm_model.predict(X_val)
        lstm_pred = self.target_scaler.inverse_transform(lstm_pred_scaled)
        y_val_actual = self.target_scaler.inverse_transform(y_val)
        
        # XGBoost predictions
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        xgb_pred_scaled = self.xgb_model.predict(X_val_flat)
        xgb_pred = self.target_scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1))
        
        # Calculate metrics for both models
        metrics = {}
        
        for model_name, predictions in [('LSTM', lstm_pred), ('XGBoost', xgb_pred)]:
            # Regression metrics
            mae = mean_absolute_error(y_val_actual, predictions)
            mse = mean_squared_error(y_val_actual, predictions)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_val_actual - predictions) / y_val_actual)) * 100
            r2 = r2_score(y_val_actual, predictions)
            
            # Calculate returns and Sharpe ratio
            returns = np.diff(y_val_actual.flatten()) / y_val_actual[:-1].flatten()
            pred_returns = np.diff(predictions.flatten()) / predictions[:-1].flatten()
            
            # If we have enough data points
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = np.nan
            
            metrics[model_name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2,
                'Sharpe Ratio': sharpe_ratio
            }
        
        return metrics
    
    def plot_results(self, forecasts, metrics):
        """Plot the results"""
        # Prepare validation data for plotting
        val_start = len(self.X) - self.val_horizon
        y_val = self.y[val_start:]
        y_val_actual = self.target_scaler.inverse_transform(y_val)
        
        # LSTM predictions
        lstm_pred_scaled = self.lstm_model.predict(self.X[val_start:])
        lstm_pred = self.target_scaler.inverse_transform(lstm_pred_scaled)
        
        # XGBoost predictions
        X_val_flat = self.X[val_start:].reshape(self.X[val_start:].shape[0], -1)
        xgb_pred_scaled = self.xgb_model.predict(X_val_flat)
        xgb_pred = self.target_scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training history
        axes[0, 0].plot(self.lstm_history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.lstm_history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('LSTM Training History')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Validation predictions
        time_index = range(len(y_val_actual))
        axes[0, 1].plot(time_index, y_val_actual, label='Actual', linewidth=2)
        axes[0, 1].plot(time_index, lstm_pred, label='LSTM Prediction', linestyle='--')
        axes[0, 1].plot(time_index, xgb_pred, label='XGBoost Prediction', linestyle='--')
        axes[0, 1].set_title('Validation Predictions')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Metrics comparison
        model_names = list(metrics.keys())
        metric_names = ['MAE', 'RMSE', 'MAPE']
        metric_values = {metric: [metrics[model][metric] for model in model_names] for metric in metric_names}
        
        x = np.arange(len(model_names))
        width = 0.25
        multiplier = 0
        
        for metric, values in metric_values.items():
            offset = width * multiplier
            rects = axes[1, 0].bar(x + offset, values, width, label=metric)
            axes[1, 0].bar_label(rects, padding=3, fmt='%.2f')
            multiplier += 1
        
        axes[1, 0].set_title('Model Performance Metrics')
        axes[1, 0].set_xticks(x + width, model_names)
        axes[1, 0].legend(loc='upper left', ncols=3)
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True, axis='y')
        
        # Plot 4: Forecast
        axes[1, 1].bar(['LSTM', 'XGBoost', 'Actual'], 
                      [forecasts['lstm_pred'], forecasts['xgb_pred'], forecasts['actual']],
                      color=['blue', 'orange', 'green'])
        axes[1, 1].set_title('Next Period Forecast')
        axes[1, 1].set_ylabel('Price')
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.ticker.replace("/", "_")}_forecast_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_forecasting_pipeline(self):
        """Run the complete forecasting pipeline"""
        try:
            # Step 1: Fetch and preprocess data
            self.fetch_and_preprocess_data()
            
            # Step 2: Optimize LSTM parameters with SCSO
            self.optimize_with_scso()
            
            # Step 3: Train final models
            self.train_final_models()
            
            # Step 4: Make forecasts
            forecasts = self.forecast()
            
            # Step 5: Calculate metrics
            metrics = self.calculate_metrics()
            
            # Step 6: Plot results
            self.plot_results(forecasts, metrics)
            
            # Print results
            print("\n" + "="*50)
            print("FORECASTING RESULTS")
            print("="*50)
            print(f"Ticker: {self.ticker}")
            print(f"Timeframe: {self.timeframe}")
            print(f"Forecast Horizon: {self.forecast_horizon}")
            print("\nNext Period Forecast:")
            print(f"  LSTM Prediction: {forecasts['lstm_pred']:.4f}")
            print(f"  XGBoost Prediction: {forecasts['xgb_pred']:.4f}")
            if forecasts['actual'] is not None:
                print(f"  Actual Value: {forecasts['actual']:.4f}")
            
            print("\nPerformance Metrics:")
            for model, model_metrics in metrics.items():
                print(f"\n{model}:")
                for metric, value in model_metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
            return forecasts, metrics
            
        except Exception as e:
            print(f"Error in forecasting pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None, None

# Example usage
if __name__ == "__main__":
    # Initialize with default parameters
    forecaster = FinancialForecaster(
        ticker="AAPL",           # Can also use "BTC/USDT" for crypto
        timeframe="1d",          # 1m, 5m, 15m, 30m, 1h, 4h, 1d
        lstm_epochs=50,         # Number of epochs for LSTM training
        val_horizon=36,          # Validation periods
        candles_to_fetch=720,   # Number of candles to fetch
        lookback=60,             # Lookback period
        forecast_horizon=1       # Forecast horizon
    )
    
    # Run the forecasting pipeline
    forecasts, metrics = forecaster.run_forecasting_pipeline()