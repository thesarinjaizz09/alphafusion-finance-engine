# features/feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple
from utils.logging import logger

class FeatureEngineer:
    """Class for feature engineering and technical indicators"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            logger.info("Adding technical indicators to dataframe")
            df = df.copy()
            
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['volatility_30'] = df['returns'].rolling(window=30).std()
            
            # Moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Price momentum
            df['momentum'] = df['close'] / df['close'].shift(5) - 1
            df['volatility_ratio'] = df['volatility'] / df['volatility'].shift(5)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            # Store feature columns for later use
            self.feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Added {len(self.feature_columns)} technical indicators")
            return df
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            raise
    
    def scale_features(self, df: pd.DataFrame, feature_columns: list) -> Tuple[pd.DataFrame, dict]:
        """Scale features using StandardScaler"""
        try:
            logger.info("Scaling features")
            scaled_df = df.copy()
            scalers = {}
            
            for col in feature_columns:
                if col in scaled_df.columns:
                    scaler = StandardScaler()
                    scaled_df[col] = scaler.fit_transform(scaled_df[col].values.reshape(-1, 1))
                    scalers[col] = scaler
            
            logger.info("Features scaled successfully")
            return scaled_df, scalers
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            raise