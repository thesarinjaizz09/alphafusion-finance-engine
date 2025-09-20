# models/hmm_model.py
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Optional
import joblib
from utils.logging import logger
from config import config

class HMMMarketRegimeDetector:
    """Hidden Markov Model for market regime detection"""
    
    def __init__(self, n_regimes: int = config.model.HMM_N_REGIMES):
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",
            n_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM"""
        features = df[['returns', 'volatility', 'rsi', 'macd']].copy()
        features = features.dropna()
        return features.values
    
    def fit(self, df: pd.DataFrame) -> 'HMMMarketRegimeDetector':
        """Fit the HMM model"""
        try:
            logger.info("Fitting HMM model for market regime detection")
            features = self.prepare_features(df)
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled)
            self.is_fitted = True
            logger.info("HMM model fitted successfully")
            return self
        except Exception as e:
            logger.error(f"Error fitting HMM model: {e}")
            raise
    
    def predict_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict market regimes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            logger.info("Predicting market regimes with HMM")
            features = self.prepare_features(df)
            features_scaled = self.scaler.transform(features)
            regimes = self.model.predict(features_scaled)
            
            # Add regime as a feature to the dataframe
            df_with_regime = df.copy()
            df_with_regime = df_with_regime.iloc[-len(regimes):].copy()
            df_with_regime['market_regime'] = regimes
            
            logger.info("Market regime prediction completed")
            return df_with_regime
        except Exception as e:
            logger.error(f"Error predicting market regimes: {e}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk"""
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }, filepath)
            logger.info(f"HMM model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving HMM model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_fitted = model_data['is_fitted']
            logger.info(f"HMM model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading HMM model: {e}")
            raise