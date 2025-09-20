# services/forecast_service.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from data.data_fetcher import DataFetcher
from features.feature_engineer import FeatureEngineer
from models.hmm_model import HMMMarketRegimeDetector
from models.pso_lstm import PSOLSTM
from models.tft_model import TFTForecaster
from models.ensemble import EnsembleForecaster
from utils.logging import logger
from utils.cache import cached
from config import config

class ForecastService:
    """Main service for the AI forecasting system"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.hmm_detector = HMMMarketRegimeDetector()
        self.pso_lstm = PSOLSTM()
        self.tft_forecaster = TFTForecaster()
        self.ensemble = EnsembleForecaster()
        self.raw_data = None
        self.features_df = None
        self.regime_df = None
    
    @cached("forecast_data_{asset_type}_{symbol}", expire=1800)
    def fetch_data(self, asset_type: str, symbol: str) -> pd.DataFrame:
        """Fetch data based on asset type"""
        try:
            if asset_type == 'stock':
                self.raw_data = self.data_fetcher.fetch_stock_data(symbol)
            elif asset_type == 'crypto':
                self.raw_data = self.data_fetcher.fetch_crypto_data(symbol)
            else:
                raise ValueError("Asset type must be 'stock' or 'crypto'")
            
            return self.raw_data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features"""
        try:
            self.features_df = self.feature_engineer.add_technical_indicators(df)
            return self.features_df
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise
    
    def detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regime using HMM"""
        try:
            # Fit HMM
            self.hmm_detector.fit(df)
            
            # Predict regimes
            self.regime_df = self.hmm_detector.predict_regimes(df)
            
            return self.regime_df
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            raise
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models"""
        try:
            logger.info("Training all models")
            
            # Train PSO-LSTM
            pso_lstm_result = self.pso_lstm.fit(df)
            
            # Train TFT
            tft_result = self.tft_forecaster.fit(df)
            
            return {
                'pso_lstm': pso_lstm_result,
                'tft': tft_result
            }
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def predict_and_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with all models and ensemble them"""
        try:
            # Get predictions from each model
            # actuals_lstm, predictions_lstm = []
            actuals_lstm, predictions_lstm = self.pso_lstm.predict(df)
            actuals_tft, predictions_tft = self.tft_forecaster.predict(df)
            
            # Ensure we have the same number of predictions
            min_len = min(len(actuals_lstm), len(actuals_tft))
            actuals_lstm = actuals_lstm[:min_len]
            predictions_lstm = predictions_lstm[:min_len]
            actuals_tft = actuals_tft[:min_len]
            predictions_tft = predictions_tft[:min_len]
            
            # Calculate weights based on performance
            weights = self.ensemble.calculate_weights(
                [actuals_lstm, actuals_tft],
                [predictions_lstm, predictions_tft]
            )
            
            # Create ensemble prediction
            ensemble_pred = self.ensemble.ensemble_predictions([predictions_lstm, predictions_tft])
            
            # Evaluate ensemble
            metrics = self.ensemble.evaluate_ensemble(actuals_lstm, ensemble_pred)
            
            return {
                'actuals': actuals_lstm,
                'predictions_lstm': predictions_lstm,
                'predictions_tft': predictions_tft,
                'ensemble_pred': ensemble_pred,
                'weights': weights,
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"Error in prediction and ensembling: {e}")
            raise
    
    def forecast_future(self, df: pd.DataFrame, steps: int = 5) -> Dict[str, Any]:
        """Forecast future values"""
        try:
            # Use the most recent data for forecasting
            recent_data = df.iloc[-self.pso_lstm.lookback:].copy()
            
            # Scale the data
            scaled_data = self.pso_lstm.scaler.transform(recent_data['close'].values.reshape(-1, 1))
            
            # Reshape for LSTM input
            X = scaled_data.reshape(1, self.pso_lstm.lookback, 1)
            
            # Predict with LSTM
            lstm_pred_scaled = self.pso_lstm.model.predict(X)
            lstm_pred = self.pso_lstm.scaler.inverse_transform(lstm_pred_scaled)
            
            # For TFT, we would need to implement a similar future forecasting method
            # This is simplified for demonstration
            
            return {
                'lstm_forecast': lstm_pred.flatten(),
                'tft_forecast': None,  # Would be implemented in a real production system
            }
        except Exception as e:
            logger.error(f"Error forecasting future values: {e}")
            raise
    
    def save_models(self, path: str = config.MODEL_STORAGE_PATH) -> None:
        """Save all trained models to disk"""
        try:
            import os
            os.makedirs(path, exist_ok=True)
            
            self.hmm_detector.save_model(f"{path}/hmm_model.pkl")
            self.pso_lstm.save_model(f"{path}/pso_lstm_model.pkl")
            self.tft_forecaster.save_model(f"{path}/tft_model.pkl")
            
            logger.info(f"All models saved to {path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, path: str = config.MODEL_STORAGE_PATH) -> None:
        """Load all trained models from disk"""
        try:
            self.hmm_detector.load_model(f"{path}/hmm_model.pkl")
            self.pso_lstm.load_model(f"{path}/pso_lstm_model.pkl")
            self.tft_forecaster.load_model(f"{path}/tft_model.pkl")
            
            logger.info(f"All models loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise