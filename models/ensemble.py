# models/ensemble.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Tuple, Dict, Any
from utils.logging import logger
from config import config

class EnsembleForecaster:
    """Ensemble model that combines predictions from multiple models"""
    
    def __init__(self, forecast_horizon: int = config.data.FORECAST_HORIZON):
        self.forecast_horizon = forecast_horizon
        self.weights = None
        self.models = {}
    
    def calculate_weights(self, actuals_list: List[np.ndarray], predictions_list: List[np.ndarray]) -> np.ndarray:
        """Calculate weights based on model performance"""
        try:
            model_errors = []
            
            for actuals, predictions in zip(actuals_list, predictions_list):
                # Calculate RMSE for each model
                rmse = np.sqrt(mean_squared_error(actuals.flatten(), predictions.flatten()))
                model_errors.append(rmse)
            
            # Invert errors (lower error = higher weight)
            inverted_errors = 1 / np.array(model_errors)
            
            # Normalize to get weights
            self.weights = inverted_errors / np.sum(inverted_errors)
            
            logger.info(f"Ensemble weights calculated: {self.weights}")
            return self.weights
        except Exception as e:
            logger.error(f"Error calculating ensemble weights: {e}")
            raise
    
    def ensemble_predictions(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Combine predictions using calculated weights"""
        if self.weights is None:
            raise ValueError("Weights must be calculated before ensembling")
        
        try:
            # Weighted average of predictions
            ensemble_pred = np.zeros_like(predictions_list[0])
            
            for i, pred in enumerate(predictions_list):
                ensemble_pred += self.weights[i] * pred
            
            return ensemble_pred
        except Exception as e:
            logger.error(f"Error ensembling predictions: {e}")
            raise
    
    def evaluate_ensemble(self, actuals: np.ndarray, ensemble_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        try:
            rmse = np.sqrt(mean_squared_error(actuals.flatten(), ensemble_pred.flatten()))
            mae = mean_absolute_error(actuals.flatten(), ensemble_pred.flatten())
            mape = np.mean(np.abs((actuals.flatten() - ensemble_pred.flatten()) / actuals.flatten())) * 100
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
            
            logger.info(f"Ensemble metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            raise