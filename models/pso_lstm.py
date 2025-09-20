# models/pso_lstm.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from pyswarm import pso
import joblib
from typing import Tuple
from utils.logging import logger
from config import config

class PSOLSTM:
    """PSO-optimized LSTM model for time series forecasting"""

    def __init__(self, lookback: int = config.data.LOOKBACK_WINDOW,
                 forecast_horizon: int = config.data.FORECAST_HORIZON,
                 n_particles: int = config.model.PSO_N_PARTICLES,
                 max_iter: int = config.model.PSO_MAX_ITER,
                 patience: int = 5):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.patience = patience
        self.scaler = MinMaxScaler()
        self.model = None
        self.best_params = None
        self.is_fitted = False

        # Track PSO progress
        self._best_loss = np.inf
        self._best_params = None
        self._no_improve = 0
        self._stop_pso = False  # flag to emulate early stopping in PSO

    def create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(lookback, len(data) - self.forecast_horizon + 1):
            X.append(data[i-lookback:i])
            y.append(data[i:i+self.forecast_horizon])
        return np.array(X), np.array(y)

    def build_lstm(self, params: list, n_features: int) -> Sequential:
        units1, units2, dropout_rate, learning_rate = params
        model = Sequential([
            Input(shape=(self.lookback, n_features)),
            LSTM(int(units1), return_sequences=True),
            Dropout(dropout_rate),
            LSTM(int(units2)),
            Dropout(dropout_rate),
            Dense(self.forecast_horizon)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model

    def objective_function(self, params: list, X_train, y_train, X_val, y_val, n_features):
        if self._stop_pso:
            return self._best_loss  # freeze swarm after patience reached

        try:
            model = self.build_lstm(params, n_features)
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=config.model.BATCH_SIZE,
                validation_data=(X_val, y_val),
                verbose=0,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
            )
            val_loss = history.history['val_loss'][-1]

            logger.info(f"Trial params: units1={int(params[0])}, units2={int(params[1])}, "
                        f"dropout={params[2]:.3f}, lr={params[3]:.5f} -> val_loss={val_loss:.6f}")

            # Track best
            if val_loss < self._best_loss:
                self._best_loss = val_loss
                self._best_params = params
                self._no_improve = 0
                logger.info(f"New best found: val_loss={val_loss:.6f}")
            else:
                self._no_improve += 1
                if self._no_improve >= self.patience:
                    logger.warning("PSO patience reached, halting further improvements.")
                    self._stop_pso = True

            return val_loss
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return float('inf')

    def optimize_with_pso(self, X_train, y_train, X_val, y_val, n_features) -> Tuple[list, float]:
        lb = [32, 16, 0.1, 0.0001]
        ub = [256, 128, 0.5, 0.01]

        self._best_loss = float("inf")
        self._best_params = None
        self._no_improve = 0
        self._stop_pso = False

        def wrapped_obj(params):
            return self.objective_function(params, X_train, y_train, X_val, y_val, n_features)

        best_params, best_val_loss = pso(
            wrapped_obj, lb, ub,
            args=(),
            swarmsize=self.n_particles,
            maxiter=self.max_iter,
            debug=True,
            f_ieqcons=None
        )

        # return best tracked values
        if self._best_params is not None:
            return self._best_params, self._best_loss
        return best_params, best_val_loss

    def prepare_data(self, features_df, target_col='close') -> Tuple:
        target = features_df[target_col].values.reshape(-1, 1)
        target_scaled = self.scaler.fit_transform(target)
        X, y = self.create_sequences(target_scaled, self.lookback)
        split_idx = int((1 - config.data.TEST_SIZE) * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        val_split_idx = int((1 - config.data.VALIDATION_SIZE) * len(X_train))
        X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def fit(self, features_df, target_col='close') -> dict:
        logger.info("Training PSO-LSTM model")
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(features_df, target_col)
        n_features = X_train.shape[2]

        logger.info("Optimizing LSTM with PSO...")
        best_params, best_val_loss = self.optimize_with_pso(X_train, y_train, X_val, y_val, n_features)
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation loss: {best_val_loss}")

        self.best_params = best_params
        self.model = self.build_lstm(best_params, n_features)

        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=config.model.EARLY_STOPPING_PATIENCE,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=5,
                              min_lr=1e-7)
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs=config.model.LSTM_EPOCHS,
            batch_size=config.model.BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=callbacks
        )

        self.is_fitted = True
        logger.info("PSO-LSTM model training completed")

        return {'history': history.history, 'best_params': best_params, 'best_val_loss': best_val_loss}

    def predict(self, features_df, target_col='close') -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        _, _, _, _, X_test, y_test = self.prepare_data(features_df, target_col)
        predictions_scaled = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
        predictions = predictions.reshape(predictions_scaled.shape)
        actuals = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        actuals = actuals.reshape(y_test.shape)
        return actuals, predictions

    def save_model(self, filepath: str) -> None:
        joblib.dump({'model': self.model, 'scaler': self.scaler,
                     'best_params': self.best_params, 'is_fitted': self.is_fitted}, filepath)
        logger.info(f"PSO-LSTM model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.best_params = model_data['best_params']
        self.is_fitted = model_data['is_fitted']
        logger.info(f"PSO-LSTM model loaded from {filepath}")
