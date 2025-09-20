# models/tft_model.py
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd
from typing import Tuple
import joblib
from utils.logging import logger
from config import config

class TFTForecaster:
    """Temporal Fusion Transformer for time series forecasting"""

    def __init__(self, lookback: int = config.data.LOOKBACK_WINDOW,
                 forecast_horizon: int = config.data.FORECAST_HORIZON,
                 max_encoder_length: int = 32,
                 hidden_size: int = config.model.TFT_HIDDEN_SIZE):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.max_encoder_length = max_encoder_length
        self.hidden_size = hidden_size
        self.model = None
        self.trainer = None
        self.training = None
        self.is_fitted = False

    def prepare_data(self, features_df: pd.DataFrame, target_col: str = 'close') -> Tuple:
        """Prepare data for TFT"""
        try:
            df = features_df.copy().reset_index()
            df['time_idx'] = range(len(df))
            df['group'] = 0  # single time series

            training_cutoff = int((1 - config.data.TEST_SIZE) * len(df))

            training = TimeSeriesDataSet(
                df[df.time_idx <= training_cutoff],
                time_idx="time_idx",
                target=target_col,
                group_ids=["group"],
                min_encoder_length=self.max_encoder_length // 2,
                max_encoder_length=self.max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=self.forecast_horizon,
                static_categoricals=[],
                static_reals=[],
                time_varying_known_categoricals=[],
                time_varying_known_reals=['time_idx'],
                time_varying_unknown_categoricals=[],
                time_varying_unknown_reals=[c for c in df.columns if c not in ['timestamp','time_idx','group',target_col]],
                target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )

            validation = TimeSeriesDataSet.from_dataset(training, df[df.time_idx > training_cutoff], predict=True, stop_randomization=True)
            print('Validation', validation, flush=True)

            train_dataloader = training.to_dataloader(train=True, batch_size=config.model.BATCH_SIZE, num_workers=3)
            val_dataloader = validation.to_dataloader(train=False, batch_size=config.model.BATCH_SIZE, num_workers=3)
            logger.info('Here')
            for batch in val_dataloader:
                print('Batch', batch, flush=True)
            # print(val_dataloader, flush=True)

            return training, train_dataloader, val_dataloader
        except Exception as e:
            logger.error(f"Error preparing data for TFT: {e}")
            raise

    def fit(self, features_df: pd.DataFrame, target_col: str = 'close',
            max_epochs: int = config.model.TFT_MAX_EPOCHS) -> Trainer:
        """Train the TFT model"""
        try:
            logger.info("Training TFT model")
            training, train_dataloader, val_dataloader = self.prepare_data(features_df, target_col)
            self.training = training

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-4,
                    patience=config.model.EARLY_STOPPING_PATIENCE,
                    verbose=True,
                    mode="min"
                ),
                LearningRateMonitor()
            ]

            # ✅ Lightning 2.x compatible
            self.trainer = Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=1,
                enable_model_summary=True,
                gradient_clip_val=0.1,
                callbacks=callbacks,
                limit_train_batches=50,
                enable_progress_bar=True
            )

            self.model = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=0.03,
                hidden_size=self.hidden_size,
                attention_head_size=4,
                dropout=0.1,
                hidden_continuous_size=8,
                output_size=7,
                loss=QuantileLoss(),
                log_interval=10,
                reduce_on_plateau_patience=4,
            )

            # ✅ Fit
            self.trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

            self.is_fitted = True
            logger.info("TFT model training completed")
            return self.trainer
        except Exception as e:
            logger.error(f"Error training TFT model: {e}")
            raise

    def predict(self, features_df: pd.DataFrame, target_col: str = 'close') -> Tuple:
        """Make predictions using the trained TFT model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        try:
            _, _, val_dataloader = self.prepare_data(features_df, target_col)
            logger.debug(val_dataloader)

            predictions = self.trainer.predict(self.model, dataloaders=val_dataloader)
            raw_predictions = torch.cat(predictions, dim=0)

            # Extract actual values from dataset
            actuals = torch.tensor(val_dataloader.dataset.get_target())

            return actuals.numpy(), raw_predictions.numpy()
        except Exception as e:
            logger.error(f"Error making predictions with TFT: {e}")
            raise

    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'training': self.training,
                'is_fitted': self.is_fitted
            }, filepath)
            logger.info(f"TFT model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving TFT model: {e}")
            raise

    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        try:
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.training = checkpoint['training']
            self.is_fitted = checkpoint['is_fitted']
            logger.info(f"TFT model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading TFT model: {e}")
            raise
