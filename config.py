# config.py
import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    LOOKBACK_WINDOW: int = 60
    FORECAST_HORIZON: int = 5
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.2
    ASSET_TYPES: List[str] = field(default_factory=lambda: ["stock", "crypto"])
    DEFAULT_STOCK: str = "AAPL"
    DEFAULT_CRYPTO: str = "BTC/USDT"

@dataclass
class ModelConfig:
    HMM_N_REGIMES: int = 3
    PSO_N_PARTICLES: int = 10
    PSO_MAX_ITER: int = 20
    TFT_HIDDEN_SIZE: int = 16
    TFT_MAX_EPOCHS: int = 2
    LSTM_EPOCHS: int = 50
    BATCH_SIZE: int = 32
    EARLY_STOPPING_PATIENCE: int = 10
    OPTUNA_N_TRIALS: int = 20

@dataclass
class APIConfig:
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    WORKERS: int = 4
    API_PREFIX: str = "/api/v1"

@dataclass
class RedisConfig:
    HOST: str = "localhost"
    PORT: int = 6379
    DB: int = 0
    PASSWORD: Optional[str] = None

@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    MODEL_STORAGE_PATH: str = "./models"
    CACHE_EXPIRATION: int = 3600  # 1 hour


config = AppConfig()