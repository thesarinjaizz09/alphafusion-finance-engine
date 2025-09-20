# api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"

class ForecastRequest(BaseModel):
    asset_type: AssetType
    symbol: str
    forecast_horizon: Optional[int] = Field(5, ge=1, le=30)
    retrain: Optional[bool] = False

class ForecastResponse(BaseModel):
    success: bool
    predictions: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    weights: Optional[List[float]] = None
    error: Optional[str] = None

class TrainRequest(BaseModel):
    asset_type: AssetType
    symbol: str

class TrainResponse(BaseModel):
    success: bool
    message: str
    error: Optional[str] = None