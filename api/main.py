# api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import ForecastRequest, ForecastResponse, TrainRequest, TrainResponse
from services.forecast_service import ForecastService
from utils.logging import logger
from config import config

app = FastAPI(title="AI Stock & Crypto Forecasting API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize forecast service
forecast_service = ForecastService()

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """Generate forecasts for a given asset"""
    try:
        logger.info(f"Forecast request received for {request.asset_type}: {request.symbol}")
        
        # Fetch data
        data = forecast_service.fetch_data(request.asset_type, request.symbol)
        
        # Engineer features
        features_df = forecast_service.engineer_features(data)
        
        # Detect market regime
        regime_df = forecast_service.detect_market_regime(features_df)
        
        # Train models if requested
        if request.retrain:
            forecast_service.train_models(regime_df)
        
        # Make predictions and ensemble
        results = forecast_service.predict_and_ensemble(regime_df)
        
        # Prepare response
        response = ForecastResponse(
            success=True,
            predictions={
                'ensemble': results['ensemble_pred'].tolist(),
                'pso_lstm': results['predictions_lstm'].tolist(),
                'tft': results['predictions_tft'].tolist()
            },
            metrics=results['metrics'],
            weights=results['weights'].tolist()
        )
        
        logger.info(f"Forecast completed for {request.asset_type}: {request.symbol}")
        return response
    
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train models for a given asset (can be run in background)"""
    try:
        logger.info(f"Training request received for {request.asset_type}: {request.symbol}")
        
        def train_task():
            try:
                # Fetch data
                data = forecast_service.fetch_data(request.asset_type, request.symbol)
                
                # Engineer features
                features_df = forecast_service.engineer_features(data)
                
                # Detect market regime
                regime_df = forecast_service.detect_market_regime(features_df)
                
                # Train models
                forecast_service.train_models(regime_df)
                
                # Save models
                forecast_service.save_models()
                
                logger.info(f"Training completed for {request.asset_type}: {request.symbol}")
            except Exception as e:
                logger.error(f"Error in training task: {e}")
        
        # Run training in background
        background_tasks.add_task(train_task)
        
        return TrainResponse(
            success=True,
            message="Training started in background"
        )
    
    except Exception as e:
        logger.error(f"Error in train endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.api.HOST,
        port=config.api.PORT,
        reload=config.api.DEBUG,
        workers=config.api.WORKERS
    )