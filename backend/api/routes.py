"""API routes for ML-Algorithm-Explorer."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from backend.config import API_VERSION
from backend.models import ModelLoader, ModelPredictor, ALGORITHM_REGISTRY
from backend.utils.logger import get_logger
from backend.utils.exceptions import MLVerseException, ModelNotFoundError, PredictionError
from .models import (
    PredictionRequest,
    PredictionResponse,
    ModelsResponse,
    ModelInfo,
    FeatureInfo,
    HealthResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["predictions"])

# Global instances
_model_loader: ModelLoader = None
_model_predictor: ModelPredictor = None


def get_model_loader() -> ModelLoader:
    """Dependency injection for model loader."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def get_model_predictor() -> ModelPredictor:
    """Dependency injection for model predictor."""
    global _model_predictor
    if _model_predictor is None:
        loader = get_model_loader()
        _model_predictor = ModelPredictor(loader)
    return _model_predictor


@router.on_event("startup")
async def startup_event():
    """Load all models on startup."""
    logger.info("Loading models on startup...")
    loader = get_model_loader()
    status = loader.load_all_models()
    
    loaded_count = sum(1 for v in status.values() if v)
    logger.info(f"Startup complete: {loaded_count}/{len(status)} models loaded")


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(loader: ModelLoader = Depends(get_model_loader)):
    """
    Health check endpoint.
    
    Returns:
        Health status and model information
    """
    status = loader.get_loaded_models()
    loaded_count = sum(1 for v in status.values() if v)
    
    return HealthResponse(
        status="healthy",
        total_models=len(ALGORITHM_REGISTRY),
        loaded_models=loaded_count,
        version=API_VERSION,
    )


@router.get("/models", response_model=ModelsResponse)
async def get_models(loader: ModelLoader = Depends(get_model_loader)):
    """
    Get all available models with their configurations.
    
    Returns:
        Information about all models
    """
    result = {}
    
    for model_id, config in ALGORITHM_REGISTRY.items():
        is_loaded = loader.is_loaded(model_id)
        
        # Get feature information if model is loaded
        features = {}
        if is_loaded:
            try:
                predictor = get_model_predictor()
                features = predictor.get_feature_info(model_id)
            except Exception as e:
                logger.warning(f"Could not get features for {model_id}: {str(e)}")
        
        result[model_id] = ModelInfo(
            name=config["name"],
            algorithm=config["algorithm"],
            dataset=config["dataset"],
            target=config["target"],
            description=config.get("description"),
            loaded=is_loaded,
            features=features,
        )
    
    loaded_count = sum(1 for m in result.values() if m.loaded)
    
    return ModelsResponse(
        total=len(result),
        loaded=loaded_count,
        models=result,
    )


@router.get("/models/{model_id}")
async def get_model_detail(
    model_id: str,
    loader: ModelLoader = Depends(get_model_loader)
):
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Model identifier
    
    Returns:
        Detailed model information
    """
    config = ALGORITHM_REGISTRY.get(model_id)
    
    if not config:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    is_loaded = loader.is_loaded(model_id)
    features = {}
    
    if is_loaded:
        try:
            predictor = get_model_predictor()
            features = predictor.get_feature_info(model_id)
        except Exception as e:
            logger.warning(f"Could not get features for {model_id}: {str(e)}")
    
    return ModelInfo(
        name=config["name"],
        algorithm=config["algorithm"],
        dataset=config["dataset"],
        target=config["target"],
        description=config.get("description"),
        loaded=is_loaded,
        features=features,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    predictor: ModelPredictor = Depends(get_model_predictor)
):
    """
    Make a prediction using the specified model.
    
    Args:
        request: Prediction request with model ID and input data
        predictor: Model predictor instance
    
    Returns:
        Prediction result
    
    Raises:
        HTTPException: If prediction fails
    """
    try:
        prediction = predictor.predict(request.model, request.data)
        
        return PredictionResponse(
            model=request.model,
            prediction=prediction,
        )
    
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=e.message)
    
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction"
        )


@router.get("/chart-data/{model_id}")
async def get_chart_data(
    model_id: str,
    loader: ModelLoader = Depends(get_model_loader)
):
    """
    Get pre-computed chart data for a model.
    
    Args:
        model_id: Model identifier
    
    Returns:
        Chart data if available
    """
    model = loader.get_model(model_id)
    
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    if isinstance(model, dict) and 'chart_data' in model:
        return model['chart_data']
    
    return {"error": "No chart data available for this model"}
