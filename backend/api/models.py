"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class PredictionRequest(BaseModel):
    """Request model for making predictions."""
    
    model: str = Field(..., description="Model identifier")
    data: Dict[str, Any] = Field(..., description="Input features for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "linear",
                "data": {"feature1": 10.5, "feature2": "category_a"}
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    model: str = Field(..., description="Model used for prediction")
    prediction: float = Field(..., description="Predicted value")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "linear",
                "prediction": 150000.5
            }
        }


class FeatureInfo(BaseModel):
    """Information about a model feature."""
    
    label: str = Field(..., description="Display label for the feature")
    type: str = Field(..., description="Feature type: 'number' or 'select'")
    default: Any = Field(..., description="Default value")
    options: Optional[List[str]] = Field(None, description="Options for select type")


class ModelInfo(BaseModel):
    """Information about a model."""
    
    name: str = Field(..., description="Model display name")
    algorithm: str = Field(..., description="Algorithm name")
    dataset: str = Field(..., description="Dataset used for training")
    target: str = Field(..., description="Target variable")
    description: Optional[str] = Field(None, description="Model description")
    loaded: bool = Field(..., description="Whether model is loaded")
    features: Dict[str, FeatureInfo] = Field(..., description="Model features")


class ModelsResponse(BaseModel):
    """Response model for listing all models."""
    
    total: int = Field(..., description="Total number of models")
    loaded: int = Field(..., description="Number of loaded models")
    models: Dict[str, ModelInfo] = Field(..., description="Model information")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="API status")
    total_models: int = Field(..., description="Total number of models")
    loaded_models: int = Field(..., description="Number of loaded models")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[str] = Field(None, description="Additional error details")
