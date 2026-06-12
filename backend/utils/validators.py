"""Input validation utilities."""

from typing import Dict, Any, List
from backend.utils.exceptions import ValidationError


def validate_prediction_data(data: Dict[str, Any], required_features: List[str]) -> bool:
    """
    Validate prediction input data.
    
    Args:
        data: Input data dictionary
        required_features: List of required feature names
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError("data", "Must be a dictionary")
    
    if not data:
        raise ValidationError("data", "Cannot be empty")
    
    if len(data) > 100:
        raise ValidationError("data", "Too many features (max 100)")
    
    return True


def validate_model_name(model_name: str) -> bool:
    """
    Validate model name format.
    
    Args:
        model_name: Model identifier
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(model_name, str):
        raise ValidationError("model", "Must be a string")
    
    if not model_name.strip():
        raise ValidationError("model", "Cannot be empty")
    
    if len(model_name) > 50:
        raise ValidationError("model", "Name too long (max 50 characters)")
    
    return True
