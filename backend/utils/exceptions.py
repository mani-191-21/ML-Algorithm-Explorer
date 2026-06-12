"""Custom exceptions for the ML-Algorithm-Explorer backend."""


class MLVerseException(Exception):
    """Base exception for all MLVerse errors."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelNotFoundError(MLVerseException):
    """Raised when a requested model is not found or not loaded."""
    
    def __init__(self, model_name: str):
        message = f"Model '{model_name}' not found or not loaded"
        super().__init__(message, status_code=404)


class PredictionError(MLVerseException):
    """Raised when prediction fails."""
    
    def __init__(self, model_name: str, details: str):
        message = f"Prediction failed for model '{model_name}': {details}"
        super().__init__(message, status_code=500)


class ValidationError(MLVerseException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, reason: str):
        message = f"Validation error in field '{field}': {reason}"
        super().__init__(message, status_code=400)


class ModelLoadError(MLVerseException):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, details: str):
        message = f"Failed to load model '{model_name}': {details}"
        super().__init__(message, status_code=500)
