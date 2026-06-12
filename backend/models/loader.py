"""Model loading and management."""

import joblib
from pathlib import Path
from typing import Dict, Any, Optional

from backend.config import ML_MODELS_DIR
from backend.utils.logger import get_logger
from backend.utils.exceptions import ModelLoadError
from .registry import ALGORITHM_REGISTRY

logger = get_logger(__name__)


class ModelLoader:
    """Handles loading and caching of ML models."""
    
    def __init__(self):
        """Initialize the model loader."""
        self.models: Dict[str, Any] = {}
        self.loaded_models: Dict[str, bool] = {}
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all models from the registry.
        
        Returns:
            Dictionary with model IDs as keys and load status as values
        """
        for model_id, config in ALGORITHM_REGISTRY.items():
            self.load_model(model_id)
        
        return self.loaded_models
    
    def load_model(self, model_id: str) -> bool:
        """
        Load a single model by ID.
        
        Args:
            model_id: Model identifier
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if model_id in self.models:
            logger.info(f"Model '{model_id}' already loaded")
            return True
        
        config = ALGORITHM_REGISTRY.get(model_id)
        if not config:
            logger.error(f"Model '{model_id}' not found in registry")
            self.loaded_models[model_id] = False
            return False
        
        pkl_path = config.get("pkl_path")
        if not pkl_path:
            logger.error(f"No pkl_path configured for model '{model_id}'")
            self.loaded_models[model_id] = False
            return False
        
        full_path = ML_MODELS_DIR / pkl_path
        
        try:
            logger.info(f"Loading model '{model_id}' from {full_path}...")
            model = joblib.load(full_path)
            self.models[model_id] = model
            self.loaded_models[model_id] = True
            logger.info(f"✓ Model '{model_id}' loaded successfully")
            return True
        
        except FileNotFoundError:
            logger.error(f"Model file not found: {full_path}")
            self.loaded_models[model_id] = False
            raise ModelLoadError(model_id, f"File not found: {full_path}")
        
        except Exception as e:
            logger.error(f"Failed to load model '{model_id}': {str(e)}")
            self.loaded_models[model_id] = False
            raise ModelLoadError(model_id, str(e))
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """
        Get a loaded model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Model object or None if not loaded
        """
        return self.models.get(model_id)
    
    def is_loaded(self, model_id: str) -> bool:
        """
        Check if a model is loaded.
        
        Args:
            model_id: Model identifier
        
        Returns:
            True if model is loaded
        """
        return model_id in self.models
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """
        Get status of all models.
        
        Returns:
            Dictionary with model IDs and their load status
        """
        return self.loaded_models.copy()
