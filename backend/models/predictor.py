"""Prediction logic for loaded models."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from backend.utils.logger import get_logger
from backend.utils.exceptions import PredictionError, ModelNotFoundError
from .loader import ModelLoader

logger = get_logger(__name__)


class ModelPredictor:
    """Handles predictions using loaded models."""
    
    def __init__(self, model_loader: ModelLoader):
        """
        Initialize the predictor.
        
        Args:
            model_loader: ModelLoader instance
        """
        self.loader = model_loader
    
    def predict(self, model_id: str, input_data: Dict[str, Any]) -> float:
        """
        Make a prediction using the specified model.
        
        Args:
            model_id: Model identifier
            input_data: Input features as dictionary
        
        Returns:
            Prediction value
        
        Raises:
            ModelNotFoundError: If model is not loaded
            PredictionError: If prediction fails
        """
        model = self.loader.get_model(model_id)
        
        if model is None:
            raise ModelNotFoundError(model_id)
        
        try:
            if isinstance(model, dict):
                return self._predict_dict_model(model, input_data, model_id)
            else:
                return self._predict_raw_model(model, input_data, model_id)
        
        except Exception as e:
            logger.error(f"Prediction error for model '{model_id}': {str(e)}")
            raise PredictionError(model_id, str(e))
    
    def _predict_dict_model(
        self, 
        model: Dict[str, Any], 
        input_data: Dict[str, Any],
        model_id: str
    ) -> float:
        """
        Predict using a dictionary-based model (with preprocessing).
        
        Args:
            model: Model dictionary with preprocessing info
            input_data: Input features
            model_id: Model identifier for logging
        
        Returns:
            Prediction value
        """
        actual_model = model.get('model')
        if actual_model is None:
            raise PredictionError(model_id, "Model object not found in pipeline")
        
        scaler = model.get('scaler')
        le_dict = model.get('label_encoders', {})
        feature_names = model.get('feature_names', [])
        
        # Prepare input data
        row = {}
        for col in feature_names:
            val = input_data.get(col, 0)
            
            if col in le_dict:
                # Handle categorical features
                le = le_dict[col]
                s = str(val)
                if s in list(le.classes_):
                    row[col] = int(le.transform([s])[0])
                else:
                    row[col] = 0
            else:
                # Handle numerical features
                try:
                    row[col] = float(val)
                except (ValueError, TypeError):
                    row[col] = 0.0
        
        # Create DataFrame
        df = pd.DataFrame([row])[feature_names]
        
        # Apply scaling if available
        if scaler is not None:
            X = scaler.transform(df)
        else:
            X = df.values
        
        # Make prediction
        prediction = actual_model.predict(X)
        return round(float(prediction[0]), 4)
    
    def _predict_raw_model(
        self,
        model: Any,
        input_data: Dict[str, Any],
        model_id: str
    ) -> float:
        """
        Predict using a raw model (CatBoost/XGBoost legacy).
        
        Args:
            model: Raw model object
            input_data: Input features
            model_id: Model identifier for logging
        
        Returns:
            Prediction value
        """
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        return round(float(prediction[0]), 4)
    
    def get_feature_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get feature information for a model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Dictionary with feature information
        
        Raises:
            ModelNotFoundError: If model is not loaded
        """
        model = self.loader.get_model(model_id)
        
        if model is None:
            raise ModelNotFoundError(model_id)
        
        features = {}
        
        if isinstance(model, dict):
            feature_names = model.get('feature_names', [])
            display_names = model.get('feature_names_display', {})
            le_dict = model.get('label_encoders', {})
            
            for fname in feature_names:
                label = display_names.get(fname, fname)
                
                if fname in le_dict:
                    le = le_dict[fname]
                    features[fname] = {
                        "label": label,
                        "type": "select",
                        "options": [str(c) for c in le.classes_],
                        "default": str(le.classes_[0]),
                    }
                else:
                    features[fname] = {
                        "label": label,
                        "type": "number",
                        "default": 0,
                    }
        
        return features
