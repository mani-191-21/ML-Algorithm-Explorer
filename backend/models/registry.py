"""
Model registry — Central configuration for all 11 algorithms.
pkl_path must point to relative path under ml_models/Supervised Learning folder.
"""

from typing import Dict, Any, Optional

ALGORITHM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "adaboost": {
        "name": "Insurance Charges Predictor",
        "algorithm": "AdaBoost Regressor",
        "dataset": "Medical Insurance Costs",
        "target": "Medical Insurance Charges ($)",
        "pkl_path": "Supervised Learning/AdaBoostRegression/insurance_charges_full_pipeline.pkl",
        "description": "Predicts medical insurance charges based on patient demographics and health factors.",
        "features_count": None,  # Will be populated on load
    },
    "catboost": {
        "name": "Advanced House Price Estimator",
        "algorithm": "CatBoost (Gradient Boosting)",
        "dataset": "Ames Housing Dataset (80+ features)",
        "target": "House Sale Price ($)",
        "pkl_path": "Supervised Learning/CatGBMRegression/house_price_full_pipeline.pkl",
        "description": "Estimates house prices using advanced gradient boosting with 80+ features.",
        "features_count": None,
    },
    "xgboost": {
        "name": "Podcast Engagement Predictor",
        "algorithm": "XGBoost Regressor",
        "dataset": "Podcast Episodes Dataset",
        "target": "Expected Listening Time (minutes)",
        "pkl_path": "Supervised Learning/XGboostRegression/podcast_listening_full_pipeline.pkl",
        "description": "Predicts podcast episode listening time based on episode characteristics.",
        "features_count": None,
    },
    "bayesian": {
        "name": "Building Energy Efficiency Predictor",
        "algorithm": "Bayesian Ridge Regression",
        "dataset": "ENB2012 Energy Efficiency Dataset",
        "target": "Heating Load (kWh)",
        "pkl_path": "Supervised Learning/BayesianRegression/energy_efficiency_full_pipeline.pkl",
        "description": "Predicts building heating load for energy efficiency optimization.",
        "features_count": None,
    },
    "elasticnet": {
        "name": "Bike Rental Demand Forecaster",
        "algorithm": "ElasticNet Regression",
        "dataset": "UCI Bike Sharing (Hourly)",
        "target": "Total Bike Rentals (count)",
        "pkl_path": "Supervised Learning/ElasticNetRegression/bike_sharing_full_pipeline.pkl",
        "description": "Forecasts hourly bike rental demand using ElasticNet regularization.",
        "features_count": None,
    },
    "lasso": {
        "name": "Video Game Global Sales Predictor",
        "algorithm": "Lasso Regression",
        "dataset": "VGSales Dataset",
        "target": "Global Sales (million units)",
        "pkl_path": "Supervised Learning/LassoRegression/video_game_sales_full_pipeline.pkl",
        "description": "Predicts video game global sales with feature selection via Lasso.",
        "features_count": None,
    },
    "linear": {
        "name": "Simple House Price Calculator",
        "algorithm": "Linear Regression",
        "dataset": "House Prices (Simple)",
        "target": "House Price ($)",
        "pkl_path": "Supervised Learning/LinearRegression/house_price_simple_full_pipeline.pkl",
        "description": "Basic linear regression model for simple house price estimation.",
        "features_count": None,
    },
    "polynomial": {
        "name": "Manufacturing Quality Predictor",
        "algorithm": "Polynomial Regression",
        "dataset": "Manufacturing Process Dataset",
        "target": "Quality Rating Score",
        "pkl_path": "Supervised Learning/PolynomialRegression/manufacturing_full_pipeline.pkl",
        "description": "Predicts manufacturing quality using polynomial regression.",
        "features_count": None,
    },
    "randomforest": {
        "name": "Used Car Price Estimator",
        "algorithm": "Random Forest Regressor",
        "dataset": "CarDekho Used Car Prices",
        "target": "Selling Price (₹ Lakhs)",
        "pkl_path": "Supervised Learning/RandomForestRegression/car_price_full_pipeline.pkl",
        "description": "Estimates used car prices using ensemble Random Forest method.",
        "features_count": None,
    },
    "ridge": {
        "name": "Cryptocurrency Price Predictor",
        "algorithm": "Ridge Regression",
        "dataset": "Crypto-to-USD Daily Prices",
        "target": "Closing Price (USD)",
        "pkl_path": "Supervised Learning/RidgeRegression/crypto_full_pipeline.pkl",
        "description": "Predicts cryptocurrency closing prices using Ridge regularization.",
        "features_count": None,
    },
    "svr": {
        "name": "Stock Price Forecaster",
        "algorithm": "Support Vector Regression",
        "dataset": "Google (GOOGL) Stock 2020–2025",
        "target": "Adjusted Closing Price (USD)",
        "pkl_path": "Supervised Learning/SupportVectorRegression/stock_price_full_pipeline.pkl",
        "description": "Forecasts stock prices using Support Vector Regression.",
        "features_count": None,
    },
}


def get_model_config(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific model.
    
    Args:
        model_id: Model identifier
    
    Returns:
        Model configuration dictionary or None if not found
    """
    return ALGORITHM_REGISTRY.get(model_id)


def get_all_models() -> Dict[str, Dict[str, Any]]:
    """
    Get all model configurations.
    
    Returns:
        Dictionary of all model configurations
    """
    return ALGORITHM_REGISTRY.copy()


def list_model_ids() -> list:
    """
    Get list of all model IDs.
    
    Returns:
        List of model identifiers
    """
    return list(ALGORITHM_REGISTRY.keys())
