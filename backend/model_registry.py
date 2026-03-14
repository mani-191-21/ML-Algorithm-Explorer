"""
model_registry.py — Central config for all 11 algorithms.
pkl_path must point to relative path under Supervised Learning folder.
"""

ALGORITHM_REGISTRY = {
    "adaboost": {
        "name": "Insurance Charges Predictor",
        "algorithm": "AdaBoost Regressor",
        "dataset": "Medical Insurance Costs",
        "target": "Medical Insurance Charges ($)",
        "pkl_path": "AdaBoostRegression/insurance_charges_full_pipeline.pkl",
    },
    "catboost": {
        "name": "Advanced House Price Estimator",
        "algorithm": "CatBoost (Gradient Boosting)",
        "dataset": "Ames Housing Dataset (80+ features)",
        "target": "House Sale Price ($)",
        "pkl_path": "CatGBMRegression/house_price_full_pipeline.pkl",
    },
    "xgboost": {
        "name": "Podcast Engagement Predictor",
        "algorithm": "XGBoost Regressor",
        "dataset": "Podcast Episodes Dataset",
        "target": "Expected Listening Time (minutes)",
        "pkl_path": "XGboostRegression/podcast_listening_full_pipeline.pkl",
    },
    "bayesian": {
        "name": "Building Energy Efficiency Predictor",
        "algorithm": "Bayesian Ridge Regression",
        "dataset": "ENB2012 Energy Efficiency Dataset",
        "target": "Heating Load (kWh)",
        "pkl_path": "BayesianRegression/energy_efficiency_full_pipeline.pkl",
    },
    "elasticnet": {
        "name": "Bike Rental Demand Forecaster",
        "algorithm": "ElasticNet Regression",
        "dataset": "UCI Bike Sharing (Hourly)",
        "target": "Total Bike Rentals (count)",
        "pkl_path": "ElasticNetRegression/bike_sharing_full_pipeline.pkl",
    },
    "lasso": {
        "name": "Video Game Global Sales Predictor",
        "algorithm": "Lasso Regression",
        "dataset": "VGSales Dataset",
        "target": "Global Sales (million units)",
        "pkl_path": "LassoRegression/video_game_sales_full_pipeline.pkl",
    },
    "linear": {
        "name": "Simple House Price Calculator",
        "algorithm": "Linear Regression",
        "dataset": "House Prices (Simple)",
        "target": "House Price ($)",
        "pkl_path": "LinearRegression/house_price_simple_full_pipeline.pkl",
    },
    "polynomial": {
        "name": "Manufacturing Quality Predictor",
        "algorithm": "Polynomial Regression",
        "dataset": "Manufacturing Process Dataset",
        "target": "Quality Rating Score",
        "pkl_path": "PolynomialRegression/manufacturing_full_pipeline.pkl",
    },
    "randomforest": {
        "name": "Used Car Price Estimator",
        "algorithm": "Random Forest Regressor",
        "dataset": "CarDekho Used Car Prices",
        "target": "Selling Price (₹ Lakhs)",
        "pkl_path": "RandomForestRegression/car_price_full_pipeline.pkl",
    },
    "ridge": {
        "name": "Cryptocurrency Price Predictor",
        "algorithm": "Ridge Regression",
        "dataset": "Crypto-to-USD Daily Prices",
        "target": "Closing Price (USD)",
        "pkl_path": "RidgeRegression/crypto_full_pipeline.pkl",
    },
    "svr": {
        "name": "Stock Price Forecaster",
        "algorithm": "Support Vector Regression",
        "dataset": "Google (GOOGL) Stock 2020–2025",
        "target": "Adjusted Closing Price (USD)",
        "pkl_path": "SupportVectorRegression/stock_price_full_pipeline.pkl",
    },
}
