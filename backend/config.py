"""
Configuration management for the ML-Algorithm-Explorer backend.
Handles environment variables, paths, and application settings.
"""

import os
from pathlib import Path
from typing import Optional

# ── Base Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
BACKEND_DIR = Path(__file__).parent
ML_MODELS_DIR = BASE_DIR / "ml_models"
SUPERVISED_LEARNING_DIR = ML_MODELS_DIR / "Supervised Learning"

# ── Environment Configuration ───────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# ── API Configuration ───────────────────────────────────────────────────
API_TITLE = "MLVerse — Supervised Learning API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Unified API for 11 supervised learning algorithms with pre-trained models"
API_DOCS_URL = "/docs"
API_REDOC_URL = "/redoc"

# ── Server Configuration ────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
RELOAD = DEBUG

# ── CORS Configuration ──────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# ── Model Configuration ─────────────────────────────────────────────────
MODEL_LOAD_TIMEOUT = int(os.getenv("MODEL_LOAD_TIMEOUT", 30))
MAX_PREDICTION_FEATURES = 100

# ── Logging Configuration ───────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ── Feature Flags ───────────────────────────────────────────────────────
ENABLE_MODEL_COMPARISON = os.getenv("ENABLE_MODEL_COMPARISON", "true").lower() == "true"
ENABLE_CHART_DATA = os.getenv("ENABLE_CHART_DATA", "true").lower() == "true"


def validate_config() -> bool:
    """
    Validate that all required directories and configurations exist.
    Returns True if valid, raises exception otherwise.
    """
    if not SUPERVISED_LEARNING_DIR.exists():
        raise ValueError(f"ML models directory not found: {SUPERVISED_LEARNING_DIR}")
    return True


# Validate on import
validate_config()
