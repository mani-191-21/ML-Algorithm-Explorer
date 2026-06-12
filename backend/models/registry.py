"""
Model registry — Central configuration for all algorithms.
Uses expanded registry with 40+ models across all categories.
"""

from .registry_expanded import EXPANDED_ALGORITHM_REGISTRY, get_models_by_category, get_all_categories

# Use expanded registry as default
ALGORITHM_REGISTRY = EXPANDED_ALGORITHM_REGISTRY


def get_model_config(model_id: str):
    """Get configuration for a specific model."""
    return ALGORITHM_REGISTRY.get(model_id)


def get_all_models():
    """Get all model configurations."""
    return ALGORITHM_REGISTRY.copy()


def list_model_ids():
    """Get list of all model IDs."""
    return list(ALGORITHM_REGISTRY.keys())


def get_models_by_category_name(category: str):
    """Get all models in a specific category."""
    return get_models_by_category(category)


def get_all_model_categories():
    """Get list of all model categories."""
    return get_all_categories()
