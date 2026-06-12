"""
Vercel serverless API handler for ML-Algorithm-Explorer.
This file serves as the entry point for Vercel deployments.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.main import app

# Export app for Vercel
__all__ = ['app']
