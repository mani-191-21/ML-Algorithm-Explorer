"""
Main FastAPI application for ML-Algorithm-Explorer.
Entry point for the backend server.
"""

import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    API_DOCS_URL,
    API_REDOC_URL,
    CORS_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS,
    HOST,
    PORT,
    RELOAD,
)
from backend.api import router
from backend.utils.logger import get_logger

# Suppress warnings
warnings.filterwarnings("ignore")

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url=API_DOCS_URL,
    redoc_url=API_REDOC_URL,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# Include API routes
app.include_router(router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": API_TITLE,
        "version": API_VERSION,
        "docs": API_DOCS_URL,
        "redoc": API_REDOC_URL,
    }


@app.on_event("startup")
async def startup():
    """Startup event handler."""
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"CORS origins: {CORS_ORIGINS}")


@app.on_event("shutdown")
async def shutdown():
    """Shutdown event handler."""
    logger.info("Shutting down application")


if __name__ == "__main__":
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
    )
