"""
FraudMap API - FastAPI Application

Main entry point for the REST API backend.
Run with: uvicorn src.api.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logger import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    setup_logging(
        log_level="DEBUG" if settings.app_debug else "INFO",
        json_format=settings.app_env == "production",
    )
    logger.info(
        "api_starting",
        env=settings.app_env,
        debug=settings.app_debug,
    )
    yield
    logger.info("api_shutdown")


app = FastAPI(
    title="FraudMap API",
    description=(
        "GNN-Based Financial Fraud & Anti-Money Laundering Detection System. "
        "GraphSentinel Team — TEKNOFEST 2026"
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Health Check =====
@app.get("/health", tags=["system"])
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "fraudmap-api",
        "version": "0.1.0",
    }


@app.get("/api/v1/stats/summary", tags=["dashboard"])
async def get_summary_stats():
    """Get dashboard summary statistics (placeholder)."""
    # TODO: Connect to real data sources
    return {
        "total_transactions": 0,
        "total_alerts": 0,
        "pending_alerts": 0,
        "fraud_detection_rate": 0.0,
        "false_positive_rate": 0.0,
        "model_version": "0.1.0",
    }
