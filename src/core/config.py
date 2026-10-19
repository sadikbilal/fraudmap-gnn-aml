"""
FraudMap Configuration Module

Centralized configuration using Pydantic Settings.
All environment variables are loaded from .env file.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- General ---
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")

    # --- Model ---
    model_checkpoint_dir: Path = Field(
        default=ROOT_DIR / "src" / "model" / "checkpoints",
        alias="MODEL_CHECKPOINT_DIR",
    )
    model_fraud_threshold: float = Field(default=0.85, alias="MODEL_FRAUD_THRESHOLD")
    model_device: str = Field(default="cuda", alias="MODEL_DEVICE")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # --- Paths ---
    data_raw_dir: Path = ROOT_DIR / "data" / "raw"
    data_processed_dir: Path = ROOT_DIR / "data" / "processed"

    model_config = {"env_file": str(ROOT_DIR / ".env"), "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Singleton instance
settings = get_settings()
