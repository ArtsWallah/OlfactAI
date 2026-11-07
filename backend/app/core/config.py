"""Application configuration settings for OlfactAI."""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings validated via Pydantic."""

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project metadata
    PROJECT_NAME: str = "OlfactAI"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = (
        "AI-powered molecular smell intelligence platform. "
        "Predicts smell profiles, analyzes molecular properties, and simulates an e-nose on CPU."
    )
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, v: Union[bool, str]) -> bool:
        """Parse debug boolean safely even if host environment sets DEBUG=release."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            lower = v.strip().lower()
            if lower in ("true", "1", "yes", "y", "debug", "development", "dev"):
                return True
            if lower in ("false", "0", "no", "n", "release", "production", "prod"):
                return False
        return True

    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    API_V1_STR: str = "/api/v1"

    # CORS configuration
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated string or preserve list of origins."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return [str(v)]

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "text"

    # Chemistry engine parameters
    DEFAULT_MORGAN_RADIUS: int = 2
    DEFAULT_MORGAN_NBITS: int = 2048
    MAX_SMILES_LENGTH: int = 1000

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "models"
    DATASETS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "datasets"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
