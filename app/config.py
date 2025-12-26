"""
Application configuration using environment variables.
"""
import os
import secrets
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "NubHQ API"
    debug: bool = False
    environment: str = "development"

    # Security
    secret_key: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60  # 1 hour
    refresh_token_expire_days: int = 7

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./nubhq.db")

    # CORS
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "https://web-pi-livid.vercel.app",
    ]

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    login_rate_limit: str = "5/minute"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Validate secret key on startup
settings = get_settings()
if settings.environment == "production" and settings.secret_key == secrets.token_urlsafe(32):
    raise ValueError(
        "SECRET_KEY must be set in production! "
        "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
    )
