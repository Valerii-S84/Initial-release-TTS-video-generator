from __future__ import annotations

from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Video Generator API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    APP_ENV: str = "dev"  # dev|staging|prod

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # Database
    DATABASE_URL: str = "sqlite:///backend/users.db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Auth
    JWT_SECRET_KEY: str = "change_me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # External APIs
    ELEVENLABS_API_KEY: str = ""

    # Storage
    STORAGE_INPUT: str = "backend/input_videos"
    STORAGE_OUTPUT: str = "backend/output_videos"
    STORAGE_TMP: str = "backend/tmp"
    MAX_UPLOAD_SIZE_MB: int = 500
    MAX_CHUNK_SIZE_MB: int = 8
    ENFORCE_UPLOAD_TOKEN: bool = False
    # Media
    MUSIC_DIR: str = "backend/music"
    MUSIC_PREVIEWS: str = "backend/music_previews"

    # Rate Limiting
    RATE_LIMIT_UPLOAD: str = "20/hour"
    RATE_LIMIT_GENERATE: str = "5/minute"

    # Monitoring
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENABLED: bool = True

    # Security
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
