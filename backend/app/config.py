import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "TalentMatch AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database paths
    DATABASE_URL: str = "sqlite:///./data/talentmatch.db"
    DATA_DIR: str = "./data"
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # JWT Authentication
    SECRET_KEY: str = "talentmatch-super-secret-production-key-change-in-prod-2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # LLM Settings (Grok or Hugging Face)
    GROK_API_KEY: Optional[str] = None
    GROK_BASE_URL: str = "https://api.x.ai/v1"
    GROK_MODEL: str = "grok-2-latest"
    
    SLM_MODEL_NAME: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    HF_TOKEN: Optional[str] = None
    LLM_PROVIDER: str = "auto"
    LLM_TIMEOUT_SECONDS: float = 30.0

    # Vector search & embeddings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384
    QDRANT_COLLECTION_NAME: str = "cv_chunks"

    # Logging
    LOG_FILE: str = "./data/talentmatch.log"

settings = Settings()
