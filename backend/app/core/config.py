import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Settings
    PROJECT_NAME: str = "DocuNexus"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True

    # Qdrant Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "docunexus_knowledge_base"
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None

    # Embedding Model Configuration
    EMBEDDING_PROVIDER: str = "gemini"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    EMBEDDING_DIM: int = 768

    # Generation LLM Configuration
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-3.5-flash"
    GEMINI_API_KEY: Optional[str] = None

    # Cohere Rerank Configuration
    COHERE_API_KEY: Optional[str] = None
    COHERE_RERANK_MODEL: str = "rerank-v3.5"
    COHERE_TOP_N: int = 5

    # Document Storage & Ingestion
    UPLOAD_DIR: str = "./storage/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    @property
    def upload_dir_path(self) -> Path:
        """Returns the resolved upload directory path and ensures it exists."""
        p = Path(self.UPLOAD_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

settings = Settings()
