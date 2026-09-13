"""Define environment-backed configuration for the Loom application."""

import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "granite4:tiny-h"
    OLLAMA_EMBEDDING_MODEL: str = "qwen3-embedding:4b"
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_NUM_CTX: int = 32768

    LANGCHAIN_PROJECT: str = "Loom"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: bool = False

    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    OUTPUT_DIR: str = "./outputs"

    NEO4J_URI: str = "bolt://127.0.0.1:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "Loom-Weave-Threads"
    NEO4J_DATABASE: Optional[str] = None

    RAG_TOP_K: int = 5
    RAG_FETCH_K: int = 10
    RAG_MMR_LAMBDA: float = 0.4

settings = Settings()


def configure_langsmith() -> bool:
    """Enable LangSmith tracing if credentials are configured."""
    if settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        return True
    return False
