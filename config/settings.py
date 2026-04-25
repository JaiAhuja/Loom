import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3.5:latest"
    OLLAMA_EMBEDDING_MODEL: str = "qwen3-embedding:4b"
    OLLAMA_TEMPERATURE: float = 0.1

    # LangSmith
    LANGCHAIN_PROJECT: str = "Loom"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: bool = True

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # Output
    OUTPUT_DIR: str = "./outputs"

    # Web Search
    WEB_SEARCH_MAX_RESULTS: int = 5

    # Neo4j
    # NEO4J_DATABASE defaults to None so the driver uses the server's default
    # database (``neo4j`` for Community edition).  Set it explicitly in .env
    # only if you are using Enterprise edition with a named database.
    NEO4J_URI: str = "neo4j://[local_instance_id]:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "Loom-Weave-Threads"
    NEO4J_DATABASE: Optional[str] = None

    # RAG
    RAG_TOP_K: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


# Module-level singleton
settings = Settings()


def configure_langsmith() -> bool:
    """Configure LangSmith tracing if credentials are provided.

    Returns True if tracing was enabled, False otherwise.
    """
    if settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        return True
    return False
