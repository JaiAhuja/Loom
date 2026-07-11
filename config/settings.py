import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "granite4:tiny-h"
    OLLAMA_EMBEDDING_MODEL: str = "qwen3-embedding:4b"
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_NUM_CTX: int = 32768  # context-window tokens; reduce for models with smaller limits

    # LangSmith
    LANGCHAIN_PROJECT: str = "Loom"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: bool = False

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # Output
    OUTPUT_DIR: str = "./outputs"

    # Neo4j
    # Use bolt:// for standalone instances; neo4j:// is for Causal Clusters.
    NEO4J_URI: str = "bolt://127.0.0.1:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "Loom-Weave-Threads"
    NEO4J_DATABASE: Optional[str] = "loom"

    # RAG
    RAG_TOP_K: int = 5
    # MMR retrieval: number of candidate docs to fetch before re-ranking
    RAG_FETCH_K: int = 10
    # MMR lambda: 1.0 = pure relevance, 0.0 = pure diversity
    RAG_MMR_LAMBDA: float = 0.4

# Module-level singleton
settings = Settings()


def configure_langsmith() -> bool:
    """Enable LangSmith tracing if credentials are configured."""
    if settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        return True
    return False
