"""Define environment-backed configuration for the Loom application."""

import os
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "granite4:tiny-h"
    OLLAMA_EMBEDDING_MODEL: str = "qwen3-embedding:4b"
    OLLAMA_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=1.0)
    OLLAMA_NUM_CTX: int = Field(default=32768, ge=256)

    LANGCHAIN_PROJECT: str = "Loom"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: bool = False

    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    OUTPUT_DIR: str = "./outputs"

    NEO4J_URI: str = "bolt://127.0.0.1:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: Optional[str] = None
    NEO4J_DATABASE: Optional[str] = None

    RAG_TOP_K: int = Field(default=5, ge=1)
    RAG_FETCH_K: int = Field(default=10, ge=1)
    RAG_MMR_LAMBDA: float = Field(default=0.4, ge=0.0, le=1.0)
    RAG_MAX_QUERY_LENGTH: int = Field(default=4000, ge=32, le=16000)
    AGENT_MAX_TOOL_ITERATIONS: int = Field(default=4, ge=1, le=20)

    @field_validator("OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_EMBEDDING_MODEL")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _validate_retrieval_settings(self):
        if self.RAG_FETCH_K < self.RAG_TOP_K:
            raise ValueError("RAG_FETCH_K must be greater than or equal to RAG_TOP_K")
        return self

    def validate_runtime(
        self,
        *,
        require_ollama: bool = False,
        require_embeddings: bool = False,
        require_neo4j: bool = False,
    ) -> None:
        """Validate only the services required by the selected runtime path.

        Loom has one local deployment mode, so this deliberately does not use
        environment profiles. Optional services are validated when a feature
        opts into them, keeping offline startup possible while making enabled
        features fail with an actionable error.
        """
        errors: list[str] = []
        if require_ollama:
            if not self.OLLAMA_BASE_URL.strip():
                errors.append("OLLAMA_BASE_URL is required")
            if not self.OLLAMA_MODEL.strip():
                errors.append("OLLAMA_MODEL is required")
        if require_embeddings and not self.OLLAMA_EMBEDDING_MODEL.strip():
            errors.append("OLLAMA_EMBEDDING_MODEL is required when RAG is enabled")
        if require_neo4j and not self.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD is required when the knowledge graph is enabled")
        if errors:
            raise ValueError("Invalid Loom configuration: " + "; ".join(errors))


settings = Settings()


def configure_langsmith() -> bool:
    """Enable LangSmith tracing if credentials are configured."""
    if settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        return True
    return False
