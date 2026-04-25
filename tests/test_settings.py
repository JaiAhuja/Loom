"""Smoke tests for config/settings.py."""

import os

from config.settings import Settings, configure_langsmith


# ---- defaults ---------------------------------------------------------------

def test_settings_loads_with_defaults():
    """Settings can be instantiated and exposes expected default values."""
    s = Settings()
    assert s.OLLAMA_BASE_URL == "http://localhost:11434"
    assert s.OLLAMA_MODEL == "gemma4:e4b"
    assert s.OLLAMA_EMBEDDING_MODEL == "qwen3-embedding:4b"
    assert s.OLLAMA_TEMPERATURE == 0.1
    assert s.CHROMA_PERSIST_DIR == "./data/chroma_db"
    assert s.OUTPUT_DIR == "./outputs"
    assert s.NEO4J_URI == "neo4j://127.0.0.1:7687"
    assert s.NEO4J_DATABASE is None
    assert s.RAG_TOP_K == 5


def test_langsmith_defaults_are_off():
    """LangSmith tracing is disabled by default."""
    s = Settings()
    assert s.LANGSMITH_API_KEY == ""
    assert s.LANGSMITH_TRACING is False
    # LangSmith project is configured via LANGCHAIN_PROJECT, not LANGSMITH_PROJECT.
    assert s.LANGCHAIN_PROJECT == "Loom"


# ---- configure_langsmith ----------------------------------------------------

def test_configure_langsmith_noop_when_disabled(monkeypatch):
    """configure_langsmith returns False when tracing is off (the default)."""
    monkeypatch.setattr("config.settings.settings.LANGSMITH_TRACING", False)
    monkeypatch.setattr("config.settings.settings.LANGSMITH_API_KEY", "")
    assert configure_langsmith() is False


def test_configure_langsmith_noop_without_key(monkeypatch):
    """configure_langsmith returns False when API key is empty even if tracing is on."""
    monkeypatch.setattr("config.settings.settings.LANGSMITH_TRACING", True)
    monkeypatch.setattr("config.settings.settings.LANGSMITH_API_KEY", "")
    assert configure_langsmith() is False


def test_configure_langsmith_sets_env_vars(monkeypatch):
    """configure_langsmith sets LANGCHAIN env vars and returns True when fully configured."""
    monkeypatch.setattr("config.settings.settings.LANGSMITH_TRACING", True)
    monkeypatch.setattr("config.settings.settings.LANGSMITH_API_KEY", "lsv2_test_key")
    monkeypatch.setattr("config.settings.settings.LANGCHAIN_PROJECT", "test-project")

    result = configure_langsmith()

    assert result is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "lsv2_test_key"
    assert os.environ["LANGCHAIN_PROJECT"] == "test-project"
