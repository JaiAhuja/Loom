"""Shared service-connection checks and cached component factories."""

from __future__ import annotations

import json
import urllib.request

import streamlit as st

from config.settings import settings
from src.graph_db import get_neo4j_connection
from src.rag import DocumentProcessor, VectorStoreManager


@st.cache_data(ttl=30)
def check_ollama_status() -> tuple[bool, list[str]]:
    """Return (connected, [model_name...]) for the configured Ollama host."""
    try:
        # Only follow http/https — reject file://, ftp://, etc. so a tampered
        # OLLAMA_BASE_URL cannot be abused to read local files (Bandit B310).
        base_url = settings.OLLAMA_BASE_URL
        if not base_url.startswith(("http://", "https://")):
            return False, []
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 — scheme validated above
            data = json.loads(resp.read().decode())
            return True, [m["name"] for m in data.get("models", [])]
    except Exception:
        return False, []


def check_neo4j_status() -> bool:
    try:
        return get_neo4j_connection().is_connected()
    except Exception:
        return False


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor()
