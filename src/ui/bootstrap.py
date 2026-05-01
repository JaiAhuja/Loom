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
    """Return (connected, [model_names]) for the configured Ollama host."""
    try:
        base_url = settings.OLLAMA_BASE_URL
        if not base_url.startswith(("http://", "https://")):
            return False, []
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
            return True, [m["name"] for m in data.get("models", [])]
    except Exception:
        return False, []


@st.cache_data(ttl=15)
def check_neo4j_status() -> tuple[bool, str]:
    """Return (connected, error_message) for the configured Neo4j instance."""
    try:
        get_neo4j_connection().driver.verify_connectivity()
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor()
