"""Integration tests for UI service helpers and Neo4j transaction adapters."""

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("neo4j")

from src.services.knowledge_graph.connection import Neo4jConnection
from src.services.ui.chrome import format_chat_error, render_status_bar

pytestmark = pytest.mark.integration


def test_chat_error_formatter_classifies_connection_and_model_failures():
    assert "Connection Error" in format_chat_error(RuntimeError("connection refused"), "model")
    assert "Model Error" in format_chat_error(RuntimeError("model not found"), "model")
    assert "**Error:**" in format_chat_error(RuntimeError("other"), "model")


def test_status_bar_escapes_user_supplied_labels(monkeypatch):
    rendered = []
    monkeypatch.setattr("src.services.ui.chrome.st.markdown", lambda text, **kwargs: rendered.append(text))
    render_status_bar("model<1>", collection_name="notes&more", paper_filter="Paper <A>", use_graph=True)
    assert "model&lt;1&gt;" in rendered[0]
    assert "notes&amp;more" in rendered[0]
    assert "Paper &lt;A&gt;" in rendered[0]


class SyncResult:
    def data(self):
        return {"value": 1}


class SyncTx:
    def run(self, query, params):
        return [SyncResult()]


class SyncSession:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute_read(self, callback):
        return callback(SyncTx())

    def execute_write(self, callback):
        return callback(SyncTx())


class SyncDriver:
    def session(self, database=None):
        return SyncSession()

    def verify_connectivity(self):
        return None

    def close(self):
        return None


def test_neo4j_connection_adapts_driver_transactions(monkeypatch):
    connection = Neo4jConnection(uri="bolt://test", username="user", password="pass")
    monkeypatch.setattr(connection, "_driver", SyncDriver())
    assert connection.execute_read("RETURN 1") == [{"value": 1}]
    assert connection.execute_write("CREATE (n)") == [{"value": 1}]
    assert connection.is_connected() is True
