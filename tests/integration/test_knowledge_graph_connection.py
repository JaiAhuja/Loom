"""Tests for Neo4j connection lifecycle helpers."""

from concurrent.futures import ThreadPoolExecutor
import time

import pytest

pytest.importorskip("neo4j")

from src.services.knowledge_graph import connection as connection_module

pytestmark = pytest.mark.integration


class _DummyDriver:
    def close(self):
        pass


def test_driver_lazy_init_is_thread_safe(monkeypatch):
    calls = 0

    def fake_driver(*args, **kwargs):
        nonlocal calls
        time.sleep(0.01)
        calls += 1
        return _DummyDriver()

    monkeypatch.setattr(connection_module.GraphDatabase, "driver", fake_driver)

    conn = connection_module.Neo4jConnection(
        uri="bolt://example.invalid:7687",
        username="neo4j",
        password="password",
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        drivers = list(pool.map(lambda _: conn.driver, range(20)))

    assert calls == 1
    assert len({id(driver) for driver in drivers}) == 1


def test_singleton_init_is_thread_safe(monkeypatch):
    calls = 0

    class DummyConnection:
        def __init__(self):
            nonlocal calls
            time.sleep(0.01)
            calls += 1

        def close(self):
            pass

    connection_module.reset_neo4j_connection()
    monkeypatch.setattr(connection_module, "Neo4jConnection", DummyConnection)

    with ThreadPoolExecutor(max_workers=8) as pool:
        conns = list(
            pool.map(lambda _: connection_module.get_neo4j_connection(), range(20))
        )

    assert calls == 1
    assert len({id(conn) for conn in conns}) == 1
    connection_module.reset_neo4j_connection()
