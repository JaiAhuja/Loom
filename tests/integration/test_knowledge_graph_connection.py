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


def test_factory_creates_independent_owned_connections():
    first = connection_module.create_neo4j_connection(
        uri="bolt://example.invalid:7687",
        username="neo4j",
        password="password",
    )
    second = connection_module.create_neo4j_connection(
        uri="bolt://example.invalid:7687",
        username="neo4j",
        password="password",
    )
    assert first is not second
