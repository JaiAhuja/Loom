"""Intent-based graph query service — safe, typed access to the knowledge graph.

Maps named intents to pre-built parameterised queries.
No free-form Cypher is accepted.
"""

from __future__ import annotations

from typing import Any, Callable

from src.graph_db.connection import Neo4jConnection
from src.graph_db.queries import KnowledgeGraphQueries

# ---- Dispatch table: intent → (required_params, callable(queries, params)) ----

_DISPATCH: dict[str, tuple[tuple[str, ...], Callable]] = {
    "graph_stats":        ((), lambda q, _: q.get_graph_stats()),
    "paper_list":         ((), lambda q, _: q.get_all_papers()),
    "paper_details":      (("title",),              lambda q, p: q.get_paper_details(p["title"])),
    "shared_concepts":    (("paper_a", "paper_b"),   lambda q, p: q.get_shared_concepts(p["paper_a"], p["paper_b"])),
    "concept_papers":     (("concept_name",),        lambda q, p: q.get_concept_papers(p["concept_name"])),
    "related_concepts":   (("concept_name",),        lambda q, p: q.get_related_concepts(p["concept_name"])),
    "contradiction_list": ((), lambda q, _: q.get_cross_paper_findings("CONTRADICTS")),
    "support_list":       ((), lambda q, _: q.get_cross_paper_findings("SUPPORTS")),
    "extension_list":     ((), lambda q, _: q.get_cross_paper_findings("EXTENDS")),
}

# Async dispatch table — mirrors _DISPATCH but calls async methods.
_ASYNC_DISPATCH: dict[str, tuple[tuple[str, ...], Callable]] = {
    "graph_stats":        ((), lambda q, _: q.aget_graph_stats()),
    "paper_list":         ((), lambda q, _: q.aget_all_papers()),
    "paper_details":      (("title",),              lambda q, p: q.aget_paper_details(p["title"])),
    "shared_concepts":    (("paper_a", "paper_b"),   lambda q, p: q.aget_shared_concepts(p["paper_a"], p["paper_b"])),
    "concept_papers":     (("concept_name",),        lambda q, p: q.aget_concept_papers(p["concept_name"])),
    "related_concepts":   (("concept_name",),        lambda q, p: q.aget_related_concepts(p["concept_name"])),
    "contradiction_list": ((), lambda q, _: q.aget_cross_paper_findings("CONTRADICTS")),
    "support_list":       ((), lambda q, _: q.aget_cross_paper_findings("SUPPORTS")),
    "extension_list":     ((), lambda q, _: q.aget_cross_paper_findings("EXTENDS")),
}

ALL_INTENTS: frozenset[str] = frozenset(_DISPATCH)

REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    intent: entry[0] for intent, entry in _DISPATCH.items()
}


class GraphQueryService:
    """Intent-based access to the Neo4j knowledge graph."""

    def __init__(self, conn: Neo4jConnection) -> None:
        self._queries = KnowledgeGraphQueries(conn)

    def execute(self, intent: str, params: dict[str, Any] | None = None) -> dict:
        """Dispatch intent to the matching query. Raises ValueError for unknown intents or missing params."""
        if intent not in _DISPATCH:
            raise ValueError(
                f"Unknown intent {intent!r}. "
                f"Supported: {', '.join(sorted(ALL_INTENTS))}"
            )

        required, call = _DISPATCH[intent]
        params = dict(params) if params else {}

        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Intent {intent!r} requires {required}; missing: {missing}")

        data = call(self._queries, params)
        return {"intent": intent, "params": params, "data": data}

    async def aexecute(self, intent: str, params: dict[str, Any] | None = None) -> dict:
        """Async dispatch intent to the matching query."""
        if intent not in _ASYNC_DISPATCH:
            raise ValueError(
                f"Unknown intent {intent!r}. "
                f"Supported: {', '.join(sorted(ALL_INTENTS))}"
            )

        required, call = _ASYNC_DISPATCH[intent]
        params = dict(params) if params else {}

        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Intent {intent!r} requires {required}; missing: {missing}")

        data = await call(self._queries, params)
        return {"intent": intent, "params": params, "data": data}
