"""Intent-based graph query service — safe, typed access to the knowledge graph.

Maps named intents to pre-built parameterised queries.
No free-form Cypher is accepted.
"""

from __future__ import annotations

from typing import Any

from src.services.knowledge_graph.connection import Neo4jConnection
from src.services.knowledge_graph.queries import KnowledgeGraphQueries

_INTENTS: dict[str, tuple[tuple[str, ...], str, tuple[str, ...]]] = {
    "graph_stats": ((), "get_graph_stats", ()),
    "paper_list": ((), "get_all_papers", ()),
    "paper_details": (("title",), "get_paper_details", ()),
    "shared_concepts": (("paper_a", "paper_b"), "get_shared_concepts", ()),
    "concept_papers": (("concept_name",), "get_concept_papers", ()),
    "related_concepts": (("concept_name",), "get_related_concepts", ()),
    "contradiction_list": ((), "get_cross_paper_findings", ("CONTRADICTS",)),
    "support_list": ((), "get_cross_paper_findings", ("SUPPORTS",)),
    "extension_list": ((), "get_cross_paper_findings", ("EXTENDS",)),
}

ALL_INTENTS: frozenset[str] = frozenset(_INTENTS)
REQUIRED_PARAMS = {intent: spec[0] for intent, spec in _INTENTS.items()}


class GraphQueryService:
    """Intent-based access to the Neo4j knowledge graph."""

    def __init__(self, conn: Neo4jConnection) -> None:
        self._queries = KnowledgeGraphQueries(conn)

    @staticmethod
    def _resolve(intent: str, params: dict[str, Any] | None):
        if intent not in _INTENTS:
            raise ValueError(f"Unknown intent {intent!r}. Supported: {', '.join(sorted(ALL_INTENTS))}")
        params = dict(params) if params else {}
        required, method, fixed_args = _INTENTS[intent]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Intent {intent!r} requires {required}; missing: {missing}")
        return method, (*fixed_args, *(params[name] for name in required)), params

    def execute(self, intent: str, params: dict[str, Any] | None = None) -> dict:
        """Dispatch an intent to its synchronous query method."""
        method, args, params = self._resolve(intent, params)
        data = getattr(self._queries, method)(*args)
        return {"intent": intent, "params": params, "data": data}

    async def aexecute(self, intent: str, params: dict[str, Any] | None = None) -> dict:
        """Async dispatch intent to the matching query."""
        method, args, params = self._resolve(intent, params)
        data = await getattr(self._queries, f"a{method}")(*args)
        return {"intent": intent, "params": params, "data": data}
