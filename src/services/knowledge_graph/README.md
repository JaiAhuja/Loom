# Knowledge graph service

The knowledge-graph service is Loom's Neo4j boundary. It owns connection
lifecycle, schema constraints, parameterised queries, graph writes, and the
intent-based query service used by the agent tool.

## Public API

- `get_neo4j_connection()` / `Neo4jConnection` — create and use the shared
  Neo4j client.
- `KnowledgeGraphWriter` — write paper profiles and cross-paper concept or
  finding relationships.
- `KnowledgeGraphQueries` — parameterised queries for papers, concepts,
  findings, statistics, visualization, and deletion.
- `GraphQueryService` — maps approved named intents to query methods.

## Safety and data model

Graph access must remain parameterised. The agent receives named intents through
`tools.safe_graph_tool`; it must never submit raw Cypher. Concepts are scoped
by domain, and schema initialization is handled by the service. Neo4j is an
optional local dependency, so callers should handle connection failures
gracefully.

Integration coverage is in `tests/integration/test_knowledge_graph_connection.py`
and `tests/integration/test_retrieval_and_graph.py`; query and tool behaviour is
also covered by `tests/unit/test_agent_and_tools.py`.
