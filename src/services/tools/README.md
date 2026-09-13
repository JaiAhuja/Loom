# Agent tools

This package adapts Loom capabilities to LangChain tools. It is the narrow
boundary between the agent and the retrieval or knowledge-graph services.

## Public API

- `create_rag_tool(...)` — creates the `query_documents` tool from an
  explicitly owned retrieval service, optionally scoped to a document ID.
- `create_safe_graph_tool(...)` — creates the `query_knowledge_graph` tool
  backed by approved graph intents and parameter validation.

Tool factories should format service results into useful model-facing text and
return actionable errors without leaking implementation details. New agent
capabilities should be added here, then registered by `agent.GraphBuilder` and
covered by unit tests in `tests/unit/test_agent_and_tools.py`.

Raw Cypher is not an accepted tool input. Graph requests must go through the
named-intent contract exposed by the knowledge-graph service.
