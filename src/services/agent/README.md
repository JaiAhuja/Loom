# Agent service

The agent service builds Loom's LangGraph conversation workflow. It combines an
Ollama chat model with the tools enabled by the current Streamlit session and
returns a compiled graph that can be invoked with conversation messages.

## Responsibilities

- Define the typed `AgentState` message history.
- Build the system prompt and optional RAG/knowledge-graph instructions.
- Assemble only the tools enabled by the caller.
- Route tool calls back through the agent until the response is complete.

## Public API

- `GraphBuilder.build(...)` — compiles an agent with optional RAG and graph
  tools, shared vector-store and Neo4j clients, and model overrides.
- `AgentState` — state schema used by the LangGraph workflow.
- `build_system_prompt(...)` and `should_continue(...)` — lower-level helpers
  with bounded tool iterations, individual-call limits, and repeated empty or
  failed-result protection.
  used when customizing or testing the graph.

## Dependencies and tests

The builder depends on `llm`, `tools`, and `knowledge_graph` connection types.
Feature flags must remain the source of truth for tool availability. Agent and
tool behaviour is covered primarily by `tests/unit/test_agent_and_tools.py`.
