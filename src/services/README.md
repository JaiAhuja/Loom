# Loom Services

The packages in this directory are Loom's in-process service boundaries. Loom
is a modular monolith, not a set of independently deployed microservices: the
packages share the same Python process, configuration, and local data stores.

## Service catalog

| Package | Responsibility | Main public entry points |
|---|---|---|
| [`agent`](agent/README.md) | Builds and runs the LangGraph conversation workflow | `GraphBuilder`, `AgentState` |
| [`chat`](chat/README.md) | Persists, searches, renders, and deletes saved conversations | `ChatStore`, `ChatRecord`, `ChatMetadata` |
| [`common`](common/README.md) | Shared low-level helpers used across services | `parse_llm_json` |
| [`evaluation`](evaluation/README.md) | Evaluates RAG answers with an LLM-as-judge | `RAGJudge`, `EvaluationResult` |
| [`ingestion`](ingestion/README.md) | Coordinates upload identity, document processing, indexing, and graph writes | `IngestionService`, `build_identity` |
| [`knowledge_graph`](knowledge_graph/README.md) | Owns Neo4j connections, schema, queries, and graph writes | `KnowledgeGraphQueries`, `KnowledgeGraphWriter`, `GraphQueryService` |
| [`llm`](llm/README.md) | Provides Ollama chat/embedding clients and paper profiling | `get_llm`, `get_embeddings`, `extract_paper_profile` |
| [`retrieval`](retrieval/README.md) | Converts documents to chunks and stores/searches them in ChromaDB | `DocumentProcessor`, `VectorStoreManager` |
| [`tools`](tools/README.md) | Exposes safe RAG and knowledge-graph capabilities to the agent | `create_rag_tool`, `create_safe_graph_tool` |
| [`ui`](ui/README.md) | Provides Streamlit bootstrap, status, styling, and confirmation helpers | `get_vector_store`, `get_document_processor`, rendering helpers |

## Runtime flow

```text
Streamlit pages
    │
    ├──► agent ──► tools ──► retrieval ──► ChromaDB
    │       │          └────► knowledge_graph ──► Neo4j
    │       └───────────────► llm ──► Ollama
    │
    ├──► ingestion ──► retrieval + llm + knowledge_graph
    ├──► chat ──► data/chat_history/
    ├──► evaluation ──► llm
    └──► ui ──► cached clients and shared Streamlit components
```

## Boundary rules

- Import configuration from `config.settings.settings`; service code should not
  read environment variables directly.
- Reuse cached clients from `src.services.ui.bootstrap` in Streamlit pages.
- Keep graph access parameterised. Agent and UI code must use named graph
  intents or query methods, never accept raw Cypher from users.
- Keep document identity deterministic so re-ingestion remains idempotent.
- Keep optional Neo4j failures graceful; chat and RAG should remain usable when
  the graph database is unavailable.
- Add or update tests in the corresponding `tests/unit`, `tests/integration`,
  or `tests/e2e` area when a service's behaviour changes.

## Where to start

- To change agent behaviour, begin with [`agent/`](agent/README.md) and
  [`tools/`](tools/README.md).
- To change PDF processing or search, read [`ingestion/`](ingestion/README.md)
  and [`retrieval/`](retrieval/README.md).
- To change graph storage or graph exploration, read
  [`knowledge_graph/`](knowledge_graph/README.md).
- To change model or embedding behaviour, read [`llm/`](llm/README.md).
- To change Streamlit wiring, read [`ui/`](ui/README.md) and the relevant page
  under `pages/`.
