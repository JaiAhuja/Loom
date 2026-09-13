# Retrieval service

The retrieval service converts uploaded documents into searchable chunks and
stores them in ChromaDB. It also provides scoped lookup and metadata operations
for the chat and knowledge-graph pages.

## Public API

- `DocumentProcessor` — converts PDFs through Docling, chunks the resulting
  Markdown, and creates deterministic chunk IDs.
- `VectorStoreManager` — manages ChromaDB collections, adds documents, creates
  retrievers, lists papers, and removes collections or papers. It requires
  canonical document/chunk IDs and exposes explicit legacy metadata migration.

Embeddings come from the `llm` service. The configured local Granite tokenizer
is used by the chunking path; it must remain available at runtime. Streamlit
owns reuse of app-level services through `ui.bootstrap`; the retrieval and
provider modules do not keep process-global resource caches. Tests are in
`tests/integration/test_retrieval_and_graph.py` and
`tests/e2e/test_ingestion_to_chat.py`.
