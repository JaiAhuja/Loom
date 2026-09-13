# Ingestion service

The ingestion service coordinates the end-to-end document pipeline. It turns
uploaded PDFs into stable document identities, searchable chunks, optional
paper profiles, and optional knowledge-graph records.

## Pipeline

```text
uploaded PDF
    ├──► DocumentIdentity ──► content-addressed local storage
    ├──► DocumentProcessor ──► chunks ──► VectorStoreManager / ChromaDB
    ├──► paper profile ──► KnowledgeGraphWriter / Neo4j
    └──► deterministic status ──► FileResult and IngestionResult
```

## Public API

- `IngestionService.ingest_files(...)` — processes one or more uploads and
  reports successes, skips, and failures.
- `build_identity(...)` and `save_upload(...)` — create stable filename-based
  document IDs and MD5-based storage paths.
- `IngestionResult` and `FileResult` — structured pipeline outcomes.

Ingestion is designed to be idempotent: the document ID comes from the
sanitised filename and chunk IDs are deterministic. It can use retrieval, LLM,
and graph services, while graph/profile work remains optional. Tests are in
`tests/unit/test_identity.py`, `tests/integration/test_ingestion_service.py`,
and `tests/e2e/test_ingestion_to_chat.py`.
