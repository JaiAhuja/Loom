# Loom Engineering Skills

This document is the implementation guide for agents and contributors working
on Loom. Loom is a local research-assistant application with Streamlit UI,
Ollama models, Chroma retrieval, and an optional Neo4j knowledge graph.

## Operating model

- Loom has one local deployment mode. `APP_ENV` is not required and must not be
  introduced as a gate for startup.
- Optional dependencies are validated when the feature is used. A disabled
  feature must not require its service to be running.
- Unknown configuration keys are errors. Do not add permissive configuration
  handling that silently ignores misspellings.
- Keep configuration in `config/settings.py`; feature-specific validation
  belongs in `Settings.validate_runtime(...)` or the owning service boundary.

## Resource ownership

- The application or service owner must create and own external resources.
- Neo4j connections are created with `create_neo4j_connection(...)` and passed
  into graph services and tools. Do not add process-wide connection singletons.
- Ollama chat and embedding clients are created by the provider functions but
  are not cached at module scope. If reuse is needed, the caller owns that
  lifecycle, such as Streamlit's app-level resource cache.
- `VectorStoreManager` owns its clients and stores for its lifetime. Do not add
  module-level collection or embedding caches.
- Resource cleanup must be explicit and testable. Avoid hidden initialization
  triggered by importing a module.

## Document identity and retrieval

- `document_id` is the canonical identity shared by ingestion, Chroma, RAG,
  and Neo4j.
- Every normal Chroma row must have a valid `document_id` and deterministic
  `chunk_id`. Invalid canonical metadata is an integrity error.
- Rows without canonical identity are legacy migration data, not a normal
  runtime path. Migrate them through `migrate_legacy_metadata(...)` with an
  explicit mapping; never guess identity from an inconsistent field.
- RAG tools must validate collection names, query input, query length, and
  optional document scope before retrieving.
- Empty results, stale document IDs, and retrieval failures must remain
  distinguishable. Do not turn dependency failures into “no results.”

## Tool contracts

- Tools receive explicit service instances and must fail fast when their
  required dependency is absent or malformed.
- Tool outcomes use explicit markers:
  - `[TOOL_RESULT status=empty]` for a valid query with no matches.
  - `[TOOL_RESULT status=stale_document]` for a missing scoped document.
  - `[TOOL_ERROR kind=invalid_input]` for malformed caller/model input.
  - `[TOOL_ERROR kind=dependency_unavailable]` for service connectivity issues.
  - `[TOOL_ERROR kind=execution_failed]` for other runtime failures.
- The graph tool exposes named intents and parameterized queries only. Never
  accept raw Cypher from the model.
- Keep tool formatting deterministic and include source/identity information
  in successful retrieval output where available.

## Agent safety

- Agent graphs must bound both tool iterations and individual tool calls.
- Repeated identical tool calls must stop with a clear response.
- Repeated empty or failed tool results must stop the tool loop rather than
  allowing the model to retry indefinitely.
- Preserve the distinction between a valid empty answer and a failed tool in
  prompts, tool output, and tests.
- Add regression tests for loop limits, repeated calls, empty results, stale
  IDs, malformed parameters, and dependency failures.

## Testing and verification

Before handing off a change, run:

```bash
ruff check config src tests
python -m compileall -q config src tests
pytest -q
git diff --check
```

Tests that require optional packages or external services may skip, but new
behavior should still have unit coverage that runs without those services.
When changing a service contract, update its README and the relevant tests in
the same change.

## Documentation honesty

Describe Streamlit-cached resources as explicitly app-owned resources, not as
globally available service singletons. Keep architecture diagrams and service
READMEs aligned with the actual ownership and failure behavior. If a behavior
is intentionally best-effort—for example, optional UI status/listing paths—say
so directly instead of presenting it as a guaranteed health signal.
