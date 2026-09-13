# LLM service

The LLM service centralises Ollama client creation and paper-profile
extraction. It keeps model, embedding, temperature, and JSON-mode settings
consistent across chat, retrieval, evaluation, ingestion, and graph writes.

## Public API

- `get_llm(...)` — returns the configured Ollama chat model, with optional model
  and temperature overrides.
- `get_embeddings(...)` — returns the configured Ollama embedding model.
- `extract_paper_profile(...)` / `aextract_paper_profile(...)` — extract a
  structured `PaperProfile` from document text.
- `PaperProfile` — typed paper title, domain, summary, concepts, methods, and
  findings.

The provider functions create clients without a module-level cache. Resource
reuse is owned by the caller (for example Streamlit's app resource cache), so
separate services and tests do not share hidden model instances.

Configuration comes from `config.settings.settings`; callers should not read
environment variables directly. Paper profiling is intentionally a single LLM
call and uses `common.parse_llm_json` to handle model output. See
`tests/unit/test_llm_and_evaluation.py` and
`tests/integration/test_retrieval_and_graph.py`.
