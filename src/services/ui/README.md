# UI service

The UI service contains shared Streamlit wiring used by the application pages.
It keeps connection checks, cached component factories, common visual chrome,
and destructive-action confirmation consistent across the app.

## Public API

- `bootstrap.check_ollama_status()` and `check_neo4j_status()` — cached health
  checks for optional local services.
- `get_vector_store()` and `get_document_processor()` — cached shared service
  instances for Streamlit reruns.
- `chrome` helpers — stylesheet injection, brand/hero/status rendering, and
  user-facing chat errors.
- `confirm_destructive(...)` — two-step confirmation for destructive actions.

Keep Streamlit-specific concerns here or in the page modules. Reuse the cached
factories rather than creating a new Chroma or Neo4j client on each rerun, and
use the confirmation helper for deletes or other irreversible UI actions.
Tests are in `tests/integration/test_ui_and_connection.py`.
