# Common service helpers

This package contains small, dependency-light helpers shared by multiple
services. It is intentionally narrow so domain or feature-specific behaviour
does not accumulate here.

## Public API

- `parse_llm_json(response_text)` — extracts and parses JSON returned by an LLM,
  including common fenced or surrounding-text formats.
- `failures.py` — shared structured markers for tool input errors, dependency
  outages, execution failures, empty results, and stale document scopes.

The parser is used by paper profiling, graph writing, and evaluation. Changes
should preserve its tolerant handling of model output and should be covered by
the tests in `tests/unit/test_domain_and_json.py`.
