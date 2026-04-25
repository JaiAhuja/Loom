"""Domain layer — shared entity definitions used by RAG and KG pipelines."""

from src.domain.paper import Paper
from src.domain.taxonomy import (
    DOMAINS,
    canonicalize_domain,
    domains_as_string,
    normalize_concept_domain,
)

__all__ = [
    "Paper",
    "DOMAINS",
    "canonicalize_domain",
    "domains_as_string",
    "normalize_concept_domain",
]
