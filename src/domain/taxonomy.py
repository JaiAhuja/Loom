"""Canonical domain taxonomy shared by RAG processing and KG extraction.

Before this module existed the domain taxonomy was defined in three
separate places (``DocumentProcessor.DOMAIN_CATEGORIES``, the extractor
prompt, and ``_DOMAIN_BUCKETS`` in ``schema.py``), which meant the same
paper could be classified inconsistently.  This module is the single
source of truth.
"""

from __future__ import annotations

# Display-friendly canonical list (Title Case).  Processor + extractor
# prompts both reference this list.
DOMAINS: tuple[str, ...] = (
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Reinforcement Learning",
    "Data Engineering",
    "Data Science",
    "Statistics",
    "Databases",
    "Distributed Systems",
    "Cloud Computing",
    "Cybersecurity",
    "Supply Chain",
    "Optimization",
    "Operations Research",
    "Robotics",
    "Finance & Economics",
    "Healthcare & Biotech",
    "Other",
)

# Aliases used by legacy snake_case prompts and .env files — normalised
# to the canonical Title-Case form above.
_ALIASES: dict[str, str] = {
    "ai_general": "Artificial Intelligence",
    "ai": "Artificial Intelligence",
    "machine_learning": "Machine Learning",
    "deep_learning": "Deep Learning",
    "nlp": "Natural Language Processing",
    "natural_language_processing": "Natural Language Processing",
    "computer_vision": "Computer Vision",
    "cv": "Computer Vision",
    "reinforcement_learning": "Reinforcement Learning",
    "rl": "Reinforcement Learning",
    "data_engineering": "Data Engineering",
    "data_science": "Data Science",
    "data_infra": "Data Engineering",
    "analytics": "Data Science",
    "statistics": "Statistics",
    "databases": "Databases",
    "db": "Databases",
    "distributed_systems": "Distributed Systems",
    "cloud": "Cloud Computing",
    "cloud_computing": "Cloud Computing",
    "cybersecurity": "Cybersecurity",
    "security": "Cybersecurity",
    "supply_chain": "Supply Chain",
    "optimization": "Optimization",
    "operations_research": "Operations Research",
    "or": "Operations Research",
    "robotics": "Robotics",
    "finance": "Finance & Economics",
    "economics": "Finance & Economics",
    "healthcare": "Healthcare & Biotech",
    "biotech": "Healthcare & Biotech",
    "general": "Other",
    "": "Other",
}

# Broad buckets used by the graph layer to compare concepts across
# closely-related fine-grained domains.
_BUCKETS: dict[str, str] = {
    "Artificial Intelligence": "ai",
    "Machine Learning": "ai",
    "Deep Learning": "ai",
    "Natural Language Processing": "ai",
    "Computer Vision": "ai",
    "Reinforcement Learning": "ai",
    "Robotics": "ai",
    "Data Engineering": "data_infra",
    "Databases": "data_infra",
    "Distributed Systems": "data_infra",
    "Cloud Computing": "data_infra",
    "Data Science": "analytics",
    "Statistics": "analytics",
    "Operations Research": "analytics",
    "Optimization": "analytics",
    "Supply Chain": "analytics",
    "Cybersecurity": "general",
    "Finance & Economics": "general",
    "Healthcare & Biotech": "general",
    "Other": "general",
}


def canonicalize_domain(raw: str) -> str:
    """Map any domain string (Title-Case, snake_case, alias) to the canonical form.

    Returns ``"Other"`` when no match is found.
    """
    if not raw:
        return "Other"
    trimmed = raw.strip()
    # Exact canonical match (case-insensitive)
    for d in DOMAINS:
        if d.lower() == trimmed.lower():
            return d
    # Alias map (snake_case)
    return _ALIASES.get(trimmed.lower(), "Other")


def normalize_concept_domain(raw_domain: str) -> str:
    """Map a fine-grained domain label to a broad bucket.

    Returns one of ``"ai"``, ``"data_infra"``, ``"analytics"``, or ``"general"``.
    Works on both canonical Title-Case and legacy snake_case inputs.
    """
    canonical = canonicalize_domain(raw_domain)
    return _BUCKETS.get(canonical, "general")


def domains_as_string() -> str:
    """Return a comma-separated string of canonical domains for prompts."""
    return ", ".join(DOMAINS)
