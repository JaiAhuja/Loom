"""Single-call LLM extraction producing the full paper profile.

Replaces the two prior calls (``_classify_and_summarise`` + ``_extract_entities``)
with one structured call.  Both the RAG processor and the KG extractor
consume the same :class:`PaperProfile`, eliminating:

* duplicate domain classification (processor vs KG),
* duplicate summary generation, and
* divergent title extraction paths.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from src.domain.taxonomy import canonicalize_domain, domains_as_string
from src.utils.json_parser import parse_llm_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured data model (Pydantic)
# ---------------------------------------------------------------------------

class ConceptItem(BaseModel):
    name: str
    description: str = ""
    domain: str = ""
    depth: str = "mentions"  # "core" | "mentions"


class MethodItem(BaseModel):
    name: str
    description: str = ""


class FindingItem(BaseModel):
    claim: str
    evidence_type: str = "empirical"  # "empirical" | "theoretical" | "survey"


class PaperProfile(BaseModel):
    """LLM-extracted profile covering everything RAG + KG need from one call."""

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    domain: str = "Other"
    summary: str = ""
    concepts: list[ConceptItem] = Field(default_factory=list)
    methods: list[MethodItem] = Field(default_factory=list)
    findings: list[FindingItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Section-aware text selection (moved from extraction.py so both
# pipelines benefit from it without duplication)
# ---------------------------------------------------------------------------

_SKIP_KEYWORDS = {
    "reference", "bibliography", "appendix", "acknowledgment",
    "acknowledgement", "vita", "biograph",
}
_PRIORITY_KEYWORDS = {
    "abstract", "introduction", "conclusion", "summary",
    "results", "findings", "discussion",
}


def select_key_sections(text: str, budget: int = 12000) -> str:
    """Pick the most informative sections from a Markdown paper.

    Splits on Markdown headings, skips reference/appendix sections,
    prioritises abstract/intro/conclusion/results, then fills with
    remaining sections until *budget* characters are reached.
    """

    parts = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
    prioritised: list[str] = []
    rest: list[str] = []

    for part in parts:
        header = part.split("\n", 1)[0].lower().strip("# \t")
        if any(kw in header for kw in _SKIP_KEYWORDS):
            continue
        if any(kw in header for kw in _PRIORITY_KEYWORDS):
            prioritised.append(part)
        else:
            rest.append(part)

    selected = ""
    for section in prioritised + rest:
        if len(selected) + len(section) <= budget:
            selected += section
        else:
            remaining = budget - len(selected)
            if remaining > 200:
                selected += section[:remaining]
            break

    return selected or text[:budget]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """Analyze this research paper/article and extract structured information.

PAPER TEXT:
---
{text}
---

SOURCE FILE: {file_name}

Return a JSON object with EXACTLY this structure (no other text, just JSON):
{{
    "title": "exact paper title",
    "authors": ["Author Name 1", "Author Name 2"],
    "year": 2024,
    "domain": "one category from: {domains}",
    "summary": "3-5 sentence summary capturing motivation, methodology, findings, and contributions",
    "concepts": [
        {{
            "name": "lowercase concept name",
            "description": "one sentence description",
            "domain": "same categories as above",
            "depth": "core or mentions"
        }}
    ],
    "methods": [
        {{"name": "method name", "description": "one sentence description"}}
    ],
    "findings": [
        {{"claim": "specific finding from the paper", "evidence_type": "empirical or theoretical or survey"}}
    ]
}}

Rules:
- Extract 5-15 key concepts (not too many, not too few).
- Extract 1-5 methods (algorithms, techniques, frameworks used).
- Extract 2-8 key findings (specific claims, results, conclusions).
- Use lowercase for concept names for consistency.
- "depth": "core" if the paper deeply discusses it, "mentions" if just referenced.
- If year/authors are unknown, use null / empty array.
- Return ONLY valid JSON, no markdown formatting."""


def extract_paper_profile(
    llm: Any,
    markdown_text: str,
    file_name: str,
    budget: int = 12000,
) -> Optional[PaperProfile]:
    """Run the single extraction call and return a validated profile.

    Args:
        llm: A chat LLM exposing ``invoke([...messages])``.
        markdown_text: Paper content in Markdown.
        file_name: Source filename for context.
        budget: Max characters of paper text to send to the LLM.

    Returns:
        A :class:`PaperProfile` (possibly with empty fields) or ``None``
        if extraction/parsing failed.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    text = select_key_sections(markdown_text, budget=budget)
    prompt = _PROMPT_TEMPLATE.format(
        text=text,
        file_name=file_name,
        domains=domains_as_string(),
    )

    messages = [
        SystemMessage(content=(
            "You are a precise academic knowledge extractor. "
            "Return only valid JSON matching the requested structure."
        )),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        parsed = parse_llm_json(response.content)
    except Exception as exc:
        logger.warning("Paper profile extraction failed: %s", exc)
        return None

    if not isinstance(parsed, dict):
        return None

    # Normalise domains before validation so Pydantic doesn't have to
    parsed["domain"] = canonicalize_domain(parsed.get("domain", ""))
    for c in parsed.get("concepts", []) or []:
        if isinstance(c, dict):
            c["domain"] = canonicalize_domain(c.get("domain", ""))

    try:
        return PaperProfile(**parsed)
    except Exception as exc:
        logger.warning("Paper profile validation failed: %s", exc)
        return None
