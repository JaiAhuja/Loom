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
import os
import re
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.domain.taxonomy import canonicalize_domain, domains_as_string
from src.services.common.json_parser import parse_llm_json

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
_MAX_FILE_NAME_LEN = 240
_DETAIL_FIELDS = (
    "contributions",
    "stands_for",
    "builds_on",
    "does_not_support",
    "limitations",
)


class ConceptItem(BaseModel):
    name: str
    description: str = ""
    domain: str = ""
    depth: str = "mentions"


class MethodItem(BaseModel):
    name: str
    description: str = ""


class FindingItem(BaseModel):
    claim: str
    evidence_type: str = "empirical"


class DetailItem(BaseModel):
    text: str
    evidence: str = ""


class PaperProfile(BaseModel):
    """LLM-extracted profile covering paper-level summary, claims, and graph details."""

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    domain: str = "Other"
    summary: str = ""
    contributions: list[DetailItem] = Field(default_factory=list)
    stands_for: list[DetailItem] = Field(default_factory=list)
    builds_on: list[DetailItem] = Field(default_factory=list)
    does_not_support: list[DetailItem] = Field(default_factory=list)
    limitations: list[DetailItem] = Field(default_factory=list)
    concepts: list[ConceptItem] = Field(default_factory=list)
    methods: list[MethodItem] = Field(default_factory=list)
    findings: list[FindingItem] = Field(default_factory=list)


_STRICT_PRIORITY_KEYWORDS = {
    "abstract",
    "introduction",
    "summary",
    "synopsis",
    "executive summary",
    "highlights",
    "in brief",
    "key points",
    "results",
    "findings",
    "outcomes",
    "observations",
    "analysis",
    "conclusion",
    "concluding remarks",
    "conclusions",
    "summary of findings",
    "implications",
    "contributions",
}


def select_key_sections(text: str, budget: int = 75000) -> str:
    """Pick front matter and high-signal sections (abstract, results, conclusions) up to budget chars."""
    if not isinstance(text, str):
        return ""
    budget = max(1_000, min(int(budget or 75_000), 150_000))
    parts = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)

    front_matter: list[str] = []
    prioritised: list[str] = []
    seen_first_priority = False

    for part in parts:
        if not part.strip():
            continue

        header = part.split("\n", 1)[0].lower().strip("# \t")
        if any(kw in header for kw in _STRICT_PRIORITY_KEYWORDS):
            seen_first_priority = True
            prioritised.append(part)
        else:
            if not seen_first_priority:
                front_matter.append(part)

    selected = ""
    for section in front_matter + prioritised:
        if len(selected) + len(section) <= budget:
            selected += section
        else:
            remaining = budget - len(selected)
            if remaining > 200:
                selected += section[:remaining]
            break

    return selected or text[:budget]


def _safe_file_name(file_name: str) -> str:
    """Keep prompt context to a basename-sized filename."""
    name = os.path.basename(str(file_name or "").replace("\\", "/")).replace("\x00", "")
    return (name or "document")[:_MAX_FILE_NAME_LEN]


def _response_text(response: Any) -> str:
    """Extract text from a LangChain response object."""
    content = getattr(response, "content", response)
    return (
        "" if content is None else content if isinstance(content, str) else str(content)
    )


def _prompt_inputs(markdown_text: str, file_name: str, budget: int) -> dict:
    return {
        "text": select_key_sections(markdown_text or "", budget=budget),
        "file_name": _safe_file_name(file_name),
        "domains": domains_as_string(),
    }


def _parse_profile_response(response: Any) -> dict | None:
    """Parse and validate the JSON object returned by the LLM."""
    parsed = parse_llm_json(_response_text(response))

    if not isinstance(parsed, dict):
        logger.warning(
            "Paper profile extraction returned non-object JSON: %s",
            type(parsed).__name__,
        )
        return None
    if "error" in parsed:
        logger.warning("Failed to parse JSON: %s", parsed.get("error", "Unknown error"))
        return None
    return parsed


def _profile_from_response(response: Any) -> Optional[PaperProfile]:
    parsed = _parse_profile_response(response)
    return _build_profile(parsed) if parsed is not None else None


def _normalise_detail(item):
    if isinstance(item, str):
        return {"text": item, "evidence": ""}
    return item if isinstance(item, dict) else None


_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """### ROLE
You are a Senior Research Analyst. Your goal is to deconstruct academic papers into high-fidelity knowledge graphs.

### EXTRACTION RULES
1. **CONCEPTS (5-15)**: Focus on technical terms, theories, or novel entities. Use snake_case or lowercase.
2. **METHODS (1-5)**: Identify the 'how'. (e.g., "Randomized Controlled Trial", "Transformer Architecture", "Qualitative Interviews").
3. **FINDINGS (2-8)**: Each finding must include a 'claim' and the 'evidence_type'. Prefer findings that include statistical results or specific outcomes.
4. **DETAILS**:
   - contributions: what the paper adds.
   - stands_for: the central thesis, position, or model the paper represents.
   - builds_on: prior work, assumptions, methods, datasets, theories, or baselines it explicitly builds on.
   - does_not_support: claims, methods, generalisations, or interpretations the paper explicitly rejects, weakens, cautions against, or fails to establish.
   - limitations: boundaries, threats to validity, missing evidence, or open questions.
   Each detail must be a concise sentence with optional evidence text.
5. **DOMAIN**: Strictly use one from: {domains}.
6. **DEPTH**: 'core' is for the primary subject; 'mentions' is for background context.

### OUTPUT INSTRUCTIONS
- Return ONLY valid JSON.
- No conversational filler (e.g., "Sure, here is...")
- If a field is missing, use null (for numbers) or [] (for arrays).
""",
        ),
        (
            "human",
            """### INPUT DATA
File: {file_name}
Content:
---
{text}
---

### TASK
Analyze the text above and populate the following JSON schema. Ensure the 'summary' is a comprehensive 15-20 sentence narrative of the paper's lifecycle, including motivation, method, evidence, contributions, boundaries, and implications.

{{
    "internal_analysis": "Briefly list the 3 most important keywords from the paper here before filling the rest",
    "title": "Full academic title",
    "authors": [],
    "year": null,
    "domain": "",
    "summary": "",
    "contributions": [
        {{"text": "", "evidence": ""}}
    ],
    "stands_for": [
        {{"text": "", "evidence": ""}}
    ],
    "builds_on": [
        {{"text": "", "evidence": ""}}
    ],
    "does_not_support": [
        {{"text": "", "evidence": ""}}
    ],
    "limitations": [
        {{"text": "", "evidence": ""}}
    ],
    "concepts": [
        {{"name": "", "description": "", "domain": "", "depth": ""}}
    ],
    "methods": [
        {{"name": "", "description": ""}}
    ],
    "findings": [
        {{"claim": "", "evidence_type": ""}}
    ]
}}
""",
        ),
    ]
)


def extract_paper_profile(
    llm: Any,
    markdown_text: str,
    file_name: str,
    budget: int = 75000,
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
    chain = _PROMPT_TEMPLATE | llm
    try:
        return _profile_from_response(
            chain.invoke(_prompt_inputs(markdown_text, file_name, budget))
        )
    except Exception as exc:
        logger.warning("Paper profile extraction failed: %s", exc, exc_info=True)
        return None


async def aextract_paper_profile(
    llm: Any,
    markdown_text: str,
    file_name: str,
    budget: int = 75000,
) -> Optional[PaperProfile]:
    """Async version of extract_paper_profile.

    Uses ainvoke on the LLM chain for non-blocking execution.
    """
    chain = _PROMPT_TEMPLATE | llm
    try:
        return _profile_from_response(
            await chain.ainvoke(_prompt_inputs(markdown_text, file_name, budget))
        )
    except Exception as exc:
        logger.warning("Async paper profile extraction failed: %s", exc, exc_info=True)
        return None


def _build_profile(parsed: dict) -> Optional[PaperProfile]:
    """Validate and normalise parsed JSON into a PaperProfile."""
    if not isinstance(parsed, dict):
        return None
    parsed = dict(parsed)
    parsed["domain"] = canonicalize_domain(parsed.get("domain", ""))
    for c in parsed.get("concepts", []) or []:
        if isinstance(c, dict):
            c["domain"] = canonicalize_domain(c.get("domain", ""))
    for field_name in _DETAIL_FIELDS:
        parsed[field_name] = [
            detail
            for item in parsed.get(field_name, []) or []
            if (detail := _normalise_detail(item)) is not None
        ]

    try:
        return PaperProfile(**parsed)
    except Exception as exc:
        logger.warning("Paper profile validation failed: %s", exc)
        return None
