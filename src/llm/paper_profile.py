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
from langchain_core.prompts import ChatPromptTemplate

from src.domain.taxonomy import canonicalize_domain, domains_as_string
from src.utils.json_parser import parse_llm_json

if TYPE_CHECKING:
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
    """LLM-extracted profile covering title, domain, summary, concepts, methods, and findings."""

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

_STRICT_PRIORITY_KEYWORDS = {
    # The Gist/Summary
    "abstract", "introduction", "summary", "synopsis", "executive summary", "highlights", 
    "in brief", "key points",
    "results", "findings", "outcomes", "observations", "analysis",
    "conclusion", "concluding remarks", "conclusions", "summary of findings",
    "implications", "contributions"
}

def select_key_sections(text: str, budget: int = 75000) -> str:
    """Pick front matter and high-signal sections (abstract, results, conclusions) up to budget chars."""
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
    for section in front_matter + prioritised: # Reconstruct: Front matter at the top, then the key sections
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

_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        """### ROLE
You are a Senior Research Analyst. Your goal is to deconstruct academic papers into high-fidelity knowledge graphs.

### EXTRACTION RULES
1. **CONCEPTS (5-15)**: Focus on technical terms, theories, or novel entities. Use snake_case or lowercase.
2. **METHODS (1-5)**: Identify the 'how'. (e.g., "Randomized Controlled Trial", "Transformer Architecture", "Qualitative Interviews").
3. **FINDINGS (2-8)**: Each finding must include a 'claim' and the 'evidence_type'. Prefer findings that include statistical results or specific outcomes.
4. **DOMAIN**: Strictly use one from: {domains}.
5. **DEPTH**: 'core' is for the primary subject; 'mentions' is for background context.

### OUTPUT INSTRUCTIONS
- Return ONLY valid JSON.
- No conversational filler (e.g., "Sure, here is...")
- If a field is missing, use null (for numbers) or [] (for arrays).
"""
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
Analyze the text above and populate the following JSON schema. Ensure the 'summary' is a comprehensive 10-15 sentence narrative of the paper's lifecycle.

{{
    "internal_analysis": "Briefly list the 3 most important keywords from the paper here before filling the rest",
    "title": "Full academic title",
    "authors": [],
    "year": null,
    "domain": "",
    "summary": "",
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
"""
    )
])


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
    text = select_key_sections(markdown_text, budget=budget)
    chain = _PROMPT_TEMPLATE | llm
    try:
        response = chain.invoke(
            {
                "text": text,
                "file_name": file_name,
                "domains": domains_as_string()
            }
        )
        parsed = parse_llm_json(response.content)

        if not isinstance(parsed, dict) or "error" in parsed:
            logger.warning("Failed to parse JSON: %s", (parsed or {}).get("error", "Unknown error"))
            return None
    except Exception as exc:
        logger.warning("Paper profile extraction failed: %s", exc)
        return None

    return _build_profile(parsed)


async def aextract_paper_profile(
    llm: Any,
    markdown_text: str,
    file_name: str,
    budget: int = 75000,
) -> Optional[PaperProfile]:
    """Async version of extract_paper_profile.

    Uses ainvoke on the LLM chain for non-blocking execution.
    """
    text = select_key_sections(markdown_text, budget=budget)
    chain = _PROMPT_TEMPLATE | llm
    try:
        response = await chain.ainvoke(
            {
                "text": text,
                "file_name": file_name,
                "domains": domains_as_string()
            }
        )
        parsed = parse_llm_json(response.content)

        if not isinstance(parsed, dict) or "error" in parsed:
            logger.warning("Failed to parse JSON: %s", (parsed or {}).get("error", "Unknown error"))
            return None
    except Exception as exc:
        logger.warning("Async paper profile extraction failed: %s", exc)
        return None

    return _build_profile(parsed)


def _build_profile(parsed: dict) -> Optional[PaperProfile]:
    """Validate and normalise parsed JSON into a PaperProfile."""
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
