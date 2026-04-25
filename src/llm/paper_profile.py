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

_STRICT_PRIORITY_KEYWORDS = {
    # The Gist/Summary
    "abstract", "summary", "synopsis", "executive summary", "highlights", 
    "in brief", "key points",
    "results", "findings", "outcomes", "observations", "analysis",
    "conclusion", "concluding remarks", "conclusions", "summary of findings",
    "implications", "contributions"
}

def select_key_sections(text: str, budget: int = 75000) -> str:
    """Strictly picks front matter plus high-signal summary/outcome sections.
    
    Excludes high-volume 'filler' like Introduction, Methodology, 
    Literature Review, and Discussion.
    """
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
        """You are a precise academic knowledge extractor. 
Your task is to analyze research papers and extract structured information.

RULES:
- Extract 5-15 key concepts (not too many, not too few).
- Extract 1-5 methods (algorithms, techniques, frameworks used).
- Extract 2-8 key findings (specific claims, results, conclusions).
- Use lowercase for concept names for consistency.
- For concept 'depth': use "core" if the paper deeply discusses it, or "mentions" if just referenced.
- If year/authors are unknown, use null or an empty array.
- You MUST return ONLY a valid JSON object. No markdown formatting, no explanations.

EXPECTED JSON SCHEMA:
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
}}"""
    ),
    (
        "human",
        """Extract the profile for the following paper.

SOURCE FILE: {file_name}

PAPER TEXT:
---
{text}
---"""
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
    from langchain_core.messages import HumanMessage, SystemMessage

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
        
        if not parsed or "error" in parsed:
            logger.warning(f"Failed to parse JSON: {parsed.get('error', 'Unknown error')}")
            return None
    except Exception as exc:
        logger.warning("Paper profile extraction failed: %s", exc)
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
