"""Shared LLM JSON response parser."""

import json
import re


def parse_llm_json(text: str) -> dict | None:
    """Extract a JSON object from an LLM response, handling code blocks."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"\{.*\}"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pattern else match.group(0))
            except (json.JSONDecodeError, IndexError):
                continue
    return None
