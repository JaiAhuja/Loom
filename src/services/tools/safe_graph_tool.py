"""Safe, intent-based knowledge-graph tool for the LangGraph agent.

The tool exposes the :class:`GraphQueryService` intent registry to the LLM
via a single ``@tool`` function.  The LLM picks an intent name and supplies
parameters; no Cypher is generated or accepted.

Usage (from builder.py)::

    from src.services.tools.safe_graph_tool import create_safe_graph_tool

    tool = create_safe_graph_tool(conn)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool

from src.services.knowledge_graph.connection import Neo4jConnection
from src.services.knowledge_graph.service import GraphQueryService, ALL_INTENTS, REQUIRED_PARAMS

logger = logging.getLogger(__name__)

_INTENT_HELP = "\n".join(
    f"  - {intent}  (params: {', '.join(req)})" if (req := REQUIRED_PARAMS.get(intent, ()))
    else f"  - {intent}"
    for intent in sorted(ALL_INTENTS)
)


def create_safe_graph_tool(conn: Neo4jConnection) -> Any:
    """Create the intent-based knowledge graph query tool."""
    svc = GraphQueryService(conn)

    _TOOL_DESCRIPTION = (
        "Query the knowledge graph using a pre-defined intent.\n\n"
        "Pick one intent from the supported list and supply the required\n"
        "parameters as a JSON object string.\n\n"
        "Supported intents:\n"
        f"{_INTENT_HELP}\n\n"
        "Args:\n"
        '    intent: The intent name (e.g. "graph_stats", "paper_details").\n'
        "    params: A JSON string of parameters, e.g. '{\"title\": \"My Paper\"}'.\n"
        "            Use '{}' or omit for intents that take no parameters.\n\n"
        "Returns:\n"
        "    A structured text summary of the query results."
    )

    def _query_knowledge_graph(intent: str, params: str = "{}") -> str:
        parsed_params, error = _parse_params(params)
        if error:
            return error

        try:
            result = svc.execute(intent, parsed_params)
        except ValueError as exc:
            return f"Graph query error: {exc}\nSupported intents:\n{_INTENT_HELP}"
        except Exception as exc:
            return _graph_error(exc, intent, async_label=False)
        return _format_service_result(result, intent, async_label=False)

    async def _aquery_knowledge_graph(intent: str, params: str = "{}") -> str:
        """Async version of the knowledge graph query tool."""
        parsed_params, error = _parse_params(params)
        if error:
            return error

        try:
            result = await svc.aexecute(intent, parsed_params)
        except ValueError as exc:
            return f"Graph query error: {exc}\nSupported intents:\n{_INTENT_HELP}"
        except Exception as exc:
            return _graph_error(exc, intent, async_label=True)
        return _format_service_result(result, intent, async_label=True)

    return StructuredTool.from_function(
        func=_query_knowledge_graph,
        coroutine=_aquery_knowledge_graph,
        name="query_knowledge_graph",
        description=_TOOL_DESCRIPTION,
    )


def _parse_params(params: str) -> tuple[dict, str | None]:
    try:
        parsed = json.loads(params) if isinstance(params, str) else params
    except (json.JSONDecodeError, TypeError):
        return {}, (
            f"Invalid params — expected a JSON object string, got: {params!r}\n"
            f"Supported intents:\n{_INTENT_HELP}"
        )
    return (parsed if isinstance(parsed, dict) else {}), None


def _graph_error(exc: Exception, intent: str, async_label: bool) -> str:
    prefix = "Async graph" if async_label else "Graph"
    logger.error(
        "%s query service error for intent '%s': %s: %s",
        prefix, intent, type(exc).__name__, exc, exc_info=True,
    )
    if "Connection" in type(exc).__name__ or "connection" in str(exc).lower():
        return "Knowledge graph unavailable (connection error). Please try again."
    return f"Graph query failed: {exc}"


def _format_service_result(result: Any, intent: str, async_label: bool) -> str:
    if not isinstance(result, dict):
        prefix = "Async graph" if async_label else "Graph"
        logger.error("%s query returned non-dict result for intent %s: %r", prefix, intent, result)
        return "Graph query failed: invalid result format."
    return _format_result(result.get("intent") or intent, result.get("data"))



def _format_result(intent: str, data: Any) -> str:
    """Convert structured query result data to a readable string."""
    if not data:
        return f"No results for intent '{intent}'."

    formatter = _FORMATTERS.get(intent, _format_generic)
    return formatter(data)


def _format_graph_stats(data: dict) -> str:
    lines = ["**Knowledge Graph Statistics**"]
    lines.extend(f"- {k.replace('_', ' ').title()}: {v}" for k, v in data.items())
    return "\n".join(lines)


def _format_list(data: list[dict], header: str, empty_msg: str, fmt) -> str:
    """Generic list formatter — header + per-item format function."""
    if not data:
        return empty_msg
    lines = [f"**{header} ({len(data)}):**"]
    lines.extend(fmt(item) for item in data)
    return "\n".join(lines)


def _format_paper_details(data: dict) -> str:
    lines = ["**Paper Details**"]
    for section in ("concepts", "methods", "findings"):
        items = data.get(section, [])
        lines.append(f"\n*{section.title()} ({len(items)}):*")
        for item in items:
            if section == "concepts":
                lines.append(f"  - {item.get('name', '?')} ({item.get('depth', '')}): {item.get('description', '')}")
            elif section == "methods":
                lines.append(f"  - {item.get('name', '?')}: {item.get('description', '')}")
            else:
                lines.append(f"  - [{item.get('evidence_type', '')}] {item.get('claim', '')}")
    return "\n".join(lines)


def _format_finding_list(data: list[dict], relationship: str) -> str:
    if not data:
        return f"No {relationship.lower()} findings found."
    lines = [f"**{relationship} Findings ({len(data)}):**"]
    for f in data:
        lines.append(
            f"- Paper 1 ({f.get('paper_1', '?')}): {f.get('finding_1', '?')}\n"
            f"  {relationship.upper()} Paper 2 ({f.get('paper_2', '?')}): {f.get('finding_2', '?')}"
        )
        if f.get("reason"):
            lines.append(f"  Reason: {f['reason']}")
    return "\n".join(lines)


def _format_generic(data: Any) -> str:
    """Fallback for any intent without a custom formatter."""
    if isinstance(data, list):
        if not data:
            return "No results."
        return json.dumps(data, indent=2, default=str)
    if isinstance(data, dict):
        return json.dumps(data, indent=2, default=str)
    return str(data)


_FORMATTERS: dict[str, Any] = {
    "graph_stats": _format_graph_stats,
    "paper_list": lambda d: _format_list(
        d, "Papers in the graph", "No papers in the knowledge graph.",
        lambda p: f"- {p.get('title', 'Untitled')} [{p.get('domain', '')}] ({p.get('concept_count', 0)} concepts)",
    ),
    "paper_details": _format_paper_details,
    "shared_concepts": lambda d: _format_list(
        d, "Shared Concepts", "No shared concepts found between these papers.",
        lambda c: f"- {c.get('concept', '?')} [{c.get('domain', '')}]: {c.get('description', '')}",
    ),
    "concept_papers": lambda d: _format_list(
        d, "Papers discussing this concept", "No papers discuss this concept.",
        lambda p: f"- {p.get('title', '?')} [{p.get('domain', '')}] (depth: {p.get('depth', '?')})",
    ),
    "related_concepts": lambda d: _format_list(
        d, "Related Concepts", "No related concepts found.",
        lambda c: f"- {c.get('name', '?')} (strength: {(c.get('strength') or 0):.1f})",
    ),
    "contradiction_list": lambda d: _format_finding_list(d, "Contradicting"),
    "support_list": lambda d: _format_finding_list(d, "Supporting"),
    "extension_list": lambda d: _format_finding_list(d, "Extending"),
}
