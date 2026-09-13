"""Shared failure/result markers used across agent-facing service boundaries."""

from enum import StrEnum


class ToolErrorKind(StrEnum):
    """Failure categories that a tool can expose to the agent."""

    INVALID_INPUT = "invalid_input"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    EXECUTION_FAILED = "execution_failed"
    INVALID_RESULT = "invalid_result"


class ToolResultStatus(StrEnum):
    """Non-error result states that still need agent-visible semantics."""

    EMPTY = "empty"
    STALE_DOCUMENT = "stale_document"


def tool_error(kind: ToolErrorKind, message: str) -> str:
    """Render a consistent structured tool failure."""
    return f"[TOOL_ERROR kind={kind.value}] {message}"


def tool_result(status: ToolResultStatus, message: str) -> str:
    """Render a consistent structured non-error tool result."""
    return f"[TOOL_RESULT status={status.value}] {message}"
