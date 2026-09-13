from src.services.common.failures import (
    ToolErrorKind,
    ToolResultStatus,
    tool_error,
    tool_result,
)


def test_tool_failure_markers_are_consistent():
    assert tool_error(ToolErrorKind.INVALID_INPUT, "bad") == "[TOOL_ERROR kind=invalid_input] bad"
    assert tool_result(ToolResultStatus.STALE_DOCUMENT, "missing") == (
        "[TOOL_RESULT status=stale_document] missing"
    )
