import json

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.settings import settings
from src.services.agent.state import AgentState


SYSTEM_PROMPT_BASE = """You are Loom, a careful tutor for data engineering, data science, and AI.
Answer the user's actual question first. Match the depth and format to the task: use
headings, lists, tables, or code only when they improve clarity. Be precise, state
uncertainty honestly, and never invent citations or document facts. When tools are
available, use them only when they add evidence; explain which sources informed the answer.
For technical explanations, add a short plain-English analogy when it genuinely helps.
"""

TOOL_PROMPT_RAG = """
Document search is available through `query_documents`. Use it for questions about
uploaded material or a named paper. Ground claims in returned excerpts and identify
the source document. If there are no relevant results, say that plainly.
"""

TOOL_PROMPT_GRAPH = """
The `query_knowledge_graph` tool searches the uploaded papers' knowledge graph.
Consult its description, provide all required parameters, and do not guess when no
supported intent matches. Explain graph results clearly and surface tool failures.
"""


def build_system_prompt(use_rag: bool = False, use_graph: bool = False) -> str:
    """Build a concise system prompt with only the capabilities in this graph."""
    prompt = SYSTEM_PROMPT_BASE
    if use_rag:
        prompt += "\n" + TOOL_PROMPT_RAG
    if use_graph:
        prompt += "\n" + TOOL_PROMPT_GRAPH
    if not use_rag and not use_graph:
        prompt += "\nOffline mode is active; answer from model knowledge only."
    return prompt


def _tool_call_signature(tool_calls: list[dict]) -> str:
    """Return a stable signature used to detect repeated model tool calls."""
    normalized = [{"name": call.get("name"), "args": call.get("args", {})} for call in tool_calls]
    return json.dumps(normalized, sort_keys=True, default=str)


def _consecutive_failed_tool_results(messages: list) -> int:
    """Count the trailing empty/error ToolMessages in graph state."""
    count = 0
    for message in reversed(messages):
        if message.__class__.__name__ != "ToolMessage":
            break
        content = str(getattr(message, "content", ""))
        if (
            "[TOOL_RESULT status=empty]" in content
            or "[TOOL_RESULT status=stale_document]" in content
            or "[TOOL_ERROR kind=" in content
        ):
            count += 1
        else:
            break
    return count


def create_agent_node(
    llm,
    tools: list,
    system_prompt: str,
    max_tool_iterations: int | None = None,
    max_tool_calls: int | None = None,
):
    """Create the agent node with an explicit tool set and prompt."""
    if max_tool_iterations is None:
        max_tool_iterations = settings.AGENT_MAX_TOOL_ITERATIONS
    if max_tool_iterations < 1:
        raise ValueError("max_tool_iterations must be positive")
    if max_tool_calls is None:
        max_tool_calls = settings.AGENT_MAX_TOOL_CALLS
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be positive")
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder(variable_name="messages")]
    )
    chain = prompt | llm_with_tools

    def agent_node(state: AgentState) -> dict:
        messages = state.get("messages") or []
        iterations = state.get("tool_iterations", 0)
        tool_calls_used = state.get("tool_calls_used", 0)
        failed_results = max(
            _consecutive_failed_tool_results(messages),
            state.get("empty_tool_result_count", 0),
        )
        if (
            iterations >= max_tool_iterations
            or tool_calls_used >= max_tool_calls
            or failed_results >= settings.AGENT_MAX_REPEATED_EMPTY_RESULTS
        ):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I stopped tool use after reaching the safety limit. "
                            "Please refine the question and try again."
                        )
                    )
                ]
            }

        response = chain.invoke({"messages": messages})
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return {"messages": [response]}

        signature = _tool_call_signature(tool_calls)
        seen_signatures = state.get("tool_call_signatures", [])
        if signature in seen_signatures:
            return {
                "messages": [
                    AIMessage(
                        content="I stopped because the same tool request was repeated without new progress."
                    )
                ]
            }
        return {
            "messages": [response],
            "tool_iterations": iterations + 1,
            "tool_calls_used": tool_calls_used + len(tool_calls),
            "tool_call_signatures": [*seen_signatures, signature],
        }

    return agent_node


def should_continue(state: AgentState, max_tool_iterations: int | None = None) -> str:
    """Route to tools only when the model explicitly requested a tool call."""
    if max_tool_iterations is None:
        max_tool_iterations = settings.AGENT_MAX_TOOL_ITERATIONS
    messages = state.get("messages") or []
    if not messages:
        return "end"
    last_message = messages[-1]
    if state.get("tool_iterations", 0) >= max_tool_iterations:
        return "end"
    if state.get("tool_calls_used", 0) >= settings.AGENT_MAX_TOOL_CALLS:
        return "end"

    empty_results = _consecutive_failed_tool_results(messages)
    empty_results = max(empty_results, state.get("empty_tool_result_count", 0))
    if empty_results >= settings.AGENT_MAX_REPEATED_EMPTY_RESULTS:
        return "end"
    return "tools" if getattr(last_message, "tool_calls", None) else "end"
