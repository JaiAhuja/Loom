from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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


def create_agent_node(llm, tools: list, system_prompt: str):
    """Create the agent node with an explicit tool set and prompt."""
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder(variable_name="messages")]
    )
    chain = prompt | llm_with_tools

    def agent_node(state: AgentState) -> dict:
        messages = state.get("messages") or []
        return {"messages": [chain.invoke({"messages": messages})]}

    return agent_node


def should_continue(state: AgentState) -> str:
    """Route to tools only when the model explicitly requested a tool call."""
    messages = state.get("messages") or []
    if not messages:
        return "end"
    last_message = messages[-1]
    return "tools" if getattr(last_message, "tool_calls", None) else "end"
