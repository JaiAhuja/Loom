import logging

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.services.agent.nodes import build_system_prompt, create_agent_node, should_continue
from src.services.agent.state import AgentState
from src.services.llm import get_llm
from src.services.knowledge_graph.connection import Neo4jConnection
from src.services.tools.rag_tool import create_rag_tool
from src.services.tools.safe_graph_tool import create_safe_graph_tool

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds configurable LangGraph agents with optional tools.

    Dynamically assembles a ReAct-style agent graph based on
    which features (RAG, knowledge graph) are enabled.

    """

    def build(
        self,
        use_rag: bool = False,
        use_graph: bool = False,
        collection_name: str = None,
        model: str = None,
        temperature: float = None,
        neo4j_conn: Neo4jConnection = None,
        vector_store=None,
        document_id: str | None = None,
    ):
        """Build and compile the Loom agent graph.

        Args:
            use_rag: Enable RAG document query tool.
            use_graph: Enable Neo4j knowledge graph query tool.
            collection_name: ChromaDB collection name (required if use_rag=True).
            model: Ollama model name override.
            temperature: LLM temperature override.
            neo4j_conn: Shared :class:`Neo4jConnection` for the graph tool.
            vector_store: Shared :class:`VectorStoreManager` for the RAG tool
                (reuses the app-level cached instance instead of creating a
                new Chroma client per chat turn).
            document_id: Canonical paper identity (filename-based, e.g.
                ``"Self-Supervised Learning"``) used to filter retrieval to a
                scope RAG retrieval to a single paper.

        Returns:
            Compiled LangGraph StateGraph ready for invocation.
        """
        llm = get_llm(model=model, temperature=temperature)
        tools = self._gather_tools(
            use_rag, use_graph, collection_name, model,
            neo4j_conn, vector_store, document_id,
        )
        system_prompt = build_system_prompt(use_rag, use_graph)

        graph = StateGraph(AgentState)
        graph.add_node("agent", create_agent_node(llm, tools, system_prompt))

        if tools:
            graph.add_node("tools", ToolNode(tools))
            graph.add_conditional_edges(
                "agent",
                should_continue,
                {"tools": "tools", "end": END},
            )
            graph.add_edge("tools", "agent")
        else:
            graph.add_edge("agent", END)

        graph.add_edge(START, "agent")
        return graph.compile()

    def _gather_tools(
        self,
        use_rag: bool,
        use_graph: bool,
        collection_name: str,
        model: str = None,
        neo4j_conn: Neo4jConnection = None,
        vector_store=None,
        document_id: str | None = None,
    ) -> list:
        """Collect tools based on feature flags.

        Override this method to add custom tools to the agent.
        """
        tools = []

        if use_rag and collection_name:
            tools.append(create_rag_tool(
                collection_name,
                store=vector_store,
                document_id=document_id,
            ))

        if use_graph:
            if neo4j_conn is not None:
                tools.append(create_safe_graph_tool(neo4j_conn))
            else:
                logger.warning(
                    "use_graph=True but no Neo4j connection provided. "
                    "Graph tool will not be available."
                )

        return tools
