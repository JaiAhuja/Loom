"""Unit tests for agent routing and safe tool formatting."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.services.agent.nodes import build_system_prompt, should_continue
from src.services.knowledge_graph.service import GraphQueryService
from src.services.tools.rag_tool import (
    _format_doc_result,
    _format_search,
    create_rag_tool,
)
from src.services.tools.safe_graph_tool import (
    _parse_params,
    _format_result,
    create_safe_graph_tool,
)

pytestmark = pytest.mark.unit


def test_system_prompt_and_agent_routing_reflect_enabled_tools():
    assert "offline mode" in build_system_prompt()
    assert "query_documents" in build_system_prompt(use_rag=True)
    assert "query_knowledge_graph" in build_system_prompt(use_graph=True)
    assert should_continue({"messages": []}) == "end"
    assert should_continue({"messages": [AIMessage(content="done")]}) == "end"
    assert (
        should_continue(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "x", "args": {}, "id": "1", "type": "tool_call"}
                        ],
                    )
                ]
            }
        )
        == "tools"
    )


def test_document_tool_formatters_include_source_and_empty_state():
    doc = Document(
        page_content="retrieved text",
        metadata={
            "paper": "Paper A",
            "domain": "AI",
            "paper_chunk": "summary",
            "chunk_type": "summary",
        },
    )
    formatted = _format_doc_result(doc, 1)
    assert "Paper A" in formatted and "[SUMMARY]" in formatted
    assert "No relevant content" in _format_search([], " (Paper A)")


def test_rag_tool_uses_the_bound_retriever_for_sync_and_async_queries():
    class Retriever:
        def invoke(self, query):
            return [
                Document(
                    page_content=f"answer for {query}", metadata={"paper": "Paper"}
                )
            ]

        async def ainvoke(self, query):
            return self.invoke(query)

    class Store:
        def get_retriever(self, collection, top_k, document_id):
            assert (collection, top_k, document_id) == ("notes", 5, "paper-1")
            return Retriever()

    tool = create_rag_tool("notes", store=Store(), document_id="paper-1")
    assert "answer for question" in tool.invoke({"query": "question"})
    assert "answer for async" in asyncio.run(tool.ainvoke({"query": "async"}))


def test_graph_service_validates_intents_and_dispatches_to_queries():
    service = GraphQueryService(SimpleNamespace())

    class Queries:
        def get_paper_details(self, title):
            return [{"title": title}]

        async def aget_paper_details(self, title):
            return [{"title": title}]

    service._queries = Queries()
    assert service.execute("paper_details", {"title": "Paper"})["data"] == [
        {"title": "Paper"}
    ]
    assert (
        asyncio.run(service.aexecute("paper_details", {"title": "Paper"}))["intent"]
        == "paper_details"
    )
    try:
        service.execute("unknown")
        assert False, "Expected unknown intents to be rejected"
    except ValueError as exc:
        assert "Unknown intent" in str(exc)
    try:
        service.execute("paper_details")
        assert False, "Expected missing parameters to be rejected"
    except ValueError as exc:
        assert "missing" in str(exc)


def test_safe_graph_tool_rejects_invalid_params_and_formats_results():
    params, error = _parse_params('{"title": "Paper"}')
    assert params == {"title": "Paper"} and error is None
    _, error = _parse_params("not json")
    assert "Invalid params" in error
    assert "Papers in the graph" in _format_result(
        "paper_list", [{"title": "Paper", "domain": "AI"}]
    )
    assert "No results" in _format_result("graph_stats", {})

    class Queries:
        def get_graph_stats(self):
            return {"papers": 1, "concepts": 2}

    tool = create_safe_graph_tool(SimpleNamespace())
    tool._svc = Queries() if hasattr(tool, "_svc") else None
    assert tool.name == "query_knowledge_graph"
