import logging

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool

from config.settings import settings
from src.services.retrieval.store import VectorStoreManager

logger = logging.getLogger(__name__)


def _format_doc_result(doc: Document, index: int) -> str:
    """Format a retrieved document with null-safe metadata access."""
    metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
    paper = metadata.get("paper") or metadata.get("source") or "Unknown"
    domain = metadata.get("domain") or ""
    chunk_id = metadata.get("paper_chunk") or "?"
    chunk_type = metadata.get("chunk_type") or "content"
    content = getattr(doc, "page_content", "") or ""

    type_badge = " [SUMMARY]" if chunk_type == "summary" else ""
    domain_badge = f" | {domain}" if domain else ""
    return (
        f"**Document {index}** (Paper: {paper}{domain_badge}, "
        f"Chunk: {chunk_id}{type_badge}):\n\n"
        f"{content}"
    )


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        _format_doc_result(doc, i) for i, doc in enumerate(docs, 1)
    )


def _search_error(exc: Exception) -> str:
    if "Connection" in type(exc).__name__ or "Timeout" in type(exc).__name__:
        return "Document retrieval unavailable (connection error). Please try again."
    return f"Document search failed: {exc}"


def _format_search(docs: list[Document], scope_label: str) -> str:
    return (
        _format_docs(docs)
        if docs
        else f"No relevant content found in the uploaded documents{scope_label}."
    )


def create_rag_tool(
    collection_name: str,
    store: VectorStoreManager | None = None,
    document_id: str | None = None,
):
    """Create a RAG document query tool bound to a specific collection.

    Returns a LangChain tool that retrieves relevant document chunks from
    the specified ChromaDB collection, optionally scoped to a single paper.

    Args:
        collection_name: The ChromaDB collection to query.
        store: Optional pre-built :class:`VectorStoreManager` to reuse.
            When ``None`` a fresh instance is created (legacy behaviour).
            Passing the app-level cached store avoids spinning up a new
            Chroma client + embedding cache on every chat turn.
        document_id: Canonical paper identity (filename-based, e.g.
            ``"Self-Supervised Learning"``) that scopes
            retrieval by the stable content-addressed key.

    Returns:
        A LangChain @tool decorated function.
    """
    if store is None:
        store = VectorStoreManager()
    retriever = store.get_retriever(
        collection_name,
        top_k=settings.RAG_TOP_K,
        document_id=document_id,
    )

    scope_label = ""
    if document_id:
        scope_label = f" (filtered to paper id: {document_id})"

    def _query_documents(query: str) -> str:
        """Search through uploaded PDF documents for relevant information.

        Use this tool to find specific content from the user's uploaded
        study materials, papers, or textbooks. Returns the most relevant
        excerpts along with source document information.

        Args:
            query: The search query to find relevant document content.
        """
        try:
            return _format_search(retriever.invoke(query), scope_label)
        except Exception as e:
            logger.error(
                f"Document search failed for query '{query}': {type(e).__name__}: {e}",
                exc_info=True,
            )
            return _search_error(e)

    async def _aquery_documents(query: str) -> str:
        """Async search through uploaded PDF documents for relevant information."""
        try:
            return _format_search(await retriever.ainvoke(query), scope_label)
        except Exception as e:
            logger.error(
                f"Async document search failed for query '{query}': {type(e).__name__}: {e}",
                exc_info=True,
            )
            return _search_error(e)

    return StructuredTool.from_function(
        func=_query_documents,
        coroutine=_aquery_documents,
        name="query_documents",
        description=(
            "Search through uploaded PDF documents for relevant information. "
            "Use this tool to find specific content from the user's uploaded "
            "study materials, papers, or textbooks. Returns the most relevant "
            "excerpts along with source document information."
        ),
    )
