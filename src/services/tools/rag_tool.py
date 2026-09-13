import logging

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool

from config.settings import Settings, settings
from src.services.common.failures import ToolErrorKind, ToolResultStatus, tool_error, tool_result
from src.services.ingestion.identity import validate_document_id
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
    return f"**Document {index}** (Paper: {paper}{domain_badge}, Chunk: {chunk_id}{type_badge}):\n\n{content}"


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(_format_doc_result(doc, i) for i, doc in enumerate(docs, 1))


def _search_error(exc: Exception) -> str:
    if "Connection" in type(exc).__name__ or "Timeout" in type(exc).__name__:
        return tool_error(
            ToolErrorKind.DEPENDENCY_UNAVAILABLE,
            "Document retrieval is unavailable (connection error).",
        )
    return tool_error(
        ToolErrorKind.EXECUTION_FAILED,
        f"Document search failed: {type(exc).__name__}: {exc}",
    )


def _format_search(docs: list[Document], scope_label: str) -> str:
    return (
        _format_docs(docs)
        if docs
        else tool_result(
            ToolResultStatus.EMPTY,
            f"No relevant content found in the uploaded documents{scope_label}.",
        )
    )


def create_rag_tool(
    collection_name: str,
    store: VectorStoreManager | None,
    document_id: str | None = None,
    settings_obj: Settings | None = None,
):
    """Create a RAG document query tool bound to a specific collection.

    Returns a LangChain tool that retrieves relevant document chunks from
    the specified ChromaDB collection, optionally scoped to a single paper.

    Args:
        collection_name: The ChromaDB collection to query.
        store: Explicitly owned retrieval service supplied by the application/service owner.
        document_id: Canonical paper identity (filename-based, e.g.
            ``"Self-Supervised Learning"``) that scopes
            retrieval by the stable content-addressed key.

    Returns:
        A LangChain @tool decorated function.
    """
    if not collection_name:
        raise ValueError("collection_name is required")
    if store is None or not callable(getattr(store, "get_retriever", None)):
        raise TypeError("store must be an explicitly owned retrieval service")
    VectorStoreManager._validate_collection_name(collection_name)
    if document_id is not None and (not isinstance(document_id, str) or not document_id.strip()):
        raise ValueError("document_id must be non-empty when provided")
    if document_id is not None:
        validate_document_id(document_id)

    config = settings_obj or settings
    scope_label = ""
    if document_id:
        scope_label = f" (filtered to paper id: {document_id})"

    def _scope_status() -> str | None:
        """Return an explicit stale-scope result when the store can verify it."""
        checker = getattr(store, "is_document_indexed", None)
        if document_id and callable(checker) and not checker(collection_name, document_id):
            return tool_result(
                ToolResultStatus.STALE_DOCUMENT,
                f"No indexed document matches document id {document_id!r}.",
            )
        return None

    def _query_documents(query: str) -> str:
        """Search through uploaded PDF documents for relevant information.

        Use this tool to find specific content from the user's uploaded
        study materials, papers, or textbooks. Returns the most relevant
        excerpts along with source document information.

        Args:
            query: The search query to find relevant document content.
        """
        try:
            if not isinstance(query, str) or not query.strip():
                return tool_error(ToolErrorKind.INVALID_INPUT, "Document query must be non-empty.")
            if len(query) > config.RAG_MAX_QUERY_LENGTH:
                return tool_error(
                    ToolErrorKind.INVALID_INPUT,
                    f"Document query exceeds the {config.RAG_MAX_QUERY_LENGTH}-character limit.",
                )
            scope_status = _scope_status()
            if scope_status:
                return scope_status
            retriever = store.get_retriever(collection_name, top_k=config.RAG_TOP_K, document_id=document_id)
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
            if not isinstance(query, str) or not query.strip():
                return tool_error(ToolErrorKind.INVALID_INPUT, "Document query must be non-empty.")
            if len(query) > config.RAG_MAX_QUERY_LENGTH:
                return tool_error(
                    ToolErrorKind.INVALID_INPUT,
                    f"Document query exceeds the {config.RAG_MAX_QUERY_LENGTH}-character limit.",
                )
            scope_status = _scope_status()
            if scope_status:
                return scope_status
            retriever = store.get_retriever(collection_name, top_k=config.RAG_TOP_K, document_id=document_id)
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
