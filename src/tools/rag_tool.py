from langchain_core.tools import tool

from config.settings import settings
from src.rag.store import VectorStoreManager


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

    @tool
    def query_documents(query: str) -> str:
        """Search through uploaded PDF documents for relevant information.

        Use this tool to find specific content from the user's uploaded
        study materials, papers, or textbooks. Returns the most relevant
        excerpts along with source document information.

        Args:
            query: The search query to find relevant document content.
        """
        try:
            docs = retriever.invoke(query)

            if not docs:
                return f"No relevant content found in the uploaded documents{scope_label}."

            formatted = []
            for i, doc in enumerate(docs, 1):
                paper = doc.metadata.get("paper", doc.metadata.get("source", "Unknown"))
                domain = doc.metadata.get("domain", "")
                chunk_id = doc.metadata.get("paper_chunk", "?")
                chunk_type = doc.metadata.get("chunk_type", "content")
                type_badge = " [SUMMARY]" if chunk_type == "summary" else ""
                domain_badge = f" | {domain}" if domain else ""
                formatted.append(
                    f"**Document {i}** (Paper: {paper}{domain_badge}, "
                    f"Chunk: {chunk_id}{type_badge}):\n\n"
                    f"{doc.page_content}"
                )

            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            return f"Document search failed: {str(e)}"

    return query_documents
