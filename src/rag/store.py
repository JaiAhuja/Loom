import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from src.llm import get_embeddings


logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage ChromaDB vector store collections for RAG.

    Handles creating, querying, and managing document collections
    persisted locally in ChromaDB.

    Usage:
        store = VectorStoreManager()
        store.add_documents(chunks, collection_name="my_notes")

        retriever = store.get_retriever("my_notes")
        results = retriever.invoke("What is MapReduce?")
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._embeddings = None
        self._client = None

    @property
    def embeddings(self):
        """Lazy-initialize embeddings to avoid startup overhead."""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def client(self):
        """Lazy-initialize the ChromaDB persistent client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def get_or_create_store(self, collection_name: str) -> Chroma:
        """Get or create a ChromaDB-backed vector store.

        Args:
            collection_name: Name of the collection.

        Returns:
            LangChain Chroma vector store instance.
        """
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def add_documents(
        self,
        documents: list[Document],
        collection_name: str = "default",
    ) -> None:
        """Add document chunks to a collection.

        When chunks carry a ``chunk_id`` metadata key the method uses
        ChromaDB's native **upsert**, making repeated ingestion of the
        same file idempotent (no duplicate rows).  Without ``chunk_id``
        it falls back to the LangChain wrapper's default ``add``.

        Args:
            documents: List of LangChain Document objects to store.
            collection_name: Target collection name.
        """
        ids = [doc.metadata.get("chunk_id") for doc in documents]

        if all(ids):
            # Fast path: deterministic IDs → upsert via raw ChromaDB client
            collection = self.client.get_or_create_collection(collection_name)
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)
            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            # Fallback: no chunk_ids, use LangChain wrapper (legacy path)
            store = self.get_or_create_store(collection_name)
            store.add_documents(documents)

    def get_retriever(
        self,
        collection_name: str,
        top_k: int = None,
        document_id: str | None = None,
    ):
        """Get a retriever for a specific collection.

        Args:
            collection_name: Collection to search.
            top_k: Number of results to return. Defaults to settings.RAG_TOP_K.
            document_id: If set, restrict results by canonical ``document_id``.

        Returns:
            LangChain retriever instance.
        """
        top_k = top_k or settings.RAG_TOP_K
        store = self.get_or_create_store(collection_name)
        search_kwargs: dict = {"k": top_k}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}
        return store.as_retriever(search_kwargs=search_kwargs)

    def list_papers(self, collection_name: str) -> list[dict]:
        """List all distinct papers stored in a collection.

        Papers are keyed by ``document_id`` (content-addressed MD5) so the
        same paper in RAG and KG can be joined unambiguously.  When a chunk
        predates identity tracking the filename-derived title is used as a
        synthetic ``document_id`` prefixed with ``legacy:``.

        Args:
            collection_name: Collection to inspect.

        Returns:
            Sorted list of dicts with keys: ``document_id``, ``title``,
            ``paper`` (alias of ``title`` for backwards compat),
            ``domain``, ``chunk_count``.
        """
        try:
            collection = self.client.get_collection(collection_name)
            results = collection.get(include=["metadatas"])
            # document_id -> aggregate dict
            by_doc: dict[str, dict] = {}
            for meta in results.get("metadatas", []):
                if not meta:
                    continue
                title = meta.get("paper") or meta.get("source") or "Unknown"
                doc_id = meta.get("document_id") or f"legacy:{title}"
                entry = by_doc.setdefault(doc_id, {
                    "document_id": doc_id,
                    "title": title,
                    "paper": title,
                    "domain": meta.get("domain", "Other"),
                    "chunk_count": 0,
                })
                entry["chunk_count"] += 1
            return sorted(by_doc.values(), key=lambda x: x["title"].lower())
        except Exception:
            logger.debug("Failed to list papers from collection %s", collection_name, exc_info=True)
            return []

    def list_collections(self) -> list[str]:
        """List all available collection names.

        Returns:
            List of collection name strings.
        """
        try:
            collections = self.client.list_collections()
            if not collections:
                return []
            return [c.name for c in collections]
        except Exception:
            logger.debug("Failed to list collections", exc_info=True)
            return []

    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of documents in a collection.

        Args:
            collection_name: Collection to check.

        Returns:
            Number of stored chunks, or 0 if collection doesn't exist.
        """
        try:
            collection = self.client.get_collection(collection_name)
            return collection.count()
        except Exception:
            logger.debug("Failed to get count for collection %s", collection_name, exc_info=True)
            return 0

    def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection.

        Args:
            collection_name: Collection to delete.
        """
        self.client.delete_collection(collection_name)

    def delete_paper(self, collection_name: str, document_id: str) -> int:
        """Delete all chunks belonging to a single paper.

        Uses the canonical ``document_id`` metadata filter so that every
        chunk of the paper (including the summary chunk) is removed.  Older
        chunks that pre-date identity tracking fall back to matching the
        filename-derived ``paper`` title when ``document_id`` is missing.

        Args:
            collection_name: Collection holding the chunks.
            document_id: Canonical document identity to remove.

        Returns:
            Number of chunks deleted (best-effort).
        """
        try:
            collection = self.client.get_collection(collection_name)
            existing = collection.get(where={"document_id": document_id})
            ids = existing.get("ids") or []
            if not ids and document_id.startswith("legacy:"):
                # Legacy chunks lack document_id — match the title fallback.
                title = document_id[len("legacy:"):]
                existing = collection.get(where={"paper": title})
                ids = existing.get("ids") or []
            if ids:
                collection.delete(ids=ids)
            return len(ids)
        except Exception:
            logger.debug(
                "Failed to delete paper %s from collection %s",
                document_id, collection_name, exc_info=True,
            )
            return 0
