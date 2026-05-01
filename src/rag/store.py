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
    persisted locally in ChromaDB. Each collection is stored in its own
    directory under data/chroma_db/<collection_name>/ for easier traversal.

    Usage:
        store = VectorStoreManager()
        store.add_documents(chunks, collection_name="my_notes")

        retriever = store.get_retriever("my_notes")
        results = retriever.invoke("What is MapReduce?")
    """

    def __init__(self, persist_dir: str = None):
        self.base_persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._embeddings = None
        self._clients: dict[str, chromadb.PersistentClient] = {}  # One client per collection

    @property
    def embeddings(self):
        """Lazy-initialize embeddings to avoid startup overhead."""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def _get_collection_persist_dir(self, collection_name: str) -> str:
        """Get the persist directory for a specific collection.
        
        Directory structure:
            data/chroma_db/
            ├── collection_1/
            │   ├── chroma.sqlite3
            │   └── <hash-based-dirs>/
            ├── collection_2/
            │   ├── chroma.sqlite3
            │   └── <hash-based-dirs>/
        """
        import os
        collection_dir = os.path.join(self.base_persist_dir, collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        return collection_dir

    def _get_client_for_collection(self, collection_name: str) -> chromadb.PersistentClient:
        """Get or create a ChromaDB PersistentClient for the given collection."""
        if collection_name not in self._clients:
            persist_dir = self._get_collection_persist_dir(collection_name)
            self._clients[collection_name] = chromadb.PersistentClient(path=persist_dir)
        return self._clients[collection_name]

    @property
    def client(self):
        """Lazy-initialize the default ChromaDB persistent client.
        
        Deprecated: Use _get_client_for_collection() instead for collection-specific clients.
        """
        # For backwards compatibility, return a client for "default" collection
        return self._get_client_for_collection("default")

    def get_or_create_store(self, collection_name: str) -> Chroma:
        """Get or create a ChromaDB-backed vector store.

        Args:
            collection_name: Name of the collection.

        Returns:
            LangChain Chroma vector store instance.
        """
        persist_dir = self._get_collection_persist_dir(collection_name)
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
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
            client = self._get_client_for_collection(collection_name)
            collection = client.get_or_create_collection(collection_name)
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
        search_kwargs: dict = {
            "k": top_k,
            "fetch_k": settings.RAG_FETCH_K,
            "lambda_mult": settings.RAG_MMR_LAMBDA,
        }
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}
        return store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

    def list_papers(self, collection_name: str) -> list[dict]:
        """List all distinct papers stored in a collection.

        Papers are keyed by ``document_id`` (derived from filename) so the
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
            client = self._get_client_for_collection(collection_name)
            collection = client.get_collection(collection_name)
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
        
        Returns the list of subdirectories under base_persist_dir that are
        collection directories (contain chroma.sqlite3), each representing
        a collection.

        Returns:
            List of collection name strings, sorted alphabetically.
        """
        import os
        try:
            if not os.path.isdir(self.base_persist_dir):
                return []
            return sorted(
                entry for entry in os.listdir(self.base_persist_dir)
                if os.path.isfile(os.path.join(self.base_persist_dir, entry, "chroma.sqlite3"))
            )
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
            client = self._get_client_for_collection(collection_name)
            collection = client.get_collection(collection_name)
            return collection.count()
        except Exception:
            logger.debug("Failed to get count for collection %s", collection_name, exc_info=True)
            return 0

    def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection and its directory.

        Args:
            collection_name: Collection to delete.
        """
        import os
        import shutil
        
        # Delete from ChromaDB
        client = self._get_client_for_collection(collection_name)
        client.delete_collection(collection_name)
        
        # Clean up the collection directory
        collection_dir = self._get_collection_persist_dir(collection_name)
        if os.path.isdir(collection_dir):
            try:
                shutil.rmtree(collection_dir)
                logger.debug("Deleted collection directory: %s", collection_dir)
            except Exception as e:
                logger.debug("Failed to delete collection directory %s: %s", collection_dir, e)
        self._clients.pop(collection_name, None)

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
            client = self._get_client_for_collection(collection_name)
            collection = client.get_collection(collection_name)
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
