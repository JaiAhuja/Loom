import logging
import re

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from src.services.ingestion.identity import make_chunk_id, validate_chunk_id, validate_document_id
from src.services.llm import get_embeddings


logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage ChromaDB vector store collections for RAG.

    Handles creating, querying, and managing document collections
    persisted locally in ChromaDB. Each collection is stored in its own
    directory under data/chroma_db/<collection_name>/ for easier traversal.

    """

    def __init__(self, persist_dir: str = None):
        self.base_persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._embeddings = None
        self._clients: dict[str, chromadb.PersistentClient] = {}
        self._stores: dict[str, Chroma] = {}

    @staticmethod
    def _validate_collection_name(collection_name: str) -> str:
        if not isinstance(collection_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,63}", collection_name):
            raise ValueError("collection_name must be 1-63 characters: letters, numbers, '_' or '-'")
        return collection_name

    @staticmethod
    def _normalize_metadata(doc: Document, index: int) -> dict:
        metadata = getattr(doc, "metadata", None)
        if not isinstance(metadata, dict):
            raise ValueError("Every document must have dict metadata")
        normalized = dict(metadata)
        document_id = normalized.get("document_id")
        if not document_id:
            raise ValueError("Every document must include canonical document_id metadata")
        document_id = validate_document_id(document_id)
        normalized["document_id"] = document_id
        chunk_id = normalized.get("chunk_id")
        if not chunk_id:
            label = normalized.get("paper_chunk") or f"chunk_{index:06d}"
            chunk_id = make_chunk_id(document_id, label)
        normalized["chunk_id"] = validate_chunk_id(chunk_id)
        for key, value in normalized.items():
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Metadata field {key!r} must be a scalar value")
        doc.metadata = normalized
        return normalized

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

        self._validate_collection_name(collection_name)

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
        return self._get_client_for_collection("default")

    def get_or_create_store(self, collection_name: str) -> Chroma:
        """Get or create a ChromaDB-backed vector store.

        Args:
            collection_name: Name of the collection.

        Returns:
            LangChain Chroma vector store instance.
        """
        self._validate_collection_name(collection_name)
        store = self._stores.get(collection_name)
        if store is None:
            persist_dir = self._get_collection_persist_dir(collection_name)
            store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
            )
            self._stores[collection_name] = store
        return store

    def add_documents(
        self,
        documents: list[Document],
        collection_name: str = "default",
    ) -> None:
        """Add document chunks to a collection.

        ChromaDB's native **upsert** is used with canonical chunk IDs, making
        repeated ingestion of the same file idempotent (no duplicate rows).

        Args:
            documents: List of LangChain Document objects to store.
            collection_name: Target collection name.
        """
        self._validate_collection_name(collection_name)
        if not documents:
            logger.debug("No documents to add to collection %s", collection_name)
            return

        for index, doc in enumerate(documents):
            self._normalize_metadata(doc, index)

        client = self._get_client_for_collection(collection_name)
        collection = client.get_or_create_collection(collection_name)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        collection.upsert(
            ids=[metadata["chunk_id"] for metadata in metadatas],
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

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
        self._validate_collection_name(collection_name)
        top_k = settings.RAG_TOP_K if top_k is None else top_k
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if document_id is not None:
            validate_document_id(document_id)
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

    def migrate_legacy_metadata(
        self,
        collection_name: str,
        document_ids_by_title: dict[str, str],
    ) -> int:
        """Migrate legacy rows with an explicit title-to-ID mapping.

        Legacy rows are never guessed into a canonical identity. Callers must
        provide the mapping from the old paper/source value to the
        filename-derived ID. Existing Chroma row IDs are preserved.
        """
        self._validate_collection_name(collection_name)
        if not isinstance(document_ids_by_title, dict):
            raise ValueError("document_ids_by_title must be a mapping")
        mapping = {title: validate_document_id(doc_id) for title, doc_id in document_ids_by_title.items()}
        client = self._get_client_for_collection(collection_name)
        collection = client.get_collection(collection_name)
        results = collection.get(include=["metadatas"])
        migrated = 0
        for row_id, metadata in zip(results.get("ids", []), results.get("metadatas", [])):
            if not isinstance(metadata, dict) or metadata.get("document_id"):
                continue
            title = metadata.get("paper") or metadata.get("source")
            canonical_id = mapping.get(title)
            if canonical_id is None:
                logger.warning("Skipping legacy row %s with unmapped title %r", row_id, title)
                continue
            updated = dict(metadata)
            updated["document_id"] = canonical_id
            chunk_id = updated.get("chunk_id")
            try:
                validate_chunk_id(chunk_id)
            except ValueError:
                label = updated.get("paper_chunk") or f"legacy_{row_id}"
                updated["chunk_id"] = make_chunk_id(canonical_id, label)
            collection.update(ids=[row_id], metadatas=[updated])
            migrated += 1
        return migrated

    def list_papers(self, collection_name: str) -> list[dict]:
        """List all distinct papers stored in a collection.

        Papers are keyed by canonical ``document_id`` so the same paper in RAG
        and KG can be joined unambiguously. Legacy rows are omitted until an
        explicit migration maps them.

        Args:
            collection_name: Collection to inspect.

        Returns:
            Sorted list of dicts with keys: ``document_id``, ``title``,
            ``domain``, ``chunk_count``.
        """
        try:
            client = self._get_client_for_collection(collection_name)
            collection = client.get_collection(collection_name)
            results = collection.get(include=["metadatas"])
            by_doc: dict[str, dict] = {}
            for meta in results.get("metadatas", []):
                if not meta:
                    continue
                doc_id = meta.get("document_id")
                if not doc_id:
                    logger.warning("Ignoring legacy Chroma metadata without document_id")
                    continue
                validate_document_id(doc_id)
                title = meta.get("paper") or meta.get("source") or doc_id
                entry = by_doc.setdefault(
                    doc_id,
                    {
                        "document_id": doc_id,
                        "title": title,
                        "domain": meta.get("domain", "Other"),
                        "chunk_count": 0,
                    },
                )
                entry["chunk_count"] += 1
            return sorted(by_doc.values(), key=lambda x: x["title"].lower())
        except Exception:
            logger.debug(
                "Failed to list papers from collection %s",
                collection_name,
                exc_info=True,
            )
            return []

    def get_paper(self, collection_name: str, document_id: str) -> dict | None:
        """Return metadata for one paper without scanning the whole collection.

        This is used by ingestion's already-indexed path, where loading every
        chunk metadata row for every file makes repeated batch ingestion scale
        with the entire collection instead of the matching paper.
        """
        validate_document_id(document_id)
        try:
            client = self._get_client_for_collection(collection_name)
            collection = client.get_collection(collection_name)
            results = collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            metadatas = results.get("metadatas", [])
            if not metadatas:
                return None

            first = next((metadata for metadata in metadatas if metadata), {})
            title = first.get("paper") or first.get("source") or "Unknown"
            return {
                "document_id": document_id,
                "title": title,
                "domain": first.get("domain", "Other"),
                "chunk_count": len(metadatas),
            }
        except Exception:
            logger.debug(
                "Failed to get paper %s from collection %s",
                document_id,
                collection_name,
                exc_info=True,
            )
            return None

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
                entry
                for entry in os.listdir(self.base_persist_dir)
                if os.path.isfile(os.path.join(self.base_persist_dir, entry, "chroma.sqlite3"))
            )
        except Exception:
            logger.debug("Failed to list collections", exc_info=True)
            return []

    def is_document_indexed(self, collection_name: str, document_id: str) -> bool:
        """Return True if the collection already contains chunks for this document_id."""
        validate_document_id(document_id)
        try:
            client = self._get_client_for_collection(collection_name)
            collection = client.get_collection(collection_name)
            results = collection.get(where={"document_id": document_id}, limit=1, include=[])
            return bool(results.get("ids"))
        except Exception:
            logger.debug(
                "Failed to check indexed document %s in collection %s",
                document_id,
                collection_name,
                exc_info=True,
            )
            return False

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

        client = self._get_client_for_collection(collection_name)
        client.delete_collection(collection_name)

        collection_dir = self._get_collection_persist_dir(collection_name)
        if os.path.isdir(collection_dir):
            try:
                shutil.rmtree(collection_dir)
                logger.debug("Deleted collection directory: %s", collection_dir)
            except Exception as e:
                logger.debug("Failed to delete collection directory %s: %s", collection_dir, e)
        self._clients.pop(collection_name, None)
        self._stores.pop(collection_name, None)

    def delete_paper(self, collection_name: str, document_id: str) -> int:
        """Delete all chunks belonging to a single paper.

        Uses the canonical ``document_id`` metadata filter so that every
        chunk of the paper (including the summary chunk) is removed. Legacy
        rows must be migrated before they can be deleted by canonical ID.

        Args:
            collection_name: Collection holding the chunks.
            document_id: Canonical document identity to remove.

        Returns:
            Number of chunks deleted (best-effort).
        """
        validate_document_id(document_id)
        try:
            client = self._get_client_for_collection(collection_name)
            collection = client.get_collection(collection_name)
            existing = collection.get(where={"document_id": document_id})
            ids = existing.get("ids") or []
            if ids:
                collection.delete(ids=ids)
            return len(ids)
        except Exception:
            logger.debug(
                "Failed to delete paper %s from collection %s",
                document_id,
                collection_name,
                exc_info=True,
            )
            return 0
