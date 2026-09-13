"""Integration tests for retrieval and knowledge-graph service boundaries."""

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("chromadb")
pytest.importorskip("langchain_chroma")

from langchain_core.documents import Document

from src.services.knowledge_graph import schema
from src.services.knowledge_graph.queries import KnowledgeGraphQueries
from src.services.knowledge_graph.writer import KnowledgeGraphWriter
from src.services.llm import PaperProfile
from src.services.retrieval.processor import DocumentProcessor
from src.services.retrieval.store import VectorStoreManager

pytestmark = pytest.mark.integration


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def upsert(self, ids, documents, metadatas, embeddings):
        for row_id, text, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self.rows[row_id] = {
                "id": row_id,
                "document": text,
                "metadata": metadata,
                "embedding": embedding,
            }

    def get(self, where=None, include=None, limit=None):
        rows = list(self.rows.values())
        if where:
            rows = [row for row in rows if all(row["metadata"].get(k) == v for k, v in where.items())]
        if limit:
            rows = rows[:limit]
        result = {"ids": [row["id"] for row in rows]}
        if include and "metadatas" in include:
            result["metadatas"] = [row["metadata"] for row in rows]
        return result

    def count(self):
        return len(self.rows)

    def delete(self, ids):
        for row_id in ids:
            self.rows.pop(row_id, None)


class FakeClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, FakeCollection())

    def get_collection(self, name):
        return self.collections[name]

    def delete_collection(self, name):
        self.collections.pop(name, None)


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text))] for text in texts]


def test_vector_store_persists_and_manages_document_metadata(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr("src.services.retrieval.store.chromadb.PersistentClient", lambda path: client)
    store = VectorStoreManager(persist_dir=str(tmp_path))
    store._embeddings = FakeEmbeddings()
    documents = [
        Document("summary", {"chunk_id": "one", "document_id": "paper", "paper": "Paper", "domain": "AI"}),
        Document("body", {"chunk_id": "two", "document_id": "paper", "paper": "Paper", "domain": "AI"}),
    ]

    store.add_documents(documents, collection_name="notes")
    assert store.get_collection_count("notes") == 2
    assert store.list_papers("notes") == [{
        "document_id": "paper",
        "title": "Paper",
        "domain": "AI",
        "chunk_count": 2,
    }]
    assert store.is_document_indexed("notes", "paper") is True
    assert store.delete_paper("notes", "paper") == 2
    assert store.get_collection_count("notes") == 0


def test_document_processor_builds_summary_and_deterministic_chunks(monkeypatch, tmp_path):
    class FakeDocument:
        def export_to_markdown(self):
            return "# Paper\nBody"

    class FakeChunker:
        def chunk(self, dl_doc):
            return ["first", "second"]

        def contextualize(self, chunk):
            return f"context: {chunk}"

    monkeypatch.chdir(tmp_path)
    processor = DocumentProcessor()
    processor._converter = SimpleNamespace(convert=lambda path: SimpleNamespace(document=FakeDocument()))
    processor._chunker = FakeChunker()
    progress = []

    result = processor.process(
        "papers/My Paper.pdf",
        extra_metadata={"document_id": "paper-1", "ingest_id": "batch-1"},
        on_step=progress.append,
    )

    assert result["paper_title"] == "My Paper"
    assert result["domain"] == "Other"
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].metadata["paper_chunk"] == "chunk_001"
    assert result["chunks"][0].metadata["document_id"] == "paper-1"
    assert result["chunks"][0].metadata["chunk_id"] == processor.generate_chunk_id("paper-1", "chunk_001")
    assert (tmp_path / "data/txt/My Paper.md").read_text() == "# Paper\nBody"
    assert progress == [
        "Converting PDF → document structure (My Paper.pdf)...",
        "Chunking document (structure-aware)...",
    ]


def test_graph_schema_initialization_and_queries_use_parameterized_contracts():
    calls = []

    class Connection:
        def execute_write(self, query, parameters=None):
            calls.append((query, parameters))

        def execute_read(self, query, parameters=None):
            if query.lstrip().startswith("MATCH (c:Concept)"):
                return [{"name": "attention", "paper_count": 2}]
            return [{"document_id": "paper-1"}]

    connection = Connection()
    schema.initialize_schema(connection)
    assert len(calls) == 12
    assert schema.make_concept_key(" Attention ", "nlp") == "ai:attention"
    assert schema.make_finding_key("Paper", "Claim") == schema.make_finding_key(" paper ", " claim ")

    queries = KnowledgeGraphQueries(connection)
    assert queries.get_all_papers() == [{"document_id": "paper-1"}]
    assert queries.get_all_concepts() == [{"name": "attention", "paper_count": 2}]
    try:
        queries.get_cross_paper_findings("DELETE")
        assert False, "Unsupported relationships must be rejected"
    except ValueError:
        pass


def test_knowledge_graph_writer_persists_profile_entities():
    class Connection:
        def __init__(self):
            self.writes = []

        def execute_write(self, query, parameters=None):
            self.writes.append((query, parameters))

        def execute_write_tx(self, queries):
            self.writes.extend(queries)

        def execute_read(self, query, parameters=None):
            return []

    connection = Connection()
    profile = PaperProfile(
        title="A Paper",
        domain="Machine Learning",
        summary="Summary",
        contributions=[{"text": "Contribution"}],
        concepts=[{"name": "attention", "domain": "Machine Learning"}],
        methods=[{"name": "Transformer"}],
        findings=[{"claim": "It works", "evidence_type": "empirical"}],
    )
    KnowledgeGraphWriter(connection).write_paper_profile(profile, "paper-1")
    assert len(connection.writes) >= 12
    assert any("MERGE (p:Paper" in query for query, _ in connection.writes)
    assert any("PaperDetail" in query for query, _ in connection.writes)
