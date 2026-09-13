"""End-to-end tests for the local ingestion, retrieval, graph, and chat flow."""

from dataclasses import dataclass, field

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from src.services.chat import ChatStore
from src.services.ingestion import IngestionService
from src.services.tools.rag_tool import create_rag_tool

pytestmark = pytest.mark.e2e


@dataclass
class InMemoryRetriever:
    documents: list[Document]

    def invoke(self, query: str):
        return [
            doc for doc in self.documents if query.lower() in doc.page_content.lower()
        ]

    async def ainvoke(self, query: str):
        return self.invoke(query)


@dataclass
class InMemoryVectorStore:
    documents: dict[str, list[Document]] = field(default_factory=dict)

    def is_document_indexed(self, collection_name: str, document_id: str) -> bool:
        return any(
            doc.metadata.get("document_id") == document_id
            for doc in self.documents.get(collection_name, [])
        )

    def add_documents(
        self, documents: list[Document], collection_name: str = "default"
    ) -> None:
        self.documents.setdefault(collection_name, []).extend(documents)

    def get_retriever(
        self, collection_name: str, top_k: int, document_id: str | None = None
    ):
        documents = self.documents.get(collection_name, [])
        if document_id:
            documents = [
                doc
                for doc in documents
                if doc.metadata.get("document_id") == document_id
            ]
        return InMemoryRetriever(documents[:top_k])


class FakeProcessor:
    def process(self, file_path, model=None, extra_metadata=None, on_step=None):
        if on_step:
            on_step("converted")
            on_step("indexed")
        metadata = {
            "paper": "Attention Paper",
            "domain": "Machine Learning",
            **(extra_metadata or {}),
        }
        return {
            "chunks": [
                Document(
                    "Attention improves sequence modeling",
                    {**metadata, "chunk_type": "content"},
                ),
                Document("Attention summary", {**metadata, "chunk_type": "summary"}),
            ],
            "markdown": "# Attention Paper\nAttention improves sequence modeling",
            "paper_title": "Attention Paper",
            "domain": "Machine Learning",
            "profile": None,
        }


def test_uploaded_pdf_can_be_ingested_retrieved_and_saved_to_chat(tmp_path):
    pdf = tmp_path / "attention.pdf"
    pdf.write_bytes(b"fake pdf bytes")
    vector_store = InMemoryVectorStore()
    service = IngestionService(processor=FakeProcessor(), store=vector_store)

    result = service.ingest_files([str(pdf)], collection_name="research")

    assert result.succeeded == 1
    document_id = result.file_results[0].identity.document_id
    assert vector_store.is_document_indexed("research", document_id)

    tool = create_rag_tool("research", store=vector_store, document_id=document_id)
    answer = tool.invoke({"query": "sequence"})
    assert "Attention improves sequence modeling" in answer
    assert "Attention Paper" in answer

    chat_store = ChatStore(directory=str(tmp_path / "chat_history"))
    path = chat_store.save(
        [
            {
                "role": "user",
                "content": "What does the paper say about sequence modeling?",
            },
            {"role": "assistant", "content": answer},
        ]
    )
    record = chat_store.load(path.split("/")[-1])
    assert record.messages[-1]["content"] == answer
    assert "Attention Paper" in chat_store.render_markdown(path.split("/")[-1])


def test_reingesting_same_document_skips_processing(tmp_path):
    pdf = tmp_path / "attention.pdf"
    pdf.write_bytes(b"fake pdf bytes")
    processor = FakeProcessor()
    vector_store = InMemoryVectorStore()
    service = IngestionService(processor=processor, store=vector_store)

    first = service.ingest_files([str(pdf)], collection_name="research")
    second = service.ingest_files([str(pdf)], collection_name="research")

    assert first.succeeded == 1
    assert second.skipped == 1
    assert len(vector_store.documents["research"]) == 2
