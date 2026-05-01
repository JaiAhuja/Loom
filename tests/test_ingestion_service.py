"""Tests for src.ingestion.service — no Streamlit dependency."""

import re
from unittest.mock import MagicMock

from src.ingestion.service import FileResult, IngestionResult, IngestionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_type="content", paper="Test Paper"):
    """Return a mock LangChain Document with typical metadata."""
    doc = MagicMock()
    doc.metadata = {"chunk_type": chunk_type, "paper": paper}
    return doc


def _fake_processor(chunks=None, paper_title="Test Paper", markdown="# Test", profile=None):
    """Return a mock DocumentProcessor whose .process() returns canned data.

    The mock accepts the extra_metadata kwarg added by the identity step.
    """
    proc = MagicMock()
    if chunks is None:
        chunks = [_make_chunk("content"), _make_chunk("content"), _make_chunk("summary")]
    proc.process.return_value = {
        "chunks": chunks,
        "markdown": markdown,
        "paper_title": paper_title,
        "domain": "Other",
        "profile": profile,
    }
    return proc


def _fake_store(already_indexed: bool = False):
    """Return a mock VectorStoreManager.

    ``already_indexed`` controls what ``is_document_indexed`` returns so tests
    can exercise both the normal processing path and the skip path.
    """
    store = MagicMock()
    store.is_document_indexed.return_value = already_indexed
    return store


# ---------------------------------------------------------------------------
# IngestionResult dataclass
# ---------------------------------------------------------------------------

def test_ingestion_result_counts():
    """succeeded / failed / skipped properties compute correctly."""
    res = IngestionResult(
        file_results=[
            FileResult(file_name="a.pdf", success=True),
            FileResult(file_name="b.pdf", success=False, error="boom"),
            FileResult(file_name="c.pdf", success=True),
            FileResult(file_name="d.pdf", success=True, skipped=True),
        ]
    )
    assert res.succeeded == 3
    assert res.failed == 1
    assert res.skipped == 1


# ---------------------------------------------------------------------------
# ingest_files — RAG-only path (no KG extractor)
# ---------------------------------------------------------------------------

def test_ingest_single_file_rag_only(tmp_path):
    """A single PDF is processed and indexed; results are recorded."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-fake")

    proc = _fake_processor()
    store = _fake_store(already_indexed=False)
    svc = IngestionService(processor=proc, store=store)

    result = svc.ingest_files(
        file_paths=[str(pdf)],
        collection_name="my-collection",
        model="llama3.2",
    )

    assert result.succeeded == 1
    assert result.failed == 0
    # ingest_id is a UUID on the result
    assert re.match(r"^[0-9a-f-]{36}$", result.ingest_id)

    fr = result.file_results[0]
    assert fr.success is True
    assert fr.paper_title == "Test Paper"
    assert fr.content_chunks == 2
    assert fr.has_summary is True

    # Identity fields
    assert fr.identity is not None
    assert not fr.identity.document_id.startswith("md5:")  # Now filename-based
    assert fr.identity.ingest_id == result.ingest_id
    assert fr.identity.original_filename == "paper.pdf"

    # Processor receives extra_metadata with identity keys
    proc.process.assert_called_once()
    _, kwargs = proc.process.call_args
    assert kwargs["model"] == "llama3.2"
    em = kwargs["extra_metadata"]
    assert em["document_id"] == fr.identity.document_id
    assert em["source_md5"] == fr.identity.source_md5
    assert em["ingest_id"] == result.ingest_id

    store.add_documents.assert_called_once()


def test_ingest_multiple_files(tmp_path):
    """Multiple files are processed sequentially and share an ingest_id."""
    files = []
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        p = tmp_path / name
        p.write_bytes(b"%PDF-fake")
        files.append(str(p))

    svc = IngestionService(processor=_fake_processor(), store=_fake_store(already_indexed=False))
    result = svc.ingest_files(file_paths=files, collection_name="col")

    assert result.succeeded == 3
    assert len(result.file_results) == 3

    # All files share the batch ingest_id
    for fr in result.file_results:
        assert fr.identity is not None
        assert fr.identity.ingest_id == result.ingest_id


def test_ingest_processor_failure(tmp_path):
    """When processor.process() raises, the file is marked failed."""
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"%PDF-fake")

    proc = MagicMock()
    proc.process.side_effect = RuntimeError("corrupt PDF")

    svc = IngestionService(processor=proc, store=_fake_store(already_indexed=False))
    result = svc.ingest_files(file_paths=[str(pdf)], collection_name="col")

    assert result.failed == 1
    fr = result.file_results[0]
    assert fr.success is False
    assert fr.error is not None


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

def test_progress_callback_called(tmp_path):
    """The on_progress callback is invoked for each step."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-fake")

    calls = []

    def recorder(current, total, msg):
        calls.append((current, total, msg))

    svc = IngestionService(processor=_fake_processor(), store=_fake_store(already_indexed=False))
    svc.ingest_files(
        file_paths=[str(pdf)],
        collection_name="col",
        on_progress=recorder,
    )

    # hash step (step 0) + up to 3 sub-steps + final "Done!" — at minimum 2 calls
    assert len(calls) >= 2
    assert calls[-1][2] == "Done!"


# ---------------------------------------------------------------------------
# Default collection name
# ---------------------------------------------------------------------------

def test_default_collection_name(tmp_path):
    """When collection_name is omitted, 'default' is used."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-fake")

    store = _fake_store(already_indexed=False)
    svc = IngestionService(processor=_fake_processor(), store=store)
    svc.ingest_files(file_paths=[str(pdf)])

    store.add_documents.assert_called_once()
    _, kwargs = store.add_documents.call_args
    assert kwargs["collection_name"] == "default"


# ---------------------------------------------------------------------------
# Already-indexed skip
# ---------------------------------------------------------------------------

def test_already_indexed_skips_all_processing(tmp_path):
    """When is_document_indexed returns True, no processing or storing occurs."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-fake")

    proc = _fake_processor()
    store = _fake_store(already_indexed=True)
    svc = IngestionService(processor=proc, store=store)

    result = svc.ingest_files(file_paths=[str(pdf)], collection_name="col")

    assert result.succeeded == 1
    assert result.skipped == 1
    assert result.failed == 0

    fr = result.file_results[0]
    assert fr.skipped is True
    assert fr.success is True

    # Expensive steps must NOT have run
    proc.process.assert_not_called()
    store.add_documents.assert_not_called()
