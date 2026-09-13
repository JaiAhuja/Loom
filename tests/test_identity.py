"""Tests for src.ingestion.identity — document identity helpers."""

import hashlib
import os

from src.ingestion.identity import (
    DocumentIdentity,
    DocumentIdentityError,
    build_identity,
    compute_file_hash,
    generate_ingest_id,
    hash_bytes,
    make_document_id,
    save_upload,
)


# ---------------------------------------------------------------------------
# hash_bytes / compute_file_hash
# ---------------------------------------------------------------------------

def test_hash_bytes_deterministic():
    """Same bytes always produce the same hash."""
    data = b"hello world"
    expected = hashlib.md5(data).hexdigest()
    assert hash_bytes(data) == expected
    assert hash_bytes(data) == hash_bytes(data)


def test_hash_bytes_different_for_different_data():
    assert hash_bytes(b"aaa") != hash_bytes(b"bbb")


def test_compute_file_hash(tmp_path):
    """compute_file_hash matches hash_bytes on the same content."""
    content = b"PDF content here"
    p = tmp_path / "test.pdf"
    p.write_bytes(content)
    assert compute_file_hash(str(p)) == hash_bytes(content)


def test_compute_file_hash_missing_file_raises_contextual_error(tmp_path):
    missing = tmp_path / "missing.pdf"
    try:
        compute_file_hash(str(missing))
        assert False, "Should have raised DocumentIdentityError"
    except DocumentIdentityError as exc:
        assert "Unable to read file" in str(exc)


# ---------------------------------------------------------------------------
# make_document_id
# ---------------------------------------------------------------------------

def test_make_document_id():
    assert make_document_id("Self-Supervised Learning.pdf") == "Self-Supervised Learning"


# ---------------------------------------------------------------------------
# generate_ingest_id
# ---------------------------------------------------------------------------

def test_generate_ingest_id_unique():
    """Each call returns a different UUID."""
    ids = {generate_ingest_id() for _ in range(50)}
    assert len(ids) == 50


def test_generate_ingest_id_format():
    """The ID looks like a UUID-4 (36-char, 5 groups separated by hyphens)."""
    iid = generate_ingest_id()
    parts = iid.split("-")
    assert len(parts) == 5
    assert len(iid) == 36


# ---------------------------------------------------------------------------
# build_identity
# ---------------------------------------------------------------------------

def test_build_identity():
    data = b"my pdf bytes"
    iid = generate_ingest_id()
    identity = build_identity(data, "paper.pdf", iid)

    assert isinstance(identity, DocumentIdentity)
    assert identity.source_md5 == hash_bytes(data)
    assert identity.document_id == os.path.splitext(identity.original_filename)[0]
    assert identity.ingest_id == iid
    assert identity.original_filename == "paper.pdf"


def test_build_identity_same_filename_same_document_id():
    """Two uploads of the same filename yield the same document_id."""
    data1 = b"content A"
    data2 = b"content B"
    id1 = build_identity(data1, "paper.pdf", generate_ingest_id())
    id2 = build_identity(data2, "paper.pdf", generate_ingest_id())
    # Same filename → same document_id
    assert id1.document_id == id2.document_id == "paper"
    # But source_md5 and ingest_ids differ (file content and ingest run differ)
    assert id1.source_md5 != id2.source_md5
    assert id1.ingest_id != id2.ingest_id


def test_build_identity_different_filename_different_document_id():
    """Different filenames yield different document_ids."""
    data = b"identical content"
    iid = generate_ingest_id()
    id1 = build_identity(data, "paper_a.pdf", iid)
    id2 = build_identity(data, "paper_b.pdf", iid)
    # Different filename → different document_id
    assert id1.document_id != id2.document_id
    # But source_md5 is the same (same content)
    assert id1.source_md5 == id2.source_md5


# ---------------------------------------------------------------------------
# save_upload — content-addressed storage
# ---------------------------------------------------------------------------

def test_save_upload_creates_file(tmp_path):
    data = b"some bytes"
    identity = build_identity(data, "doc.pdf", generate_ingest_id())
    path = save_upload(data, identity, str(tmp_path))

    assert os.path.isfile(path)
    assert path.endswith("doc.pdf")
    with open(path, "rb") as f:
        assert f.read() == data


def test_save_upload_content_addressed_layout(tmp_path):
    """File is stored under <sha[:2]>/<sha>/<filename>."""
    data = b"layout test"
    identity = build_identity(data, "paper.pdf", generate_ingest_id())
    path = save_upload(data, identity, str(tmp_path))

    sha = identity.source_md5
    expected_suffix = os.path.join(sha[:2], sha, "paper.pdf")
    assert path.endswith(expected_suffix)


def test_save_upload_idempotent(tmp_path):
    """Re-uploading the same file is a no-op (same path, same content)."""
    data = b"idempotent"
    identity = build_identity(data, "same.pdf", generate_ingest_id())
    path1 = save_upload(data, identity, str(tmp_path))
    path2 = save_upload(data, identity, str(tmp_path))
    assert path1 == path2


def test_save_upload_different_filenames_same_hash(tmp_path):
    """Same content with different filenames land in the same sha directory."""
    data = b"shared content"
    iid = generate_ingest_id()
    id1 = build_identity(data, "v1.pdf", iid)
    id2 = build_identity(data, "v2.pdf", iid)

    p1 = save_upload(data, id1, str(tmp_path))
    p2 = save_upload(data, id2, str(tmp_path))

    # Same parent directory
    assert os.path.dirname(p1) == os.path.dirname(p2)
    # Different filenames
    assert os.path.basename(p1) == "v1.pdf"
    assert os.path.basename(p2) == "v2.pdf"


def test_save_upload_rejects_hand_built_traversal_identity(tmp_path):
    identity = DocumentIdentity(
        document_id="evil",
        source_md5=hash_bytes(b"payload"),
        ingest_id=generate_ingest_id(),
        original_filename="../../evil.pdf",
    )

    path = save_upload(b"payload", identity, str(tmp_path))
    assert os.path.basename(path) == "evil.pdf"
    assert os.path.commonpath([str(tmp_path), path]) == str(tmp_path)


# ---------------------------------------------------------------------------
# DocumentIdentity is frozen (immutable)
# ---------------------------------------------------------------------------

def test_document_identity_is_immutable():
    identity = build_identity(b"test", "f.pdf", generate_ingest_id())
    try:
        identity.document_id = "changed"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass
