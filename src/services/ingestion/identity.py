"""Document identity helpers — filename-based identity and path management.

Document IDs are now derived from filenames (not content hashes) for simpler,
more intuitive uniqueness. Files are still organized by MD5 on disk for deduplication.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._\- ]+")
_DOCUMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\- ]{0,199}\Z")
_CHUNK_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_MAX_FILENAME_LEN = 200


class DocumentIdentityError(RuntimeError):
    """Raised when document identity or upload storage cannot be completed."""


class MetadataIntegrityError(ValueError):
    """Raised when stored metadata cannot identify a document chunk."""


def validate_document_id(document_id: str) -> str:
    """Validate and return the canonical cross-store document identity."""
    if not isinstance(document_id, str) or not _DOCUMENT_ID_RE.fullmatch(document_id):
        raise ValueError(
            "document_id must be 1-200 characters and contain only letters, numbers, '.', '-', '_', or spaces"
        )
    return document_id


def validate_chunk_id(chunk_id: str) -> str:
    """Validate a deterministic content chunk ID."""
    if not isinstance(chunk_id, str) or not _CHUNK_ID_RE.fullmatch(chunk_id):
        raise ValueError("chunk_id must be a 32-character lowercase hexadecimal ID")
    return chunk_id


def validate_metadata_identity(metadata: dict) -> dict:
    """Validate the canonical metadata contract shared by RAG and ingestion."""
    if not isinstance(metadata, dict):
        raise MetadataIntegrityError("document metadata must be a dictionary")
    try:
        validate_document_id(metadata.get("document_id"))
        validate_chunk_id(metadata.get("chunk_id"))
    except ValueError as exc:
        raise MetadataIntegrityError(f"invalid canonical document metadata: {exc}") from exc
    for key in ("source_md5", "ingest_id"):
        if key in metadata and (not isinstance(metadata[key], str) or not metadata[key].strip()):
            raise MetadataIntegrityError(f"metadata field {key!r} must be a non-empty string")
    if "paper_chunk" in metadata and (
        not isinstance(metadata["paper_chunk"], str) or not metadata["paper_chunk"].strip()
    ):
        raise MetadataIntegrityError("metadata field 'paper_chunk' must be a non-empty string")
    return metadata


def make_chunk_id(document_id: str, chunk_label: str) -> str:
    """Create the canonical deterministic ID for one document chunk."""
    validate_document_id(document_id)
    if not isinstance(chunk_label, str) or not chunk_label.strip():
        raise ValueError("chunk label must be non-empty")
    return hashlib.md5(f"{document_id}:{chunk_label}".encode(), usedforsecurity=False).hexdigest()


def _sanitize_filename(name: str) -> str:
    """Return a safe basename, stripping path components and unsafe characters."""
    base = os.path.basename(name.replace("\\", "/"))
    base = base.replace("/", "_").replace("\x00", "")
    base = base.lstrip(".")
    base = _SAFE_FILENAME_RE.sub("_", base).strip() or "upload"
    return base[:_MAX_FILENAME_LEN]


@dataclass(frozen=True)
class DocumentIdentity:
    """Immutable identity for a single ingested document."""

    document_id: str
    source_md5: str
    ingest_id: str
    original_filename: str


def generate_ingest_id() -> str:
    return str(uuid.uuid4())


def hash_bytes(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """Return the hex MD5 of the file at file_path, reading in 64 KiB blocks."""
    h = hashlib.md5(usedforsecurity=False)
    try:
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65_536), b""):
                h.update(block)
    except OSError as exc:
        raise DocumentIdentityError(f"Unable to read file for hashing: {file_path!r}") from exc
    return h.hexdigest()


def make_document_id(original_filename: str) -> str:
    """Derive a document ID from the filename (no extension)."""
    safe_name = _sanitize_filename(original_filename)
    name_without_ext = os.path.splitext(safe_name)[0]
    return validate_document_id(name_without_ext or "document")


def build_identity(file_bytes: bytes, original_filename: str, ingest_id: str) -> DocumentIdentity:
    """Construct a DocumentIdentity from raw upload bytes.

    The provided ``original_filename`` is sanitised — path components and
    unsafe characters are stripped — before being stored on the identity
    so every downstream disk write is safe from path-traversal.

    The document_id is now based on the filename for simpler, deterministic identity.
    """
    md5 = hash_bytes(file_bytes)
    safe_filename = _sanitize_filename(original_filename)
    return DocumentIdentity(
        document_id=make_document_id(original_filename),
        source_md5=md5,
        ingest_id=ingest_id,
        original_filename=safe_filename,
    )


def save_upload(file_bytes: bytes, identity: DocumentIdentity, base_dir: str) -> str:
    """Persist upload bytes under a content-addressed path.

    Layout: ``<base_dir>/<md5[:2]>/<md5>/<original_filename>``.

    The final destination is verified to live *within* ``base_dir`` — any
    attempt at path-traversal (symlink races or a filename that slipped
    past sanitisation) raises :class:`ValueError`.
    """
    md5 = identity.source_md5
    safe_name = _sanitize_filename(identity.original_filename)

    try:
        base_path = Path(base_dir).resolve()
        dest_dir = base_path / md5[:2] / md5
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = (dest_dir / safe_name).resolve()
    except OSError as exc:
        raise DocumentIdentityError(f"Unable to prepare upload directory: {base_dir!r}") from exc

    try:
        dest_path.relative_to(base_path)
    except ValueError as exc:
        raise ValueError(f"Refusing to write upload outside base dir: {str(dest_path)!r}") from exc

    try:
        if not dest_path.exists():
            with open(dest_path, "wb") as f:
                f.write(file_bytes)
    except OSError as exc:
        raise DocumentIdentityError(f"Unable to save upload: {str(dest_path)!r}") from exc

    return str(dest_path)
