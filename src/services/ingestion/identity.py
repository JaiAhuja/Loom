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
_MAX_FILENAME_LEN = 200


class DocumentIdentityError(RuntimeError):
    """Raised when document identity or upload storage cannot be completed."""


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
    return name_without_ext or "document"


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
