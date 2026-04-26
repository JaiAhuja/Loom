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


# Keep only letters, digits, dots, hyphens, underscores and spaces.
# Anything else (path separators, NUL, shell metacharacters, …) is replaced
# with an underscore.  Applied before joining an uploaded filename into any
# on-disk path to prevent directory-traversal via crafted ``file.name``.
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._\- ]+")
_MAX_FILENAME_LEN = 200


def _sanitize_filename(name: str) -> str:
    """Return a safe basename suitable for joining with a trusted directory.

    Strips any directory components, drops NUL bytes, replaces path-unsafe
    characters and collapses leading dots (to avoid dotfile / traversal
    tricks such as ``..`` or ``.git``).  Falls back to ``"upload"`` if
    sanitisation leaves nothing usable.
    """
    # Take basename against both separators so Windows paths on POSIX
    # (or vice-versa) are handled consistently.
    base = os.path.basename(name.replace("\\", "/"))
    # Defense-in-depth: replace any residual separator or NUL.
    base = base.replace("/", "_").replace("\x00", "")
    # Collapse leading dots to avoid "..", ".env", etc.
    base = base.lstrip(".")
    # Whitelist-replace everything else.
    base = _SAFE_FILENAME_RE.sub("_", base).strip() or "upload"
    return base[:_MAX_FILENAME_LEN]


@dataclass(frozen=True)
class DocumentIdentity:
    """Immutable identity for a single ingested document."""
    document_id: str       # e.g. "Self-Supervised Learning" (derived from filename, no extension)
    source_md5: str        # hex MD5 of original file bytes (for storage organization)
    ingest_id: str         # UUID-4 for the ingest run
    original_filename: str


def generate_ingest_id() -> str:
    """Return a fresh UUID-4 string for an ingestion run."""
    return str(uuid.uuid4())


def hash_bytes(data: bytes) -> str:
    """Return the hex MD5 of *data*."""
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """Return the hex MD5 of the file at *file_path* (reads in 64 KiB blocks)."""
    h = hashlib.md5(usedforsecurity=False)
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65_536), b""):
            h.update(block)
    return h.hexdigest()


def make_document_id(original_filename: str) -> str:
    """Derive a deterministic document ID from the filename.
    
    This ensures uniqueness based on filename only, avoiding content-based hashing overhead.
    The filename is sanitized to be path-safe.
    """
    safe_name = _sanitize_filename(original_filename)
    # Remove extension for cleaner document_id
    name_without_ext = os.path.splitext(safe_name)[0]
    return name_without_ext or "document"


def build_identity(file_bytes: bytes, original_filename: str, ingest_id: str) -> DocumentIdentity:
    """Construct a DocumentIdentity from raw upload bytes.

    The provided ``original_filename`` is sanitised — path components and
    unsafe characters are stripped — before being stored on the identity
    so every downstream disk write is safe from path-traversal.
    
    The document_id is now based on the filename for simpler, deterministic identity.
    """
    md5 = hash_bytes(file_bytes)  # Still computed for file storage organization
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
    # Re-sanitise defensively in case the identity was constructed by hand.
    safe_name = _sanitize_filename(identity.original_filename)

    base_abs = os.path.abspath(base_dir)
    dest_dir = os.path.join(base_abs, md5[:2], md5)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.abspath(os.path.join(dest_dir, safe_name))

    # Hard boundary check — final path must be inside base_abs.
    if os.path.commonpath([base_abs, dest_path]) != base_abs:
        raise ValueError(f"Refusing to write upload outside base dir: {dest_path!r}")

    if not os.path.exists(dest_path):
        with open(dest_path, "wb") as f:
            f.write(file_bytes)

    return dest_path
