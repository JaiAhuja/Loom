"""Ingestion service — orchestrates PDF processing and RAG indexing.

This module contains no Streamlit dependency so it can be tested and
called from CLI scripts, notebooks, or the Streamlit app alike.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from src.services.ingestion.identity import (
    DocumentIdentity,
    compute_file_hash,
    generate_ingest_id,
    make_document_id,
)

if TYPE_CHECKING:
    from src.services.knowledge_graph.connection import Neo4jConnection
    from src.services.retrieval.processor import DocumentProcessor
    from src.services.retrieval.store import VectorStoreManager

logger = logging.getLogger(__name__)

_TXT_DIR = os.path.join("data", "txt")
_DETAIL_FIELDS = (
    "contributions",
    "stands_for",
    "builds_on",
    "does_not_support",
    "limitations",
)


def _safe_sidecar_stem(document_id: str) -> str:
    """Return a path-safe sidecar stem for a document id."""
    return os.path.basename(str(document_id or "document").replace("\\", "/")) or "document"


def _profile_sidecar_path(document_id: str) -> str:
    """Return the path for the JSON sidecar of a PaperProfile."""
    return os.path.join(_TXT_DIR, f"{_safe_sidecar_stem(document_id)}_profile.json")


def _save_profile_sidecar(profile, document_id: str) -> None:
    """Persist a PaperProfile as a JSON sidecar alongside the Markdown cache.

    Silently swallows write errors so a disk hiccup never fails an ingestion run.
    """
    try:
        os.makedirs(_TXT_DIR, exist_ok=True)
        path = _profile_sidecar_path(document_id)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(profile.model_dump_json(indent=2))
    except Exception as exc:
        logger.warning("Could not save profile sidecar for %s: %s", document_id, exc)


def _load_profile_sidecar(document_id: str):
    """Load a previously saved PaperProfile sidecar.  Returns None if absent."""
    from src.services.llm.paper_profile import PaperProfile

    path = _profile_sidecar_path(document_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return PaperProfile.model_validate_json(fh.read())
    except Exception as exc:
        logger.warning("Could not load profile sidecar %s: %s", path, exc)
        return None


def _profile_has_details(profile) -> bool:
    """Return True if the profile contains any typed paper-detail entries."""
    return bool(profile) and any(getattr(profile, field_name, None) for field_name in _DETAIL_FIELDS)


@dataclass
class FileResult:
    """Outcome of ingesting a single file."""

    file_name: str
    success: bool = False
    skipped: bool = False
    paper_title: str = ""
    content_chunks: int = 0
    has_summary: bool = False
    kg_indexed: bool = False
    error: Optional[str] = None
    identity: Optional[DocumentIdentity] = None


@dataclass
class IngestionResult:
    """Aggregate outcome of an ingestion run."""

    ingest_id: str = ""
    file_results: list[FileResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.file_results if r.success)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.file_results if not r.success and not r.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.file_results if r.skipped)


ProgressCallback = Callable[[int, int, str], None]

_SUB_STEPS = 4


class IngestionService:
    """Coordinate PDF → RAG chunks (and optionally the Neo4j knowledge graph)."""

    def __init__(
        self,
        processor: "DocumentProcessor",
        store: "VectorStoreManager",
        neo4j_conn: "Neo4jConnection | None" = None,
    ):
        self.processor = processor
        self.store = store
        self._kg_writer = None
        if neo4j_conn is not None:
            from src.services.knowledge_graph.writer import KnowledgeGraphWriter

            self._kg_writer = KnowledgeGraphWriter(neo4j_conn)

    def ingest_files(
        self,
        file_paths: list[str],
        collection_name: str = "default",
        model: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        """Process and index a batch of PDF files.

        Parameters
        ----------
        file_paths:
            Absolute or relative paths to PDF files already on disk.
        collection_name:
            Target ChromaDB collection.
        model:
            Ollama model name forwarded to the processor / KG extractor.
        on_progress:
            Optional callback ``(current, total, message)`` invoked before
            each major step so callers (e.g. a Streamlit progress bar) can
            report status.  ``total`` equals ``len(file_paths) * _SUB_STEPS``
            so the bar advances smoothly through PDF conversion, LLM
            profiling, and chunking rather than jumping file-by-file.

        Returns
        -------
        IngestionResult
            Per-file outcomes plus aggregate counts.
        """
        ingest_id = generate_ingest_id()
        n_files = len(file_paths)
        total_steps = n_files * _SUB_STEPS
        result = IngestionResult(ingest_id=ingest_id)

        for idx, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            step_base = idx * _SUB_STEPS
            fr = FileResult(file_name=file_name)

            sub_step = 1

            def _step_cb(msg: str) -> None:
                nonlocal sub_step
                full_msg = f"[{idx + 1}/{n_files}] {msg}"
                print(f"  {full_msg}", flush=True)
                if on_progress:
                    on_progress(step_base + sub_step, total_steps, full_msg)
                sub_step = min(sub_step + 1, _SUB_STEPS - 1)

            hash_msg = f"[{idx + 1}/{n_files}] Computing hash for {file_name}..."
            print(f"\n{hash_msg}", flush=True)
            if on_progress:
                on_progress(step_base, total_steps, hash_msg)

            try:
                sha = compute_file_hash(file_path)
                identity = DocumentIdentity(
                    document_id=make_document_id(file_name),
                    source_md5=sha,
                    ingest_id=ingest_id,
                    original_filename=file_name,
                )
                fr.identity = identity
            except Exception as exc:
                logger.error("Hash computation failed for %s: %s", file_path, exc)
                fr.error = f"Hash computation failed for {file_name}"
                result.file_results.append(fr)
                continue

            if self.store.is_document_indexed(collection_name, identity.document_id):
                fr.skipped = True
                fr.success = True

                paper_meta = self.store.get_paper(collection_name, identity.document_id)
                fr.paper_title = paper_meta["title"] if paper_meta else identity.document_id
                fr.content_chunks = paper_meta["chunk_count"] if paper_meta else 0

                kg_needs_write = False
                if self._kg_writer is not None:
                    in_kg = self._kg_writer.is_paper_in_kg(identity.document_id)
                    kg_needs_write = not in_kg or (
                        in_kg and not self._kg_writer.has_paper_details(identity.document_id)
                    )

                if self._kg_writer is not None and kg_needs_write:
                    skip_msg = f"[{idx + 1}/{n_files}] RAG already indexed — writing KG for {file_name}"
                    print(f"\n{skip_msg}", flush=True)
                    if on_progress:
                        on_progress(step_base + _SUB_STEPS, total_steps, skip_msg)
                    fr.kg_indexed = self._write_kg_only(file_path, file_name, model, identity.document_id)
                else:
                    skip_msg = f"[{idx + 1}/{n_files}] Already indexed — skipping {file_name}"
                    print(f"\n{skip_msg}", flush=True)
                    if on_progress:
                        on_progress(step_base + _SUB_STEPS, total_steps, skip_msg)
                    if self._kg_writer is not None:
                        fr.kg_indexed = self._kg_writer.is_paper_in_kg(identity.document_id)

                result.file_results.append(fr)
                continue

            extra_metadata = {
                "document_id": identity.document_id,
                "source_md5": identity.source_md5,
                "ingest_id": ingest_id,
            }

            proc_result = self._process_and_index(
                file_path,
                collection_name,
                model,
                extra_metadata=extra_metadata,
                on_step=_step_cb,
            )
            if proc_result is None:
                fr.error = f"Processing failed for {file_name}"
                result.file_results.append(fr)
                continue

            chunks, raw_result = proc_result
            content_count = 0
            has_summary = False
            for chunk in chunks:
                chunk_type = chunk.metadata.get("chunk_type")
                content_count += chunk_type == "content"
                has_summary = has_summary or chunk_type == "summary"
            fr.success = True
            fr.paper_title = raw_result.get("paper_title", file_name)
            fr.content_chunks = content_count
            fr.has_summary = has_summary

            profile = raw_result.get("profile")
            if profile is not None:
                _save_profile_sidecar(profile, identity.document_id)
            if self._kg_writer is not None and profile is not None:
                try:
                    self._kg_writer.write_paper_profile(profile, identity.document_id)
                    fr.kg_indexed = True
                    self._link_cross_edges(profile, identity.document_id, model)
                except Exception as kg_exc:
                    logger.warning(
                        "KG indexing failed for %s (RAG indexing succeeded): %s",
                        file_path,
                        kg_exc,
                    )

            result.file_results.append(fr)

        if on_progress:
            on_progress(total_steps, total_steps, "Done!")

        return result

    def _process_and_index(
        self,
        file_path: str,
        collection_name: str,
        model: str | None,
        extra_metadata: dict | None = None,
        on_step=None,
    ):
        """Run the processor and store chunks in the RAG vector store.

        Returns ``(chunks, raw_result)`` or *None* on failure.  KG writing
        is intentionally NOT done here so the caller can set ``fr.kg_indexed``
        accurately from the outcome.
        """
        try:
            raw_result = self.processor.process(
                file_path,
                model=model,
                extra_metadata=extra_metadata,
                on_step=on_step,
            )
            chunks = raw_result["chunks"]
            self.store.add_documents(chunks, collection_name=collection_name)
            return chunks, raw_result
        except Exception as exc:
            logger.error("Ingestion failed for %s: %s", file_path, exc)
            return None

    def _link_cross_edges(self, profile, document_id: str, model: str | None) -> None:
        if model is None:
            return
        for label, method_name in (
            ("finding", "link_findings"),
            ("concept", "link_concepts"),
        ):
            try:
                n_links = getattr(self._kg_writer, method_name)(profile, document_id, model)
                if n_links:
                    logger.info(
                        "KG: wrote %d cross-%s edge(s) for %s",
                        n_links,
                        label,
                        document_id,
                    )
            except Exception as exc:
                logger.warning("Cross-%s linking failed for %s: %s", label, document_id, exc)

    def _write_kg_only(
        self,
        file_path: str,
        file_name: str,
        model: str | None,
        document_id: str,
    ) -> bool:
        """Write or refresh KG data when RAG was already indexed.

        Prefers the saved PaperProfile sidecar (written during the original ingest)
        over re-running the LLM.  Falls back to LLM re-extraction from the cached
        Markdown when the sidecar is absent or predates newer detail fields and
        a model is available.

        Returns True on success, False on any failure.
        """
        if self._kg_writer is None:
            return False

        profile = _load_profile_sidecar(document_id)

        should_reextract = profile is None or (not _profile_has_details(profile) and model is not None)
        if should_reextract:
            if model is None:
                logger.warning(
                    "No complete profile sidecar for %s and no model provided; skipping KG write.",
                    document_id,
                )
                return False

            stem = os.path.splitext(file_name)[0]
            txt_path = os.path.join(_TXT_DIR, f"{stem}.md")
            if not os.path.isfile(txt_path):
                logger.warning(
                    "Sidecar and Markdown both absent for %s; cannot write KG.",
                    document_id,
                )
                return False

            try:
                with open(txt_path, "r", encoding="utf-8") as fh:
                    markdown_text = fh.read()
            except OSError as exc:
                logger.warning("Could not read Markdown %s: %s", txt_path, exc)
                return False

            from src.services.llm import extract_paper_profile, get_llm

            llm = get_llm(model=model, temperature=0.1, require_json=True)
            profile = extract_paper_profile(llm, markdown_text, file_name)
            if profile is None:
                logger.warning("Profile extraction returned None for %s", document_id)
                return False

            _save_profile_sidecar(profile, document_id)

        try:
            self._kg_writer.write_paper_profile(profile, document_id)
            self._link_cross_edges(profile, document_id, model)
            return True
        except Exception as exc:
            logger.warning("KG write failed for %s: %s", document_id, exc)
            return False
