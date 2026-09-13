"""Canonical :class:`Paper` identity.

A single representation of a paper used across the RAG retriever, the
knowledge graph explorer, and the UI listing.  The canonical key is
``document_id`` (derived from filename, without extension), ensuring
uniqueness based on filename only.  ``title`` is a human-readable display
property that may differ between the filename-derived name (legacy RAG) and
the LLM-extracted title (KG); both sides should key on ``document_id`` for
joins.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    """Identity + display metadata for a single paper.

    Attributes
    ----------
    document_id:
        Canonical key (e.g. ``"Self-Supervised Learning"``).  Stable across RAG and KG.
    title:
        Human-readable title.  May be LLM-extracted (preferred) or
        filename-derived (fallback).
    domain:
        Broad domain classification (e.g. ``"Machine Learning"``).
    in_rag:
        ``True`` if the paper has chunks in the active Chroma collection.
    in_kg:
        ``True`` if the paper has a node in the Neo4j knowledge graph.
    chunk_count:
        Number of chunks stored in RAG (``0`` when not in RAG).
    concept_count:
        Number of concepts linked in the KG (``0`` when not in KG).
    """

    document_id: str
    title: str
    domain: str = ""
    in_rag: bool = False
    in_kg: bool = False
    chunk_count: int = 0
    concept_count: int = 0

    @property
    def display_label(self) -> str:
        """Short label suitable for dropdowns: ``"Title  [Domain]"``."""
        if self.domain:
            return f"{self.title}  [{self.domain}]"
        return self.title

    @property
    def status_badge(self) -> str:
        """Compact indicator showing which stores contain this paper."""
        flags = []
        if self.in_rag:
            flags.append("RAG")
        if self.in_kg:
            flags.append("KG")
        return "+".join(flags) if flags else "—"


def merge_paper_sources(
    rag_papers: list[dict] | None = None,
    kg_papers: list[dict] | None = None,
) -> list[Paper]:
    """Merge per-store paper listings into a single set keyed by ``document_id``.

    ``rag_papers`` items are expected to provide at minimum
    ``document_id``, ``title``, ``domain``, ``chunk_count``.  ``kg_papers``
    items should provide ``document_id``, ``title``, ``domain``,
    ``concept_count``.  Missing fields default to empty/0.
    """
    merged: dict[str, dict] = {}

    for r in rag_papers or []:
        doc_id = r.get("document_id")
        if not doc_id:
            continue
        merged[doc_id] = {
            "document_id": doc_id,
            "title": r.get("title") or r.get("paper") or doc_id,
            "domain": r.get("domain", ""),
            "in_rag": True,
            "in_kg": False,
            "chunk_count": r.get("chunk_count", 0),
            "concept_count": 0,
        }

    for k in kg_papers or []:
        doc_id = k.get("document_id")
        if not doc_id:
            continue
        entry = merged.setdefault(doc_id, {
            "document_id": doc_id,
            "title": k.get("title", doc_id),
            "domain": k.get("domain", ""),
            "in_rag": False,
            "in_kg": False,
            "chunk_count": 0,
            "concept_count": 0,
        })
        entry["in_kg"] = True
        entry["concept_count"] = k.get("concept_count", 0)
        if not entry.get("title"):
            entry["title"] = k.get("title", doc_id)
        if not entry.get("domain"):
            entry["domain"] = k.get("domain", "")

    return [Paper(**entry) for entry in merged.values()]
