"""Knowledge-graph writer — upserts a PaperProfile into Neo4j.

Used by the ingestion pipeline to populate the Neo4j graph database
whenever a new PDF is processed and a PaperProfile has been extracted.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.graph_db.connection import Neo4jConnection
from src.graph_db.schema import (
    CONCEPT,
    DISCUSSES,
    FINDING,
    HAS_FINDING,
    METHOD,
    PAPER,
    USES_METHOD,
    initialize_schema,
    make_concept_key,
)

if TYPE_CHECKING:
    from src.llm.paper_profile import PaperProfile

logger = logging.getLogger(__name__)


class KnowledgeGraphWriter:
    """Write PaperProfile data into the Neo4j knowledge graph.

    All write methods are idempotent — safe to call on re-ingestion of the
    same paper.

    Usage::

        writer = KnowledgeGraphWriter(conn)
        writer.write_paper_profile(profile, document_id="my-paper")
    """

    def __init__(self, conn: Neo4jConnection) -> None:
        self._conn = conn
        self._schema_ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_paper_in_kg(self, document_id: str) -> bool:
        """Return True if a Paper node for this document_id already exists in Neo4j."""
        try:
            rows = self._conn.execute_read(
                f"MATCH (p:{PAPER} {{document_id: $id}}) RETURN count(p) AS n",
                {"id": document_id},
            )
            return bool(rows and rows[0]["n"] > 0)
        except Exception:
            return False

    def write_paper_profile(self, profile: "PaperProfile", document_id: str) -> None:
        """Upsert a full PaperProfile into Neo4j.

        Creates or updates the Paper node, its Concept / Method neighbours,
        and its Finding nodes.  Safe to call multiple times for the same
        ``document_id`` (re-ingestion).

        Args:
            profile: LLM-extracted paper profile.
            document_id: Canonical, filename-derived paper identity.
        """
        self._ensure_schema()
        try:
            self._write_paper(profile, document_id)
            self._write_concepts(profile, document_id)
            self._write_methods(profile, document_id)
            self._write_findings(profile, document_id)
            logger.info(
                "KG: wrote profile for document_id=%r  title=%r  "
                "concepts=%d  methods=%d  findings=%d",
                document_id,
                profile.title,
                len(profile.concepts),
                len(profile.methods),
                len(profile.findings),
            )
        except Exception as exc:
            logger.error("KG write failed for document_id=%r: %s", document_id, exc)
            raise

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        if not self._schema_ready:
            initialize_schema(self._conn)
            self._schema_ready = True

    def _write_paper(self, profile: "PaperProfile", document_id: str) -> None:
        authors_str = ", ".join(profile.authors) if profile.authors else ""
        self._conn.execute_write(
            f"""MERGE (p:{PAPER} {{document_id: $document_id}})
            SET p.title      = $title,
                p.domain     = $domain,
                p.summary    = $summary,
                p.year       = $year,
                p.authors_str = $authors_str""",
            {
                "document_id": document_id,
                "title": profile.title or document_id,
                "domain": profile.domain,
                "summary": profile.summary,
                "year": profile.year,
                "authors_str": authors_str,
            },
        )

    def _write_concepts(self, profile: "PaperProfile", document_id: str) -> None:
        for concept in profile.concepts:
            concept_key = make_concept_key(
                concept.name, concept.domain or profile.domain
            )
            self._conn.execute_write(
                f"""MERGE (c:{CONCEPT} {{concept_key: $concept_key}})
                SET c.name        = $name,
                    c.domain      = $domain,
                    c.description = $description
                WITH c
                MATCH (p:{PAPER} {{document_id: $document_id}})
                MERGE (p)-[r:{DISCUSSES}]->(c)
                SET r.depth = $depth""",
                {
                    "concept_key": concept_key,
                    "name": concept.name,
                    "domain": concept.domain or profile.domain,
                    "description": concept.description,
                    "document_id": document_id,
                    "depth": concept.depth,
                },
            )

    def _write_methods(self, profile: "PaperProfile", document_id: str) -> None:
        for method in profile.methods:
            self._conn.execute_write(
                f"""MERGE (m:{METHOD} {{name: $name}})
                SET m.description = $description
                WITH m
                MATCH (p:{PAPER} {{document_id: $document_id}})
                MERGE (p)-[:{USES_METHOD}]->(m)""",
                {
                    "name": method.name,
                    "description": method.description,
                    "document_id": document_id,
                },
            )

    def _write_findings(self, profile: "PaperProfile", document_id: str) -> None:
        # Delete existing findings first so re-ingestion stays idempotent.
        self._conn.execute_write(
            f"""MATCH (p:{PAPER} {{document_id: $document_id}})-[:{HAS_FINDING}]->(f:{FINDING})
            DETACH DELETE f""",
            {"document_id": document_id},
        )
        paper_title = profile.title or document_id
        for finding in profile.findings:
            self._conn.execute_write(
                f"""MATCH (p:{PAPER} {{document_id: $document_id}})
                CREATE (f:{FINDING} {{
                    claim:          $claim,
                    evidence_type:  $evidence_type,
                    paper_title:    $paper_title
                }})
                CREATE (p)-[:{HAS_FINDING}]->(f)""",
                {
                    "document_id": document_id,
                    "claim": finding.claim,
                    "evidence_type": finding.evidence_type,
                    "paper_title": paper_title,
                },
            )
