"""Knowledge-graph writer — upserts a PaperProfile into Neo4j.

Used by the ingestion pipeline to populate the Neo4j graph database
whenever a new PDF is processed and a PaperProfile has been extracted.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

from src.graph_db.connection import Neo4jConnection
from src.graph_db.schema import (
    CONCEPT,
    CONTRADICTS,
    CROSS_CONCEPT_RELS,
    CROSS_FINDING_RELS,
    DISCUSSES,
    EXTENDS,
    FINDING,
    HAS_FINDING,
    METHOD,
    PAPER,
    RELATED_TO,
    SUBTOPIC_OF,
    SUPPORTS,
    USES_METHOD,
    initialize_schema,
    make_concept_key,
    make_finding_key,
)

if TYPE_CHECKING:
    from src.llm.paper_profile import PaperProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template for cross-finding relationship detection
# ---------------------------------------------------------------------------

_FINDING_LINKS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a scientific knowledge graph assistant specialising in cross-paper relationship detection.

Your task is to compare two sets of research findings and identify semantic relationships between them.

RELATIONSHIP TYPES:
- SUPPORTS: The new finding confirms or reinforces the existing finding.
- CONTRADICTS: The new finding directly opposes the existing finding.
- EXTENDS: The new finding builds on, broadens, or adds nuance to the existing finding.

OUTPUT INSTRUCTIONS:
- Return ONLY a valid JSON array.
- Each element must be an object with exactly three keys:
    "source" — the finding_key of the new finding
    "target" — the finding_key of the existing finding
    "relation" — one of SUPPORTS, CONTRADICTS, or EXTENDS
- If no relationships exist, return an empty array: []
- No markdown fences, no prose, no commentary — raw JSON only."""
    ),
    (
        "human",
        """NEW FINDINGS (just added to the graph):
{new_findings_json}

EXISTING FINDINGS (from other papers already in the graph):
{existing_findings_json}

Identify all pairs where a new finding SUPPORTS, CONTRADICTS, or EXTENDS an existing finding."""
    ),
])

# ---------------------------------------------------------------------------
# Prompt template for cross-concept relationship detection
# ---------------------------------------------------------------------------

_CONCEPT_LINKS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a scientific knowledge graph assistant specialising in concept relationship detection.

Your task is to compare two sets of research concepts and identify semantic relationships between them.

RELATIONSHIP TYPES:
- RELATED_TO: The concepts are related but neither is prerequisite to or subsumed by the other (e.g., "attention" RELATED_TO "optimization").
- SUBTOPIC_OF: The new concept is a specific instance or subtype of an existing concept (e.g., "self-attention" SUBTOPIC_OF "attention").
- EXTENDS: The new concept builds on or extends an existing concept with additional nuance or technique (e.g., "multi-head attention" EXTENDS "self-attention").

GUIDANCE:
- Only suggest relationships where the concepts are semantically related in the academic domain.
- Avoid spurious links between unrelated concepts.
- Use domain context (the "domain" field) to disambiguate.

OUTPUT INSTRUCTIONS:
- Return ONLY a valid JSON array.
- Each element must be an object with exactly three keys:
    "source" — the concept_key of the new concept
    "target" — the concept_key of the existing concept
    "relation" — one of RELATED_TO, SUBTOPIC_OF, or EXTENDS
- If no relationships exist, return an empty array: []
- No markdown fences, no prose, no commentary — raw JSON only."""
    ),
    (
        "human",
        """NEW CONCEPTS (just added to the graph):
{new_concepts_json}

EXISTING CONCEPTS (from other papers already in the graph, same domain):
{existing_concepts_json}

Identify all pairs where a new concept is RELATED_TO, SUBTOPIC_OF, or EXTENDS an existing concept. 
Focus on scientifically meaningful relationships."""
    ),
])


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

    def link_findings(
        self,
        profile: "PaperProfile",
        document_id: str,
        model: str,
    ) -> int:
        """Detect cross-paper SUPPORTS/CONTRADICTS/EXTENDS relationships between Finding nodes.

        Compares the findings in *profile* against all findings already in the graph
        from *other* papers.  One LLM call produces a JSON list of ``{source,
        target, relation}`` triples that are then written as directed edges.

        Args:
            profile: The paper profile whose findings were just written.
            document_id: The paper’s document_id (used to exclude its own findings).
            model: Ollama model name used for the comparison LLM call.

        Returns:
            Number of cross-finding edges created.  Returns 0 if there are no
            existing findings to compare against or if the LLM returns nothing useful.
        """
        paper_title = profile.title or document_id
        new_findings = [
            {
                "finding_key": make_finding_key(paper_title, f.claim),
                "claim": f.claim,
                "evidence_type": f.evidence_type,
            }
            for f in profile.findings
        ]
        if not new_findings:
            return 0

        # Fetch existing findings from *other* papers.
        rows = self._conn.execute_read(
            f"""MATCH (p:{PAPER})-[:{HAS_FINDING}]->(f:{FINDING})
            WHERE p.document_id <> $document_id
            RETURN f.finding_key AS finding_key,
                   f.claim        AS claim,
                   f.evidence_type AS evidence_type,
                   p.title        AS paper_title
            LIMIT 200""",
            {"document_id": document_id},
        )
        if not rows:
            return 0

        existing_findings = [
            {
                "finding_key": r["finding_key"],
                "claim": r["claim"],
                "evidence_type": r["evidence_type"],
                "paper_title": r["paper_title"],
            }
            for r in rows
            if r.get("finding_key")  # skip legacy nodes without a key
        ]
        if not existing_findings:
            return 0

        relations = self._call_llm_for_finding_links(
            new_findings, existing_findings, model
        )
        if not relations:
            return 0

        return self._write_finding_links(relations)

    def link_concepts(
        self,
        profile: "PaperProfile",
        document_id: str,
        model: str,
    ) -> int:
        """Detect cross-concept RELATED_TO/SUBTOPIC_OF/EXTENDS relationships.

        Compares the concepts in *profile* against all concepts already in the graph
        from *other* papers within the same domain.  One LLM call produces a JSON
        list of ``{source, target, relation}`` triples that are then written as
        directed edges.

        Args:
            profile: The paper profile whose concepts were just written.
            document_id: The paper's document_id (used to exclude its own concepts).
            model: Ollama model name used for the comparison LLM call.

        Returns:
            Number of cross-concept edges created.  Returns 0 if there are no
            existing concepts to compare against (in the same domain) or if the
            LLM returns nothing useful.
        """
        new_concepts = [
            {
                "concept_key": make_concept_key(c.name, c.domain or profile.domain),
                "name": c.name,
                "description": c.description,
                "domain": c.domain or profile.domain,
                "depth": c.depth,
            }
            for c in profile.concepts
        ]
        if not new_concepts:
            return 0

        # Fetch existing concepts from *other* papers in the SAME domain.
        # Only consider concepts from papers with the same domain to keep the
        # comparison semantically meaningful.
        rows = self._conn.execute_read(
            f"""MATCH (p1:{PAPER})-[:{DISCUSSES}]->(c:{CONCEPT})
            WHERE p1.domain = $domain AND p1.document_id <> $document_id
            RETURN DISTINCT c.concept_key AS concept_key,
                           c.name        AS name,
                           c.description AS description,
                           c.domain      AS domain,
                           count(DISTINCT p1) AS paper_count
            LIMIT 200""",
            {"domain": profile.domain, "document_id": document_id},
        )
        if not rows:
            return 0

        existing_concepts = [
            {
                "concept_key": r["concept_key"],
                "name": r["name"],
                "description": r["description"],
                "domain": r["domain"],
                "paper_count": r["paper_count"],
            }
            for r in rows
            if r.get("concept_key")  # skip legacy nodes without a key
        ]
        if not existing_concepts:
            return 0

        relations = self._call_llm_for_concept_links(
            new_concepts, existing_concepts, model
        )
        if not relations:
            return 0

        return self._write_concept_links(relations)

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
        """Upsert all concepts and DISCUSSES edges in a single round-trip via UNWIND."""
        if not profile.concepts:
            return
        items = [
            {
                "concept_key": make_concept_key(c.name, c.domain or profile.domain),
                "name": c.name,
                "domain": c.domain or profile.domain,
                "description": c.description,
                "depth": c.depth,
            }
            for c in profile.concepts
        ]
        self._conn.execute_write(
            f"""UNWIND $items AS item
            MERGE (c:{CONCEPT} {{concept_key: item.concept_key}})
            SET c.name        = item.name,
                c.domain      = item.domain,
                c.description = item.description
            WITH c, item
            MATCH (p:{PAPER} {{document_id: $document_id}})
            MERGE (p)-[r:{DISCUSSES}]->(c)
            SET r.depth = item.depth""",
            {"items": items, "document_id": document_id},
        )

    def _write_methods(self, profile: "PaperProfile", document_id: str) -> None:
        """Upsert all methods and USES_METHOD edges in a single round-trip via UNWIND."""
        if not profile.methods:
            return
        items = [
            {"name": m.name, "description": m.description}
            for m in profile.methods
        ]
        self._conn.execute_write(
            f"""UNWIND $items AS item
            MERGE (m:{METHOD} {{name: item.name}})
            SET m.description = item.description
            WITH m
            MATCH (p:{PAPER} {{document_id: $document_id}})
            MERGE (p)-[:{USES_METHOD}]->(m)""",
            {"items": items, "document_id": document_id},
        )

    def _write_findings(self, profile: "PaperProfile", document_id: str) -> None:
        """Delete existing findings and create new ones in a single transaction.

        Each Finding node is stamped with a stable ``finding_key`` derived from
        the paper title + claim so cross-finding edges can reference nodes by a
        reliable identity handle.

        Batching the DELETE and all CREATEs into one ``execute_write_tx`` call
        prevents data loss if the process crashes between the delete and the
        inserts (the old behaviour left the paper with zero findings).
        """
        paper_title = profile.title or document_id

        # Build the full batch: DELETE first, then one CREATE per finding.
        queries: list[tuple[str, dict]] = [
            (
                f"""MATCH (p:{PAPER} {{document_id: $document_id}})-[:{HAS_FINDING}]->(f:{FINDING})
                DETACH DELETE f""",
                {"document_id": document_id},
            )
        ]
        for finding in profile.findings:
            finding_key = make_finding_key(paper_title, finding.claim)
            queries.append((
                f"""MATCH (p:{PAPER} {{document_id: $document_id}})
                CREATE (f:{FINDING} {{
                    finding_key:    $finding_key,
                    claim:          $claim,
                    evidence_type:  $evidence_type,
                    paper_title:    $paper_title
                }})
                CREATE (p)-[:{HAS_FINDING}]->(f)""",
                {
                    "document_id": document_id,
                    "finding_key": finding_key,
                    "claim": finding.claim,
                    "evidence_type": finding.evidence_type,
                    "paper_title": paper_title,
                },
            ))

        self._conn.execute_write_tx(queries)

    def _call_llm_for_finding_links(
        self,
        new_findings: list[dict],
        existing_findings: list[dict],
        model: str,
    ) -> list[dict]:
        """Ask an LLM to identify SUPPORTS/CONTRADICTS/EXTENDS pairs.

        Uses the module-level ``_FINDING_LINKS_PROMPT`` (a ``ChatPromptTemplate``)
        so the system role and human content are clearly separated and composable.

        Returns a (possibly empty) list of ``{source, target, relation}`` dicts
        where ``source`` and ``target`` are ``finding_key`` strings.
        """
        import json
        from src.llm import get_llm
        from src.utils.json_parser import parse_llm_json

        llm = get_llm(model=model, temperature=0.0, require_json=True)
        chain = _FINDING_LINKS_PROMPT | llm

        try:
            response = chain.invoke({
                "new_findings_json": json.dumps(new_findings, indent=2),
                "existing_findings_json": json.dumps(existing_findings, indent=2),
            })
            content = response.content
            parsed = parse_llm_json(content)
            if not isinstance(parsed, list):
                return []
            valid = [
                r for r in parsed
                if isinstance(r, dict)
                and r.get("relation") in CROSS_FINDING_RELS
                and r.get("source")
                and r.get("target")
            ]
            return valid
        except Exception as exc:
            logger.warning("LLM call for finding links failed: %s", exc)
            return []

    def _write_finding_links(self, relations: list[dict]) -> int:
        """Write cross-finding edges returned by the LLM.

        Uses MERGE so repeated ingestion does not create duplicate edges.
        Returns the number of edge write-attempts (some may merge onto existing).
        """
        queries: list[tuple[str, dict]] = []
        for rel in relations:
            rel_type = rel["relation"]  # already validated in CROSS_FINDING_RELS
            queries.append((
                f"""MATCH (src:{FINDING} {{finding_key: $source}})
                MATCH (tgt:{FINDING} {{finding_key: $target}})
                MERGE (src)-[:{rel_type}]->(tgt)""",
                {"source": rel["source"], "target": rel["target"]},
            ))
        if not queries:
            return 0
        try:
            self._conn.execute_write_tx(queries)
            return len(queries)
        except Exception as exc:
            logger.warning("Writing finding links failed: %s", exc)
            return 0

    def _call_llm_for_concept_links(
        self,
        new_concepts: list[dict],
        existing_concepts: list[dict],
        model: str,
    ) -> list[dict]:
        """Ask an LLM to identify RELATED_TO/SUBTOPIC_OF/EXTENDS concept pairs.

        Uses the module-level ``_CONCEPT_LINKS_PROMPT`` (a ``ChatPromptTemplate``)
        so the system role and human content are clearly separated and composable.

        Returns a (possibly empty) list of ``{source, target, relation}`` dicts
        where ``source`` and ``target`` are ``concept_key`` strings.
        """
        import json
        from src.llm import get_llm
        from src.utils.json_parser import parse_llm_json

        llm = get_llm(model=model, temperature=0.0, require_json=True)
        chain = _CONCEPT_LINKS_PROMPT | llm

        try:
            response = chain.invoke({
                "new_concepts_json": json.dumps(new_concepts, indent=2),
                "existing_concepts_json": json.dumps(existing_concepts, indent=2),
            })
            content = response.content
            parsed = parse_llm_json(content)
            if not isinstance(parsed, list):
                return []
            valid = [
                r for r in parsed
                if isinstance(r, dict)
                and r.get("relation") in CROSS_CONCEPT_RELS
                and r.get("source")
                and r.get("target")
            ]
            return valid
        except Exception as exc:
            logger.warning("LLM call for concept links failed: %s", exc)
            return []

    def _write_concept_links(self, relations: list[dict]) -> int:
        """Write cross-concept edges returned by the LLM.

        Uses MERGE so repeated ingestion does not create duplicate edges.
        Returns the number of edge write-attempts (some may merge onto existing).
        """
        queries: list[tuple[str, dict]] = []
        for rel in relations:
            rel_type = rel["relation"]  # already validated in CROSS_CONCEPT_RELS
            queries.append((
                f"""MATCH (src:{CONCEPT} {{concept_key: $source}})
                MATCH (tgt:{CONCEPT} {{concept_key: $target}})
                MERGE (src)-[:{rel_type}]->(tgt)""",
                {"source": rel["source"], "target": rel["target"]},
            ))
        if not queries:
            return 0
        try:
            self._conn.execute_write_tx(queries)
            return len(queries)
        except Exception as exc:
            logger.warning("Writing concept links failed: %s", exc)
            return 0
