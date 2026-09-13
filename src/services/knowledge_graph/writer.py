"""Knowledge-graph writer — upserts a PaperProfile into Neo4j.

Used by the ingestion pipeline to populate the Neo4j graph database
whenever a new PDF is processed and a PaperProfile has been extracted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

from src.services.knowledge_graph.connection import Neo4jConnection
from src.services.knowledge_graph.schema import (
    CONCEPT,
    CROSS_CONCEPT_RELS,
    CROSS_FINDING_RELS,
    DETAIL,
    DISCUSSES,
    FINDING,
    HAS_DETAIL,
    HAS_FINDING,
    METHOD,
    PAPER,
    USES_METHOD,
    initialize_schema,
    make_concept_key,
    make_finding_key,
)
from src.services.llm import get_llm
from src.services.common.json_parser import parse_llm_json

if TYPE_CHECKING:
    from src.services.llm.paper_profile import PaperProfile

logger = logging.getLogger(__name__)


_FINDING_LINKS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a scientific knowledge graph assistant specialising in cross-paper
relationship detection.

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
- No markdown fences, no prose, no commentary — raw JSON only.""",
        ),
        (
            "human",
            """NEW FINDINGS (just added to the graph):
{new_findings_json}

EXISTING FINDINGS (from other papers already in the graph):
{existing_findings_json}

Identify all pairs where a new finding SUPPORTS, CONTRADICTS, or EXTENDS an existing finding.""",
        ),
    ]
)


_CONCEPT_LINKS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a scientific knowledge graph assistant specialising in concept relationship detection.

Your task is to compare two sets of research concepts and identify semantic relationships between them.

RELATIONSHIP TYPES:
- RELATED_TO: The concepts are related but neither is prerequisite to or subsumed by the other
  (e.g., "attention" RELATED_TO "optimization").
- SUBTOPIC_OF: The new concept is a specific instance or subtype of an existing concept
  (e.g., "self-attention" SUBTOPIC_OF "attention").
- EXTENDS: The new concept builds on or extends an existing concept with additional nuance or
  technique (e.g., "multi-head attention" EXTENDS "self-attention").

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
- No markdown fences, no prose, no commentary — raw JSON only.""",
        ),
        (
            "human",
            """NEW CONCEPTS (just added to the graph):
{new_concepts_json}

EXISTING CONCEPTS (from other papers already in the graph, same domain):
{existing_concepts_json}

Identify all pairs where a new concept is RELATED_TO, SUBTOPIC_OF, or EXTENDS an existing concept.
Focus on scientifically meaningful relationships.""",
        ),
    ]
)


class KnowledgeGraphWriter:
    """Write PaperProfile data into the Neo4j knowledge graph.

    All write methods are idempotent — safe to call on re-ingestion of the
    same paper.

    """

    def __init__(self, conn: Neo4jConnection) -> None:
        self._conn = conn
        self._schema_ready = False

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

    def has_paper_details(self, document_id: str) -> bool:
        """Return True when a paper has the typed detail child nodes."""
        try:
            rows = self._conn.execute_read(
                f"""MATCH (p:{PAPER} {{document_id: $id}})-[:{HAS_DETAIL}]->(d:{DETAIL})
                RETURN count(d) AS n""",
                {"id": document_id},
            )
            return bool(rows and rows[0]["n"] > 0)
        except Exception:
            return False

    def write_paper_profile(self, profile: "PaperProfile", document_id: str) -> None:
        """Upsert a full PaperProfile into Neo4j.

        Creates or updates the Paper node, typed PaperDetail children,
        Concept / Method neighbours, and Finding nodes.  Safe to call
        multiple times for the same ``document_id`` (re-ingestion).

        Args:
            profile: LLM-extracted paper profile.
            document_id: Canonical, filename-derived paper identity.
        """
        self._ensure_schema()
        try:
            queries: list[tuple[str, dict]] = []
            self._write_paper(profile, document_id, queries)
            self._write_details(profile, document_id, queries)
            self._write_concepts(profile, document_id, queries)
            self._write_methods(profile, document_id, queries)
            self._write_findings(profile, document_id, queries)
            self._conn.execute_write_tx(queries)
            logger.info(
                "KG: wrote profile for document_id=%r  title=%r  concepts=%d  methods=%d  findings=%d",
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
            if r.get("finding_key")
        ]
        if not existing_findings:
            return 0

        relations = self._call_llm_for_links(
            _FINDING_LINKS_PROMPT,
            new_findings,
            existing_findings,
            model,
            CROSS_FINDING_RELS,
            "findings",
        )
        return self._write_links(relations, FINDING, "finding_key", "finding")

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
            if r.get("concept_key")
        ]
        if not existing_concepts:
            return 0

        relations = self._call_llm_for_links(
            _CONCEPT_LINKS_PROMPT,
            new_concepts,
            existing_concepts,
            model,
            CROSS_CONCEPT_RELS,
            "concepts",
        )
        return self._write_links(relations, CONCEPT, "concept_key", "concept")

    def _ensure_schema(self) -> None:
        if not self._schema_ready:
            initialize_schema(self._conn)
            self._schema_ready = True

    def _write_paper(
        self,
        profile: "PaperProfile",
        document_id: str,
        write_queries: list[tuple[str, dict]] | None = None,
    ) -> None:
        authors_str = ", ".join(profile.authors) if profile.authors else ""
        query = (
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
        if write_queries is None:
            self._conn.execute_write(*query)
        else:
            write_queries.append(query)

    def _write_details(
        self,
        profile: "PaperProfile",
        document_id: str,
        write_queries: list[tuple[str, dict]] | None = None,
    ) -> None:
        """Upsert paper-owned detail nodes for the central Paper node."""
        detail_fields = [
            ("contribution", "CONTRIBUTES", getattr(profile, "contributions", [])),
            ("stands_for", "STANDS_FOR", getattr(profile, "stands_for", [])),
            ("builds_on", "BUILDS_ON", getattr(profile, "builds_on", [])),
            (
                "does_not_support",
                "DOES_NOT_SUPPORT",
                getattr(profile, "does_not_support", []),
            ),
            ("limitation", "HAS_LIMITATION", getattr(profile, "limitations", [])),
        ]
        items: list[dict] = []
        for category, edge_label, details in detail_fields:
            for index, detail in enumerate(details or []):
                text = (getattr(detail, "text", "") or "").strip()
                if not text:
                    continue
                raw_key = f"{document_id}:{category}:{index}:{text.lower()}"
                items.append(
                    {
                        "detail_key": hashlib.md5(raw_key.encode(), usedforsecurity=False).hexdigest(),
                        "category": category,
                        "edge_label": edge_label,
                        "text": text,
                        "evidence": (getattr(detail, "evidence", "") or "").strip(),
                        "paper_document_id": document_id,
                        "paper_title": profile.title or document_id,
                        "position": index,
                    }
                )

        queries: list[tuple[str, dict]] = [
            (
                f"""MATCH (p:{PAPER} {{document_id: $document_id}})-[:{HAS_DETAIL}]->(d:{DETAIL})
                DETACH DELETE d""",
                {"document_id": document_id},
            )
        ]
        if items:
            queries.append(
                (
                    f"""UNWIND $items AS item
                MATCH (p:{PAPER} {{document_id: $document_id}})
                MERGE (d:{DETAIL} {{detail_key: item.detail_key}})
                SET d.category = item.category,
                    d.text = item.text,
                    d.evidence = item.evidence,
                    d.paper_document_id = item.paper_document_id,
                    d.paper_title = item.paper_title,
                    d.position = item.position
                MERGE (p)-[r:{HAS_DETAIL}]->(d)
                SET r.label = item.edge_label,
                    r.category = item.category""",
                    {"document_id": document_id, "items": items},
                )
            )
        if write_queries is None:
            self._conn.execute_write_tx(queries)
        else:
            write_queries.extend(queries)

    def _write_concepts(
        self,
        profile: "PaperProfile",
        document_id: str,
        write_queries: list[tuple[str, dict]] | None = None,
    ) -> None:
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
        query = (
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
        if write_queries is None:
            self._conn.execute_write(*query)
        else:
            write_queries.append(query)

    def _write_methods(
        self,
        profile: "PaperProfile",
        document_id: str,
        write_queries: list[tuple[str, dict]] | None = None,
    ) -> None:
        """Upsert all methods and USES_METHOD edges in a single round-trip via UNWIND."""
        if not profile.methods:
            return
        items = [{"name": m.name, "description": m.description} for m in profile.methods]
        query = (
            f"""UNWIND $items AS item
            MERGE (m:{METHOD} {{name: item.name}})
            SET m.description = item.description
            WITH m
            MATCH (p:{PAPER} {{document_id: $document_id}})
            MERGE (p)-[:{USES_METHOD}]->(m)""",
            {"items": items, "document_id": document_id},
        )
        if write_queries is None:
            self._conn.execute_write(*query)
        else:
            write_queries.append(query)

    def _write_findings(
        self,
        profile: "PaperProfile",
        document_id: str,
        write_queries: list[tuple[str, dict]] | None = None,
    ) -> None:
        """Replace a paper's findings atomically using one batched UNWIND."""
        paper_title = profile.title or document_id
        items = [
            {
                "finding_key": make_finding_key(paper_title, finding.claim),
                "claim": finding.claim,
                "evidence_type": finding.evidence_type,
            }
            for finding in profile.findings
        ]
        queries = [
            (
                f"""MATCH (p:{PAPER} {{document_id: $document_id}})-[:{HAS_FINDING}]->(f:{FINDING})
                DETACH DELETE f""",
                {"document_id": document_id},
            )
        ]
        if items:
            queries.append(
                (
                    f"""UNWIND $items AS item
                MATCH (p:{PAPER} {{document_id: $document_id}})
                CREATE (f:{FINDING} {{
                    finding_key: item.finding_key,
                    claim: item.claim,
                    evidence_type: item.evidence_type,
                    paper_document_id: $document_id,
                    paper_title: $paper_title
                }})
                CREATE (p)-[:{HAS_FINDING}]->(f)""",
                    {
                        "document_id": document_id,
                        "paper_title": paper_title,
                        "items": items,
                    },
                )
            )
        if write_queries is None:
            self._conn.execute_write_tx(queries)
        else:
            write_queries.extend(queries)

    @staticmethod
    def _call_llm_for_links(
        prompt: ChatPromptTemplate,
        new_items: list[dict],
        existing_items: list[dict],
        model: str,
        allowed_relations: set[str] | frozenset[str],
        item_name: str,
    ) -> list[dict]:
        """Ask the LLM for validated relationship triples."""
        chain = prompt | get_llm(model=model, temperature=0.0, require_json=True)
        try:
            parsed = parse_llm_json(
                chain.invoke(
                    {
                        f"new_{item_name}_json": json.dumps(new_items, indent=2),
                        f"existing_{item_name}_json": json.dumps(existing_items, indent=2),
                    }
                ).content
            )
            if not isinstance(parsed, list):
                return []
            return [
                relation
                for relation in parsed
                if isinstance(relation, dict)
                and relation.get("relation") in allowed_relations
                and relation.get("source")
                and relation.get("target")
            ]
        except Exception as exc:
            logger.warning("LLM call for %s links failed: %s", item_name, exc)
            return []

    def _write_links(
        self,
        relations: list[dict],
        node_label: str,
        key_property: str,
        link_name: str,
    ) -> int:
        """Write validated relationship triples with idempotent MERGE queries."""
        if not relations:
            return 0
        queries = [
            (
                f"""MATCH (src:{node_label} {{{key_property}: $source}})
                MATCH (tgt:{node_label} {{{key_property}: $target}})
                MERGE (src)-[:{relation["relation"]}]->(tgt)""",
                {"source": relation["source"], "target": relation["target"]},
            )
            for relation in relations
        ]
        try:
            self._conn.execute_write_tx(queries)
            return len(queries)
        except Exception as exc:
            logger.warning("Writing %s links failed: %s", link_name, exc)
            return 0
