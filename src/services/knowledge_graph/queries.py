from src.services.knowledge_graph.connection import Neo4jConnection
from src.services.knowledge_graph.schema import (
    AUTHOR,
    AUTHORED_BY,
    CONCEPT,
    CONTRADICTS,
    DETAIL,
    DISCUSSES,
    EXTENDS,
    FINDING,
    HAS_DETAIL,
    HAS_FINDING,
    METHOD,
    PAPER,
    RELATED_TO,
    SUBTOPIC_OF,
    SUPPORTS,
    USES_METHOD,
)

_PAPERS_QUERY = f"""MATCH (p:{PAPER})
OPTIONAL MATCH (p)-[:{DISCUSSES}]->(c:{CONCEPT})
RETURN p.document_id AS document_id, p.title AS title,
       p.domain AS domain, p.year AS year, p.summary AS summary,
       p.authors_str AS authors, count(c) AS concept_count
ORDER BY p.title"""

_DETAIL_KEYS = ("concepts", "methods", "findings", "details")
_PAPER_DETAILS_QUERY = f"""
CALL {{
    MATCH (p:{PAPER})-[r:{DISCUSSES}]->(c:{CONCEPT})
    WHERE p.document_id = $key OR p.title = $key
    WITH c, r ORDER BY r.depth, c.name
    RETURN collect({{name: c.name, description: c.description,
                    domain: c.domain, depth: r.depth}}) AS concepts
}}
CALL {{
    MATCH (p:{PAPER})-[:{USES_METHOD}]->(m:{METHOD})
    WHERE p.document_id = $key OR p.title = $key
    RETURN collect({{name: m.name, description: m.description}}) AS methods
}}
CALL {{
    MATCH (p:{PAPER})-[:{HAS_FINDING}]->(f:{FINDING})
    WHERE p.document_id = $key OR p.title = $key
    RETURN collect({{claim: f.claim, evidence_type: f.evidence_type}}) AS findings
}}
CALL {{
    MATCH (p:{PAPER})-[r:{HAS_DETAIL}]->(d:{DETAIL})
    WHERE p.document_id = $key OR p.title = $key
    WITH d, r ORDER BY d.category, d.position
    RETURN collect({{category: d.category, text: d.text,
                    evidence: d.evidence, edge_label: r.label,
                    position: d.position}}) AS details
}}
RETURN concepts, methods, findings, details
"""

_CONCEPTS_QUERY = f"""MATCH (c:{CONCEPT})
OPTIONAL MATCH (p:{PAPER})-[:{DISCUSSES}]->(c)
RETURN c.name AS name, c.description AS description,
       c.domain AS domain, count(p) AS paper_count
ORDER BY paper_count DESC"""

_RELATED_CONCEPTS_QUERY = f"""MATCH (c1:{CONCEPT} {{name: $name}})-[r]-(c2:{CONCEPT})
WHERE type(r) IN ['{RELATED_TO}', '{SUBTOPIC_OF}', '{EXTENDS}']
RETURN c2.name AS name, c2.description AS description,
       c2.domain AS domain, type(r) AS relation_type
ORDER BY c2.name"""

_CONCEPT_HIERARCHY_QUERY = f"""
CALL {{
    MATCH (c1:{CONCEPT})<-[:{SUBTOPIC_OF}]-(c2:{CONCEPT} {{name: $name}})
    RETURN collect({{name: c1.name, description: c1.description,
                    domain: c1.domain}}) AS prerequisites
}}
CALL {{
    MATCH (c1:{CONCEPT} {{name: $name}})-[:{RELATED_TO}]-(c2:{CONCEPT})
    RETURN collect({{name: c2.name, description: c2.description,
                    domain: c2.domain}}) AS related
}}
CALL {{
    MATCH (c1:{CONCEPT} {{name: $name}})-[:{EXTENDS}]->(c2:{CONCEPT})
    RETURN collect({{name: c2.name, description: c2.description,
                    domain: c2.domain}}) AS extensions
}}
RETURN prerequisites, related, extensions
"""

_STATS_QUERY = f"""OPTIONAL MATCH (p:{PAPER}) WITH count(p) AS papers
OPTIONAL MATCH (c:{CONCEPT}) WITH papers, count(c) AS concepts
OPTIONAL MATCH (m:{METHOD}) WITH papers, concepts, count(m) AS methods
OPTIONAL MATCH (f:{FINDING}) WITH papers, concepts, methods, count(f) AS findings
OPTIONAL MATCH (d:{DETAIL}) WITH papers, concepts, methods, findings, count(d) AS details
RETURN papers, concepts, methods, findings, details"""
_EMPTY_STATS = dict.fromkeys(("papers", "concepts", "methods", "findings", "details"), 0)


class KnowledgeGraphQueries:
    """Pre-built Cypher queries for common knowledge graph operations.

    Provides both the queries themselves (for the Streamlit UI) and
    convenience methods that return formatted results.

    """

    def __init__(self, conn: Neo4jConnection):
        self.conn = conn

    def get_all_papers(self) -> list[dict]:
        """Get all papers with their metadata and concept count."""
        return self.conn.execute_read(_PAPERS_QUERY)

    def get_paper_details(self, document_id: str) -> dict:
        """Get full details for a paper including all linked entities.

        ``document_id`` may be the canonical filename-derived ID **or** the
        paper's title — both are matched so the agent (which sees titles) and
        the UI (which uses document_ids) call the same method correctly.
        """
        if not document_id:
            return {key: [] for key in _DETAIL_KEYS}
        rows = self.conn.execute_read(_PAPER_DETAILS_QUERY, {"key": document_id})
        if not rows:
            return {key: [] for key in _DETAIL_KEYS}
        return {key: rows[0].get(key) or [] for key in _DETAIL_KEYS}

    def get_shared_concepts(self, paper_a: str, paper_b: str) -> list[dict]:
        """Find concepts discussed by both papers."""
        return self.conn.execute_read(
            f"""MATCH (p1:{PAPER} {{title: $a}})-[:{DISCUSSES}]->
            (c:{CONCEPT})<-[:{DISCUSSES}]-(p2:{PAPER} {{title: $b}})
            RETURN c.name AS concept, c.description AS description, c.domain AS domain""",
            {"a": paper_a, "b": paper_b},
        )

    def get_cross_paper_findings(self, rel_type: str) -> list[dict]:
        """Get findings linked by CONTRADICTS/SUPPORTS/EXTENDS. rel_type is allow-listed."""
        allowed = {SUPPORTS, CONTRADICTS, EXTENDS}
        if rel_type not in allowed:
            raise ValueError(f"Unsupported rel_type {rel_type!r}; expected one of {sorted(allowed)}")
        return self.conn.execute_read(
            f"""MATCH (f1:{FINDING})-[r:{rel_type}]->(f2:{FINDING})
            RETURN f1.claim AS finding_1, f1.paper_title AS paper_1,
                   f2.claim AS finding_2, f2.paper_title AS paper_2,
                   r.reason AS reason"""
        )

    def get_all_concepts(self) -> list[dict]:
        """Get all concepts with the count of papers that discuss them."""
        return self.conn.execute_read(_CONCEPTS_QUERY)

    def get_concept_papers(self, concept_name: str) -> list[dict]:
        """Get all papers that discuss a specific concept."""
        return self.conn.execute_read(
            f"""MATCH (p:{PAPER})-[r:{DISCUSSES}]->(c:{CONCEPT} {{name: $name}})
            RETURN p.title AS title, p.domain AS domain, r.depth AS depth""",
            {"name": concept_name},
        )

    def get_related_concepts(self, concept_name: str) -> list[dict]:
        """Get concepts related to a given concept.

        Returns all concepts connected via RELATED_TO, SUBTOPIC_OF, or EXTENDS relationships.
        """
        return self.conn.execute_read(_RELATED_CONCEPTS_QUERY, {"name": concept_name})

    def get_concept_hierarchy(self, concept_name: str) -> dict:
        """Get hierarchical relationships for a concept.

        Returns:
            - 'prerequisites': concepts this one depends on (SUBTOPIC_OF incoming)
            - 'related': related concepts (RELATED_TO both directions)
            - 'extensions': concepts that extend this one (EXTENDS outgoing)
        """
        rows = self.conn.execute_read(_CONCEPT_HIERARCHY_QUERY, {"name": concept_name})
        if not rows:
            return {"prerequisites": [], "related": [], "extensions": []}
        return {key: rows[0].get(key) or [] for key in ("prerequisites", "related", "extensions")}

    def get_graph_stats(self) -> dict:
        """Get high-level statistics about the knowledge graph."""
        result = self.conn.execute_read(_STATS_QUERY)
        return result[0] if result else _EMPTY_STATS.copy()

    def get_graph_for_visualization(
        self,
        limit: int = 100,
        domains: list[str] | None = None,
        document_ids: list[str] | None = None,
        rel_types: list[str] | None = None,
    ) -> dict:
        """Get nodes and edges for pyvis visualization.

        Parameters
        ----------
        limit:
            Max DISCUSSES / RELATED_TO rows fetched.
        domains:
            Optional list of canonical domain labels; only concepts whose
            ``domain`` is in this list (and the papers that discuss them)
            are included.
        document_ids:
            Optional list of paper ``document_id`` values; restricts the
            visualisation to these papers.
        rel_types:
            Subset of ``["DISCUSSES", "HAS_DETAIL", "USES_METHOD",
            "HAS_FINDING", "RELATED_TO", "SUPPORTS", "CONTRADICTS",
            "EXTENDS"]`` to include. When *None*, all are included.

        Returns a dict with 'nodes' and 'edges' lists suitable for
        building a pyvis network.
        """
        nodes: list[dict] = []
        edges: list[dict] = []
        seen_nodes: set[str] = set()

        allowed_rels = set(rel_types) if rel_types else None

        def _include(rel_label: str) -> bool:
            return allowed_rels is None or rel_label in allowed_rels

        def _add_paper_node(doc_id: str, title: str, domain: str | None = None):
            """Helper to add or update a paper node with cached metadata."""
            paper_id = f"paper:{doc_id}"
            if paper_id not in seen_nodes:
                meta = self.get_paper_details(document_id=doc_id)
                paper_rows = self.conn.execute_read(
                    f"""MATCH (p:{PAPER})
                    WHERE p.document_id = $key OR p.title = $key
                    RETURN p.summary AS summary, p.year AS year,
                           p.authors_str AS authors
                    LIMIT 1""",
                    {"key": doc_id},
                )
                paper_meta = paper_rows[0] if paper_rows else {}
                tooltip = "\n\n".join(
                    filter(
                        None,
                        [
                            title,
                            paper_meta.get("summary") or "",
                        ],
                    )
                )
                node_data = {
                    "id": paper_id,
                    "label": (title or "")[:40],
                    "title": tooltip or title,
                    "group": "paper",
                    "size": 25,
                    "doc_id": doc_id,
                    "domain": domain or "unknown",
                    "summary": paper_meta.get("summary") or "",
                    "year": paper_meta.get("year"),
                    "authors": paper_meta.get("authors") or "",
                    "concepts_count": len(meta.get("concepts", [])),
                    "methods_count": len(meta.get("methods", [])),
                    "findings_count": len(meta.get("findings", [])),
                    "details_count": len(meta.get("details", [])),
                }
                nodes.append(node_data)
                seen_nodes.add(paper_id)

        def _paper_id(row: dict) -> str:
            doc_id = row["paper_doc"] or row["paper"]
            paper_id = f"paper:{doc_id}"
            if paper_id not in seen_nodes:
                _add_paper_node(doc_id, row["paper"], row.get("paper_domain"))
            return paper_id

        def _add_node(node_id: str, **attributes) -> None:
            if node_id not in seen_nodes:
                nodes.append({"id": node_id, **attributes})
                seen_nodes.add(node_id)

        filters = []
        params: dict = {"limit": limit}
        if domains:
            filters.append("c.domain IN $domains")
            params["domains"] = domains
        if document_ids:
            filters.append("p.document_id IN $document_ids")
            params["document_ids"] = document_ids
        where_clause = (" WHERE " + " AND ".join(filters)) if filters else ""

        paper_filters = []
        if domains:
            paper_filters.append("p.domain IN $domains")
        if document_ids:
            paper_filters.append("p.document_id IN $document_ids")
        paper_where_clause = (" WHERE " + " AND ".join(paper_filters)) if paper_filters else ""

        if _include(DISCUSSES):
            results = self.conn.execute_read(
                f"""MATCH (p:{PAPER})-[r:{DISCUSSES}]->(c:{CONCEPT}){where_clause}
                RETURN p.title AS paper, p.document_id AS paper_doc, p.domain AS paper_domain,
                       p.summary AS paper_summary,
                       c.name AS concept, c.domain AS concept_domain,
                       c.description AS concept_description,
                       r.depth AS depth
                LIMIT $limit""",
                params,
            )

            for row in results:
                paper_id = _paper_id(row)
                concept_id = f"concept:{row['concept']}"
                _add_node(
                    concept_id,
                    label=row["concept"],
                    title="\n".join(
                        filter(
                            None,
                            [
                                f"{row['concept']} ({row.get('concept_domain', '')})",
                                row.get("concept_description") or "",
                            ],
                        )
                    ),
                    group="concept",
                    size=15,
                )

                edges.append(
                    {
                        "from": paper_id,
                        "to": concept_id,
                        "label": "DISCUSSES",
                        "color": "#4CAF50" if row["depth"] == "core" else "#9E9E9E",
                    }
                )

        if _include(HAS_DETAIL):
            detail_results = self.conn.execute_read(
                f"""MATCH (p:{PAPER})-[r:{HAS_DETAIL}]->(d:{DETAIL}){paper_where_clause}
                RETURN p.title AS paper, p.document_id AS paper_doc, p.domain AS paper_domain,
                       d.detail_key AS detail_key, d.category AS category,
                       d.text AS text, d.evidence AS evidence, r.label AS edge_label
                LIMIT $limit""",
                params,
            )
            for row in detail_results:
                paper_id = _paper_id(row)
                detail_id = f"detail:{row['detail_key']}"
                _add_node(
                    detail_id,
                    label=row.get("category", "detail").replace("_", " ").title(),
                    title="\n\n".join(
                        filter(
                            None,
                            [
                                row.get("text") or "",
                                f"Evidence: {row.get('evidence')}" if row.get("evidence") else "",
                            ],
                        )
                    ),
                    group="detail",
                    size=13,
                )
                edges.append(
                    {
                        "from": paper_id,
                        "to": detail_id,
                        "label": row.get("edge_label") or "HAS_DETAIL",
                        "color": "#B388FF",
                    }
                )

        if _include(USES_METHOD):
            method_results = self.conn.execute_read(
                f"""MATCH (p:{PAPER})-[:{USES_METHOD}]->(m:{METHOD}){paper_where_clause}
                RETURN p.title AS paper, p.document_id AS paper_doc, p.domain AS paper_domain,
                       m.name AS method, m.description AS description
                LIMIT $limit""",
                params,
            )
            for row in method_results:
                paper_id = _paper_id(row)
                method_id = f"method:{row['method']}"
                _add_node(
                    method_id,
                    label=row["method"],
                    title=row.get("description") or row["method"],
                    group="method",
                    size=14,
                )
                edges.append(
                    {
                        "from": paper_id,
                        "to": method_id,
                        "label": "USES_METHOD",
                        "color": "#FFD54F",
                    }
                )

        if _include(HAS_FINDING):
            finding_results = self.conn.execute_read(
                f"""MATCH (p:{PAPER})-[:{HAS_FINDING}]->(f:{FINDING}){paper_where_clause}
                RETURN p.title AS paper, p.document_id AS paper_doc, p.domain AS paper_domain,
                       f.finding_key AS finding_key, f.claim AS claim,
                       f.evidence_type AS evidence_type
                LIMIT $limit""",
                params,
            )
            for row in finding_results:
                paper_id = _paper_id(row)
                finding_id = f"finding:{row['finding_key']}"
                _add_node(
                    finding_id,
                    label="Finding",
                    title="\n".join(
                        filter(
                            None,
                            [
                                row.get("claim") or "",
                                f"Evidence type: {row.get('evidence_type')}"
                                if row.get("evidence_type")
                                else "",
                            ],
                        )
                    ),
                    group="finding",
                    size=13,
                )
                edges.append(
                    {
                        "from": paper_id,
                        "to": finding_id,
                        "label": "HAS_FINDING",
                        "color": "#80CBC4",
                    }
                )

        for concept_rel_type, concept_color in (
            (RELATED_TO, "#2196F3"),
            (SUBTOPIC_OF, "#7E57C2"),
            (EXTENDS, "#FF9800"),
        ):
            if not _include(concept_rel_type):
                continue
            concept_rels = self.conn.execute_read(
                f"""MATCH (c1:{CONCEPT})-[r:{concept_rel_type}]->(c2:{CONCEPT})
                RETURN c1.name AS from_concept, c2.name AS to_concept,
                       r.strength AS strength
                LIMIT $limit""",
                {"limit": limit},
            )
            for row in concept_rels:
                from_id = f"concept:{row['from_concept']}"
                to_id = f"concept:{row['to_concept']}"
                if from_id in seen_nodes and to_id in seen_nodes:
                    edges.append(
                        {
                            "from": from_id,
                            "to": to_id,
                            "label": concept_rel_type,
                            "color": concept_color,
                            "dashes": concept_rel_type == RELATED_TO,
                        }
                    )

        finding_rel_specs = [
            (SUPPORTS, "#4CAF50"),
            (CONTRADICTS, "#F44336"),
            (EXTENDS, "#FF9800"),
        ]
        for rel_type, color in finding_rel_specs:
            if not _include(rel_type):
                continue
            finding_rels = self.conn.execute_read(
                f"""MATCH (f1:{FINDING})-[r:{rel_type}]->(f2:{FINDING})
                RETURN f1.paper_title AS paper1, f2.paper_title AS paper2,
                       f1.paper_document_id AS paper1_doc,
                       f2.paper_document_id AS paper2_doc
                LIMIT 50"""
            )
            for row in finding_rels:
                p1_id = f"paper:{row['paper1_doc'] or row['paper1']}"
                p2_id = f"paper:{row['paper2_doc'] or row['paper2']}"

                if p1_id not in seen_nodes:
                    _add_paper_node(row["paper1_doc"] or row["paper1"], row["paper1"])
                if p2_id not in seen_nodes:
                    _add_paper_node(row["paper2_doc"] or row["paper2"], row["paper2"])

                edges.append(
                    {
                        "from": p1_id,
                        "to": p2_id,
                        "label": rel_type,
                        "color": color,
                        "width": 3,
                    }
                )

        return {"nodes": nodes, "edges": edges}

    def delete_paper(self, document_id: str) -> dict:
        """Detach-delete a paper and clean up orphaned entities.

        Removes the :class:`Paper` node plus its owned :class:`Finding`
        and :class:`PaperDetail` nodes along with any
        :class:`Concept`/:class:`Method`/:class:`Author` nodes that are
        no longer referenced by any other paper.

        Returns a dict with per-entity deletion counts.
        """
        orphan_query = f"""
        MATCH (p:{PAPER} {{document_id: $document_id}})
        OPTIONAL MATCH (p)-[:{DISCUSSES}]->(c:{CONCEPT})
        OPTIONAL MATCH (p)-[:{USES_METHOD}]->(m:{METHOD})
        OPTIONAL MATCH (p)-[:{AUTHORED_BY}]->(a:{AUTHOR})
        RETURN
            collect(DISTINCT c.concept_key) AS concept_keys,
            collect(DISTINCT m.name) AS method_names,
            collect(DISTINCT a.name) AS author_names
        """
        rows = self.conn.execute_read(orphan_query, {"document_id": document_id})
        if not rows:
            return {
                "papers": 0,
                "findings": 0,
                "concepts": 0,
                "methods": 0,
                "authors": 0,
            }
        concept_keys = [k for k in rows[0]["concept_keys"] if k]
        method_names = [n for n in rows[0]["method_names"] if n]
        author_names = [n for n in rows[0]["author_names"] if n]

        finding_count_rows = self.conn.execute_read(
            f"""MATCH (p:{PAPER} {{document_id: $document_id}})-[:{HAS_FINDING}]->(f:{FINDING})
            RETURN count(f) AS n""",
            {"document_id": document_id},
        )
        finding_count = finding_count_rows[0]["n"] if finding_count_rows else 0
        detail_count_rows = self.conn.execute_read(
            f"""MATCH (p:{PAPER} {{document_id: $document_id}})-[:{HAS_DETAIL}]->(d:{DETAIL})
            RETURN count(d) AS n""",
            {"document_id": document_id},
        )
        detail_count = detail_count_rows[0]["n"] if detail_count_rows else 0

        self.conn.execute_write(
            f"""MATCH (p:{PAPER} {{document_id: $document_id}})
            OPTIONAL MATCH (p)-[:{HAS_FINDING}]->(f:{FINDING})
            OPTIONAL MATCH (p)-[:{HAS_DETAIL}]->(d:{DETAIL})
            DETACH DELETE f, d, p""",
            {"document_id": document_id},
        )

        concepts_removed = 0
        methods_removed = 0
        authors_removed = 0

        if concept_keys:
            removed = self.conn.execute_write(
                f"""UNWIND $keys AS key
                MATCH (c:{CONCEPT} {{concept_key: key}})
                WHERE NOT (c)<-[:{DISCUSSES}]-(:{PAPER})
                DETACH DELETE c
                RETURN count(c) AS n""",
                {"keys": concept_keys},
            )
            concepts_removed = removed[0]["n"] if removed else 0

        if method_names:
            removed = self.conn.execute_write(
                f"""UNWIND $names AS name
                MATCH (m:{METHOD} {{name: name}})
                WHERE NOT (m)<-[:{USES_METHOD}]-(:{PAPER})
                DETACH DELETE m
                RETURN count(m) AS deleted""",
                {"names": method_names},
            )
            methods_removed = removed[0]["deleted"] if removed else 0

        if author_names:
            removed = self.conn.execute_write(
                f"""UNWIND $names AS name
                MATCH (a:{AUTHOR} {{name: name}})
                WHERE NOT (a)<-[:{AUTHORED_BY}]-(:{PAPER})
                DETACH DELETE a
                RETURN count(a) AS deleted""",
                {"names": author_names},
            )
            authors_removed = removed[0]["deleted"] if removed else 0

        return {
            "papers": 1,
            "findings": finding_count,
            "details": detail_count,
            "concepts": concepts_removed,
            "methods": methods_removed,
            "authors": authors_removed,
        }

    def evidence_for_concept(self, concept_name: str) -> dict:
        """Collect cross-paper evidence for a concept.

        For every pair of findings ``(f1)-[:SUPPORTS|CONTRADICTS|EXTENDS]->(f2)``
        where either paper discusses *concept_name*, return the pair with
        its relationship and paper titles.  The result groups the
        relationships for easy rendering:

        ``{"supports": [...], "contradicts": [...], "extends": [...]}``

        Each entry is a dict with ``paper_1``, ``finding_1``, ``paper_2``,
        ``finding_2``, ``reason``.
        """
        query = f"""
        MATCH (f1:{FINDING})-[r]->(f2:{FINDING})
        WHERE type(r) IN ["{SUPPORTS}", "{CONTRADICTS}", "{EXTENDS}"]
        AND EXISTS {{
            MATCH (p:{PAPER})-[:{DISCUSSES}]->(c:{CONCEPT} {{name: $name}})
            WHERE p.document_id IN [f1.paper_document_id, f2.paper_document_id]
               OR p.title IN [f1.paper_title, f2.paper_title]
        }}
        RETURN f1.paper_title AS paper_1, f1.claim AS finding_1,
               type(r) AS rel_type,
               f2.paper_title AS paper_2, f2.claim AS finding_2,
               r.reason AS reason
        """
        rows = self.conn.execute_read(query, {"name": concept_name})
        grouped: dict[str, list[dict]] = {
            "supports": [],
            "contradicts": [],
            "extends": [],
        }
        for row in rows:
            bucket = row["rel_type"].lower()
            if bucket not in grouped:
                continue
            grouped[bucket].append(
                {
                    "paper_1": row["paper_1"],
                    "finding_1": row["finding_1"],
                    "paper_2": row["paper_2"],
                    "finding_2": row["finding_2"],
                    "reason": row.get("reason") or "",
                }
            )
        return grouped

    def get_conflict_matrix(self) -> list[dict]:
        """Return per-paper-pair counts of SUPPORTS/CONTRADICTS/EXTENDS edges.

        Each row is ``{paper_a, paper_b, supports, contradicts, extends}``
        and only unordered pairs are returned (A < B by title).
        """
        query = f"""
        MATCH (f1:{FINDING})-[r]->(f2:{FINDING})
        WHERE type(r) IN ["{SUPPORTS}", "{CONTRADICTS}", "{EXTENDS}"]
          AND f1.paper_title < f2.paper_title
        RETURN f1.paper_title AS paper_a, f2.paper_title AS paper_b,
               type(r) AS rel_type, count(r) AS cnt
        """
        rows = self.conn.execute_read(query)
        matrix: dict[tuple[str, str], dict] = {}
        for row in rows:
            key = (row["paper_a"], row["paper_b"])
            entry = matrix.setdefault(
                key,
                {
                    "paper_a": row["paper_a"],
                    "paper_b": row["paper_b"],
                    "supports": 0,
                    "contradicts": 0,
                    "extends": 0,
                },
            )
            bucket = row["rel_type"].lower()
            if bucket in entry:
                entry[bucket] += row["cnt"]
        return list(matrix.values())

    async def aget_all_papers(self) -> list[dict]:
        return await self.conn.aexecute_read(_PAPERS_QUERY)

    async def aget_paper_details(self, document_id: str) -> dict:
        """Async version of get_paper_details — matches on document_id or title."""
        if not document_id:
            return {key: [] for key in _DETAIL_KEYS}
        rows = await self.conn.aexecute_read(_PAPER_DETAILS_QUERY, {"key": document_id})
        if not rows:
            return {key: [] for key in _DETAIL_KEYS}
        return {key: rows[0].get(key) or [] for key in _DETAIL_KEYS}

    async def aget_shared_concepts(self, paper_a: str, paper_b: str) -> list[dict]:
        return await self.conn.aexecute_read(
            f"""MATCH (p1:{PAPER} {{title: $a}})-[:{DISCUSSES}]->
            (c:{CONCEPT})<-[:{DISCUSSES}]-(p2:{PAPER} {{title: $b}})
            RETURN c.name AS concept, c.description AS description, c.domain AS domain""",
            {"a": paper_a, "b": paper_b},
        )

    async def aget_cross_paper_findings(self, rel_type: str) -> list[dict]:
        allowed = {SUPPORTS, CONTRADICTS, EXTENDS}
        if rel_type not in allowed:
            raise ValueError(f"Unsupported rel_type {rel_type!r}; expected one of {sorted(allowed)}")
        return await self.conn.aexecute_read(
            f"""MATCH (f1:{FINDING})-[r:{rel_type}]->(f2:{FINDING})
            RETURN f1.claim AS finding_1, f1.paper_title AS paper_1,
                   f2.claim AS finding_2, f2.paper_title AS paper_2,
                   r.reason AS reason"""
        )

    async def aget_all_concepts(self) -> list[dict]:
        return await self.conn.aexecute_read(_CONCEPTS_QUERY)

    async def aget_concept_papers(self, concept_name: str) -> list[dict]:
        return await self.conn.aexecute_read(
            f"""MATCH (p:{PAPER})-[r:{DISCUSSES}]->(c:{CONCEPT} {{name: $name}})
            RETURN p.title AS title, p.domain AS domain, r.depth AS depth""",
            {"name": concept_name},
        )

    async def aget_related_concepts(self, concept_name: str) -> list[dict]:
        return await self.conn.aexecute_read(_RELATED_CONCEPTS_QUERY, {"name": concept_name})

    async def aget_graph_stats(self) -> dict:
        result = await self.conn.aexecute_read(_STATS_QUERY)
        return result[0] if result else _EMPTY_STATS.copy()
