from src.graph_db.connection import Neo4jConnection
from src.graph_db.schema import (
    AUTHOR,
    AUTHORED_BY,
    CONCEPT,
    CONTRADICTS,
    DISCUSSES,
    EXTENDS,
    FINDING,
    HAS_FINDING,
    METHOD,
    PAPER,
    RELATED_TO,
    SUPPORTS,
    USES_METHOD,
)


class KnowledgeGraphQueries:
    """Pre-built Cypher queries for common knowledge graph operations.

    Provides both the queries themselves (for the Streamlit UI) and
    convenience methods that return formatted results.

    Usage:
        conn = Neo4jConnection()
        queries = KnowledgeGraphQueries(conn)
        papers = queries.get_all_papers()
        shared = queries.get_shared_concepts("Paper A", "Paper B")
    """

    def __init__(self, conn: Neo4jConnection):
        self.conn = conn

    # ----- Paper Queries -----

    def get_all_papers(self) -> list[dict]:
        """Get all papers with their metadata and concept count."""
        return self.conn.execute_read(
            f"""MATCH (p:{PAPER})
            OPTIONAL MATCH (p)-[:{DISCUSSES}]->(c:{CONCEPT})
            RETURN p.document_id AS document_id, p.title AS title,
                   p.domain AS domain, p.year AS year,
                   p.summary AS summary, p.authors_str AS authors,
                   count(c) AS concept_count
            ORDER BY p.title"""
        )

    def get_paper_details(self, document_id: str) -> dict:
        """Get full details for a paper including all linked entities."""
        if not document_id:
            return {"concepts": [], "methods": [], "findings": []}

        params = {"key": document_id}
        concepts = self.conn.execute_read(
            f"""MATCH (p:{PAPER} {{document_id: $key}})-[r:{DISCUSSES}]->(c:{CONCEPT})
            RETURN c.name AS name, c.description AS description,
                   c.domain AS domain, r.depth AS depth
            ORDER BY r.depth, c.name""",
            params,
        )
        methods = self.conn.execute_read(
            f"""MATCH (p:{PAPER} {{document_id: $key}})-[:{USES_METHOD}]->(m:{METHOD})
            RETURN m.name AS name, m.description AS description""",
            params,
        )
        findings = self.conn.execute_read(
            f"""MATCH (p:{PAPER} {{document_id: $key}})-[:{HAS_FINDING}]->(f:{FINDING})
            RETURN f.claim AS claim, f.evidence_type AS evidence_type""",
            params,
        )
        return {"concepts": concepts, "methods": methods, "findings": findings}

    # ----- Cross-Paper Queries -----

    def get_shared_concepts(self, paper_a: str, paper_b: str) -> list[dict]:
        """Find concepts discussed by both papers."""
        return self.conn.execute_read(
            f"""MATCH (p1:{PAPER} {{title: $a}})-[:{DISCUSSES}]->(c:{CONCEPT})<-[:{DISCUSSES}]-(p2:{PAPER} {{title: $b}})
            RETURN c.name AS concept, c.description AS description, c.domain AS domain""",
            {"a": paper_a, "b": paper_b},
        )

    def get_cross_paper_findings(self, rel_type: str) -> list[dict]:
        """Get findings linked by CONTRADICTS/SUPPORTS/EXTENDS. rel_type is allow-listed."""
        allowed = {SUPPORTS, CONTRADICTS, EXTENDS}
        if rel_type not in allowed:
            raise ValueError(
                f"Unsupported rel_type {rel_type!r}; expected one of {sorted(allowed)}"
            )
        return self.conn.execute_read(
            f"""MATCH (f1:{FINDING})-[r:{rel_type}]->(f2:{FINDING})
            RETURN f1.claim AS finding_1, f1.paper_title AS paper_1,
                   f2.claim AS finding_2, f2.paper_title AS paper_2,
                   r.reason AS reason"""
        )

    # ----- Concept Queries -----

    def get_all_concepts(self) -> list[dict]:
        """Get all concepts with the count of papers that discuss them."""
        return self.conn.execute_read(
            f"""MATCH (c:{CONCEPT})
            OPTIONAL MATCH (p:{PAPER})-[:{DISCUSSES}]->(c)
            RETURN c.name AS name, c.description AS description,
                   c.domain AS domain, count(p) AS paper_count
            ORDER BY paper_count DESC"""
        )

    def get_concept_papers(self, concept_name: str) -> list[dict]:
        """Get all papers that discuss a specific concept."""
        return self.conn.execute_read(
            f"""MATCH (p:{PAPER})-[r:{DISCUSSES}]->(c:{CONCEPT} {{name: $name}})
            RETURN p.title AS title, p.domain AS domain, r.depth AS depth""",
            {"name": concept_name},
        )

    def get_related_concepts(self, concept_name: str) -> list[dict]:
        """Get concepts related to a given concept."""
        return self.conn.execute_read(
            f"""MATCH (c1:{CONCEPT} {{name: $name}})-[r:{RELATED_TO}]-(c2:{CONCEPT})
            RETURN c2.name AS name, c2.description AS description,
                   r.strength AS strength
            ORDER BY r.strength DESC""",
            {"name": concept_name},
        )

    # ----- Graph Overview -----

    def get_graph_stats(self) -> dict:
        """Get high-level statistics about the knowledge graph."""
        result = self.conn.execute_read(
            f"""OPTIONAL MATCH (p:{PAPER}) WITH count(p) AS papers
            OPTIONAL MATCH (c:{CONCEPT}) WITH papers, count(c) AS concepts
            OPTIONAL MATCH (m:{METHOD}) WITH papers, concepts, count(m) AS methods
            OPTIONAL MATCH (f:{FINDING}) WITH papers, concepts, methods, count(f) AS findings
            RETURN papers, concepts, methods, findings"""
        )
        if result:
            return result[0]
        return {"papers": 0, "concepts": 0, "methods": 0, "findings": 0}

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
            Subset of ``["DISCUSSES", "RELATED_TO", "SUPPORTS",
            "CONTRADICTS", "EXTENDS"]`` to include. When *None*, all are
            included.

        Returns a dict with 'nodes' and 'edges' lists suitable for
        building a pyvis network.
        """
        nodes: list[dict] = []
        edges: list[dict] = []
        seen_nodes: set[str] = set()

        allowed_rels = set(rel_types) if rel_types else None

        def _include(rel_label: str) -> bool:
            return allowed_rels is None or rel_label in allowed_rels

        # --- Papers <-[DISCUSSES]-> Concepts (with optional domain/paper filters) ---
        filters = []
        params: dict = {"limit": limit}
        if domains:
            filters.append("c.domain IN $domains")
            params["domains"] = domains
        if document_ids:
            filters.append("p.document_id IN $document_ids")
            params["document_ids"] = document_ids
        where_clause = (" WHERE " + " AND ".join(filters)) if filters else ""

        if _include(DISCUSSES):
            results = self.conn.execute_read(
                f"""MATCH (p:{PAPER})-[r:{DISCUSSES}]->(c:{CONCEPT}){where_clause}
                RETURN p.title AS paper, p.document_id AS paper_doc,
                       c.name AS concept, c.domain AS concept_domain,
                       r.depth AS depth
                LIMIT $limit""",
                params,
            )

            for row in results:
                paper_id = f"paper:{row['paper_doc'] or row['paper']}"
                concept_id = f"concept:{row['concept']}"

                if paper_id not in seen_nodes:
                    nodes.append({
                        "id": paper_id,
                        "label": (row["paper"] or "")[:40],
                        "title": row["paper"],
                        "group": "paper",
                        "size": 25,
                    })
                    seen_nodes.add(paper_id)

                if concept_id not in seen_nodes:
                    nodes.append({
                        "id": concept_id,
                        "label": row["concept"],
                        "title": f"{row['concept']} ({row.get('concept_domain', '')})",
                        "group": "concept",
                        "size": 15,
                    })
                    seen_nodes.add(concept_id)

                edges.append({
                    "from": paper_id,
                    "to": concept_id,
                    "label": "DISCUSSES",
                    "color": "#4CAF50" if row["depth"] == "core" else "#9E9E9E",
                })

        # --- Concept <-[RELATED_TO]-> Concept (limited to concepts already shown) ---
        if _include(RELATED_TO):
            concept_rels = self.conn.execute_read(
                f"""MATCH (c1:{CONCEPT})-[r:{RELATED_TO}]->(c2:{CONCEPT})
                RETURN c1.name AS from_concept, c2.name AS to_concept,
                       r.strength AS strength
                LIMIT $limit""",
                {"limit": limit},
            )
            for row in concept_rels:
                from_id = f"concept:{row['from_concept']}"
                to_id = f"concept:{row['to_concept']}"
                if from_id in seen_nodes and to_id in seen_nodes:
                    edges.append({
                        "from": from_id,
                        "to": to_id,
                        "label": "RELATED_TO",
                        "color": "#2196F3",
                        "dashes": True,
                    })

        # --- Cross-paper finding relationships ---
        finding_rel_specs = [
            (SUPPORTS,    "#4CAF50"),
            (CONTRADICTS, "#F44336"),
            (EXTENDS,     "#FF9800"),
        ]
        for rel_type, color in finding_rel_specs:
            if not _include(rel_type):
                continue
            finding_rels = self.conn.execute_read(
                f"""MATCH (f1:{FINDING})-[r:{rel_type}]->(f2:{FINDING})
                RETURN f1.paper_title AS paper1, f2.paper_title AS paper2,
                       f1.paper_document_id AS paper1_doc,
                       f2.paper_document_id AS paper2_doc,
                       type(r) AS rel_type
                LIMIT 50"""
            )
            for row in finding_rels:
                p1_id = f"paper:{row['paper1_doc'] or row['paper1']}"
                p2_id = f"paper:{row['paper2_doc'] or row['paper2']}"
                if p1_id in seen_nodes and p2_id in seen_nodes:
                    edges.append({
                        "from": p1_id,
                        "to": p2_id,
                        "label": row["rel_type"],
                        "color": color,
                        "width": 3,
                    })

        return {"nodes": nodes, "edges": edges}

    # ----- Mutation Helpers -----

    def delete_paper(self, document_id: str) -> dict:
        """Detach-delete a paper and clean up orphaned entities.

        Removes the :class:`Paper` node (and its owned :class:`Finding`
        nodes, which are paper-specific by construction) along with any
        :class:`Concept`/:class:`Method`/:class:`Author` nodes that are
        no longer referenced by any other paper.

        Returns a dict with per-entity deletion counts.
        """
        # Step 1: capture orphan-candidate ids BEFORE deleting the paper
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
            return {"papers": 0, "findings": 0, "concepts": 0, "methods": 0, "authors": 0}
        concept_keys = [k for k in rows[0]["concept_keys"] if k]
        method_names = [n for n in rows[0]["method_names"] if n]
        author_names = [n for n in rows[0]["author_names"] if n]

        # Step 2: count findings then detach-delete the paper + findings
        finding_count_rows = self.conn.execute_read(
            f"""MATCH (p:{PAPER} {{document_id: $document_id}})-[:{HAS_FINDING}]->(f:{FINDING})
            RETURN count(f) AS n""",
            {"document_id": document_id},
        )
        finding_count = finding_count_rows[0]["n"] if finding_count_rows else 0

        self.conn.execute_write(
            f"""MATCH (p:{PAPER} {{document_id: $document_id}})
            OPTIONAL MATCH (p)-[:{HAS_FINDING}]->(f:{FINDING})
            DETACH DELETE f, p""",
            {"document_id": document_id},
        )

        # Step 3: delete orphan concepts/methods/authors (no remaining paper links)
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
                f"""UNWIND $names AS n
                MATCH (m:{METHOD} {{name: n}})
                WHERE NOT (m)<-[:{USES_METHOD}]-(:{PAPER})
                DETACH DELETE m
                RETURN count(m) AS n""",
                {"names": method_names},
            )
            methods_removed = removed[0]["n"] if removed else 0

        if author_names:
            removed = self.conn.execute_write(
                f"""UNWIND $names AS n
                MATCH (a:{AUTHOR} {{name: n}})
                WHERE NOT (a)<-[:{AUTHORED_BY}]-(:{PAPER})
                DETACH DELETE a
                RETURN count(a) AS n""",
                {"names": author_names},
            )
            authors_removed = removed[0]["n"] if removed else 0

        return {
            "papers": 1,
            "findings": finding_count,
            "concepts": concepts_removed,
            "methods": methods_removed,
            "authors": authors_removed,
        }

    # ----- Explorer Queries -----

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
        grouped: dict[str, list[dict]] = {"supports": [], "contradicts": [], "extends": []}
        for row in rows:
            bucket = row["rel_type"].lower()
            if bucket not in grouped:
                continue
            grouped[bucket].append({
                "paper_1": row["paper_1"],
                "finding_1": row["finding_1"],
                "paper_2": row["paper_2"],
                "finding_2": row["finding_2"],
                "reason": row.get("reason") or "",
            })
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
            entry = matrix.setdefault(key, {
                "paper_a": row["paper_a"],
                "paper_b": row["paper_b"],
                "supports": 0,
                "contradicts": 0,
                "extends": 0,
            })
            bucket = row["rel_type"].lower()
            if bucket in entry:
                entry[bucket] += row["cnt"]
        return list(matrix.values())

    # ------------------------------------------------------------------
    # Async counterparts (used by the LangGraph agent path)
    # ------------------------------------------------------------------

    async def aget_all_papers(self) -> list[dict]:
        return await self.conn.aexecute_read(
            f"""MATCH (p:{PAPER})
            OPTIONAL MATCH (p)-[:{DISCUSSES}]->(c:{CONCEPT})
            RETURN p.document_id AS document_id, p.title AS title,
                   p.domain AS domain, p.year AS year,
                   p.summary AS summary, p.authors_str AS authors,
                   count(c) AS concept_count
            ORDER BY p.title"""
        )

    async def aget_paper_details(self, document_id: str) -> dict:
        if not document_id:
            return {"concepts": [], "methods": [], "findings": []}
        params = {"key": document_id}
        concepts = await self.conn.aexecute_read(
            f"""MATCH (p:{PAPER} {{document_id: $key}})-[r:{DISCUSSES}]->(c:{CONCEPT})
            RETURN c.name AS name, c.description AS description,
                   c.domain AS domain, r.depth AS depth
            ORDER BY r.depth, c.name""",
            params,
        )
        methods = await self.conn.aexecute_read(
            f"""MATCH (p:{PAPER} {{document_id: $key}})-[:{USES_METHOD}]->(m:{METHOD})
            RETURN m.name AS name, m.description AS description""",
            params,
        )
        findings = await self.conn.aexecute_read(
            f"""MATCH (p:{PAPER} {{document_id: $key}})-[:{HAS_FINDING}]->(f:{FINDING})
            RETURN f.claim AS claim, f.evidence_type AS evidence_type""",
            params,
        )
        return {"concepts": concepts, "methods": methods, "findings": findings}

    async def aget_shared_concepts(self, paper_a: str, paper_b: str) -> list[dict]:
        return await self.conn.aexecute_read(
            f"""MATCH (p1:{PAPER} {{title: $a}})-[:{DISCUSSES}]->(c:{CONCEPT})<-[:{DISCUSSES}]-(p2:{PAPER} {{title: $b}})
            RETURN c.name AS concept, c.description AS description, c.domain AS domain""",
            {"a": paper_a, "b": paper_b},
        )

    async def aget_cross_paper_findings(self, rel_type: str) -> list[dict]:
        allowed = {SUPPORTS, CONTRADICTS, EXTENDS}
        if rel_type not in allowed:
            raise ValueError(
                f"Unsupported rel_type {rel_type!r}; expected one of {sorted(allowed)}"
            )
        return await self.conn.aexecute_read(
            f"""MATCH (f1:{FINDING})-[r:{rel_type}]->(f2:{FINDING})
            RETURN f1.claim AS finding_1, f1.paper_title AS paper_1,
                   f2.claim AS finding_2, f2.paper_title AS paper_2,
                   r.reason AS reason"""
        )

    async def aget_all_concepts(self) -> list[dict]:
        return await self.conn.aexecute_read(
            f"""MATCH (c:{CONCEPT})
            OPTIONAL MATCH (p:{PAPER})-[:{DISCUSSES}]->(c)
            RETURN c.name AS name, c.description AS description,
                   c.domain AS domain, count(p) AS paper_count
            ORDER BY paper_count DESC"""
        )

    async def aget_concept_papers(self, concept_name: str) -> list[dict]:
        return await self.conn.aexecute_read(
            f"""MATCH (p:{PAPER})-[r:{DISCUSSES}]->(c:{CONCEPT} {{name: $name}})
            RETURN p.title AS title, p.domain AS domain, r.depth AS depth""",
            {"name": concept_name},
        )

    async def aget_related_concepts(self, concept_name: str) -> list[dict]:
        return await self.conn.aexecute_read(
            f"""MATCH (c1:{CONCEPT} {{name: $name}})-[r:{RELATED_TO}]-(c2:{CONCEPT})
            RETURN c2.name AS name, c2.description AS description,
                   r.strength AS strength
            ORDER BY r.strength DESC""",
            {"name": concept_name},
        )

    async def aget_graph_stats(self) -> dict:
        result = await self.conn.aexecute_read(
            f"""OPTIONAL MATCH (p:{PAPER}) WITH count(p) AS papers
            OPTIONAL MATCH (c:{CONCEPT}) WITH papers, count(c) AS concepts
            OPTIONAL MATCH (m:{METHOD}) WITH papers, concepts, count(m) AS methods
            OPTIONAL MATCH (f:{FINDING}) WITH papers, concepts, methods, count(f) AS findings
            RETURN papers, concepts, methods, findings"""
        )
        if result:
            return result[0]
        return {"papers": 0, "concepts": 0, "methods": 0, "findings": 0}
