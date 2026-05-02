from src.domain.taxonomy import normalize_concept_domain
from src.graph_db.connection import Neo4jConnection

# --- Graph Schema — Node Labels and Relationship Types ---

# Node labels
PAPER = "Paper"
CONCEPT = "Concept"
METHOD = "Method"
FINDING = "Finding"
AUTHOR = "Author"

# Relationship types
DISCUSSES = "DISCUSSES"
USES_METHOD = "USES_METHOD"
HAS_FINDING = "HAS_FINDING"
AUTHORED_BY = "AUTHORED_BY"
CITES = "CITES"
RELATED_TO = "RELATED_TO"
SUBTOPIC_OF = "SUBTOPIC_OF"
SUPPORTS = "SUPPORTS"
CONTRADICTS = "CONTRADICTS"
EXTENDS = "EXTENDS"

# Allow-list for cross-finding relationships (used as a guard in queries).
CROSS_FINDING_RELS: frozenset[str] = frozenset({SUPPORTS, CONTRADICTS, EXTENDS})

# Allow-list for cross-concept relationships (used as a guard in queries).
CROSS_CONCEPT_RELS: frozenset[str] = frozenset({RELATED_TO, SUBTOPIC_OF, EXTENDS})


# ---------------------------------------------------------------------------
# Concept identity helpers
# ---------------------------------------------------------------------------

def make_concept_key(name: str, domain: str) -> str:
    """Build a composite identity key for a Concept node.

    Format: ``"<normalized_domain>:<lowercased_name>"``.
    Two concepts share a key only when they have the same broad domain
    *and* the same (lowercased) name.  The broad-domain mapping lives in
    :mod:`src.domain.taxonomy` so RAG and KG agree on what "the same
    domain" means.
    """
    norm_domain = normalize_concept_domain(domain)
    norm_name = (name or "").strip().lower()
    return f"{norm_domain}:{norm_name}"


import hashlib as _hashlib


def make_finding_key(paper_title: str, claim: str) -> str:
    """Build a stable identity key for a Finding node.

    The key is an MD5 hex of ``"<paper_title>||<claim>"`` (lowercased,
    stripped) so two Finding nodes with identical claim text in the same
    paper collapse onto the same key.  This enables ``MERGE`` semantics and
    lets cross-finding edges reference nodes by a stable handle.
    """
    raw = f"{(paper_title or '').strip().lower()}||{(claim or '').strip().lower()}"
    return _hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


def initialize_schema(conn: Neo4jConnection) -> None:
    """Create constraints and indexes for the knowledge graph.

    Call this once when first setting up the database. Constraints
    ensure entity uniqueness and indexes speed up lookups.

    Args:
        conn: An active Neo4jConnection.
    """
    constraints = [
        f"CREATE CONSTRAINT IF NOT EXISTS FOR (p:{PAPER}) REQUIRE p.document_id IS UNIQUE",
        f"CREATE CONSTRAINT IF NOT EXISTS FOR (c:{CONCEPT}) REQUIRE c.concept_key IS UNIQUE",
        f"CREATE CONSTRAINT IF NOT EXISTS FOR (m:{METHOD}) REQUIRE m.name IS UNIQUE",
        f"CREATE CONSTRAINT IF NOT EXISTS FOR (a:{AUTHOR}) REQUIRE a.name IS UNIQUE",
        f"CREATE CONSTRAINT IF NOT EXISTS FOR (f:{FINDING}) REQUIRE f.finding_key IS UNIQUE",
    ]
    indexes = [
        f"CREATE INDEX IF NOT EXISTS FOR (p:{PAPER}) ON (p.title)",
        f"CREATE INDEX IF NOT EXISTS FOR (p:{PAPER}) ON (p.domain)",
        f"CREATE INDEX IF NOT EXISTS FOR (c:{CONCEPT}) ON (c.name)",
        f"CREATE INDEX IF NOT EXISTS FOR (c:{CONCEPT}) ON (c.domain)",
        f"CREATE INDEX IF NOT EXISTS FOR (f:{FINDING}) ON (f.paper_title)",
    ]

    for cypher in constraints + indexes:
        conn.execute_write(cypher)
