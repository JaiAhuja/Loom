import atexit

from neo4j import AsyncGraphDatabase, GraphDatabase

from config.settings import settings


class Neo4jConnection:
    """Manage the Neo4j database connection and provide query execution.

    Supports context-manager protocol for automatic cleanup.

    Usage:
        with Neo4jConnection() as conn:
            result = conn.execute_read("MATCH (n) RETURN count(n) AS count")

        # Or without context manager:
        conn = Neo4jConnection()
        conn.close()
    """

    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = None,
    ):
        self.uri = uri or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.database = database or getattr(settings, "NEO4J_DATABASE", None)
        self._driver = None
        self._async_driver = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    @property
    def driver(self):
        """Lazy-initialize the Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
        return self._driver

    @property
    def async_driver(self):
        """Lazy-initialize the async Neo4j driver."""
        if self._async_driver is None:
            self._async_driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
        return self._async_driver

    def is_connected(self) -> bool:
        """Check if Neo4j is reachable."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def execute_read(self, query: str, parameters: dict = None) -> list[dict]:
        """Execute a read-only Cypher query using a managed read transaction.

        Args:
            query: Cypher query string.
            parameters: Query parameters dict.

        Returns:
            List of result records as dicts.
        """
        with self.driver.session(database=self.database) as session:
            return session.execute_read(
                lambda tx: [r.data() for r in tx.run(query, parameters or {})]
            )

    def execute_write(self, query: str, parameters: dict = None) -> list[dict]:
        """Execute a write Cypher query using a managed write transaction.

        Uses execute_write() for automatic retries on transient errors.

        Args:
            query: Cypher query string.
            parameters: Query parameters dict.

        Returns:
            List of result records as dicts.
        """
        with self.driver.session(database=self.database) as session:
            return session.execute_write(
                lambda tx: [r.data() for r in tx.run(query, parameters or {})]
            )

    def execute_write_tx(self, queries: list[tuple[str, dict]]) -> None:
        """Execute multiple write queries in a single transaction.

        Args:
            queries: List of (cypher_string, parameters_dict) tuples.
        """
        def _work(tx):
            for query, params in queries:
                tx.run(query, params or {})

        with self.driver.session(database=self.database) as session:
            session.execute_write(_work)

    # ------------------------------------------------------------------
    # Async counterparts
    # ------------------------------------------------------------------

    async def aexecute_read(self, query: str, parameters: dict = None) -> list[dict]:
        """Async version of execute_read — uses a managed read transaction with auto-retry."""
        async with self.async_driver.session(database=self.database) as session:

            async def _work(tx):
                result = await tx.run(query, parameters or {})
                return await result.data()

            return await session.execute_read(_work)

    async def aexecute_write(self, query: str, parameters: dict = None) -> list[dict]:
        """Async version of execute_write."""
        async with self.async_driver.session(database=self.database) as session:

            async def _work(tx):
                result = await tx.run(query, parameters or {})
                return await result.data()

            return await session.execute_write(_work)

    async def aexecute_write_tx(self, queries: list[tuple[str, dict]]) -> None:
        """Async version of execute_write_tx."""
        async with self.async_driver.session(database=self.database) as session:

            async def _work(tx):
                for q, params in queries:
                    result = await tx.run(q, params or {})
                    await result.consume()  # Exhaust cursor so the transaction can commit cleanly.

            await session.execute_write(_work)

    async def aclose(self) -> None:
        """Async close of the async driver."""
        if self._async_driver is not None:
            await self._async_driver.close()
            self._async_driver = None

    def close(self):
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
        if self._async_driver is not None:
            import asyncio
            driver = self._async_driver
            self._async_driver = None
            try:
                loop = asyncio.get_running_loop()
                # An event loop is already running — schedule the close as a fire-and-forget task.
                loop.create_task(driver.close())
            except RuntimeError:
                # No running event loop (e.g. atexit, tests) — close synchronously.
                asyncio.run(driver.close())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# A single process-wide Neo4jConnection is shared across the Streamlit app
# and all its pages.  Because Streamlit re-imports modules on each rerun but
# keeps module state alive, a plain module-level variable is sufficient and
# avoids the duplicate driver problem caused by per-page @st.cache_resource.

_shared_connection: "Neo4jConnection | None" = None


def get_neo4j_connection() -> Neo4jConnection:
    """Return the process-wide shared :class:`Neo4jConnection` singleton.

    The same instance is returned on every call; the underlying Bolt driver
    is lazy-initialised on first use and re-used across all Streamlit pages
    and background workers in the same process.
    """
    global _shared_connection
    if _shared_connection is None:
        _shared_connection = Neo4jConnection()
        atexit.register(reset_neo4j_connection)
    return _shared_connection


def reset_neo4j_connection() -> None:
    """Close and clear the singleton (primarily for tests)."""
    global _shared_connection
    if _shared_connection is not None:
        try:
            _shared_connection.close()
        except Exception:
            pass
        _shared_connection = None
