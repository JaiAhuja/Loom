import atexit
import logging
import threading

from neo4j import AsyncGraphDatabase, GraphDatabase

from config.settings import settings

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Manage the Neo4j database connection and provide query execution.

    Supports context-manager protocol for automatic cleanup.

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
        self._driver_lock = threading.RLock()
        self._async_driver_lock = threading.RLock()

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
            self._driver = self._init_driver(
                self._driver_lock,
                "_driver",
                GraphDatabase.driver,
                connection_acquisition_timeout=5.0,
            )
        return self._driver

    @property
    def async_driver(self):
        """Lazy-initialize the async Neo4j driver."""
        if self._async_driver is None:
            self._async_driver = self._init_driver(
                self._async_driver_lock,
                "_async_driver",
                AsyncGraphDatabase.driver,
            )
        return self._async_driver

    def _init_driver(self, lock, attr: str, factory, **kwargs):
        with lock:
            driver = getattr(self, attr)
            if driver is None:
                driver = factory(self.uri, auth=(self.username, self.password), **kwargs)
                setattr(self, attr, driver)
            return driver

    def is_connected(self) -> bool:
        """Check if Neo4j is reachable and the database instance is available.

        Unlike verify_connectivity(), this actually executes a query to ensure
        the database instance is running (not just the server).
        """
        try:
            self.driver.verify_connectivity()
            self.execute_read("RETURN 1")
            return True
        except Exception:
            logger.debug("Neo4j connectivity check failed", exc_info=True)
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


    async def aexecute_read(self, query: str, parameters: dict = None) -> list[dict]:
        """Async version of execute_read — uses a managed read transaction with auto-retry."""
        return await self._aexecute("read", query, parameters)

    async def aexecute_write(self, query: str, parameters: dict = None) -> list[dict]:
        """Async version of execute_write."""
        return await self._aexecute("write", query, parameters)

    async def _aexecute(
        self, mode: str, query: str, parameters: dict | None
    ) -> list[dict]:
        async with self.async_driver.session(database=self.database) as session:
            async def _work(tx):
                try:
                    result = await tx.run(query, parameters or {})
                    return await result.data()
                except Exception as exc:
                    logger.error(
                        "Async %s query failed: %s", mode, exc, exc_info=True
                    )
                    raise

            return await getattr(session, f"execute_{mode}")(_work)

    async def aexecute_write_tx(self, queries: list[tuple[str, dict]]) -> None:
        """Async version of execute_write_tx."""
        async with self.async_driver.session(database=self.database) as session:

            async def _work(tx):
                for i, (q, params) in enumerate(queries):
                    try:
                        result = await tx.run(q, params or {})
                        await result.consume()
                    except Exception as e:
                        logger.error(f"Write transaction failed at query {i}: {e}", exc_info=True)
                        raise

            await session.execute_write(_work)

    async def aclose(self) -> None:
        """Async close of the async driver."""
        if self._async_driver is not None:
            await self._async_driver.close()
            self._async_driver = None

    def close(self):
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            with self._driver_lock:
                if self._driver is not None:
                    self._driver.close()
                    self._driver = None
        if self._async_driver is not None:
            import asyncio
            with self._async_driver_lock:
                driver = self._async_driver
                self._async_driver = None
            if driver is None:
                return
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(driver.close())
                task.add_done_callback(self._log_async_close_failure)
            except RuntimeError:
                try:
                    asyncio.run(driver.close())
                except Exception:
                    logger.debug("Failed to close async Neo4j driver", exc_info=True)

    @staticmethod
    def _log_async_close_failure(task) -> None:
        try:
            task.result()
        except Exception:
            logger.debug("Failed to close async Neo4j driver", exc_info=True)



_shared_connection: "Neo4jConnection | None" = None
_shared_connection_lock = threading.RLock()


def get_neo4j_connection() -> Neo4jConnection:
    """Return the process-wide shared :class:`Neo4jConnection` singleton.

    The same instance is returned on every call; the underlying Neo4j driver
    is lazy-initialised on first use and re-used across all Streamlit pages
    and background workers in the same process.
    """
    global _shared_connection
    if _shared_connection is None:
        with _shared_connection_lock:
            if _shared_connection is None:
                _shared_connection = Neo4jConnection()
                atexit.register(reset_neo4j_connection)
    return _shared_connection


def reset_neo4j_connection() -> None:
    """Close and clear the singleton (primarily for tests)."""
    global _shared_connection
    with _shared_connection_lock:
        if _shared_connection is not None:
            try:
                _shared_connection.close()
            except Exception:
                logger.debug("Failed to reset Neo4j singleton", exc_info=True)
            _shared_connection = None
