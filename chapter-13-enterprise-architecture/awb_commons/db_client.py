"""awb_commons/db_client.py
Aurora Serverless v2 connection-pooled PostgreSQL client.
War story fix: default pool size caused concurrent agent
failure (Ch 13 £2.3M rework root cause).
PRA SS1/23 | 7-yr retention | DORA Art.9
"""
import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2 import pool as pg_pool

    class AWBDatabaseClient:
        """Pooled PostgreSQL client for AWB AI services.

        Args:
            dsn:       PostgreSQL DSN (from AWS Secrets Manager).
            pool_min:  Minimum connections in pool (default 2).
            pool_max:  Maximum connections in pool (default 20).
            timeout:   Connection acquire timeout in seconds.
        """

        def __init__(
            self,
            dsn: str | None = None,
            pool_min: int = 2,
            pool_max: int = 20,
            timeout: int = 30,
        ) -> None:
            _dsn = dsn or os.environ["DATABASE_URL"]
            self._pool = pg_pool.ThreadedConnectionPool(
                pool_min,
                pool_max,
                _dsn,
                connect_timeout=timeout,
                cursor_factory=psycopg2.extras.RealDictCursor,
            )
            logger.info(
                "db_pool_created",
                extra={"min": pool_min, "max": pool_max},
            )

        @contextmanager
        def connection(
            self,
        ) -> Generator[psycopg2.extensions.connection, None, None]:
            """Yield a pooled connection; return on exit."""
            conn = self._pool.getconn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._pool.putconn(conn)

        def execute(
            self, query: str, params: tuple = ()
        ) -> None:
            """Execute a write query (INSERT/UPDATE/DELETE)."""
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)

        def fetchone(
            self, query: str, params: tuple = ()
        ) -> dict[str, Any] | None:
            """Fetch single row as dict, or None."""
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchone()

        def fetchall(
            self, query: str, params: tuple = ()
        ) -> list[dict[str, Any]]:
            """Fetch all rows as list of dicts."""
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchall()

        def close(self) -> None:
            """Close all pool connections (called on shutdown)."""
            self._pool.closeall()
            logger.info("db_pool_closed")

except ImportError:
    # Stub for environments without psycopg2 (test/CI)
    class AWBDatabaseClient:  # type: ignore[no-redef]
        """In-memory stub for unit tests."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._store: dict[str, list] = {}
            logger.warning("db_client_stub_mode")

        @contextmanager
        def connection(self) -> Generator[None, None, None]:
            yield None

        def execute(
            self, query: str, params: tuple = ()
        ) -> None:
            pass

        def fetchone(
            self, query: str, params: tuple = ()
        ) -> dict[str, Any] | None:
            return None

        def fetchall(
            self, query: str, params: tuple = ()
        ) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            pass
