"""
Database abstraction layer for SpeakFit API.
Supports both SQLite (development) and PostgreSQL (production).

Usage:
    from api.database import get_db, init_db
    
    # Get a database connection
    with get_db() as db:
        db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = db.fetchone()
"""

import os
import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Any, Iterator
from functools import lru_cache

logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "")
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", "/home/melchior/speech3/speechscore.db"))

# Detect database type
IS_POSTGRES = DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")


class DatabaseConnection:
    """Unified database connection interface."""
    
    def __init__(self, conn, is_postgres: bool = False):
        self.conn = conn
        self.is_postgres = is_postgres
        self._cursor = None
    
    @property
    def cursor(self):
        if self._cursor is None:
            self._cursor = self.conn.cursor()
        return self._cursor
    
    def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a query, converting placeholders if needed."""
        if self.is_postgres:
            # Convert SQLite ? placeholders to Postgres %s
            query = self._convert_placeholders(query)
        return self.cursor.execute(query, params)
    
    def executemany(self, query: str, params_list: list) -> Any:
        """Execute query with multiple parameter sets."""
        if self.is_postgres:
            query = self._convert_placeholders(query)
        return self.cursor.executemany(query, params_list)
    
    def fetchone(self) -> Optional[dict]:
        """Fetch one row as dictionary."""
        row = self.cursor.fetchone()
        if row is None:
            return None
        if self.is_postgres:
            # psycopg2 RealDictCursor returns dict-like
            return dict(row)
        else:
            # sqlite3.Row is dict-like
            return dict(row)
    
    def fetchall(self) -> list:
        """Fetch all rows as list of dictionaries."""
        rows = self.cursor.fetchall()
        if self.is_postgres:
            return [dict(row) for row in rows]
        else:
            return [dict(row) for row in rows]
    
    def fetchmany(self, size: int = 100) -> list:
        """Fetch up to size rows."""
        rows = self.cursor.fetchmany(size)
        return [dict(row) for row in rows]
    
    def commit(self):
        """Commit the transaction."""
        self.conn.commit()
    
    def rollback(self):
        """Rollback the transaction."""
        self.conn.rollback()
    
    def close(self):
        """Close the connection."""
        if self._cursor:
            self._cursor.close()
        self.conn.close()
    
    @property
    def lastrowid(self) -> int:
        """Get the last inserted row ID."""
        if self.is_postgres:
            # Postgres needs RETURNING clause, handled differently
            # This is a fallback
            return self.cursor.fetchone()[0] if self.cursor.description else 0
        return self.cursor.lastrowid
    
    @property
    def rowcount(self) -> int:
        """Get number of affected rows."""
        return self.cursor.rowcount
    
    def _convert_placeholders(self, query: str) -> str:
        """Convert SQLite ? to Postgres %s placeholders."""
        # Simple conversion - doesn't handle ? in strings
        # For complex queries, use named parameters instead
        return query.replace("?", "%s")


def _get_sqlite_connection() -> DatabaseConnection:
    """Create SQLite connection."""
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return DatabaseConnection(conn, is_postgres=False)


def _get_postgres_connection() -> DatabaseConnection:
    """Create PostgreSQL connection."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return DatabaseConnection(conn, is_postgres=True)


@contextmanager
def get_db() -> Iterator[DatabaseConnection]:
    """
    Get a database connection as a context manager.
    
    Usage:
        with get_db() as db:
            db.execute("SELECT * FROM users")
            users = db.fetchall()
    """
    if IS_POSTGRES:
        db = _get_postgres_connection()
    else:
        db = _get_sqlite_connection()
    
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_dependency():
    """FastAPI dependency for database connection."""
    with get_db() as db:
        yield db


# Legacy compatibility - direct connection for existing code
def get_connection():
    """
    Get raw database connection (legacy compatibility).
    Caller is responsible for closing.
    """
    if IS_POSTGRES:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    else:
        conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn


def init_db():
    """Initialize database (create tables if needed)."""
    if IS_POSTGRES:
        logger.info("Using PostgreSQL database")
        # Schema should be created by migration script
    else:
        logger.info(f"Using SQLite database: {SQLITE_PATH}")
        # Existing SQLite initialization
        from api.db import init_db as init_sqlite_db
        init_sqlite_db()


# Query builders for common operations
class QueryBuilder:
    """Helper for building database-agnostic queries."""
    
    @staticmethod
    def insert_returning_id(table: str, columns: list, is_postgres: bool) -> str:
        """Build INSERT query that returns the new ID."""
        cols = ", ".join(columns)
        if is_postgres:
            placeholders = ", ".join(["%s"] * len(columns))
            return f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) RETURNING id"
        else:
            placeholders = ", ".join(["?"] * len(columns))
            return f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    
    @staticmethod
    def upsert(table: str, columns: list, conflict_column: str, is_postgres: bool) -> str:
        """Build UPSERT (INSERT ... ON CONFLICT UPDATE) query."""
        cols = ", ".join(columns)
        update_cols = ", ".join([f"{c} = EXCLUDED.{c}" for c in columns if c != conflict_column])
        
        if is_postgres:
            placeholders = ", ".join(["%s"] * len(columns))
            return f"""
                INSERT INTO {table} ({cols}) VALUES ({placeholders})
                ON CONFLICT ({conflict_column}) DO UPDATE SET {update_cols}
            """
        else:
            placeholders = ", ".join(["?"] * len(columns))
            return f"""
                INSERT INTO {table} ({cols}) VALUES ({placeholders})
                ON CONFLICT ({conflict_column}) DO UPDATE SET {update_cols}
            """
    
    @staticmethod
    def now(is_postgres: bool) -> str:
        """Get current timestamp expression."""
        if is_postgres:
            return "NOW()"
        else:
            return "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"
    
    @staticmethod 
    def json_extract(column: str, path: str, is_postgres: bool) -> str:
        """Extract value from JSON column."""
        if is_postgres:
            # Postgres JSONB uses ->> for text extraction
            return f"{column}->>{path}"
        else:
            # SQLite uses json_extract
            return f"json_extract({column}, '$.{path}')"
    
    @staticmethod
    def limit_offset(limit: int, offset: int, is_postgres: bool) -> str:
        """Build LIMIT/OFFSET clause."""
        # Same syntax for both
        return f"LIMIT {limit} OFFSET {offset}"


# Export database info
def get_db_info() -> dict:
    """Get information about current database configuration."""
    return {
        "type": "postgresql" if IS_POSTGRES else "sqlite",
        "url": DATABASE_URL if IS_POSTGRES else str(SQLITE_PATH),
        "is_postgres": IS_POSTGRES,
    }
