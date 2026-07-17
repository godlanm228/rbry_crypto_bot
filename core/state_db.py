"""
Shared SQLite storage for bot-level state.

Portfolio and training examples have their own databases. This module is for
small operational state that used to live in JSON files: access control, user
preferences, and usage analytics.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


DEFAULT_STATE_DB_PATH = Path("data/bot_state.db")


class SQLiteStateDB:
    """Thread-local SQLite connection helper with conservative defaults."""

    def __init__(self, db_path: Path = DEFAULT_STATE_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=10000")
            self._local.conn = connection
        return self._local.conn

    def close(self):
        """Close the connection owned by the current thread."""
        connection = getattr(self._local, "conn", None)
        if connection is not None:
            connection.close()
            self._local.conn = None
