"""Small risk stores in the shared state database.

EntryWatchStore — "tell me when SOL reaches $96.94": a better entry level the
trade plan suggested, checked by a job and fired once.

RiskNoticeStore — the last drawdown state a user was told about, so the
state-change notice goes out when the state changes, not on every check.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.risk.repository import DEFAULT_STATE_DB_PATH

MAX_ACTIVE_WATCHES = 5
WATCH_TTL_DAYS = 14


@dataclass(frozen=True)
class EntryWatch:
    id: int
    user_id: int
    symbol: str
    coin_id: str
    level: float
    created_at: float
    expires_at: float


class _Store:
    _schema = ""

    def __init__(self, db_path: Path = DEFAULT_STATE_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(self._schema)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10.0)


class EntryWatchStore(_Store):
    _schema = (
        "CREATE TABLE IF NOT EXISTS entry_watches ("
        " id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, symbol TEXT NOT NULL,"
        " coin_id TEXT NOT NULL, level REAL NOT NULL, created_at REAL NOT NULL,"
        " expires_at REAL NOT NULL, fired_at REAL)"
    )

    def add(self, user_id: int, symbol: str, coin_id: str, level: float,
            now: Optional[float] = None) -> Optional[EntryWatch]:
        """One watch per coin (a new level replaces the old). None at the limit."""
        now = time.time() if now is None else now
        symbol = symbol.upper()
        with self._connect() as conn:
            conn.execute("DELETE FROM entry_watches WHERE user_id = ? AND symbol = ? AND fired_at IS NULL",
                         (int(user_id), symbol))
            active = conn.execute(
                "SELECT COUNT(*) FROM entry_watches WHERE user_id = ? AND fired_at IS NULL AND expires_at > ?",
                (int(user_id), now)).fetchone()[0]
            if active >= MAX_ACTIVE_WATCHES:
                return None
            expires = now + WATCH_TTL_DAYS * 86400
            cur = conn.execute(
                "INSERT INTO entry_watches (user_id, symbol, coin_id, level, created_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?)", (int(user_id), symbol, coin_id, float(level), now, expires))
            return EntryWatch(cur.lastrowid, int(user_id), symbol, coin_id, float(level), now, expires)

    def active(self, now: Optional[float] = None) -> list[EntryWatch]:
        now = time.time() if now is None else now
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, user_id, symbol, coin_id, level, created_at, expires_at FROM entry_watches"
                " WHERE fired_at IS NULL AND expires_at > ? ORDER BY id", (now,)).fetchall()
        return [EntryWatch(*row) for row in rows]

    def mark_fired(self, watch_id: int, now: Optional[float] = None) -> bool:
        """True only for the first caller: a watch fires once."""
        with self._connect() as conn:
            cur = conn.execute("UPDATE entry_watches SET fired_at = ? WHERE id = ? AND fired_at IS NULL",
                               (time.time() if now is None else now, int(watch_id)))
            return cur.rowcount == 1

    def delete_user(self, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM entry_watches WHERE user_id = ?", (int(user_id),))


class RiskNoticeStore(_Store):
    _schema = (
        "CREATE TABLE IF NOT EXISTS risk_notices ("
        " user_id INTEGER PRIMARY KEY, state TEXT NOT NULL, notified_at REAL NOT NULL)"
    )

    def get(self, user_id: int) -> Optional[tuple[str, float]]:
        with self._connect() as conn:
            row = conn.execute("SELECT state, notified_at FROM risk_notices WHERE user_id = ?",
                               (int(user_id),)).fetchone()
        return (row[0], row[1]) if row else None

    def set(self, user_id: int, state: str, now: Optional[float] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO risk_notices (user_id, state, notified_at) VALUES (?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET state = excluded.state, notified_at = excluded.notified_at",
                (int(user_id), state, time.time() if now is None else now))

    def delete_user(self, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM risk_notices WHERE user_id = ?", (int(user_id),))
