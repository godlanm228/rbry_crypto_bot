"""learning.db: candles, the coin universe, replay runs and their trades.

A separate SQLite file next to the bot's state, never mixed with user data.
The holdout log records every time the untouched last period was looked at:
the count is part of every report, because each look spends its honesty.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "runtime" / "learning" / "learning.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL, day TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume REAL NOT NULL, PRIMARY KEY (symbol, day));
CREATE TABLE IF NOT EXISTS fear_greed (day TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS universe (
    symbol TEXT PRIMARY KEY, coin_id TEXT, pair TEXT NOT NULL, market_cap_rank INTEGER,
    added_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at REAL NOT NULL, feature_version TEXT NOT NULL,
    strategy_version TEXT NOT NULL, params TEXT NOT NULL, summary TEXT);
CREATE TABLE IF NOT EXISTS trades (
    run_id TEXT NOT NULL, strategy TEXT NOT NULL, symbol TEXT NOT NULL,
    signal_day TEXT NOT NULL, entry_day TEXT NOT NULL, exit_day TEXT NOT NULL,
    entry REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL, exit REAL NOT NULL,
    reason TEXT NOT NULL, r_net REAL NOT NULL, mfe_r REAL NOT NULL, mae_r REAL NOT NULL,
    bars INTEGER NOT NULL, overlap INTEGER NOT NULL, twin_r REAL, twin_n INTEGER,
    PRIMARY KEY (run_id, strategy, symbol, signal_day));
CREATE TABLE IF NOT EXISTS situations (
    run_id TEXT NOT NULL, symbol TEXT NOT NULL, day TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (run_id, symbol, day));
CREATE TABLE IF NOT EXISTS holdout_log (
    run_id TEXT NOT NULL, opened_at REAL NOT NULL, what TEXT NOT NULL);
"""


class LearningStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            have = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
            for column, decl in (("twin_r", "REAL"), ("twin_n", "INTEGER")):
                if column not in have:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {column} {decl}")

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # market data
    def save_candles(self, symbol: str, frame: pd.DataFrame) -> int:
        rows = [(symbol, d.strftime("%Y-%m-%d"), float(o), float(h), float(l), float(c), float(v))
                for d, o, h, l, c, v in frame[["day", "open", "high", "low", "close", "volume"]]
                .itertuples(index=False)]
        with self.connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    def candles(self, symbol: str) -> pd.DataFrame:
        with self.connect() as conn:
            frame = pd.read_sql_query(
                "SELECT day, open, high, low, close, volume FROM candles WHERE symbol = ? ORDER BY day",
                conn, params=(symbol,))
        frame["day"] = pd.to_datetime(frame["day"])
        return frame

    def last_day(self, symbol: str) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(day) FROM candles WHERE symbol = ?", (symbol,)).fetchone()
        return row[0] if row else None

    def save_fear_greed(self, rows: Iterable[tuple[str, int]]) -> None:
        with self.connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO fear_greed VALUES (?, ?)", list(rows))

    def fear_greed(self) -> pd.DataFrame:
        with self.connect() as conn:
            frame = pd.read_sql_query("SELECT day, value FROM fear_greed ORDER BY day", conn)
        frame["day"] = pd.to_datetime(frame["day"])
        return frame

    def set_universe(self, rows: list[dict]) -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute("DELETE FROM universe")
            conn.executemany(
                "INSERT INTO universe (symbol, coin_id, pair, market_cap_rank, added_at) VALUES (?, ?, ?, ?, ?)",
                [(r["symbol"], r.get("coin_id"), r["pair"], r.get("rank"), now) for r in rows])

    def universe(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT symbol, coin_id, pair, market_cap_rank FROM universe ORDER BY market_cap_rank").fetchall()
        return [{"symbol": s, "coin_id": c, "pair": p, "rank": r} for s, c, p, r in rows]

    # replay results
    def save_run(self, run_id: str, feature_version: str, strategy_version: str, params: dict) -> None:
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO runs (run_id, created_at, feature_version, strategy_version, params)"
                         " VALUES (?, ?, ?, ?, ?)",
                         (run_id, time.time(), feature_version, strategy_version, json.dumps(params)))

    def save_summary(self, run_id: str, summary: dict) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE runs SET summary = ? WHERE run_id = ?", (json.dumps(summary), run_id))

    def save_trades(self, run_id: str, trades: pd.DataFrame) -> None:
        cols = ["strategy", "symbol", "signal_day", "entry_day", "exit_day", "entry", "stop", "target",
                "exit", "reason", "r_net", "mfe_r", "mae_r", "bars", "overlap", "twin_r", "twin_n"]
        rows = [(run_id, *(None if isinstance(x, float) and x != x else x for x in r))
                for r in trades[cols].itertuples(index=False)]
        with self.connect() as conn:
            conn.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
            conn.executemany(f"INSERT INTO trades (run_id, {', '.join(cols)}) VALUES ({', '.join('?' * (len(cols) + 1))})",
                             rows)

    def trades(self, run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("SELECT * FROM trades WHERE run_id = ?", conn, params=(run_id,))

    def save_situations(self, run_id: str, situations: pd.DataFrame, columns: list[str]) -> None:
        rows = [(run_id, sym, day, json.dumps([None if pd.isna(x) else float(x) for x in vals]))
                for sym, day, *vals in situations[["symbol", "day_str", *columns]].itertuples(index=False)]
        with self.connect() as conn:
            conn.execute("DELETE FROM situations WHERE run_id = ?", (run_id,))
            conn.executemany("INSERT INTO situations VALUES (?, ?, ?, ?)", rows)

    def log_holdout(self, run_id: str, what: str) -> int:
        with self.connect() as conn:
            conn.execute("INSERT INTO holdout_log VALUES (?, ?, ?)", (run_id, time.time(), what))
            return conn.execute("SELECT COUNT(*) FROM holdout_log").fetchone()[0]

    def holdout_opens(self) -> int:
        with self.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM holdout_log").fetchone()[0]
