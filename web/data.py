"""What the site shows, read only: the paper watcher's files, C1's journal, prices from market.db.

Nothing here writes or imports the trading code (pandas and the models stay out of the site's memory): the
watcher already writes everything a page needs (runtime/learning/watch/journal.json every hour).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
LEARNING = Path(os.environ.get("RBRY_WEB_DATA", ROOT / "runtime" / "learning"))   # another folder: a test on a copy
WATCH = LEARNING / "watch"
PAPER = LEARNING / "paper"
MARKET_DB = Path(os.environ.get("RBRY_WEB_MARKET_DB", LEARNING / "market.db"))
UNITS = ("rbry-test-bot", "rbry-urgent", "rbry-panic.timer", "rbry-paper.timer", "rbry-paper-day.timer",
         "market-data.timer", "market-hourly.timer", "news-archive.timer", "brain-weekly.timer", "rbry-web")
HOUR_MS = 3_600_000


def _json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def tail(path: Path, lines: int, chunk: int = 1 << 20) -> list[dict]:
    """The last ``lines`` entries of a JSON-lines file without reading all of it; a torn line is left out."""
    try:
        with path.open("rb") as fh:
            size = fh.seek(0, 2)
            fh.seek(max(0, size - chunk))
            rows = [r for r in fh.read().decode("utf-8", errors="ignore").splitlines() if r.strip()]
    except OSError:
        return []
    if size > chunk:
        rows = rows[1:]
    out = []
    for r in rows[-lines:]:
        try:
            out.append(json.loads(r))
        except ValueError:
            pass
    return out


def state() -> dict:
    return _json(WATCH / "state.json", {})


def journal() -> dict:
    return _json(WATCH / "journal.json", {"books": {}})


def equity(book: str, points: int = 2000) -> list[tuple[int, float]]:
    return [(e["hour"], e["books"][book]) for e in tail(WATCH / "equity.jsonl", points) if book in e.get("books", {})]


def hours(n: int = 48) -> list[dict]:
    return list({h["hour"]: h for h in tail(WATCH / "hours.jsonl", n + 4)}.values())[-n:]


def places() -> list[dict]:
    return tail(WATCH / "places.jsonl", 200)


def c1_state() -> dict:
    return _json(PAPER / "state.json", {})


def c1_decisions(n: int = 7) -> list[dict]:
    return tail(PAPER / "decisions.jsonl", n)


def trade(book: str, moment: int, base: str) -> Optional[dict]:
    for t in journal().get("books", {}).get(book, {}).get("trades", []):
        if t["moment"] == moment and t["base"] == base:
            return t
    return None


def prices(symbol: str, from_ms: int, to_ms: int) -> list[tuple]:
    """Hourly (ts, open, high, low, close) of a pair, from market.db opened read only."""
    try:
        with sqlite3.connect(f"file:{MARKET_DB}?mode=ro", uri=True, timeout=5) as db:
            return db.execute("SELECT ts, open, high, low, close FROM klines WHERE symbol = ? AND tf = '1h' "
                              "AND ts >= ? AND ts <= ? ORDER BY ts", (symbol, from_ms, to_ms)).fetchall()
    except sqlite3.Error:
        return []


def services() -> list[tuple[str, str]]:
    out = []
    for unit in UNITS:
        try:
            state_ = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True,
                                    timeout=5).stdout.strip() or "?"
        except (OSError, subprocess.SubprocessError):
            state_ = "?"
        out.append((unit, state_))
    return out


def freshness() -> list[tuple[str, Optional[float]]]:
    """How long ago each file the site reads was written (seconds), None when it is missing."""
    now = time.time()
    files = [("состояние счетов", WATCH / "state.json"), ("журнал сделок", WATCH / "journal.json"),
             ("журнал часов", WATCH / "hours.jsonl"), ("баланс по часам", WATCH / "equity.jsonl"),
             ("C1", PAPER / "state.json"), ("база цен", MARKET_DB)]
    return [(name, now - os.path.getmtime(p) if p.exists() else None) for name, p in files]
