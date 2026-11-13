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


def razgon() -> dict:
    """The ladder of «Разгон v2» (core/brain/razgon.py): the versions, every step with its exam and why it was or
    was not accepted; {} until the laptop's first exam has been copied here."""
    return _json(WATCH / "razgon-versions.json", {}) or {}      # copied next to promoted.json by the laptop's night


def rebuilt() -> dict:
    """Snapshots of trades decided before 07.11.2026, rebuilt by the watcher's code (scripts/rebuild_snapshots.py):
    "<moment>:<base>" -> {hour, market, coin, logged}."""
    return (_json(WATCH / "rebuilt.json", {}) or {}).get("trades") or {}


def equity(book: str, points: int = 2000) -> list[tuple[int, float]]:
    return [(e["hour"], e["books"][book]) for e in tail(WATCH / "equity.jsonl", points) if book in e.get("books", {})]


def equity_books(points: int = 2000) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {}
    for e in tail(WATCH / "equity.jsonl", points):
        for book, value in (e.get("books") or {}).items():
            if isinstance(value, (int, float)):
                out.setdefault(book, []).append((e["hour"], value))
    return out


def all_trades(j: Optional[dict] = None) -> list[tuple[str, dict]]:
    """(book, trade) of every book, newest first."""
    j = journal() if j is None else j
    out = [(book, t) for book, b in (j.get("books") or {}).items() for t in b.get("trades", [])
           if isinstance(t, dict) and "moment" in t and "base" in t]
    return sorted(out, key=lambda bt: -bt[1]["moment"])


def coins(j: Optional[dict] = None) -> list[dict]:
    """Every coin a book traded or skipped: how often, how it went, when last."""
    j = journal() if j is None else j
    out: dict[str, dict] = {}
    for book, t in all_trades(j):
        c = out.setdefault(t["base"], {"base": t["base"], "trades": 0, "open": 0, "closed": 0, "won": 0, "pnl": 0.0,
                                       "last": 0, "skips": 0, "books": set()})
        c["trades"] += 1
        c["books"].add(book)
        c["last"] = max(c["last"], t["moment"])
        if t.get("status") == "open":
            c["open"] += 1
        else:
            c["closed"] += 1
            c["won"] += (t.get("net_pct") or 0) > 0
        if book == "account":
            c["pnl"] += t.get("pnl_usd") or 0.0
    for s in (j.get("books") or {}).get("account", {}).get("skips", []):
        if isinstance(s, dict) and s.get("base") and s.get("reason") != "no_price":
            c = out.setdefault(s["base"], {"base": s["base"], "trades": 0, "open": 0, "closed": 0, "won": 0, "pnl": 0.0,
                                           "last": 0, "skips": 0, "books": set()})
            c["skips"] += 1
            c["last"] = max(c["last"], s.get("moment") or 0)
    return sorted(out.values(), key=lambda c: -c["last"])


def hours(n: int = 48) -> list[dict]:
    return list({h["hour"]: h for h in tail(WATCH / "hours.jsonl", n + 4)}.values())[-n:]


def places() -> list[dict]:
    return tail(WATCH / "places.jsonl", 200)


def c1_state() -> dict:
    return _json(PAPER / "state.json", {})


def c1_decisions(n: int = 7) -> list[dict]:
    return tail(PAPER / "decisions.jsonl", n)


SKIP_BOOK = {"паника": "nostop", "паника без стопа": "nostop", "модель выбора": "selector"}


def skip_outcome(skip: dict, j: dict) -> Optional[dict]:
    """What a signal the account skipped did in its own strategy's book (each strategy takes every signal of its own
    on its own $100, with the same rules and prices), or None when that book does not have it."""
    book = SKIP_BOOK.get(str(skip.get("source", "")))
    for t in ((j.get("books") or {}).get(book) or {}).get("trades") or []:
        if isinstance(t, dict) and t.get("moment") == skip.get("moment") and t.get("base") == skip.get("base"):
            return {"book": book, **t}
    return None


def c1_positions(st: dict, decisions: list[dict]) -> list[dict]:
    """Every C1 position, open and closed, with the order that opened it (its coin, place, why, exit plan)."""
    orders = {(str(d.get("day", ""))[:10], o.get("history")): {**o, "decided_at": d.get("decided_at"),
                                                               "decided_ms": d.get("decided_ms")}
              for d in decisions if isinstance(d, dict) for o in d.get("orders") or [] if isinstance(o, dict)}
    out = []
    for p in st.get("open") or []:
        if isinstance(p, dict):
            day = str(p.get("day", ""))[:10]
            out.append({**p, "day": day, "status": "open", "order": orders.get((day, p.get("history")), {})})
    for p in st.get("closed_list") or []:
        if isinstance(p, dict):
            day = str(p.get("day", ""))[:10]
            out.append({**p, "day": day, "status": "closed", "order": orders.get((day, p.get("history")), {})})
    return sorted(out, key=lambda p: (p["day"], p.get("history", "")), reverse=True)


def feed(since_ms: int, limit: int = 30) -> list[dict]:
    """What happened after ``since_ms``, newest first, for the site's notifications: the account's entries, exits and
    skips, a mass panic (4+ coins in a nervous hour), the urgent watcher's stories, C1's daily decisions."""
    j = journal()
    out = []
    acc = (j.get("books") or {}).get("account") or {}
    for t in acc.get("trades") or []:
        if not isinstance(t, dict) or "moment" not in t:
            continue
        side = "лонг" if (t.get("side") or 1) > 0 else "шорт"
        href = f"/trade/account/{int(t['moment'])}/{t['base']}"
        out.append({"ts": t["moment"], "kind": "entry", "title": f"Вход: {t['base']} {side}",
                    "text": f"{t.get('source', '')} · ${t.get('size', 0):.2f} по {t.get('entry')}", "href": href})
        if t.get("status") != "open" and t.get("exit_ts"):
            out.append({"ts": t["exit_ts"], "kind": "exit", "title": f"Выход: {t['base']} {t.get('net_pct', 0):+.2f}%",
                        "text": f"{t.get('source', '')} · ${t.get('pnl_usd', 0):+.2f}", "href": href})
    for s in acc.get("skips") or []:
        if isinstance(s, dict) and s.get("moment") and s.get("reason") != "no_price":
            out.append({"ts": s["moment"], "kind": "skip", "title": f"Пропуск: {s.get('base')}",
                        "text": f"{s.get('source', '')} · {s.get('why', s.get('reason', ''))}",
                        "href": f"/coin/{s.get('base')}"})
    for h in hours(72):
        n = len(h.get("panic") or [])
        if h.get("nervous") and n >= 4:
            out.append({"ts": h["hour"] + HOUR_MS, "kind": "panic", "title": f"Массовая паника: {n} монет",
                        "text": ", ".join(map(str, (h.get("panic") or [])[:8])), "href": "/hours"})
        for u in h.get("urgent") or []:
            if isinstance(u, dict) and u.get("told_at"):
                out.append({"ts": u["told_at"], "kind": "urgent", "title": f"Срочно: {u.get('story', '')}",
                            "text": ", ".join(map(str, u.get("kinds") or [])), "href": "/hours"})
    for d in c1_decisions(3):
        if isinstance(d, dict) and d.get("decided_ms"):
            longs = [o.get("base") for o in d.get("orders") or [] if (o.get("side") or 1) > 0]
            shorts = [o.get("base") for o in d.get("orders") or [] if (o.get("side") or 1) < 0]
            out.append({"ts": d["decided_ms"], "kind": "c1", "title": f"C1: решение за {d.get('day', '')[:10]}",
                        "text": f"лонг {', '.join(map(str, longs)) or '—'}; шорт {', '.join(map(str, shorts)) or '—'}",
                        "href": "/c1"})
    seen, uniq = set(), []
    for x in sorted(out, key=lambda x: -x["ts"]):
        key = (x["ts"], x["kind"], x["title"])
        if x["ts"] > since_ms and key not in seen:
            seen.add(key)
            uniq.append(x)
    return uniq[:limit]


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


FRESH = [("состояние счетов", "state.json", 3), ("журнал сделок", "journal.json", 3), ("журнал часов", "hours.jsonl", 3),
         ("баланс по часам", "equity.jsonl", 3), ("C1", "paper", 27), ("база цен", "market", 3)]


def _fresh_files() -> list[tuple[str, Path, int]]:
    where = {"paper": PAPER / "state.json", "market": MARKET_DB}
    return [(name, where.get(f, WATCH / f), hours) for name, f, hours in FRESH]


def freshness() -> list[tuple[str, Optional[float]]]:
    """How long ago each file the site reads was written (seconds), None when it is missing."""
    now = time.time()
    return [(name, now - os.path.getmtime(p) if p.exists() else None) for name, p, _ in _fresh_files()]


def stale() -> list[str]:
    """The files older than they should be: the hourly ones after 3 hours, C1 (daily) after 27."""
    now = time.time()
    return [name for name, p, hours in _fresh_files() if not p.exists() or now - os.path.getmtime(p) > hours * 3600]


def limits() -> dict[str, float]:
    return {name: hours * 3600 for name, _, hours in _fresh_files()}
