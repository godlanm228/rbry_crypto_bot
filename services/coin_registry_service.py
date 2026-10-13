import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from core.portfolio_db import DB_PATH
from core.api.coingecko_client import get_coin_list, get_market_cap_ranks

logger = logging.getLogger(__name__)

# Lower is better. Any coin in the market-cap top list beats every unranked
# one; among unranked candidates an exact id beats a ticker beats a name.
_KIND_PRIORITY = {"id": 0, "symbol": 1, "name": 2}
_UNRANKED = 10**9


def choose_aliases(items, ranks):
    """Pick one coin per lowercase alias (ticker, CoinGecko id, name).

    Ticker collisions are common (TON, OP, TRUMP each have dozens of
    tokens); the old rule "shortest id wins" mapped them to dead tokens.
    Returns {alias: (coingecko_id, symbol, name, rank_or_None)}.
    """
    best = {}
    for it in items:
        coin_id = (it.get("id") or "").strip()
        symbol = (it.get("symbol") or "").strip().upper()
        name = (it.get("name") or "").strip()
        if not coin_id or not symbol:
            continue
        rank = ranks.get(coin_id)
        for kind, alias in (("id", coin_id), ("symbol", symbol), ("name", name)):
            alias = alias.lower()
            if not alias:
                continue
            score = (rank if rank is not None else _UNRANKED, _KIND_PRIORITY[kind], len(coin_id))
            prev = best.get(alias)
            if prev is None or score < prev[0]:
                best[alias] = (score, coin_id, symbol, name, rank)
    return {alias: entry[1:] for alias, entry in best.items()}


class CoinRegistryService:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS coin_aliases (
                    alias TEXT PRIMARY KEY,
                    coingecko_id TEXT NOT NULL,
                    symbol TEXT,
                    name TEXT,
                    market_cap_rank INTEGER,
                    updated_at TIMESTAMP
                )
                """
            )
        return self._local.conn

    def _has_aliases(self) -> bool:
        row = self._get_conn().execute("SELECT 1 FROM coin_aliases LIMIT 1").fetchone()
        return row is not None

    def _get_last_updated(self) -> Optional[datetime]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT MAX(updated_at) AS max_updated FROM coin_registry")
        row = cur.fetchone()
        if not row or not row["max_updated"]:
            return None
        try:
            return datetime.fromisoformat(str(row["max_updated"]).replace("Z", "+00:00"))
        except Exception:
            return None

    async def refresh_if_needed(self, max_age_hours: int = 24, force: bool = False) -> bool:
        last_updated = self._get_last_updated()
        now = datetime.now(timezone.utc)
        fresh = last_updated and (now - last_updated) < timedelta(hours=max_age_hours)
        if not force and fresh and self._has_aliases():
            return False

        logger.info("Refreshing coin registry from CoinGecko...")
        try:
            items = await get_coin_list()
        except Exception as e:
            logger.warning(f"Failed to refresh coin registry: {e}")
            return False
        try:
            ranks = await get_market_cap_ranks()
        except Exception as e:
            # Without ranks ticker collisions fall back to the exact-id rule;
            # still better than keeping a stale registry.
            logger.warning(f"Market-cap ranks unavailable for registry refresh: {e}")
            ranks = {}

        updated_at = now.isoformat().replace("+00:00", "Z")
        aliases = choose_aliases(items, ranks)
        alias_rows = [
            (alias, coin_id, symbol, name, rank, updated_at)
            for alias, (coin_id, symbol, name, rank) in aliases.items()
        ]
        # Legacy per-ticker table, kept for resolve_symbol_by_id and readers.
        rows = [
            (symbol, coin_id, None, name, updated_at)
            for alias, (coin_id, symbol, name, rank) in aliases.items()
            if alias == symbol.lower()
        ]

        if not rows:
            logger.warning("Coin registry refresh returned no rows")
            return False

        conn = self._get_conn()
        try:
            with conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM coin_registry")
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO coin_registry
                    (symbol, coingecko_id, binance_symbol, name, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                cur.execute("DELETE FROM coin_aliases")
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO coin_aliases
                    (alias, coingecko_id, symbol, name, market_cap_rank, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    alias_rows,
                )
        except Exception as error:
            logger.exception("Coin registry transaction failed: %s", error)
            return False
        logger.info(
            "Coin registry refreshed: %s symbols, %s aliases, %s ranked coins",
            len(rows), len(alias_rows), len(ranks),
        )
        return True

    def resolve(self, ticker: str) -> Optional[Dict[str, str]]:
        """Resolve a ticker, CoinGecko id or coin name (case-insensitive)."""
        if not ticker:
            return None
        alias = str(ticker).strip().lower()
        if not alias:
            return None
        conn = self._get_conn()
        row = conn.execute(
            "SELECT coingecko_id, symbol, name, market_cap_rank FROM coin_aliases WHERE alias = ?",
            (alias,),
        ).fetchone()
        if row:
            return {
                "symbol": row["symbol"],
                "coingecko_id": row["coingecko_id"],
                "binance_symbol": None,
                "name": row["name"],
                "market_cap_rank": row["market_cap_rank"],
            }
        # Alias table not built yet (first start before a refresh).
        row = conn.execute(
            "SELECT symbol, coingecko_id, binance_symbol, name FROM coin_registry WHERE symbol = ?",
            (alias.upper(),),
        ).fetchone()
        if not row:
            return None
        return {
            "symbol": row["symbol"],
            "coingecko_id": row["coingecko_id"],
            "binance_symbol": row["binance_symbol"],
            "name": row["name"],
            "market_cap_rank": None,
        }

    def resolve_symbol_by_id(self, coin_id: str) -> Optional[str]:
        if not coin_id:
            return None
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol FROM coin_registry WHERE coingecko_id = ? LIMIT 1",
            (coin_id,),
        )
        row = cur.fetchone()
        return row["symbol"] if row else None


_coin_registry_service: Optional[CoinRegistryService] = None
_registry_lock = threading.Lock()


def get_coin_registry_service() -> CoinRegistryService:
    global _coin_registry_service
    if _coin_registry_service is None:
        with _registry_lock:
            if _coin_registry_service is None:
                _coin_registry_service = CoinRegistryService()
    return _coin_registry_service
