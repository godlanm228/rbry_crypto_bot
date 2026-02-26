import aiohttp
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from core.portfolio_db import DB_PATH

logger = logging.getLogger(__name__)

COINGECKO_LIST_URL = "https://api.coingecko.com/api/v3/coins/list"


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
        return self._local.conn

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
        if not force and last_updated and (now - last_updated) < timedelta(hours=max_age_hours):
            return False

        logger.info("Refreshing coin registry from CoinGecko...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(COINGECKO_LIST_URL, timeout=30) as resp:
                    resp.raise_for_status()
                    items = await resp.json()
        except Exception as e:
            logger.warning(f"Failed to refresh coin registry: {e}")
            return False

        updated_at = now.isoformat().replace("+00:00", "Z")
        best_by_symbol = {}
        for it in items:
            symbol = (it.get("symbol") or "").strip().upper()
            coin_id = (it.get("id") or "").strip()
            name = (it.get("name") or "").strip()
            if not symbol or not coin_id:
                continue
            score = (len(coin_id), len(name))
            prev = best_by_symbol.get(symbol)
            if prev is None or score < prev["score"]:
                best_by_symbol[symbol] = {"score": score, "id": coin_id, "name": name}

        rows = [
            (symbol, meta["id"], None, meta["name"], updated_at)
            for symbol, meta in best_by_symbol.items()
        ]

        if not rows:
            logger.warning("Coin registry refresh returned no rows")
            return False

        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("BEGIN")
        cur.execute("DELETE FROM coin_registry")
        cur.executemany(
            """
            INSERT OR REPLACE INTO coin_registry
            (symbol, coingecko_id, binance_symbol, name, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        logger.info(f"Coin registry refreshed: {len(rows)} symbols")
        return True

    def resolve(self, ticker: str) -> Optional[Dict[str, str]]:
        if not ticker:
            return None
        symbol = str(ticker).strip().upper()
        if not symbol:
            return None
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol, coingecko_id, binance_symbol, name FROM coin_registry WHERE symbol = ?",
            (symbol,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "symbol": row["symbol"],
            "coingecko_id": row["coingecko_id"],
            "binance_symbol": row["binance_symbol"],
            "name": row["name"],
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
