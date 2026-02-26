from __future__ import annotations
import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from core.api.coingecko_client import get_session
from config import COINGECKO_API_URL
from core.rate_limiter import api_limiter

@dataclass
class Snapshot:
    ts: float
    markets: Dict[str, dict] = field(default_factory=dict)  # coin_id -> markets row
    simple: Dict[str, dict] = field(default_factory=dict)   # coin_id -> {usd, change, ...}

class MarketSnapshotService:
    """Keeps a warm in-memory snapshot for instant Telegram replies.
    - Uses /coins/markets for rich fields (price, 24h volume, changes 1h/24h/7d).
    - Provides get() accessors and ensures data is at most ttl seconds old.
    """
    def __init__(self, ids: List[str], ttl: int = 20):
        self.ids = ids
        self.ttl = ttl
        self._snapshot: Optional[Snapshot] = None
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def get_latest(self) -> Optional[Snapshot]:
        return self._snapshot

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except:
                pass

    async def _fetch_markets(self) -> dict:
        await api_limiter.wait_if_needed()  #

        session = await get_session()
        params = {
            "vs_currency": "usd",
            "ids": ",".join(self.ids),
            "price_change_percentage": "1h,24h,7d",
            "order": "market_cap_desc",
            "per_page": len(self.ids),
            "page": 1,
            "sparkline": "false",
        }
        url = f"{COINGECKO_API_URL}/coins/markets"
        async with session.get(url, params=params) as resp:
            if resp.status == 429:
                retry_after = int(resp.headers.get("Retry-After", "2"))
                await asyncio.sleep(max(2, min(retry_after, 10)))
                raise RuntimeError("429")
            if resp.status != 200:
                raise RuntimeError(f"markets {resp.status}: {await resp.text()}")
        data = await resp.json()
        return {row.get("id"): row for row in data}

    async def _run(self):
        while self._running:
            await api_limiter.wait_if_needed()
            try:
                markets = await self._fetch_markets()
                self._snapshot = Snapshot(ts=time.time(), markets=markets)
            except Exception as e:
                # Keep previous snapshot on error
                pass
            # Sleep small; handlers rely on warm cache
            await asyncio.sleep(self.ttl)
    async def fetch_top_coins(self, limit: int = 20) -> List[str]:
        """Получить актуальный топ монет по market cap"""
        await api_limiter.wait_if_needed()
        session = await get_session()
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false"
        }
        url = f"{COINGECKO_API_URL}/coins/markets"
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return self.ids  # fallback to current
            data = await resp.json()
        return [coin["id"] for coin in data]

    async def update_top_coins(self):
        """Обновить список отслеживаемых монет на актуальный топ"""
        new_ids = await self.fetch_top_coins(30)  # Топ-30 для покрытия
        if new_ids:
            self.ids = new_ids
