from __future__ import annotations
import asyncio, time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import pandas as pd
from core.api.coingecko_client import get_batch_historical_data
from core.rate_limiter import api_limiter

@dataclass
class HistorySet:
    ts: float
    by_coin: Dict[str, pd.DataFrame] = field(default_factory=dict)  # coin_id -> DataFrame

class HistorySnapshotService:
    """Periodically fetches historical data (days=14) for a set of coin_ids.
    Used to compute RSI/SMA instantly without hitting the network on user clicks.
    """
    def __init__(self, ids: List[str], days: int = 14, interval: int = 120):
        self.ids = list(dict.fromkeys(ids))
        self.days = days
        self.interval = interval
        self._snapshot: Optional[HistorySet] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def get_latest(self) -> Optional[HistorySet]:
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

    async def _fetch_histories(self) -> Dict[str, pd.DataFrame]:
        await api_limiter.wait_if_needed()
        data = await get_batch_historical_data(self.ids, days=self.days)
        result: Dict[str, pd.DataFrame] = {}
        for cid, obj in (data or {}).items():
            prices = obj.get("prices") or []
            vols = obj.get("total_volumes") or []
            if not prices:
                continue
            df = pd.DataFrame(prices, columns=["timestamp","price"])
            dfv = pd.DataFrame(vols, columns=["timestamp","volume"]) if vols else pd.DataFrame(columns=["timestamp","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            if not dfv.empty:
                dfv["timestamp"] = pd.to_datetime(dfv["timestamp"], unit="ms")
                df = df.merge(dfv, on="timestamp", how="left")
            result[cid] = df
        return result

    async def _run(self):
        while self._running:
            try:
                data = await self._fetch_histories()
                self._snapshot = HistorySet(ts=time.time(), by_coin=data)
            except Exception:
                # keep last
                pass
            await asyncio.sleep(self.interval)

    async def update_ids(self, ids: List[str]):
        """Replace tracked ids (e.g., when portfolio changed)."""
        async with self._lock:
            self.ids = list(dict.fromkeys(ids))
