from __future__ import annotations
import time
from typing import Dict, List, Optional
from core.api.coingecko_client import get_markets_data, get_batch_historical_data

class BatchAnalyzer:
    def __init__(self, ttl_sec: int = 60):
        self._cache: Dict[str, dict] = {}
        self._ts: float = 0.0
        self._ttl = ttl_sec

    def _valid(self) -> bool:
        return (time.time() - self._ts) <= self._ttl and bool(self._cache)

    async def analyze(self, coin_ids: List[str], need_history: bool = False, history_days: int = 30) -> Dict[str, dict]:
        # CoinGecko: макс ~250 ids/запрос; мы ограничим 50 для стабильности
        if len(coin_ids) > 50:
            coin_ids = coin_ids[:50]

        # 1 запрос markets
        markets = await get_markets_data(coin_ids)
        self._cache = markets
        self._ts = time.time()

        out: Dict[str, dict] = {cid: {"market": markets.get(cid) or {}} for cid in coin_ids}

        # История — только если реально нужна паттернам/RSI (и только для выбранных монет)
        if need_history:
            histories = await get_batch_historical_data(coin_ids, days=history_days)
            for cid in coin_ids:
                out[cid]["history"] = histories.get(cid)
        return out

    def get_cached(self) -> Optional[Dict[str, dict]]:
        return self._cache if self._valid() else None
