from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
import pandas as pd

import config
from core.api.coingecko_client import get_batch_historical_data
from core.api.binance_client import SYMBOL_MAP, get_klines


logger = logging.getLogger(__name__)


@dataclass
class HistorySet:
    ts: float
    by_coin: Dict[str, pd.DataFrame] = field(default_factory=dict)


class HistorySnapshotService:
    """Cache validated daily histories without publishing partial warm-ups."""

    def __init__(
        self,
        ids: List[str],
        days: int = 14,
        interval: int = 120,
        min_coverage: float | None = None,
    ):
        self.ids = list(dict.fromkeys(ids))
        self.days = days
        self.interval = interval
        self.min_coverage = (
            config.HISTORY_MIN_COVERAGE
            if min_coverage is None
            else float(min_coverage)
        )
        self._snapshot: Optional[HistorySet] = None
        self.last_partial_snapshot: Optional[HistorySet] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self.started_at = time.time()
        self.last_attempt_ts: Optional[float] = None
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0
        self.coverage_ratio = 0.0
        self.expected_count = len(self.ids)
        self.available_count = 0

    def get_latest(self) -> Optional[HistorySet]:
        """Return the last history set that passed coverage validation."""
        return self._snapshot

    def get_usable(self, max_age: float) -> Optional[HistorySet]:
        snapshot = self._snapshot
        if not snapshot or time.time() - snapshot.ts > max_age:
            return None
        return snapshot

    async def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait for the first validated history set without polling internals."""
        if self._snapshot is not None:
            return True
        try:
            if timeout is None:
                await self._ready.wait()
            else:
                await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return self._snapshot is not None
        except asyncio.TimeoutError:
            return False

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
            except asyncio.CancelledError:
                pass

    async def _fetch_binance_daily(self, ids: List[str]) -> Dict[str, pd.DataFrame]:
        """Daily candles from Binance for mapped coins: 20 coins load in seconds.

        CoinGecko's keyless budget (~4 req/min) made the 200-day history of
        20 coins take about 7 minutes, and analyses during warm-up showed no
        indicators. The last row is today's still-open candle, like the
        intraday point CoinGecko history ends with.
        """
        try:
            async with aiohttp.ClientSession() as session:
                rows = await asyncio.gather(*[
                    get_klines(SYMBOL_MAP[coin_id], "1d", limit=self.days + 1, session=session)
                    for coin_id in ids
                ])
        except Exception as exc:
            logger.warning("Binance daily history failed: %s", exc)
            return {}
        result: Dict[str, pd.DataFrame] = {}
        for coin_id, candles in zip(ids, rows):
            if not candles or len(candles) < 14:
                continue
            result[coin_id] = pd.DataFrame({
                "timestamp": pd.to_datetime([c["timestamp"] for c in candles], unit="ms"),
                "price": [c["close"] for c in candles],
                "volume": [c["quote_volume"] for c in candles],  # USD-like, as CoinGecko
            })
        return result

    async def _fetch_histories(self) -> Dict[str, pd.DataFrame]:
        async with self._lock:
            ids = list(self.ids)
        result = await self._fetch_binance_daily([c for c in ids if c in SYMBOL_MAP])
        # CoinGecko for coins Binance does not list or did not return.
        ids = [c for c in ids if c not in result]
        if not ids:
            return result
        data = await get_batch_historical_data(ids, days=self.days)
        for coin_id, payload in (data or {}).items():
            if not isinstance(payload, dict) or payload.get("error"):
                continue
            prices = payload.get("prices") or []
            volumes = payload.get("total_volumes") or []
            # RSI(14) cannot be represented truthfully with fewer observations.
            if len(prices) < 14:
                continue
            frame = pd.DataFrame(prices, columns=["timestamp", "price"])
            volume_frame = (
                pd.DataFrame(volumes, columns=["timestamp", "volume"])
                if volumes
                else pd.DataFrame(columns=["timestamp", "volume"])
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
            if not volume_frame.empty:
                volume_frame["timestamp"] = pd.to_datetime(
                    volume_frame["timestamp"], unit="ms"
                )
                frame = frame.merge(volume_frame, on="timestamp", how="left")
            result[coin_id] = frame
        return result

    async def refresh_once(self) -> bool:
        self.last_attempt_ts = time.time()
        try:
            data = await self._fetch_histories()
            completed_at = time.time()
            async with self._lock:
                expected = len(self.ids)
            self.expected_count = expected
            self.available_count = len(data)
            self.coverage_ratio = len(data) / expected if expected else 1.0
            if expected and not data:
                raise RuntimeError("history snapshot returned no tracked coins")

            candidate = HistorySet(ts=completed_at, by_coin=data)
            if self.coverage_ratio < self.min_coverage:
                self.last_partial_snapshot = candidate
                self.consecutive_failures += 1
                self.last_error = (
                    f"incomplete history coverage {len(data)}/{expected} "
                    f"(< {self.min_coverage:.0%})"
                )
                logger.warning(self.last_error)
                return False

            self._snapshot = candidate
            self._ready.set()
            self.last_partial_snapshot = None
            self.last_success_ts = completed_at
            self.last_error = None
            self.consecutive_failures = 0
            return True
        except Exception as error:
            self.consecutive_failures += 1
            self.available_count = 0
            self.coverage_ratio = 0.0
            self.last_error = f"{type(error).__name__}: {error}"
            logger.exception(
                "History snapshot refresh failed consecutive_failures=%s",
                self.consecutive_failures,
            )
            return False

    async def _run(self):
        # Until the first snapshot is published, retry on a short backoff
        # instead of waiting a full refresh interval (an hour by default).
        retry_delay = 60
        while self._running:
            ok = await self.refresh_once()
            if self._snapshot is None and not ok:
                await asyncio.sleep(min(retry_delay, self.interval))
                retry_delay = min(retry_delay * 2, 300)
                continue
            await asyncio.sleep(self.interval)

    async def update_ids(self, ids: List[str]):
        normalized = list(dict.fromkeys(ids))
        async with self._lock:
            self.ids = normalized
            self.expected_count = len(normalized)
