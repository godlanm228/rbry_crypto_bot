from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

import config
from core.api.coingecko_client import get_batch_historical_data


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

    async def _fetch_histories(self) -> Dict[str, pd.DataFrame]:
        async with self._lock:
            ids = list(self.ids)
        data = await get_batch_historical_data(ids, days=self.days)
        result: Dict[str, pd.DataFrame] = {}
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
        while self._running:
            await self.refresh_once()
            await asyncio.sleep(self.interval)

    async def update_ids(self, ids: List[str]):
        normalized = list(dict.fromkeys(ids))
        async with self._lock:
            self.ids = normalized
            self.expected_count = len(normalized)
