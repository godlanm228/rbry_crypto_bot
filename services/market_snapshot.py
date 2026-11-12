from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from core.api.coingecko_client import get_markets_data


logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    ts: float
    markets: Dict[str, dict] = field(default_factory=dict)
    simple: Dict[str, dict] = field(default_factory=dict)


class MarketSnapshotService:
    """Publish only validated market snapshots while retaining diagnostics."""

    def __init__(
        self,
        ids: List[str],
        ttl: int = 20,
        min_coverage: float | None = None,
    ):
        self.ids = list(dict.fromkeys(ids))
        self.ttl = ttl
        self.min_coverage = (
            config.SNAPSHOT_MIN_COVERAGE
            if min_coverage is None
            else float(min_coverage)
        )
        self._snapshot: Optional[Snapshot] = None
        self.last_partial_snapshot: Optional[Snapshot] = None
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.started_at = time.time()
        self.last_attempt_ts: Optional[float] = None
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0
        self.coverage_ratio = 0.0
        self.expected_count = len(self.ids)
        self.available_count = 0

    def get_latest(self) -> Optional[Snapshot]:
        """Return the last snapshot that passed coverage validation."""
        return self._snapshot

    def get_usable(self, max_age: float) -> Optional[Snapshot]:
        snapshot = self._snapshot
        if not snapshot or time.time() - snapshot.ts > max_age:
            return None
        return snapshot

    def last_ts(self) -> Optional[float]:
        return self._snapshot.ts if self._snapshot else None

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

    async def _fetch_markets(self) -> dict:
        async with self._lock:
            ids = list(self.ids)
        if not ids:
            return {}
        rows = await get_markets_data(ids)
        usable = {}
        for coin_id, row in rows.items():
            price = row.get("current_price")
            if isinstance(price, (int, float)) and math.isfinite(price) and price > 0:
                usable[coin_id] = row
        return usable

    async def refresh_once(self) -> bool:
        self.last_attempt_ts = time.time()
        try:
            markets = await self._fetch_markets()
            completed_at = time.time()
            async with self._lock:
                expected = len(self.ids)
            self.expected_count = expected
            self.available_count = len(markets)
            self.coverage_ratio = len(markets) / expected if expected else 1.0
            if expected and not markets:
                raise RuntimeError("market snapshot returned no tracked coins")

            candidate = Snapshot(ts=completed_at, markets=markets)
            if self.coverage_ratio < self.min_coverage:
                # Keep the candidate for owner diagnostics, but never expose it
                # through get_latest() as though it were a healthy snapshot.
                self.last_partial_snapshot = candidate
                self.consecutive_failures += 1
                self.last_error = (
                    f"incomplete market coverage {len(markets)}/{expected} "
                    f"(< {self.min_coverage:.0%})"
                )
                logger.warning(self.last_error)
                return False

            self._snapshot = candidate
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
                "Market snapshot refresh failed consecutive_failures=%s",
                self.consecutive_failures,
            )
            return False

    async def _run(self):
        while self._running:
            await self.refresh_once()
            await asyncio.sleep(self.ttl)

    async def update_ids(self, ids: List[str]):
        normalized = list(dict.fromkeys(ids))
        async with self._lock:
            self.ids = normalized
            self.expected_count = len(normalized)
