"""Async pacing for the shared CoinGecko request budget."""

from __future__ import annotations

import asyncio
import logging
import os
import time


logger = logging.getLogger(__name__)


class ApiRateLimiter:
    """Serialize request starts and share upstream cooldowns across callers."""

    def __init__(self, cps: float | None = None):
        if cps is None:
            configured = os.getenv("COINGECKO_RPS")
            # Keyless CoinGecko has a dynamic, IP-wide public budget. Four
            # requests/minute leaves headroom below the observed 5/minute cap.
            cps = float(configured) if configured else 1.0 / 15.0
        if cps <= 0:
            raise ValueError("CoinGecko requests per second must be positive")
        self.min_interval = 1.0 / cps
        self._last_call = 0.0
        self._blocked_until = 0.0
        self._lock: asyncio.Lock | None = None
        self._lock_loop = None

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def wait_if_needed(self) -> None:
        """Grant one request slot; concurrent callers cannot leave in a burst."""
        async with self._get_lock():
            now = time.monotonic()
            allowed_at = max(
                self._blocked_until,
                self._last_call + self.min_interval,
            )
            wait = allowed_at - now
            if wait > 0:
                logger.debug("CoinGecko pacing wait %.2fs", wait)
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    async def cooldown(self, seconds: float) -> None:
        """Apply a shared Retry-After/backoff window to every caller."""
        delay = max(0.0, float(seconds))
        async with self._get_lock():
            self._blocked_until = max(
                self._blocked_until,
                time.monotonic() + delay,
            )

    def reset(self) -> None:
        """Reset timing state (primarily useful for deterministic tests)."""
        self._last_call = 0.0
        self._blocked_until = 0.0


api_limiter = ApiRateLimiter()
