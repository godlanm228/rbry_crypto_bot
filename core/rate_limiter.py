import time, os
import logging
import asyncio
logger = logging.getLogger(__name__)

class ApiRateLimiter:
    def __init__(self, cps: float | None = None):
        if cps is None:
            cps = float(os.getenv("COINGECKO_RPS", "1.0"))  # 1 req / 5s по умолчанию
        self.min_interval = 1.0 / cps
        self._last_call = 0.0

    async def wait_if_needed(self):
        now = time.monotonic()
        delta = now - self._last_call
        if delta < self.min_interval:
            wait = self.min_interval - delta
            if wait > 0:
                logging.debug(f"Rate limiting: waiting {wait:.2f}s")  # DEBUG
                await asyncio.sleep(wait)
        self._last_call = time.monotonic()

# Глобальный лимитер для API - ОЧЕНЬ медленный для бесплатного API
api_limiter = ApiRateLimiter()
