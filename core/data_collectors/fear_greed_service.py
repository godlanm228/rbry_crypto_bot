"""
Fear & Greed Index Service

Fetches the Crypto Fear & Greed Index from alternative.me API.
This provides market sentiment data (0-100 scale) with 1-hour caching.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Optional, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class FearGreedService:
    """
    Service for fetching Crypto Fear & Greed Index

    API: https://api.alternative.me/fng/
    Free, no API key required

    Index ranges from 0 (Extreme Fear) to 100 (Extreme Greed)
    """

    API_URL = "https://api.alternative.me/fng/"
    CACHE_TTL = 3600  # 1 hour cache
    # The index updates daily; an older fallback value is no longer evidence.
    MAX_STALE_SECONDS = 36 * 3600

    def __init__(self):
        self._cache: Optional[Dict] = None
        self._cache_timestamp: float = 0

    async def close(self):
        """Kept for callers; requests use a per-call session now."""
        return None

    def _stale_value(self) -> Optional[float]:
        if self._cache and time.time() - self._cache_timestamp < self.MAX_STALE_SECONDS:
            logger.info("Using stale Fear & Greed cache")
            return self._cache.get("value")
        return None

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if self._cache is None:
            return False
        age = time.time() - self._cache_timestamp
        return age < self.CACHE_TTL

    async def get_current_index(self) -> Optional[float]:
        """
        Get current Fear & Greed Index value (0-100)

        Returns:
            float: Index value 0-100, or None if API unavailable

        Index interpretation:
            0-24: Extreme Fear
            25-44: Fear
            45-55: Neutral
            56-75: Greed
            76-100: Extreme Greed
        """
        # Check cache first
        if self._is_cache_valid():
            logger.debug("Fear & Greed: using cached value")
            return self._cache.get("value")

        try:
            # Per-call session: a shared one was bound to the first event loop,
            # never closed ("Unclosed client session") and failed in later loops.
            async with aiohttp.ClientSession() as session,                     session.get(self.API_URL, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()

                    # API returns: {"data": [{"value": "50", "value_classification": "Neutral", "timestamp": "..."}]}
                    if "data" in data and len(data["data"]) > 0:
                        current = data["data"][0]
                        value = float(current["value"])
                        classification = current.get("value_classification", "Unknown")

                        # Cache the result
                        self._cache = {
                            "value": value,
                            "classification": classification,
                            "timestamp": current.get("timestamp"),
                            "time_until_update": current.get("time_until_update")
                        }
                        self._cache_timestamp = time.time()

                        logger.info(f"Fear & Greed Index: {value} ({classification})")
                        return value
                    else:
                        logger.warning("Fear & Greed API returned empty data")
                        return None
                else:
                    logger.warning(f"Fear & Greed API error: HTTP {response.status}")
                    return self._stale_value()

        except asyncio.TimeoutError:
            logger.warning("Fear & Greed API timeout")
            return self._stale_value()

        except Exception as e:
            logger.error(f"Fear & Greed API error: {e!r}")
            return self._stale_value()


_fear_greed_service: Optional[FearGreedService] = None


def get_fear_greed_service() -> FearGreedService:
    """Get global Fear & Greed service instance"""
    global _fear_greed_service
    if _fear_greed_service is None:
        _fear_greed_service = FearGreedService()
    return _fear_greed_service


async def cleanup_fear_greed_service():
    """Cleanup global Fear & Greed service (call on shutdown)"""
    global _fear_greed_service
    if _fear_greed_service:
        await _fear_greed_service.close()
        _fear_greed_service = None
