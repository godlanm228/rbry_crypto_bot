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

    def __init__(self):
        self._cache: Optional[Dict] = None
        self._cache_timestamp: float = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()

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
            session = await self._get_session()
            async with session.get(self.API_URL, timeout=aiohttp.ClientTimeout(total=10)) as response:
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
                    # Return stale cache if available
                    if self._cache:
                        logger.info("Using stale Fear & Greed cache")
                        return self._cache.get("value")
                    return None

        except asyncio.TimeoutError:
            logger.warning("Fear & Greed API timeout")
            # Return stale cache if available
            if self._cache:
                logger.info("Using stale Fear & Greed cache after timeout")
                return self._cache.get("value")
            return None

        except Exception as e:
            logger.error(f"Fear & Greed API error: {e}")
            # Return stale cache if available (graceful degradation)
            if self._cache:
                logger.info("Using stale Fear & Greed cache after error")
                return self._cache.get("value")
            return None

    async def get_historical(self, days: int = 30) -> List[Dict]:
        """
        Get historical Fear & Greed Index data

        Args:
            days: Number of days of historical data (max 365)

        Returns:
            List of dicts with keys: value, value_classification, timestamp
        """
        try:
            # API supports limit parameter for historical data
            url = f"{self.API_URL}?limit={days}"
            session = await self._get_session()

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data:
                        # Convert string values to floats
                        historical = []
                        for entry in data["data"]:
                            historical.append({
                                "value": float(entry["value"]),
                                "classification": entry.get("value_classification"),
                                "timestamp": entry.get("timestamp")
                            })
                        logger.info(f"Fetched {len(historical)} days of Fear & Greed history")
                        return historical
                    else:
                        logger.warning("Fear & Greed historical data empty")
                        return []
                else:
                    logger.warning(f"Fear & Greed historical API error: HTTP {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching Fear & Greed historical data: {e}")
            return []

    def get_classification(self, value: float) -> str:
        """
        Get text classification for a Fear & Greed value

        Args:
            value: Fear & Greed index (0-100)

        Returns:
            Classification string: 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
        """
        if value < 25:
            return "Extreme Fear"
        elif value < 45:
            return "Fear"
        elif value < 55:
            return "Neutral"
        elif value < 75:
            return "Greed"
        else:
            return "Extreme Greed"

    def normalize_to_sentiment(self, value: Optional[float]) -> float:
        """
        Normalize Fear & Greed index to -1 to 1 sentiment scale

        Args:
            value: Fear & Greed index (0-100), or None

        Returns:
            Normalized value: -1 (extreme fear) to 1 (extreme greed)
            Returns 0.0 if value is None
        """
        if value is None:
            return 0.0
        # Convert 0-100 to -1 to 1, centered at 50
        return (value - 50.0) / 50.0


# Global singleton instance
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
