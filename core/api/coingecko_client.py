"""Resilient CoinGecko client shared by snapshots and on-demand handlers."""

from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import aiohttp
from aiohttp import ClientError, ClientSession, ClientTimeout

import config
from core.rate_limiter import api_limiter


logger = logging.getLogger(__name__)
_session: Optional[ClientSession] = None
_MAX_ATTEMPTS = max(1, int(os.getenv("COINGECKO_MAX_ATTEMPTS", "3")))
_BATCH_FAILURE_BUDGET = max(1, int(os.getenv("COINGECKO_BATCH_FAILURE_BUDGET", "3")))


class CoinGeckoError(RuntimeError):
    """Base exception for a failed CoinGecko request."""


class CoinGeckoHTTPError(CoinGeckoError):
    def __init__(self, status: int, detail: str = ""):
        super().__init__(f"HTTP {status}" + (f": {detail}" if detail else ""))
        self.status = status


class CoinGeckoRateLimitError(CoinGeckoHTTPError):
    def __init__(self, retry_after: float):
        super().__init__(429, "rate limited")
        self.retry_after = retry_after


class CoinGeckoTransientError(CoinGeckoHTTPError):
    pass


def _get_timeout() -> ClientTimeout:
    return ClientTimeout(total=25, connect=8, sock_read=20)


def _session_headers() -> dict[str, str]:
    api_key = getattr(config, "COINGECKO_API_KEY", "")
    # The configured base URL is the Demo/Public host, so the documented Demo
    # header is the safe keyed option. Keyless operation sends no auth header.
    return {"x-cg-demo-api-key": api_key} if api_key else {}


async def get_session() -> ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            timeout=_get_timeout(),
            raise_for_status=False,
            headers=_session_headers(),
        )
    return _session


def _endpoint_name(url: str) -> str:
    return urlsplit(url).path


def _retry_after_seconds(raw_value: str | None, attempt: int) -> float:
    if raw_value:
        try:
            return max(0.0, min(float(raw_value), 300.0))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(raw_value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(
                    0.0,
                    min((retry_at - datetime.now(timezone.utc)).total_seconds(), 300.0),
                )
            except (TypeError, ValueError, OverflowError):
                pass
    # Full jitter avoids every consumer retrying on the same boundary.
    ceiling = min(60.0, 2.0 ** (attempt + 2))
    return random.uniform(ceiling / 2.0, ceiling)


async def _fetch_json(
    session: ClientSession,
    url: str,
    params: Dict[str, Any] | None = None,
) -> Any:
    """Fetch JSON with shared pacing and retry only transient failures."""
    endpoint = _endpoint_name(url)
    last_error: BaseException | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        await api_limiter.wait_if_needed()
        try:
            async with session.get(url, params=params or {}) as response:
                if response.status == 429:
                    retry_after = _retry_after_seconds(
                        response.headers.get("Retry-After"),
                        attempt,
                    )
                    await response.read()
                    raise CoinGeckoRateLimitError(retry_after)
                if 500 <= response.status < 600:
                    detail = (await response.text())[:160]
                    raise CoinGeckoTransientError(response.status, detail)
                if response.status != 200:
                    detail = (await response.text())[:160]
                    raise CoinGeckoHTTPError(response.status, detail)
                return await response.json()

        except CoinGeckoRateLimitError as error:
            last_error = error
            await api_limiter.cooldown(error.retry_after)
            logger.warning(
                "CoinGecko rate limited endpoint=%s retry_in=%.1fs attempt=%s/%s",
                endpoint,
                error.retry_after,
                attempt,
                _MAX_ATTEMPTS,
            )
        except CoinGeckoTransientError as error:
            last_error = error
            delay = _retry_after_seconds(None, attempt)
            await api_limiter.cooldown(delay)
            logger.warning(
                "CoinGecko transient status=%s endpoint=%s retry_in=%.1fs attempt=%s/%s",
                error.status,
                endpoint,
                delay,
                attempt,
                _MAX_ATTEMPTS,
            )
        except (asyncio.TimeoutError, ClientError) as error:
            last_error = error
            delay = _retry_after_seconds(None, attempt)
            await api_limiter.cooldown(delay)
            logger.warning(
                "CoinGecko transport failure=%s endpoint=%s retry_in=%.1fs attempt=%s/%s",
                type(error).__name__,
                endpoint,
                delay,
                attempt,
                _MAX_ATTEMPTS,
            )
        except CoinGeckoHTTPError:
            # Authentication, invalid IDs and other permanent 4xx responses
            # must not consume more of the minute budget through blind retries.
            raise

        if attempt == _MAX_ATTEMPTS:
            break

    if isinstance(last_error, CoinGeckoError):
        raise last_error
    raise CoinGeckoError(
        f"request failed after {_MAX_ATTEMPTS} attempts: "
        f"{type(last_error).__name__ if last_error else 'unknown'}"
    ) from last_error


async def get_batch_historical_data(
    coin_ids: List[str],
    days: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """Fetch histories sequentially so market traffic is never hit by a burst."""
    session = await get_session()
    unique_ids = list(dict.fromkeys(coin_ids))
    output: Dict[str, Dict[str, Any]] = {}
    consecutive_failures = 0

    for index, coin_id in enumerate(unique_ids):
        try:
            data = await _fetch_json(
                session,
                f"{config.COINGECKO_API_URL}/coins/{coin_id}/market_chart",
                {"vs_currency": "usd", "days": days, "interval": "daily"},
            )
            if not isinstance(data, dict):
                raise CoinGeckoError("history response is not an object")
            output[coin_id] = data
            consecutive_failures = 0
        except Exception as error:
            consecutive_failures += 1
            output[coin_id] = {"error": type(error).__name__}
            logger.warning(
                "CoinGecko history failed coin=%s error=%s streak=%s",
                coin_id,
                type(error).__name__,
                consecutive_failures,
            )
            if consecutive_failures >= _BATCH_FAILURE_BUDGET:
                remaining = unique_ids[index + 1:]
                for remaining_id in remaining:
                    output[remaining_id] = {"error": "batch_circuit_open"}
                logger.warning(
                    "CoinGecko history batch stopped after %s consecutive failures; skipped=%s",
                    consecutive_failures,
                    len(remaining),
                )
                break
    return output


async def get_markets_data(coin_ids: List[str]) -> Dict[str, dict]:
    """Fetch rich market rows in bounded chunks through the shared budget."""
    ids = list(dict.fromkeys(coin_ids))
    if not ids:
        return {}
    session = await get_session()
    markets: Dict[str, dict] = {}
    url = f"{config.COINGECKO_API_URL}/coins/markets"
    for offset in range(0, len(ids), 200):
        chunk = ids[offset:offset + 200]
        data = await _fetch_json(
            session,
            url,
            {
                "vs_currency": "usd",
                "ids": ",".join(chunk),
                "price_change_percentage": "1h,24h,7d",
                "order": "market_cap_desc",
                "per_page": len(chunk),
                "page": 1,
                "sparkline": "false",
            },
        )
        if not isinstance(data, list):
            raise CoinGeckoError("markets response is not a list")
        markets.update({row.get("id"): row for row in data if row.get("id")})
    logger.info("CoinGecko markets received records=%s requested=%s", len(markets), len(ids))
    return markets


async def get_top_markets(limit: int = 20) -> List[dict]:
    session = await get_session()
    data = await _fetch_json(
        session,
        f"{config.COINGECKO_API_URL}/coins/markets",
        {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": max(1, min(limit, 250)),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h,7d",
        },
    )
    if not isinstance(data, list):
        raise CoinGeckoError("top markets response is not a list")
    return data


async def get_market_cap_ranks(pages: int = 4) -> Dict[str, int]:
    """Market-cap rank per CoinGecko id for the top ``pages * 250`` coins."""
    session = await get_session()
    ranks: Dict[str, int] = {}
    for page in range(1, pages + 1):
        data = await _fetch_json(
            session,
            f"{config.COINGECKO_API_URL}/coins/markets",
            {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": page,
                "sparkline": "false",
            },
        )
        if not isinstance(data, list):
            raise CoinGeckoError("markets response is not a list")
        for row in data:
            coin_id, rank = row.get("id"), row.get("market_cap_rank")
            if coin_id and rank:
                ranks[coin_id] = int(rank)
    return ranks


async def get_coin_list() -> List[dict]:
    session = await get_session()
    data = await _fetch_json(
        session,
        f"{config.COINGECKO_API_URL}/coins/list",
    )
    if not isinstance(data, list):
        raise CoinGeckoError("coin list response is not a list")
    return data


async def close_session() -> None:
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None
