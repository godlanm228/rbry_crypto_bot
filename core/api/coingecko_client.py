import asyncio , logging , os
from typing import List, Dict, Optional
from config import COINGECKO_API_URL
from core.rate_limiter import api_limiter
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_session: Optional[ClientSession] = None
logger = logging.getLogger(__name__)
COINGECKO_CONCURRENCY = int(os.getenv("COINGECKO_CONCURRENCY", "4"))

def _get_timeout() -> ClientTimeout:
    return ClientTimeout(total=12, connect=4, sock_read=8)

async def get_session() -> ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=_get_timeout(), raise_for_status=False)
    return _session

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
       retry=retry_if_exception_type((asyncio.TimeoutError, ClientError)))
async def _fetch_json(session: ClientSession, url: str, params: Dict) -> Dict:
    await api_limiter.wait_if_needed()
    async with session.get(url, params=params) as resp:
        await api_limiter.wait_if_needed()
        if resp.status == 429:
            retry_after = int(resp.headers.get("Retry-After", "2"))
            retry_after = max(2, min(retry_after, 10))  # не спим по минуте
            logger.warning(f"429 Too Many Requests. Спим {retry_after}с")
            await asyncio.sleep(retry_after)
            raise ClientError("429 Too Many Requests")
        elif resp.status != 200:
            text = await resp.text()
            raise ClientError(f"{resp.status}: {text[:200]}")
        return await resp.json()

async def get_multiple_prices(coin_ids: List[str]) -> Dict[str, Dict[str, float]]:
    await api_limiter.wait_if_needed()
    session = await get_session()
    url = f"{COINGECKO_API_URL}/simple/price"
    params = {
        "ids": ",".join(coin_ids),
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_last_updated_at": "true"
    }
    data = await _fetch_json(session, url, params)
    # /simple/price -> dict: { "bitcoin": {"usd":..., "usd_24h_change":...}, ... }
    if isinstance(data, dict):
        return data
    logger.warning("simple/price returned non-dict (rate-limit or error)")
    return {}

async def get_batch_historical_data(coin_ids: List[str], days: int = 30) -> Dict[str, Dict]:
    """
    Батч-запрос истории цен за N дней (daily) конкурентно.
    Возвращает: {'bitcoin': {...market_chart JSON...}, ...}
    """
    await api_limiter.wait_if_needed()  # один лимит на весь батч
    session = await get_session()
    sem = asyncio.Semaphore(COINGECKO_CONCURRENCY)

    async def fetch_one(coin_id: str):
        url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        async with sem:
            try:
                await api_limiter.wait_if_needed()       # <<< добавь
                data = await _fetch_json(session, url, params)
                return coin_id, data
            except Exception as e:
                logger.warning(f"[history] {coin_id}: {e}")
                return coin_id, {"error": str(e)}

    results = await asyncio.gather(*[fetch_one(cid) for cid in coin_ids], return_exceptions=False)

    out: Dict[str, Dict] = {}
    for cid, data in results:
        out[cid] = data
    return out

async def get_markets_data(coin_ids: List[str]) -> Dict[str, dict]:
    """Один запрос для всех монет - цены, объемы, изменения"""
    await api_limiter.wait_if_needed()
    session = await get_session()
    url = f"{COINGECKO_API_URL}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ",".join(coin_ids),
        "price_change_percentage": "1h,24h,7d",
        "order": "market_cap_desc",
        "per_page": len(coin_ids) if coin_ids else 50,
        "sparkline": "false"
    }
    logger.info(f"Запрос markets: URL={url}, params={params}")
    data = await _fetch_json(session, url, params)
    logger.info(f"Ответ markets: status=ok, records={len(data) if isinstance(data, list) else 'dict'}")
    if isinstance(data, list):
        return {row["id"]: row for row in data}
    else:
        logger.error(f"Unexpected markets response: {data}")
        return {}

async def close_session():
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None
