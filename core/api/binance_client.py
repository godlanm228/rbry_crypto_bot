import asyncio
import logging
import pandas as pd
from typing import List, Dict, Optional
import aiohttp


logger = logging.getLogger(__name__)

BINANCE_API_URL = "https://api.binance.com/api/v3"
SYMBOL_MAP = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
    "ripple": "XRPUSDT",
    "cardano": "ADAUSDT",
    "avalanche-2": "AVAXUSDT",
    "dogecoin": "DOGEUSDT",
    "polkadot": "DOTUSDT",
    "chainlink": "LINKUSDT",
    # MATIC was rebranded to POL in Sept 2024; MATICUSDT is delisted on Binance
    "polygon": "POLUSDT",
    "polygon-ecosystem-token": "POLUSDT",
    "uniswap": "UNIUSDT",
    "litecoin": "LTCUSDT",
    "cosmos": "ATOMUSDT",
    "tron": "TRXUSDT",
    "shiba-inu": "SHIBUSDT",
    "internet-computer": "ICPUSDT",
    "near": "NEARUSDT",
    "stellar": "XLMUSDT",
    "bitcoin-cash": "BCHUSDT",
    # leo-token is not traded on Binance — intentionally unmapped
}


async def get_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 500,
    session: Optional[aiohttp.ClientSession] = None,
) -> Optional[List[Dict]]:
    """Get OHLCV candles. Intervals: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M

    Returns None on any error (never an empty list for errors), so callers can
    distinguish "no data" from "request failed".
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000)
    }
    url = f"{BINANCE_API_URL}/klines"

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                logger.error(f"Binance error {resp.status} for {symbol}")
                return None

            data = await resp.json()
            return [
                {
                    "timestamp": row[0],
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": row[6],
                    "quote_volume": float(row[7]),
                    "trades": int(row[8])
                }
                for row in data
            ]
    except Exception as e:
        logger.error(f"Binance klines error for {symbol}: {e!r}")
        return None
    finally:
        if owns_session:
            await session.close()


async def get_batch_ohlcv(coin_ids: List[str], interval: str = "1h") -> Dict[str, pd.DataFrame]:
    """Get OHLCV for multiple coins in one interval (single shared HTTP session)."""
    valid_coins = []
    symbols = []

    for coin_id in coin_ids:
        symbol = SYMBOL_MAP.get(coin_id)
        if symbol:
            valid_coins.append(coin_id)
            symbols.append(symbol)
        else:
            logger.debug(f"Binance: нет символа для {coin_id}")

    if not valid_coins:
        return {}

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[get_klines(symbol, interval, session=session) for symbol in symbols]
        )

    output = {}
    for coin_id, data in zip(valid_coins, results):
        # data is None on error and may be [] when Binance has no candles —
        # an empty DataFrame has no 'timestamp' column and would crash set_index
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            output[coin_id] = df

    return output
