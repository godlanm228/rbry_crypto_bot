import asyncio
import logging
import pandas as pd
from typing import List, Dict, Optional
import aiohttp
from datetime import datetime, timedelta


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
    "polygon": "MATICUSDT",
    "uniswap": "UNIUSDT",
    "litecoin": "LTCUSDT",
    "cosmos": "ATOMUSDT"
}

async def get_klines(symbol: str, interval: str = "1h", limit: int = 500) -> List[Dict]:
    """Get OHLCV candles. Intervals: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M"""
    async with aiohttp.ClientSession() as session:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        url = f"{BINANCE_API_URL}/klines"

        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Binance error {resp.status}")
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
            logger.error(f"Binance klines error: {e}")
            return []

async def get_multi_timeframe_data(coin_id: str) -> Dict[str, pd.DataFrame]:
    """Get OHLCV for multiple timeframes from single coin"""

    symbol = SYMBOL_MAP.get(coin_id)
    if not symbol:
        logger.warning(f"Binance: нет символа для {coin_id}")
        return {}

    timeframes = {
        "1m": 300,   # 300 candles = 5 hours
        "5m": 288,   # 288 candles = 24 hours
        "15m": 192,  # 192 candles = 48 hours
        "1h": 168,   # 168 candles = 7 days
        "4h": 180,   # 180 candles = 30 days
        "1d": 90     # 90 candles = 3 months
    }

    results = {}
    for tf, limit in timeframes.items():
        data = await get_klines(symbol, tf, limit)
        if data is not None:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            results[tf] = df

    return results

async def get_batch_ohlcv(coin_ids: List[str], interval: str = "1h") -> Dict[str, pd.DataFrame]:
    """Get OHLCV for multiple coins in one interval"""
    import pandas as pd

    tasks = []
    valid_coins = []

    for coin_id in coin_ids:
        symbol = SYMBOL_MAP.get(coin_id)
        if symbol:
            valid_coins.append(coin_id)
            tasks.append(get_klines(symbol, interval))
        else:
            logger.warning(f"Binance: нет символа для {coin_id}")

    results = await asyncio.gather(*tasks)

    output = {}
    for coin_id, data in zip(valid_coins, results):
        if data is not None:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            output[coin_id] = df

    return output
