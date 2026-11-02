"""History for the sandbox: Binance daily candles, the coin universe, Fear & Greed.

Offline tooling (scripts/sandbox_replay.py), synchronous on purpose. Only
closed daily candles are stored; a re-run fetches just the missing tail.

Survivorship bias, stated once: the universe is today's top coins, so coins
that died since 2022 are missing and absolute results lean optimistic.
Comparisons against the ``control`` rule on the same coins are not affected
the same way, which is why every verdict is relative to it.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd

from core.risk.exposure import DEFAULT_STABLECOIN_SYMBOLS
from core.sandbox.store import LearningStore

logger = logging.getLogger(__name__)

KLINES_URLS = ("https://api.binance.com/api/v3/klines", "https://data-api.binance.vision/api/v3/klines")
EXCHANGE_INFO_URLS = ("https://api.binance.com/api/v3/exchangeInfo", "https://data-api.binance.vision/api/v3/exchangeInfo")
COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
HISTORY_START = "2022-01-01"
DAY_MS = 86_400_000

# Not separate bets: pegged, wrapped or staked versions of another asset.
EXCLUDED = set(DEFAULT_STABLECOIN_SYMBOLS) | {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "PYUSD", "USDD", "USD1", "RLUSD", "USDP",
    "BUSD", "EURC", "EURT", "AEUR", "XAUT", "PAXG", "WBTC", "WETH", "STETH", "WSTETH", "WEETH",
    "CBBTC", "RETH", "BNSOL", "JITOSOL", "MSOL", "LBTC", "SOLVBTC", "BTCB", "CLBTC", "WBETH", "BFUSD",
}


def _get_json(urls, params, timeout=20):
    last = None
    for url in urls if isinstance(urls, tuple) else (urls,):
        try:
            resp = httpx.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            last = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except httpx.HTTPError as exc:
            last = exc
    raise RuntimeError(f"request failed: {last}")


def tradable_usdt_pairs() -> set[str]:
    info = _get_json(EXCHANGE_INFO_URLS, {"permissions": "SPOT"}, timeout=60)
    return {s["baseAsset"] for s in info["symbols"]
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"}


def pick_universe(size: int = 50, pool: int = 150) -> list[dict]:
    """Top coins by market cap that trade against USDT on Binance."""
    markets = _get_json(COINGECKO_MARKETS, {"vs_currency": "usd", "order": "market_cap_desc",
                                            "per_page": pool, "page": 1})
    bases = tradable_usdt_pairs()
    picked = []
    for row in markets:
        symbol = str(row.get("symbol") or "").upper()
        if symbol in EXCLUDED or symbol not in bases or any(p["symbol"] == symbol for p in picked):
            continue
        picked.append({"symbol": symbol, "coin_id": row.get("id"), "pair": f"{symbol}USDT",
                       "rank": row.get("market_cap_rank")})
        if len(picked) >= size:
            break
    if "BTC" not in {p["symbol"] for p in picked}:
        picked.insert(0, {"symbol": "BTC", "coin_id": "bitcoin", "pair": "BTCUSDT", "rank": 1})
    return picked


def fetch_daily(pair: str, start_day: str, now_ms: Optional[int] = None) -> pd.DataFrame:
    """Closed daily candles from ``start_day`` (inclusive), oldest first."""
    now_ms = now_ms or int(time.time() * 1000)
    start = int(datetime.strptime(start_day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    rows: list[list] = []
    while start < now_ms:
        page = _get_json(KLINES_URLS, {"symbol": pair, "interval": "1d", "startTime": start, "limit": 1000})
        if not page:
            break
        rows.extend(page)
        start = int(page[-1][0]) + DAY_MS
        if len(page) < 1000:
            break
        time.sleep(0.2)
    closed = [r for r in rows if int(r[6]) < now_ms]
    frame = pd.DataFrame({
        "day": pd.to_datetime([int(r[0]) for r in closed], unit="ms").normalize(),
        "open": [float(r[1]) for r in closed], "high": [float(r[2]) for r in closed],
        "low": [float(r[3]) for r in closed], "close": [float(r[4]) for r in closed],
        "volume": [float(r[7]) for r in closed],   # quote volume, USDT
    })
    return frame.drop_duplicates("day").reset_index(drop=True)


def update_candles(store: LearningStore, universe: list[dict], start_day: str = HISTORY_START) -> dict:
    counts = {}
    for coin in universe:
        last = store.last_day(coin["symbol"])
        since = start_day if last is None else (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            frame = fetch_daily(coin["pair"], since)
        except RuntimeError as exc:
            logger.warning("candles %s: %r", coin["pair"], exc)
            continue
        counts[coin["symbol"]] = store.save_candles(coin["symbol"], frame) if not frame.empty else 0
    return counts


def update_fear_greed(store: LearningStore) -> int:
    data = _get_json(FEAR_GREED_URL, {"limit": 0, "format": "json"}, timeout=30).get("data", [])
    rows = [(datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d"), int(d["value"]))
            for d in data]
    store.save_fear_greed(rows)
    return len(rows)
