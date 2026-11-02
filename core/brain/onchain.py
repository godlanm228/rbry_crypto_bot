"""On-chain events (Etherscan's non-obvious use, without paid data): money moving to and from Binance.

Reading every transfer is out of reach for free (Binance's USDT hot wallet alone moves ~1,600 times an
hour, Etherscan's free plan returns 1,000 rows a call), so the history comes from DefiLlama's free
proof-of-reserves tracking of Binance's wallets (every chain, every token it holds, a day since
11.2022, market_data.py cex) and from the stablecoins' total supply (stablecoins_usd). A day's change
of a snapshot is that day's net flow. Thresholds were set before any test:
  CEX-IN       a token's amount on Binance up 10%+ in a day, worth $1M+: supply arriving to be sold
               (the free stand-in for unlock data: a team's or fund's tokens show up here when they
               are moved to be sold)
  CEX-OUT      down 10%+, worth $1M+: coins withdrawn to be held
  STABLE-IN    Binance's stablecoins up 3%+ in a day: money arriving to buy (BTC and ETH)
  STABLE-OUT   down 3%+: money leaving (BTC and ETH)
  STABLE-MINT  every stablecoin's supply up 0.5%+ in a day: new money printed (BTC and ETH)
DefiLlama rewrites a day's point through that day (at 18:00 UTC the point stamped today already held
the latest value), so a point stamped D is D's closing amount and is taken as known at 03:00 UTC of
D+1: the signal is the close of 02:00-03:00 that next day, the entry the hour after, like every library
event. The point of the unfinished day is dropped. A day when GLITCH_COINS or more coins move the same
way is DefiLlama adding or losing one of Binance's wallets (: 32 coins "in" at once; 09.10 and
10.10.2026: 15 out, then the same 15 back), not money moving, and none of that day's coin events count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.brain.dataset import DAY_MS
from core.sandbox.market_store import HOUR_MS

KNOWN_AFTER_MS = DAY_MS + 3 * HOUR_MS
GLITCH_COINS = 6
TOKEN_MOVE = 0.10
TOKEN_USD = 1e6
STABLE_MOVE = 0.03
MINT_MOVE = 0.005
STABLES = ("USDT", "USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USDE", "PYUSD", "USD1")
MARKET_COINS = ("BTC", "ETH")
TITLES = {"CEX-IN": "занос токена на Binance: +10% резерва за день",
          "CEX-OUT": "вывоз токена с Binance: −10% резерва за день",
          "STABLE-IN": "занос стейблкоинов на Binance: +3% за день",
          "STABLE-OUT": "вывоз стейблкоинов с Binance: −3% за день",
          "STABLE-MINT": "выпуск стейблкоинов: +0.5% всего предложения за день"}


def _signal(day: pd.Timestamp) -> int:
    return int(day.value // 10**6) + KNOWN_AFTER_MS


def token_events(series: pd.DataFrame, price_at) -> list[tuple[str, int]]:
    """``series``: day, value (a token's amount on Binance); ``price_at(ms)``: its dollar price then."""
    s = series.sort_values("day").reset_index(drop=True)
    out = []
    amounts = s["value"].to_numpy(dtype=float)
    for i in range(1, len(s)):
        prev, now = amounts[i - 1], amounts[i]
        if prev <= 0 or (s["day"].iat[i] - s["day"].iat[i - 1]).days != 1:
            continue
        change = now / prev - 1
        if abs(change) < TOKEN_MOVE:
            continue
        price = price_at(_signal(s["day"].iat[i]))
        if not np.isfinite(price) or abs(now - prev) * price < TOKEN_USD:
            continue
        out.append(("CEX-IN" if change > 0 else "CEX-OUT", _signal(s["day"].iat[i])))
    return out


def day_moves(series: pd.DataFrame, move: float, up: str, down: str = "") -> list[tuple[str, int]]:
    """A daily series' day-over-day moves of ``move`` or more, as events."""
    s = series.sort_values("day").reset_index(drop=True)
    v = s["value"].to_numpy(dtype=float)
    out = []
    for i in range(1, len(s)):
        if v[i - 1] <= 0 or (s["day"].iat[i] - s["day"].iat[i - 1]).days != 1:
            continue
        change = v[i] / v[i - 1] - 1
        if change >= move:
            out.append((up, _signal(s["day"].iat[i])))
        elif down and change <= -move:
            out.append((down, _signal(s["day"].iat[i])))
    return out


def stable_total(store, exchange: str = "binance") -> pd.DataFrame:
    """The exchange's stablecoins a day, summed (amounts are dollars)."""
    parts = [store.series(f"cex:{exchange}:{s}") for s in STABLES]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["day", "value"])
    total = pd.concat(parts).groupby("day")["value"].sum().reset_index()
    return total


def all_events(store, market) -> dict[str, list[tuple[str, int]]]:
    """{base: [(event, signal ms)]} of every on-chain event, for the library's coins."""
    out: dict[str, list] = {}
    crowd: dict[tuple[str, int], int] = {}
    with store.connect() as conn:
        names = [r[0] for r in conn.execute("SELECT DISTINCT series FROM daily_series WHERE series LIKE 'cex:binance:%'")]
    for name in names:
        base = name.split(":", 2)[2]
        if base in STABLES or base not in market.base_of:
            continue
        prices = market.prices(base)
        ts, close = prices[:, 0], prices[:, 4]

        def price_at(ms, ts=ts, close=close):
            k = int(np.searchsorted(ts, ms)) - 1
            return float(close[k]) if 0 <= k < len(close) else float("nan")
        events = token_events(store.series(name), price_at)
        if events:
            out.setdefault(base, []).extend(events)
            for e in events:
                crowd[e] = crowd.get(e, 0) + 1
    out = {b: [e for e in lst if crowd[e] < GLITCH_COINS] for b, lst in out.items()}
    out = {b: lst for b, lst in out.items() if lst}
    market_events = day_moves(stable_total(store), STABLE_MOVE, "STABLE-IN", "STABLE-OUT")
    market_events += day_moves(store.series("stablecoins_usd"), MINT_MOVE, "STABLE-MINT")
    for coin in MARKET_COINS:
        out.setdefault(coin, []).extend(market_events)
    return out
