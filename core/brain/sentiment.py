"""Who is buying right now (Vlad, 30.10.2026): the US, Korea's retail, and the options market's fear.

Free and keyless, by the hour (market_data.py sentiment):
  Coinbase premium  BTC-USD / ETH-USD on Coinbase over BTCUSDT / ETHUSDT on Binance, less 1: above its
                    usual, Americans (funds, the ETFs' buyers) pay up; since 08.2017
  Korean premium    KRW-BTC / KRW-ETH on Upbit in dollars (the ECB's won rate of the day before) over
                    Binance, less 1: Korea's retail overpaying; since 09.2017
  fear              Deribit's DVOL for BTC and ETH: the move options price for the next 30 days; since 03.2021
One rule for all three, set before any test: a value above all but 5% of the 30 days before it (HI) or
below all but 5% (LO), at the hour that state begins, at most once a day. Each coin's own measure goes
on that coin (BTC's on BTC, ETH's on ETH); the signal is the hour's close, the entry the next hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.brain.dataset import DAY_MS
from core.brain.events import fire
from core.sandbox import context_sources as cs
from core.sandbox.market_store import HOUR_MS

WINDOW_HOURS = 720
EDGE_Q = 0.05
COINS = ("BTC", "ETH")
FX = "fx:usdkrw"


def _ms(day: str) -> int:
    return int(pd.Timestamp(day).value // 10**6)


# hourly series name: (fetch, ticker, first hour)
SOURCES = {f"{src}:{c}": (fetch, ticker.format(c=c), _ms(first))
           for src, fetch, ticker, first in (("coinbase", cs.fetch_coinbase, "{c}-USD", "2017-08-01"),
                                             ("upbit", cs.fetch_upbit, "KRW-{c}", "2017-09-25"),
                                             ("dvol", cs.fetch_dvol, "{c}", "2021-03-01"))
           for c in COINS}


def on_grid(frame: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """An hourly series' values at the grid's hours (NaN where it has none)."""
    if not len(frame):
        return np.full(len(grid), np.nan)
    return pd.Series(frame["value"].to_numpy(float), index=frame["ts"].to_numpy(np.int64)).reindex(grid).to_numpy()


def won_per_dollar(fx: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """The won's rate known at each hour: the latest ECB business day that had ended before it."""
    if not len(fx):
        return np.full(len(grid), np.nan)
    known = fx["day"].to_numpy().astype("datetime64[ms]").astype(np.int64) + DAY_MS
    k = np.searchsorted(known, grid, side="right") - 1
    return np.where(k >= 0, fx["value"].to_numpy(float)[np.clip(k, 0, None)], np.nan)


def extremes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hours a value is above all but 5% of the 30 days before it, and below all but 5%."""
    v = pd.Series(values, dtype=float)
    past = v.shift(1).rolling(WINDOW_HOURS, min_periods=WINDOW_HOURS * 3 // 4)
    return (v > past.quantile(1 - EDGE_Q)).to_numpy(), (v < past.quantile(EDGE_Q)).to_numpy()


def measures(base: str, store, market) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The hourly grid of a coin's Binance history and its three measures on it."""
    p = market.prices(base)
    grid = np.arange(int(p[0, 0]), int(p[-1, 0]) + HOUR_MS, HOUR_MS, dtype=np.int64)
    close = pd.Series(p[:, 4], index=p[:, 0].astype(np.int64)).reindex(grid).to_numpy()
    return grid, {"CB": on_grid(store.hourly(f"coinbase:{base}"), grid) / close - 1,
                  "KR": on_grid(store.hourly(f"upbit:{base}"), grid) / won_per_dollar(store.series(FX), grid)
                  / close - 1,
                  "FEAR": on_grid(store.hourly(f"dvol:{base}"), grid)}


def all_events(store, market) -> dict[str, list[tuple[str, int]]]:
    """{base: [(event, signal ms)]} for BTC and ETH."""
    out: dict[str, list] = {}
    for base in COINS:
        if base not in market.base_of:
            continue
        grid, found = measures(base, store, market)
        for name, values in found.items():
            hi, lo = extremes(values)
            for side, mask in (("HI", hi), ("LO", lo)):
                out.setdefault(base, []).extend((f"{name}-{side}", int(grid[i]) + HOUR_MS)
                                                for i in fire(mask, edge=True))
    return out
