"""Turn stored candles into situations and simulated trades.

Two passes: situations for every coin first (the ``momentum`` rule ranks
coins against each other on the same day), then every rule on every coin.

Every firing of a rule is simulated, including ones that start while an
earlier trade of the same rule on the same coin is still open; those are
flagged ``overlap = 1``. Performance numbers use only non-overlapping trades
(one position per coin per rule, as a person would trade); the "similar
situations" memory may use all of them.

Each trade of a real rule also gets random-day twins: the same trade shape
(stop, target and limit as ATR distances from that day's close, same holding
time) on other days of the same coin and month. Their mean is the yardstick:
it carries the market drift of that month and the payoff of those exits, so
``r_net - twin_r`` is what the rule's choice of day added.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import replace
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from core.sandbox.features import build_situations
from core.sandbox.simulate import Costs, Setup, simulate
from core.sandbox.strategies import STRATEGIES, Strategy, setups

logger = logging.getLogger(__name__)

TWINS = 8                                # random-day twins per trade
CONTROLS = {"control", "control_3r"}     # already random: no twins
TRADE_COLUMNS = ["strategy", "symbol", "signal_day", "entry_day", "exit_day", "entry", "stop", "target",
                 "exit", "reason", "r_net", "mfe_r", "mae_r", "bars", "overlap"]


def all_situations(candles: dict[str, pd.DataFrame], btc: Optional[pd.DataFrame],
                   fng: Optional[pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for symbol, frame in candles.items():
        if len(frame) < 260:
            continue
        s = build_situations(frame, btc, fng)
        s.insert(0, "symbol", symbol)
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # cross-sectional: where the coin's 30-day strength vs BTC ranks that day
    out["rs_rank"] = out.groupby("day")["rs_30d"].rank(pct=True)
    out.loc[out.groupby("day")["rs_30d"].transform("count") < 10, "rs_rank"] = np.nan
    out["day_str"] = out["day"].dt.strftime("%Y-%m-%d")
    return out


def mark_overlaps(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    trades = trades.sort_values(["strategy", "symbol", "entry_bar"]).reset_index(drop=True)
    flags = np.zeros(len(trades), dtype=int)
    busy: dict[tuple[str, str], int] = {}
    for i, (strategy, symbol, entry_bar, exit_bar) in enumerate(
            trades[["strategy", "symbol", "entry_bar", "exit_bar"]].itertuples(index=False)):
        key = (strategy, symbol)
        if entry_bar <= busy.get(key, -1):
            flags[i] = 1
        else:
            busy[key] = exit_bar
    trades["overlap"] = flags
    return trades


def twin(setup: Setup, close_b: float, atr_b: float, bar_u: int, close_u: float, atr_u: float) -> Setup:
    """The same trade shape (levels in ATRs from the signal close) on another day."""
    def moved(level):
        return None if level is None else close_u + (level - close_b) / atr_b * atr_u
    return replace(setup, signal_bar=bar_u, stop=moved(setup.stop), target=moved(setup.target),
                   limit=moved(setup.limit), atr=atr_u)


def replay(candles: dict[str, pd.DataFrame], situations: pd.DataFrame,
           strategies: Iterable[Strategy] = STRATEGIES, costs: Costs = Costs(),
           twins: int = TWINS, seed: int = 7) -> pd.DataFrame:
    """Simulate every firing; non-control rules also get ``twins`` random-day twins.

    ``twin_r`` is the mean result of the same trade shape entered on random
    days of the same coin and calendar month: what the rule's exits earn
    without its timing. ``r_net - twin_r`` is the timing edge.
    """
    rows = []
    for symbol, sit in situations.groupby("symbol", sort=False):
        frame = candles[symbol].sort_values("day").reset_index(drop=True)
        o, h, l, c = (frame[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        days = frame["day"].dt.strftime("%Y-%m-%d").to_numpy()
        sit = sit.reset_index(drop=True)
        sit_bar = sit["bar"].to_numpy()
        sit_close, sit_atr = sit["close"].to_numpy(dtype=float), sit["atr"].to_numpy(dtype=float)
        month = sit["day"].dt.strftime("%Y-%m").to_numpy()
        same_month = {m: np.flatnonzero(month == m) for m in np.unique(month)}
        for strategy in strategies:
            rng = np.random.default_rng([seed, zlib.crc32(f"{strategy.name}:{symbol}".encode())])
            with_twins = twins > 0 and strategy.name not in CONTROLS
            for idx, setup in setups(sit, strategy):
                out = simulate(o, h, l, c, setup, costs)
                if out is None:
                    continue
                twin_r, twin_n = np.nan, 0
                if with_twins:
                    pool = same_month[month[idx]]
                    pool = pool[pool != idx]
                    picks = rng.choice(pool, size=min(twins, len(pool)), replace=False) if len(pool) else []
                    results = []
                    for u in picks:
                        if not (np.isfinite(sit_atr[u]) and sit_atr[u] > 0):
                            continue
                        res = simulate(o, h, l, c, twin(setup, sit_close[idx], sit_atr[idx], int(sit_bar[u]),
                                                        sit_close[u], sit_atr[u]), costs)
                        if res is not None:
                            results.append(res.r_net)
                    if results:
                        twin_r, twin_n = float(np.mean(results)), len(results)
                rows.append((strategy.name, symbol, days[setup.signal_bar], days[out.entry_bar],
                             days[out.exit_bar], out.entry, out.stop, out.target, out.exit, out.reason,
                             out.r_net, out.mfe_r, out.mae_r, out.bars, 0, twin_r, twin_n,
                             out.entry_bar, out.exit_bar))
    trades = pd.DataFrame(rows, columns=[*TRADE_COLUMNS, "twin_r", "twin_n", "entry_bar", "exit_bar"])
    return mark_overlaps(trades)
