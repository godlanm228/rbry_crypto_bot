"""Entry rules the sandbox tries on every situation.

Ten rules, fixed in advance (more rules = more chances to fool ourselves):

- ``control`` / ``control_3r``: enter every day with an ATR stop. This is
  "entering at random"; a rule that cannot beat it has no edge, however good
  its own numbers look.
- ``plan`` / ``plan_good``: the  card as the bot shows it (stop under the
  support, target at the next resistance), all days / only "good" verdicts.
- ``better_entry``: the card's "wait for $X" limit order, valid 7 days.
- ``breakout``, ``pullback``, ``oversold``, ``momentum``, ``capitulation``:
  classic swing setups built from the same candles.

Bump STRATEGY_VERSION whenever a rule changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from core.risk.trade_plan import FALLBACK_STOP_ATR, MAX_STOP_ATR, MIN_STOP_ATR, STOP_BUFFER_ATR
from core.sandbox.simulate import Setup

STRATEGY_VERSION = "s1"
HOLD_DAYS = 14
LONG_HOLD_DAYS = 20
LIMIT_VALID_DAYS = 7


def structure_stop(price: float, atr: float, level: float) -> float:
    """Below a level with the plan's buffer, 1..3 ATR away, else 2 ATR."""
    stop = min(level - STOP_BUFFER_ATR * atr, price - MIN_STOP_ATR * atr)
    if price - stop > MAX_STOP_ATR * atr:
        stop = price - FALLBACK_STOP_ATR * atr
    return stop


@dataclass(frozen=True)
class Strategy:
    name: str
    title: str
    when: Callable[[pd.DataFrame], pd.Series]
    setup: Callable[[pd.Series], Setup]


def _atr_setup(stop_atr: float, target_r: float, hold: int):
    return lambda r: Setup(int(r.bar), stop_atr=stop_atr, target_r=target_r, atr=float(r.atr), max_hold=hold)


def _level_setup(level_col: str, target_r: float = 2.0, hold: int = HOLD_DAYS):
    return lambda r: Setup(int(r.bar), stop=structure_stop(float(r.close), float(r.atr), float(r[level_col])),
                           target_r=target_r, max_hold=hold)


STRATEGIES: tuple[Strategy, ...] = (
    Strategy("control", "every day, stop 2 ATR, target 2R",
             lambda s: s["atr"].notna(), _atr_setup(2.0, 2.0, HOLD_DAYS)),
    Strategy("control_3r", "every day, stop 2 ATR, target 3R",
             lambda s: s["atr"].notna(), _atr_setup(2.0, 3.0, LONG_HOLD_DAYS)),
    Strategy("plan", "the 🎯 card, every day",
             lambda s: s["plan_stop"].notna(),
             lambda r: Setup(int(r.bar), stop=float(r.plan_stop), target=float(r.plan_target), max_hold=HOLD_DAYS)),
    Strategy("plan_good", "the 🎯 card when it says 'good'",
             lambda s: s["plan_stop"].notna() & (s["verdict"] == "good"),
             lambda r: Setup(int(r.bar), stop=float(r.plan_stop), target=float(r.plan_target), max_hold=HOLD_DAYS)),
    Strategy("better_entry", "the card's 'wait for $X' limit, 7 days",
             lambda s: s["better_entry"].notna(),
             lambda r: Setup(int(r.bar), stop=float(r.better_stop), target=float(r.plan_target),
                             limit=float(r.better_entry), valid_days=LIMIT_VALID_DAYS, max_hold=HOLD_DAYS)),
    Strategy("breakout", "close above the 20-day high",
             lambda s: s["close"] > s["high20_prev"], _level_setup("high20_prev")),
    Strategy("pullback", "uptrend, dip to the 20-day average",
             lambda s: (s["close"] > s["sma50"]) & (s["sma50_slope"] > 0)
             & (s["close"] <= s["sma20"] + 0.25 * s["atr"]),
             _level_setup("sma50")),
    Strategy("oversold", "RSI < 32 within 1 ATR of a support",
             lambda s: (s["rsi14"] < 32) & (s["support_dist"] <= 1.0) & s["support"].notna(),
             _level_setup("support")),
    Strategy("momentum", "top 20% by 30-day strength vs BTC, above SMA50",
             lambda s: (s.get("rs_rank", pd.Series(np.nan, index=s.index)) >= 0.8) & (s["close"] > s["sma50"]),
             _atr_setup(2.0, 3.0, LONG_HOLD_DAYS)),
    Strategy("capitulation", "35%+ below the 90-day high, first green week",
             lambda s: (s["dd_90"] <= -0.35) & (s["ret_7d"] > 0) & (s["rsi14"] < 50),
             _atr_setup(2.0, 2.0, HOLD_DAYS)),
)

BY_NAME = {s.name: s for s in STRATEGIES}


def setups(situations: pd.DataFrame, strategy: Strategy) -> list[tuple[int, Setup]]:
    """(row index in ``situations``, setup) for every day the rule fires."""
    mask = strategy.when(situations).fillna(False).to_numpy(dtype=bool)
    found = []
    for idx in np.flatnonzero(mask):
        row = situations.iloc[idx]
        if not (math.isfinite(row.atr) and row.atr > 0):
            continue
        found.append((int(idx), strategy.setup(row)))
    return found
