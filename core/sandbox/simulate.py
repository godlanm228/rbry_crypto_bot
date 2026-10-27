"""Resolve one long trade on daily candles, honestly.

Rules (the same as the October backtest audit, extended to limit entries):
- a market setup made at the close of day ``t`` enters at the OPEN of day
  ``t + 1``; if that open is already beyond the stop or the target the setup
  is void (the user could not have taken it as planned);
- a limit setup waits up to ``valid_days``; it fills at the limit, or at the
  open when the day opens below it. On the fill day only the stop can trigger
  (we cannot know whether the high came before the dip);
- when one candle touches both stop and target, the stop wins;
- a later candle that opens through a level exits at that open;
- otherwise the trade closes at the close of its last allowed day;
- fees and slippage are paid on both sides and the result is in R, where
  1R = entry - stop (what the user agreed to lose).

Trades that cannot finish inside the data are not returned: an unfinished
trade would bias the sample towards whatever happened last.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Costs:
    fee_pct: float = 0.10       # Binance spot taker, per side
    slippage_pct: float = 0.05  # per side


@dataclass(frozen=True)
class Setup:
    """What a strategy decided at the close of ``signal_bar``.

    Levels are absolute prices, or ``None`` to derive them from the real
    entry: ``stop_atr`` ATRs below it and ``target_r`` times that risk above.
    """
    signal_bar: int
    stop: Optional[float] = None
    target: Optional[float] = None
    stop_atr: Optional[float] = None
    target_r: Optional[float] = None
    atr: float = 0.0
    max_hold: int = 14
    limit: Optional[float] = None
    valid_days: int = 0


@dataclass(frozen=True)
class Outcome:
    entry_bar: int
    exit_bar: int
    entry: float
    stop: float
    target: float
    exit: float
    reason: str          # target | stop | target_gap | stop_gap | time
    r_net: float
    mfe_r: float         # best unrealised move, in R
    mae_r: float         # worst unrealised move, in R (positive number)

    @property
    def bars(self) -> int:
        return self.exit_bar - self.entry_bar + 1


def net_return(entry: float, exit_: float, costs: Costs) -> float:
    """Fractional long return after slippage and fees on both sides."""
    slip, fee = costs.slippage_pct / 100.0, costs.fee_pct / 100.0
    return exit_ * (1.0 - slip) * (1.0 - fee) / (entry * (1.0 + slip) * (1.0 + fee)) - 1.0


def _levels(setup: Setup, entry: float) -> Optional[tuple[float, float]]:
    stop = setup.stop if setup.stop is not None else entry - (setup.stop_atr or 0.0) * setup.atr
    if not (math.isfinite(stop) and 0 < stop < entry):
        return None
    if setup.target is not None:
        target = setup.target
    elif setup.target_r is not None:
        target = entry + setup.target_r * (entry - stop)
    else:
        return None
    if not (math.isfinite(target) and target > entry):
        return None
    return stop, target


def simulate(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, setup: Setup,
             costs: Costs = Costs()) -> Optional[Outcome]:
    n = len(o)
    first = setup.signal_bar + 1
    if first >= n:
        return None

    if setup.limit is None:
        entry_bar, entry = first, float(o[first])
        fill_day_only_stop = False
    else:
        entry_bar = None
        for k in range(first, min(n, first + max(1, setup.valid_days))):
            if l[k] <= setup.limit:
                entry_bar, entry = k, float(min(o[k], setup.limit))
                break
        if entry_bar is None:
            return None
        fill_day_only_stop = True

    levels = _levels(setup, entry)
    if levels is None:
        return None
    stop, target = levels
    last = entry_bar + setup.max_hold - 1
    if last >= n:
        return None
    if setup.limit is None and not (stop < entry < target):
        return None  # opened through a level: the planned trade never existed

    risk = entry - stop
    exit_price, reason, exit_bar = float(c[last]), "time", last
    best, worst = entry, entry
    for k in range(entry_bar, last + 1):
        if k > entry_bar:
            if o[k] <= stop:
                exit_price, reason, exit_bar = float(o[k]), "stop_gap", k
                worst = min(worst, float(o[k]))
                break
            if o[k] >= target:
                exit_price, reason, exit_bar = float(o[k]), "target_gap", k
                best = max(best, float(o[k]))
                break
        if l[k] <= stop:
            exit_price, reason, exit_bar = stop, "stop", k
            worst = min(worst, stop)
            break
        worst = min(worst, float(l[k]))
        if k == entry_bar and fill_day_only_stop:
            continue
        if h[k] >= target:
            exit_price, reason, exit_bar = target, "target", k
            best = max(best, target)
            break
        best = max(best, float(h[k]))

    ret = net_return(entry, exit_price, costs)
    return Outcome(entry_bar, exit_bar, entry, stop, target, exit_price, reason,
                   r_net=ret / (risk / entry), mfe_r=(best - entry) / risk, mae_r=(entry - worst) / risk)
