"""Money, not averages: a portfolio that follows a strategy day by day.

A strategy turns each day's scored coins into orders (buy, short, or nothing).
The portfolio is the risk manager around them:
- it starts with START_USD in USDT and puts POSITION_SHARE of its current
  value into each new position, at most MAX_POSITIONS at once and at most
  MAX_SHORT_SHARE of its value in shorts, never two positions in one coin;
- a position ends as the outcome it was graded with (the 14-day hold, or the
  fixed trade with its stop and target), after fees, shorts after funding;
- idle money stays in USDT, or sits in BTC when the strategy asks for it;
- after a fall of DRAWDOWN_BRAKE from its best value it opens half-size
  positions until it makes a new best (the brake).
A position is booked when it closes; while open it counts at what it cost, so
drawdowns inside a trade are not seen, only those between closed trades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

START_USD = 10_000.0
POSITION_SHARE = 0.10
MAX_POSITIONS = 10
MAX_SHORT_SHARE = 0.5
DRAWDOWN_BRAKE = 0.20


@dataclass(frozen=True)
class Rules:
    position_share: float = POSITION_SHARE
    max_positions: int = MAX_POSITIONS
    max_short_share: float = MAX_SHORT_SHARE
    drawdown_brake: float = DRAWDOWN_BRAKE


def run(orders: pd.DataFrame, days: pd.DatetimeIndex, btc_daily_ret: pd.Series | None = None,
        idle_in_btc: pd.Series | None = None, rules: Rules = Rules()) -> tuple[pd.Series, pd.DataFrame]:
    """(value at each day's close, closed positions).

    ``orders``: day, history, side (+1 long / -1 short), priority (higher first), ret (the
    position's net result as a share of its size), exit_day (when the money comes back), and
    optionally share (the order's own share of the portfolio instead of the rules').
    ``idle_in_btc``: per day, True to keep the idle money in BTC (needs ``btc_daily_ret``).
    """
    by_day = {d: g.sort_values("priority", ascending=False) for d, g in orders.groupby("day")} if len(orders) else {}
    cash, best = START_USD, START_USD
    open_pos: list[dict] = []
    closed, values = [], []
    for day in days:
        still = []
        for p in open_pos:
            if p["exit_day"] <= day:
                cash += p["size"] * (1.0 + p["ret"])
                closed.append({**p, "pnl": p["size"] * p["ret"], "closed": day})
            else:
                still.append(p)
        open_pos = still
        if idle_in_btc is not None and btc_daily_ret is not None and bool(idle_in_btc.get(day, False)):
            r = btc_daily_ret.get(day, 0.0)
            cash *= 1.0 + (r if math.isfinite(r) else 0.0)
        value = cash + sum(p["size"] for p in open_pos)
        best = max(best, value)
        scale = 0.5 if value < best * (1.0 - rules.drawdown_brake) else 1.0
        held = {p["history"] for p in open_pos}
        short_out = sum(p["size"] for p in open_pos if p["side"] < 0)
        for o in by_day.get(day, pd.DataFrame()).itertuples(index=False):
            if len(open_pos) >= rules.max_positions or o.history in held or not math.isfinite(o.ret):
                continue
            share = getattr(o, "share", rules.position_share)
            size = value * (share if math.isfinite(share) else rules.position_share) * scale
            if size > cash or size <= 0:
                continue
            if o.side < 0 and short_out + size > value * rules.max_short_share:
                continue
            cash -= size
            open_pos.append({"day": day, "history": o.history, "side": int(o.side), "size": size,
                             "ret": float(o.ret), "exit_day": o.exit_day})
            held.add(o.history)
            short_out += size if o.side < 0 else 0.0
        values.append(cash + sum(p["size"] for p in open_pos))
    return pd.Series(values, index=days), pd.DataFrame(closed)


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float((values / values.cummax() - 1.0).min())


def summary(values: pd.Series, trades: pd.DataFrame, btc_close: pd.Series) -> dict:
    """The plain numbers of one period: how the money did against simply holding BTC."""
    if values.empty:
        return {}
    start, end = values.index[0], values.index[-1]
    t = trades[(trades["closed"] >= start) & (trades["closed"] <= end)] if len(trades) else trades
    monthly = values.resample("ME").last().pct_change().dropna()
    btc = btc_close[(btc_close.index >= start) & (btc_close.index <= end)]
    pnl = t["pnl"].sort_values(ascending=False) if len(t) else pd.Series(dtype=float)
    first = values.iloc[0]
    return {
        "return_pct": 100 * (values.iloc[-1] / first - 1), "max_drawdown_pct": 100 * max_drawdown(values),
        "trades": len(t), "longs": int((t["side"] > 0).sum()) if len(t) else 0,
        "shorts": int((t["side"] < 0).sum()) if len(t) else 0,
        "win_of10": 10 * float((t["ret"] > 0).mean()) if len(t) else float("nan"),
        "usd_per_100": 100 * float(t["ret"].mean()) if len(t) else float("nan"),
        "without_best5_pct": 100 * (values.iloc[-1] - first - pnl.head(5).sum()) / first if len(t) else float("nan"),
        "months_up_of10": 10 * float((monthly > 0).mean()) if len(monthly) else float("nan"),
        "btc_return_pct": 100 * (btc.iloc[-1] / btc.iloc[0] - 1) if len(btc) > 1 else float("nan"),
        "btc_max_drawdown_pct": 100 * max_drawdown(btc) if len(btc) > 1 else float("nan"),
    }


def by_year(values: pd.Series, trades: pd.DataFrame, btc_close: pd.Series) -> pd.DataFrame:
    rows = []
    for year in sorted(set(values.index.year)):
        v = values[values.index.year == year]
        # each year starts where the previous one ended: the first value is the last close before it
        before = values[values.index < v.index[0]]
        if len(before):
            v = pd.concat([before.iloc[-1:], v])
        rows.append({"year": year, **summary(v, trades, btc_close)})
    return pd.DataFrame(rows)


def exit_days(day: pd.Series, hours: pd.Series) -> pd.Series:
    """When a position's money is back: the entry is the next day's first hour."""
    return day + pd.to_timedelta(1 + np.ceil(hours.astype(float) / 24.0), unit="D")
