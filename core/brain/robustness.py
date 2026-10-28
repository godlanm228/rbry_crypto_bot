"""Is a strategy's result skill or a lucky path? Run it many times with small changes.

One portfolio run depends on which coins happened to take its few slots: on
28.10.2026 removing 2% of the candidate shorts (outcomes near zero on average)
moved one strategy's 2025 from +277% to +5%. So every strategy is run in
VARIANTS versions, each with:
- its start moved by 0..27 days,
- 5, 10 or 15 positions at once (each 1/n of the portfolio),
- every order dropped with chance MISS (a signal nobody acted on in time).
A strategy is trusted only when it holds up across them: most versions in
profit, and the bad tenth of versions still bearable.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from core.brain import portfolio as pf

VARIANTS = 100
MAX_START_SHIFT = 27
POSITIONS = (5, 10, 15)
MISS = 0.10
SEED = 11


def variants(n: int = VARIANTS, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"shift": int(rng.integers(0, MAX_START_SHIFT + 1)), "positions": int(rng.choice(POSITIONS)),
             "seed": int(rng.integers(0, 2 ** 31))} for _ in range(n)]


def _miss(orders: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Drop one order in ten; a pair is dropped whole (a pair is both legs or nothing)."""
    if not len(orders):
        return orders
    if "group" in orders.columns:
        groups = orders["group"].unique()
        kept = set(groups[rng.random(len(groups)) >= MISS])
        return orders[orders["group"].isin(kept)]
    return orders[rng.random(len(orders)) >= MISS]


def label_runner(orders, days, rules, btc_ret, idle):
    """The first portfolio, positions booked as the dataset's outcomes."""
    return pf.run(orders, days, btc_ret, idle, rules)


def run_variants(orders: pd.DataFrame, days: pd.DatetimeIndex, periods, btc_ret: pd.Series,
                 idle: pd.Series | None, btc_close: pd.Series, specs: list[dict], runner=label_runner) -> pd.DataFrame:
    """One row per variant and period with the portfolio's numbers.

    ``periods``: callable splitting a value series into {name: series}; ``runner(orders, days,
    rules, btc_ret, idle) -> (values, trades)`` is the portfolio to use.
    """
    rows = []
    for k, spec in enumerate(specs):
        rng = np.random.default_rng(spec["seed"])
        kept = _miss(orders, rng)
        rules = replace(pf.Rules(), max_positions=spec["positions"], position_share=1.0 / spec["positions"])
        start = days[spec["shift"]]
        values, trades = runner(kept[kept["day"] >= start] if len(kept) else kept, days[days >= start],
                                rules, btc_ret, idle)
        for name, v in periods(values).items():
            s = pf.summary(v, trades, btc_close)
            rows.append({"variant": k, "period": name, **spec, "return_pct": s.get("return_pct"),
                         "max_drawdown_pct": s.get("max_drawdown_pct"), "trades": s.get("trades")})
    return pd.DataFrame(rows)


def spread_of(results: pd.DataFrame) -> pd.DataFrame:
    """Per period: how many versions in 10 made money, the typical one, the bad tenth, the typical drawdown."""
    g = results.groupby("period", sort=False)
    return pd.DataFrame({
        "up_of10": g["return_pct"].apply(lambda r: 10 * float((r > 0).mean())),
        "median_pct": g["return_pct"].median(),
        "bad_tenth_pct": g["return_pct"].quantile(0.1),
        "good_tenth_pct": g["return_pct"].quantile(0.9),
        "median_drawdown_pct": g["max_drawdown_pct"].median(),
        "bad_drawdown_pct": g["max_drawdown_pct"].quantile(0.1),
    })
