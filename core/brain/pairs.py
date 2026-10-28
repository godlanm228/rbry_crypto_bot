"""Pair candidates for the assessor: is this pair's balance of chances and sizes worth it?

Each day every one of the CANDIDATES best coins (long, 14-day spot hold) is
paired with every one of the CANDIDATES worst coins that can be shorted (14-day
hold on the perpetual, 1x margin), equal size: the assessor sees which
combinations work, and a day trades at most PER_DAY pairs, no coin twice. A pair's result is the mean of its legs, so it
depends on how far apart the two coins end up, which the brain can foresee,
and hardly on where the market goes, which it cannot.

The assessor (core/brain/assessor.py, same exam) learns from both legs'
numbers, their differences and the market around them: the chance the pair
ends in profit, and its typical result. A pair is left out only when that
balance is bad (Vlad: a big chance of losing, and losing a lot, against a
small chance of profit), not because it is uncertain or volatile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.brain.dataset import slippage_pct

CANDIDATES = 8
PER_DAY = 3
LEG = ("place", "atr_pct", "ret_1d", "ret_7d", "ret_30d", "ret_90d", "dd_90", "dd_365", "rsi14", "vol_regime",
       "liquidity", "age", "funding_7d", "taker_buy_7d", "volume_ratio", "dist_sma50", "delist_notice")
MARKET = ("btc_trend", "btc_ret_7d", "btc_ret_30d", "btc_atr_pct", "beat_btc_7d", "beat_btc_30d", "above_sma50",
          "alt_median_7d", "btc_volume_share", "fear_greed", "fear_greed_7d", "vix", "spx_4w", "dxy_4w", "stables_4w")
FEATURES = (tuple(f"long_{f}" for f in LEG) + tuple(f"short_{f}" for f in LEG)
            + tuple(f"gap_{f}" for f in LEG if f != "delist_notice") + MARKET + ("long_k", "short_k", "side"))


def candidates(pred: pd.DataFrame, rows: pd.DataFrame, n: int = CANDIDATES) -> pd.DataFrame:
    """Up to ``n`` pairs a day with both legs' numbers, the market, the pair's result and what orders need."""
    cols = sorted({*LEG, *MARKET} - {"place"} | {"hold_ret", "short_liq_ret", "atr", "turnover_30d", "base"})
    base = pred[["history", "day", "place"]].assign(history=pred["history"].astype(str)).merge(
        rows[["history", "day", *cols]].assign(history=rows["history"].astype(str)), on=["history", "day"])
    longs = base[base["hold_ret"].notna()].copy()
    longs["k"] = longs.groupby("day")["place"].rank(ascending=False, method="first")
    shorts = base[base["short_liq_ret"].notna()].copy()
    shorts["k"] = shorts.groupby("day")["place"].rank(ascending=True, method="first")
    longs, shorts = longs[longs["k"] <= n], shorts[shorts["k"] <= n]
    leg_cols = [*LEG, "history", "base", "atr", "turnover_30d", "k"]
    pairs = longs[["day", *leg_cols, "hold_ret", *MARKET]].rename(columns={c: f"long_{c}" for c in leg_cols}).merge(
        shorts[["day", *leg_cols, "short_liq_ret"]].rename(columns={c: f"short_{c}" for c in leg_cols}), on="day")
    pairs = pairs[pairs["long_history"] != pairs["short_history"]]
    for f in LEG:
        if f != "delist_notice":
            pairs[f"gap_{f}"] = pairs[f"long_{f}"] - pairs[f"short_{f}"]
    pairs["ret"] = (pairs["hold_ret"] + pairs["short_liq_ret"]) / 2.0
    # the columns the assessor's exam keeps next to its predictions
    pairs["history"] = pairs["long_history"] + "|" + pairs["short_history"]
    pairs["base"], pairs["place"], pairs["side"] = pairs["long_base"], pairs["long_place"], 1.0
    pairs["atr"], pairs["turnover_30d"] = np.nan, np.nan
    return pairs.reset_index(drop=True)


def choose(pred: pd.DataFrame, per_day: int = PER_DAY) -> pd.DataFrame:
    """Each day the best pairs by ``priority``, no coin in two of them."""
    out = []
    for _, day in pred.sort_values(["day", "priority"], ascending=[True, False]).groupby("day", sort=False):
        used: set = set()
        for row in day.itertuples(index=False):
            if row.long_history in used or row.short_history in used:
                continue
            out.append(row)
            used |= {row.long_history, row.short_history}
            if len(out) and sum(1 for r in out if r.day == row.day) >= per_day:
                break
    return pd.DataFrame(out)


def orders(chosen: pd.DataFrame, rule) -> pd.DataFrame:
    """Both legs of every chosen pair as one group for simulator v2."""
    legs = []
    for side, prefix in ((1, "long"), (-1, "short")):
        leg = pd.DataFrame({"day": chosen["day"], "history": chosen[f"{prefix}_history"], "side": side,
                            "priority": chosen["priority"], "atr": chosen[f"{prefix}_atr"].astype(float),
                            "slip_pct": slippage_pct(chosen[f"{prefix}_turnover_30d"].to_numpy(dtype=float)),
                            "group": chosen["day"].dt.strftime("%Y%m%d") + "-" + chosen["history"]})
        leg["rule"] = [rule] * len(leg)
        legs.append(leg)
    return pd.concat(legs, ignore_index=True).sort_values(["day", "priority"], ascending=[True, False])
