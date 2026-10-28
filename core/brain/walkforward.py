"""The exam: refit on the past only, grade on months the brain never saw.

For every month from EXAM_START to the end of 2025:
- the model learns on coin-days whose 14 days had passed before the month
  began (signal day at least GAP_DAYS earlier), all older years included;
- the month's coin-days are scored and graded on what really happened;
- calibration for that month learns on the previous CALIBRATION_MONTHS of exam
  scores whose outcome was known by then: out of sample for the models that
  made them, so a calibrated "6 of 10" is never graded on itself.

2026 stays closed, as in the sandbox L1 reports (dev up to 2024, valid 2025).
Opening it is a separate step, logged in learning.db.

Only coin-days with a 30-day median turnover of MIN_TURNOVER or more take part:
below that a market order already moves the price.

Every selection is graded against all eligible coin-days of the same months,
which carry the same market. Standard errors are clustered by month: thirty
coins on the same day are one bet, not thirty.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.brain.dataset import DAY_MS, RANKING_FEATURES, TRADE_DAYS
from core.brain.model import calibrations, train
from core.sandbox.evaluate import DEV_END, VALID_END, t_cluster

logger = logging.getLogger(__name__)

EXAM_START = "2020-01-01"
GAP_DAYS = TRADE_DAYS + 2
MIN_TURNOVER = 1e6
CALIBRATION_MONTHS = 12
MIN_TRAIN_ROWS = 20_000
MIN_CALIBRATION_ROWS = 5_000
TOP_SHARE = 0.10
TOP_PER_DAY = 3
KEYS = ["history", "base", "day"]
GRADED = ["excess_14d", "hold_ret", "r_net", "ret_net", "reason", "hours"]


def eligible(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["turnover_30d"] >= MIN_TURNOVER]


def run(rows: pd.DataFrame, refit_months: int = 1, start: str = EXAM_START, end: str = VALID_END,
        features: tuple[str, ...] = RANKING_FEATURES,
        on_month: Optional[Callable[[str, dict], None]] = None) -> pd.DataFrame:
    """Out-of-sample scores for every eligible coin-day from ``start`` to ``end``."""
    pool = eligible(rows)
    graded = pool[pool["excess_14d"].notna()]
    out = []
    for m0 in pd.date_range(start, end, freq=f"{refit_months}MS"):
        m1 = min(m0 + pd.DateOffset(months=refit_months), pd.Timestamp(end) + pd.Timedelta(days=1))
        test = pool[(pool["day"] >= m0) & (pool["day"] < m1)]
        learn = graded[graded["day"] <= m0 - pd.Timedelta(days=GAP_DAYS)]
        if test.empty or len(learn) < MIN_TRAIN_ROWS:
            logger.info("exam %s: %d rows to learn from, skipped", m0.date(), len(learn))
            continue
        brain = train(learn, features)
        part = test[KEYS + GRADED + ["turnover_30d"] + [c for c in test.columns if c.startswith("rule_")]].copy()
        part["score"], part["model_month"] = brain.score(test), m0
        out.append(part)
        if on_month:
            on_month(str(m0.date()), {"rows": len(learn), **brain.meta, "importance": brain.importance(8)})
    return calibrate(pd.concat(out, ignore_index=True))


def calibrate(pred: pd.DataFrame) -> pd.DataFrame:
    """Calibrated outcomes per model month, learned on earlier exam months already graded."""
    pred = pred.copy()
    for m0, idx in pred.groupby("model_month").groups.items():
        known = pred[(pred["day"] <= m0 - pd.Timedelta(days=GAP_DAYS))
                     & (pred["day"] >= m0 - pd.DateOffset(months=CALIBRATION_MONTHS))
                     & pred["excess_14d"].notna() & pred["r_net"].notna()]
        if len(known) < MIN_CALIBRATION_ROWS:
            continue
        for name, cal in calibrations(known["score"].to_numpy(), known).items():
            pred.loc[idx, name] = cal(pred.loc[idx, "score"].to_numpy())
    return pred


# -- grading --------------------------------------------------------------------------

def period(days: pd.Series) -> pd.Series:
    return pd.Series(np.where(days <= DEV_END, "dev", np.where(days <= VALID_END, "valid", "holdout")),
                     index=days.index)


def day_share(pred: pd.DataFrame, col: str = "score") -> pd.Series:
    """Where a coin-day's score stands among that day's coins, 0..1."""
    return pred.groupby("day")[col].rank(pct=True)


def top_per_day(pred: pd.DataFrame, n: int = TOP_PER_DAY, col: str = "score") -> pd.Series:
    return pred.groupby("day")[col].rank(ascending=False, method="first") <= n


def one_at_a_time(sel: pd.DataFrame, days: int = TRADE_DAYS) -> pd.DataFrame:
    """No new buy of a coin that is still held from an earlier one, as a person would trade."""
    sel = sel.sort_values(["history", "day"])
    keep = np.zeros(len(sel), dtype=bool)
    busy_until: dict[str, int] = {}
    day_ms = ((sel["day"] - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).to_numpy()
    for i, (h, d) in enumerate(zip(sel["history"].to_numpy(), day_ms)):
        if d >= busy_until.get(h, 0):
            keep[i], busy_until[h] = True, int(d + days * DAY_MS)
    return sel[keep]


def describe(g: pd.DataFrame, base: pd.DataFrame) -> dict:
    """Plain numbers for a group of coin-days, against all coin-days of the same months."""
    g = g[g["excess_14d"].notna()]
    if g.empty:
        return {"n": 0}
    month = g["day"].dt.to_period("M")
    month_mean = base[base["excess_14d"].notna()].groupby(base["day"].dt.to_period("M"))["excess_14d"].mean()
    vs_market = g["excess_14d"].to_numpy() - month.map(month_mean).to_numpy(dtype=float)
    ok = np.isfinite(vs_market)
    trade = g[g["r_net"].notna()]
    return {
        "n": len(g), "coins": g["base"].nunique(), "days": g["day"].nunique(),
        "beat_btc_of10": 10 * float((g["excess_14d"] > 0).mean()),
        "vs_btc_usd": 100 * float(g["excess_14d"].mean()),
        "hold_usd": 100 * float(g["hold_ret"].mean()),
        "hold_up_of10": 10 * float((g["hold_ret"] > 0).mean()),
        "vs_others_usd": 100 * float(vs_market[ok].mean()),
        "t_vs_others": t_cluster(vs_market[ok], month.to_numpy()[ok]),
        "trade_win_of10": 10 * float((trade["r_net"] > 0).mean()),
        "trade_r": float(trade["r_net"].mean()),
        "promised_beat_btc_of10": 10 * float(g["beat_btc"].mean()) if "beat_btc" in g and g["beat_btc"].notna().any()
        else float("nan"),
    }


def selections(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The groups every report grades; the sandbox rules are graded on the same days."""
    share = day_share(pred)
    out = {
        "все монеты": pred,
        "мозг: лучшие 10% дня": pred[share > 1 - TOP_SHARE],
        "мозг: 3 лучших в день": pred[top_per_day(pred)],
        "мозг: худшие 10% дня": pred[share <= TOP_SHARE],
    }
    out["мозг: 3 лучших, монету не докупаем"] = one_at_a_time(out["мозг: 3 лучших в день"])
    for rule in ("oversold", "capitulation", "plan_good", "breakout", "momentum", "pullback"):
        col = f"rule_{rule}"
        if col in pred:
            out[f"правило {rule}"] = pred[pred[col] > 0]
    return out


def grade(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.assign(period=period(pred["day"]))
    rows = []
    for name, g in selections(pred).items():
        for per in ("dev", "valid"):
            rows.append({"group": name, "period": per, **describe(g[g["period"] == per], pred[pred["period"] == per])})
    return pd.DataFrame(rows)


def by_year(pred: pd.DataFrame) -> pd.DataFrame:
    share = day_share(pred)
    rows = []
    for year, base in pred.groupby(pred["day"].dt.year):
        s = share.loc[base.index]
        top, bottom = describe(base[s > 1 - TOP_SHARE], base), describe(base[s <= TOP_SHARE], base)
        everything = describe(base, base)
        rows.append({"year": year, "coins_per_day": base.groupby("day").size().mean(),
                     "all_vs_btc": everything["vs_btc_usd"], "top_vs_btc": top["vs_btc_usd"],
                     "bottom_vs_btc": bottom["vs_btc_usd"], "top_beat_btc_of10": top["beat_btc_of10"],
                     "bottom_beat_btc_of10": bottom["beat_btc_of10"], "all_beat_btc_of10": everything["beat_btc_of10"],
                     "top_minus_bottom": top["vs_btc_usd"] - bottom["vs_btc_usd"]})
    return pd.DataFrame(rows)


def deciles(pred: pd.DataFrame) -> pd.DataFrame:
    """Ten groups by that day's score, from worst to best."""
    g = pred[pred["excess_14d"].notna()].groupby(np.minimum(9, (day_share(pred) * 10 - 1e-9).astype(int)))
    return pd.DataFrame({"vs_btc_usd": 100 * g["excess_14d"].mean(),
                         "beat_btc_of10": 10 * g["excess_14d"].apply(lambda x: float((x > 0).mean())),
                         "hold_usd": 100 * g["hold_ret"].mean(), "trade_r": g["r_net"].mean(), "n": g.size()})


def calibration_table(pred: pd.DataFrame, col: str = "beat_btc", outcome: str = "excess_14d") -> pd.DataFrame:
    """Promised vs real, in tenths of the promised chance to beat BTC."""
    known = pred[pred[col].notna() & pred[outcome].notna()]
    if known.empty:
        return pd.DataFrame()
    bins = pd.qcut(known[col], 10, duplicates="drop")
    g = known.groupby(bins, observed=True)
    return pd.DataFrame({"promised_of10": 10 * g[col].mean(),
                         "real_of10": 10 * g[outcome].apply(lambda r: float((r > 0).mean())), "n": g.size()})


def fmt(x, digits: int = 1, sign: bool = False) -> str:
    if isinstance(x, (float, np.floating)):
        return "—" if not math.isfinite(x) else (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}")
    return str(x)
