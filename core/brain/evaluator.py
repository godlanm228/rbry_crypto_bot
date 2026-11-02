"""The trade evaluator (B): how badly can a trade go, and why.

Lesson of the assessor: a trade's plain result is decided by the
market's direction, which nothing here foresees, and market-wide inputs only
let a model memorise good months. What a coin's own numbers can tell is how
wild its next two weeks may get. In 2026 that was the whole story: N5 lost
because 13 of its 120 shorts were squeezed to liquidation.

So for every tradable coin-day, from the hour after its close:
  squeeze  the price was SQUEEZE above the entry at some hour within 14 days
           (what liquidates or ruins a short),
  crash    it was CRASH below the entry at some hour (what ruins a long),
learned from the coin's own numbers and the brain's place (RANKING_FEATURES,
no market-wide ones), walk-forward like the exam: refit every REFIT_MONTHS on
coin-days finished before, calibrated on earlier exam months. Each answer
comes with the numbers that raised the risk most (the model's contributions),
in words: the "why" of a trade.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from core.brain.dataset import DAY_MS, ENTRY_WITHIN_HOURS, RANKING_FEATURES, TRADE_DAYS
from core.brain.model import fit_calibration, matrix
from core.brain.walkforward import GAP_DAYS
from core.sandbox.market_store import HOUR_MS

SQUEEZE = 0.5
CRASH = 0.3
FEATURES = tuple(dict.fromkeys(RANKING_FEATURES + ("place",)))
EXAM_START = "2022-01-01"
REFIT_MONTHS = 3
CALIBRATION_MONTHS = 12
MIN_ROWS = 50_000
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 300,
          "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 63,
          "verbose": -1, "seed": 7, "num_threads": max(1, (os.cpu_count() or 2) - 1)}
ROUNDS = 300
TARGETS = {"squeeze": "max_up", "crash": "max_down"}

WORDS = {   # what a feature means, for the "why" of a trade
    "ret_1d": "движение за день", "ret_7d": "движение за неделю", "ret_30d": "движение за месяц",
    "ret_90d": "движение за 3 месяца", "ret_365d": "движение за год", "atr_pct": "резкость (ATR)",
    "vol_regime": "резкость против обычной", "dist_sma20": "отрыв от 20-дневной средней",
    "dist_sma50": "отрыв от 50-дневной средней", "dist_sma200": "отрыв от 200-дневной средней",
    "sma50_slope": "наклон 50-дневной средней", "rsi14": "RSI", "range_pos_30": "место в месячном диапазоне",
    "dd_90": "падение от пика за 3 месяца", "dd_365": "падение от пика за год", "volume_ratio": "всплеск объёма",
    "support_dist": "до поддержки", "resistance_dist": "до сопротивления", "rs_30d": "сила против BTC за месяц",
    "plan_rr": "риск/прибыль плана", "day_range": "размах свечи", "close_pos": "закрытие в свече",
    "liquidity": "ликвидность", "taker_buy_7d": "доля покупок по рынку", "trades_ratio": "всплеск числа сделок",
    "age": "возраст монеты", "funding_1d": "фандинг за день", "funding_7d": "фандинг за неделю",
    "perp_listed": "есть фьючерс", "delist_notice": "объявлен делистинг", "rs_rank": "место по силе",
    "ret_rank_7d": "место по недельному движению", "liquidity_rank": "место по ликвидности",
    "place": "место у мозга",
}


def excursions(market, rows: pd.DataFrame, days: int = TRADE_DAYS) -> pd.DataFrame:
    """Per coin-day of ``rows``: the highest and lowest price within ``days`` from the entry, against the entry.

    The entry is the first hour after the day's close (within ENTRY_WITHIN_HOURS), like the trade's. A window
    the newest data has not finished is unknown, unless the coin stopped trading inside it (then it is over).
    """
    newest = float(market.btc()[-1, 0])
    span = days * 24
    out = []
    for history, g in rows.groupby(rows["history"].astype(str), sort=False, observed=True):
        p = market.prices(history)
        if not len(p):
            continue
        ts, o = p[:, 0], p[:, 1]
        hi = pd.Series(p[::-1, 2]).rolling(span, min_periods=1).max().to_numpy()[::-1]
        lo = pd.Series(p[::-1, 3]).rolling(span, min_periods=1).min().to_numpy()[::-1]
        day0 = ((g["day"] - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).to_numpy(dtype=np.int64) + DAY_MS
        a = np.searchsorted(ts, day0)
        ok = a < len(ts)
        a = np.minimum(a, len(ts) - 1)
        ok &= ts[a] < day0 + ENTRY_WITHIN_HOURS * HOUR_MS
        ended = ts[-1] < newest - 3 * DAY_MS
        ok &= (ts[a] + span * HOUR_MS <= newest + HOUR_MS) | ended
        entry = o[a]
        out.append(pd.DataFrame({"history": history, "day": g["day"].to_numpy(),
                                 "max_up": np.where(ok, hi[a] / entry - 1, np.nan),
                                 "max_down": np.where(ok, lo[a] / entry - 1, np.nan)}))
    return pd.concat(out, ignore_index=True)


def labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["squeeze"] = np.where(out["max_up"].notna(), (out["max_up"] >= SQUEEZE).astype(float), np.nan)
    out["crash"] = np.where(out["max_down"].notna(), (out["max_down"] <= -CRASH).astype(float), np.nan)
    return out


def _fit(x: np.ndarray, y: np.ndarray):
    import lightgbm as lgb
    return lgb.train(PARAMS, lgb.Dataset(x, y, free_raw_data=True), num_boost_round=ROUNDS)


def run(frame: pd.DataFrame, end: str, start: str = EXAM_START, features: tuple = FEATURES,
        on_refit=None) -> pd.DataFrame:
    """Out-of-sample squeeze and crash chances for every coin-day from ``start`` to ``end``, and the why."""
    days = frame["day"]
    parts = []
    for m0 in pd.date_range(start, end, freq=f"{REFIT_MONTHS}MS"):
        m1 = m0 + pd.DateOffset(months=REFIT_MONTHS)
        test = ((days >= m0) & (days < m1)).to_numpy()
        if not test.any():
            continue
        x_test = matrix(frame, features, test)
        part = frame.loc[test, ["history", "day", "place", "max_up", "max_down", "squeeze", "crash"]].copy()
        part["model_from"] = m0
        for target in TARGETS:
            learn = ((days <= m0 - pd.Timedelta(days=GAP_DAYS)) & frame[target].notna()).to_numpy()
            if learn.sum() < MIN_ROWS:
                break
            model = _fit(matrix(frame, features, learn), frame[target].to_numpy(dtype=float)[learn])
            part[f"raw_{target}"] = model.predict(x_test)
            contrib = model.predict(x_test, pred_contrib=True)[:, :-1]          # the last column is the bias
            top = np.argsort(-contrib, axis=1)[:, :3]
            part[f"why_{target}"] = ["|".join(features[j] for j in row) for row in top]
        else:
            parts.append(part)
            if on_refit:
                on_refit(str(m0.date()), int(learn.sum()))
    return calibrate(pd.concat(parts, ignore_index=True))


def calibrate(pred: pd.DataFrame) -> pd.DataFrame:
    """Raw scores -> chances, learned per refit on earlier exam months already finished."""
    pred = pred.copy()
    for target in TARGETS:
        pred[target + "_chance"] = np.nan
        for m0, idx in pred.groupby("model_from").groups.items():
            known = pred[(pred["day"] <= m0 - pd.Timedelta(days=GAP_DAYS))
                         & (pred["day"] >= m0 - pd.DateOffset(months=CALIBRATION_MONTHS)) & pred[target].notna()]
            if len(known) < 5_000:
                continue
            cal = fit_calibration(known[f"raw_{target}"].to_numpy(), known[target].to_numpy())
            pred.loc[idx, target + "_chance"] = cal(pred.loc[idx, f"raw_{target}"].to_numpy())
    return pred


def why(names: str) -> str:
    """'atr_pct|dd_365|funding_1d' -> 'резкость (ATR), падение от пика за год, фандинг за день'."""
    return ", ".join(WORDS.get(n, n) for n in names.split("|") if n)


def promise_table(pred: pd.DataFrame, target: str) -> pd.DataFrame:
    """By promised chance: how often it really happened."""
    p = pred[pred[target + "_chance"].notna() & pred[target].notna()]
    bins = pd.cut(p[target + "_chance"], [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0], include_lowest=True)
    g = p.groupby(bins, observed=True)
    return pd.DataFrame({"coin_days": g.size(), "promised_pct": 100 * g[target + "_chance"].mean(),
                         "real_pct": 100 * g[target].mean()})
