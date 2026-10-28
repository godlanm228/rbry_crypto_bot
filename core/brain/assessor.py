"""The trade assessor: will this prepared trade, held as planned, end in profit after costs?

The brain ranks coins against each other; a trade lives in the whole market.
So for every tradable coin-day and both sides, with what was known at the
close (the coin's own numbers, the brain's out-of-sample place among the day's
coins, the market regime, breadth, funding, macro), two models learn from
earlier trades:
  win   the chance the trade ends in profit (a classifier, calibrated);
  size  the typical result (a regression on the clipped result), so a trade
        that wins often but loses big when it loses is not taken for good.
The trade is the planned one: a long held 14 days on spot, a short held 14
days on the perpetual with 1x isolated margin (it can lose its size, no more).

The exam is the brain's: refit every month on trades finished before it,
calibrate on earlier exam months only, 2022-2024 to choose, 2025 to confirm.
It starts in 2022 because the brain's out-of-sample places start in 12.2020.
Market-wide numbers are allowed here (the question is about the whole trade),
which is also how a model memorises good months; only months it never saw
can tell, so that is all it is graded on.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from core.brain.dataset import MARKET_FEATURES, RANKING_FEATURES
from core.brain.model import Calibration, fit_calibration, matrix
from core.brain.walkforward import GAP_DAYS

# the day's ranks are in both lists: once is enough
FEATURES = tuple(dict.fromkeys(RANKING_FEATURES + MARKET_FEATURES + ("place", "side")))
EXAM_START = "2022-01-01"
EXAM_END = "2025-12-31"
CALIBRATION_MONTHS = 12
MIN_ROWS = 20_000
CLIP = 1.0
MAX_ROUNDS = 600
PATIENCE = 40
PARAMS = {"learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 500, "feature_fraction": 0.7,
          "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 63, "verbose": -1,
          "seed": 7, "num_threads": max(1, (os.cpu_count() or 2) - 1)}


def candidates(pred: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Every out-of-sample coin-day twice: as a long (14-day spot hold) and as a short (1x margin)."""
    feats = [f for f in dict.fromkeys(RANKING_FEATURES + MARKET_FEATURES) if f in rows.columns]
    base = pred[["history", "day", "place"]].assign(history=pred["history"].astype(str)).merge(
        rows[["history", "day", "base", *feats, "hold_ret", "short_liq_ret", "atr", "turnover_30d"]]
        .assign(history=rows["history"].astype(str)), on=["history", "day"], how="inner")
    long = base.drop(columns=["short_liq_ret"]).rename(columns={"hold_ret": "ret"}).assign(side=1.0)
    short = base.drop(columns=["hold_ret"]).rename(columns={"short_liq_ret": "ret"}).assign(side=-1.0)
    out = pd.concat([long, short[short["ret"].notna()]], ignore_index=True)
    return out[out["ret"].notna()].reset_index(drop=True)


def _lgb():
    import lightgbm
    return lightgbm


def _fit(x: np.ndarray, y: np.ndarray, days: np.ndarray, objective: dict):
    lgb = _lgb()
    params = {**PARAMS, **objective}
    full = lgb.Dataset(x, y, params=params, free_raw_data=True).construct()
    cut = np.quantile(days, 0.9)
    probe = lgb.train(params, full.subset(np.flatnonzero(days < cut)), num_boost_round=MAX_ROUNDS,
                      valid_sets=[full.subset(np.flatnonzero(days >= cut))],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    rounds = max(40, int((probe.best_iteration or MAX_ROUNDS) * 1.1))
    return lgb.train(params, full, num_boost_round=rounds)


def train(frame: pd.DataFrame, mask: np.ndarray, features: tuple[str, ...] = FEATURES):
    """(win model, size model) on the rows of ``mask``."""
    x = matrix(frame, features, mask)
    ret = frame["ret"].to_numpy(dtype=float)[mask]
    days = frame["day"].to_numpy(dtype="datetime64[D]").astype(np.int64)[mask]
    win = _fit(x, (ret > 0).astype(float), days, {"objective": "binary", "metric": "binary_logloss"})
    size = _fit(x, np.clip(ret, -CLIP, CLIP), days, {"objective": "huber", "alpha": 0.2, "metric": "l2"})
    return win, size


def run(frame: pd.DataFrame, start: str = EXAM_START, end: str = EXAM_END,
        features: tuple[str, ...] = FEATURES, on_month=None) -> pd.DataFrame:
    """Out-of-sample win chance and typical result for every candidate from ``start`` to ``end``."""
    days = frame["day"]
    parts = []
    for m0 in pd.date_range(start, end, freq="MS"):
        m1 = m0 + pd.DateOffset(months=1)
        learn = (days <= m0 - pd.Timedelta(days=GAP_DAYS)).to_numpy()
        test = ((days >= m0) & (days < m1)).to_numpy()
        if learn.sum() < MIN_ROWS or not test.any():
            continue
        win, size = train(frame, learn, features)
        x = matrix(frame, features, test)
        part = frame.loc[test, ["history", "base", "day", "side", "place", "ret", "atr", "turnover_30d"]].copy()
        part["raw_win"], part["raw_size"], part["model_month"] = win.predict(x), size.predict(x), m0
        parts.append(part)
        if on_month:
            on_month(str(m0.date()), int(learn.sum()))
    return calibrate(pd.concat(parts, ignore_index=True))


def calibrate(pred: pd.DataFrame) -> pd.DataFrame:
    """``win`` and ``size`` per model month, learned on earlier exam months already finished."""
    pred = pred.copy()
    pred["win"], pred["size"] = np.nan, np.nan
    for m0, idx in pred.groupby("model_month").groups.items():
        known = pred[(pred["day"] <= m0 - pd.Timedelta(days=GAP_DAYS))
                     & (pred["day"] >= m0 - pd.DateOffset(months=CALIBRATION_MONTHS))]
        if len(known) < 5_000:
            continue
        for side in (1.0, -1.0):
            k = known[known["side"] == side]
            rows = pred.index.isin(idx) & (pred["side"] == side).to_numpy()
            if len(k) < 2_000:
                continue
            win_cal: Calibration = fit_calibration(k["raw_win"].to_numpy(), (k["ret"] > 0).to_numpy(dtype=float))
            size_cal = fit_calibration(k["raw_size"].to_numpy(), k["ret"].to_numpy())
            pred.loc[rows, "win"] = win_cal(pred.loc[rows, "raw_win"].to_numpy())
            pred.loc[rows, "size"] = size_cal(pred.loc[rows, "raw_size"].to_numpy())
    return pred


def promise_table(pred: pd.DataFrame, period_mask: Optional[pd.Series] = None) -> pd.DataFrame:
    """By promised chance of profit: what really happened, side by side."""
    p = pred if period_mask is None else pred[period_mask]
    p = p[p["win"].notna()]
    bins = pd.cut(p["win"], [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    g = p.groupby([p["side"], bins], observed=True)
    return pd.DataFrame({"trades": g.size(), "promised_of10": 10 * g["win"].mean(),
                         "real_of10": 10 * g["ret"].apply(lambda r: float((r > 0).mean())),
                         "usd_per_100": 100 * g["ret"].mean(),
                         "avg_win_usd": 100 * g["ret"].apply(lambda r: float(r[r > 0].mean()) if (r > 0).any() else np.nan),
                         "avg_loss_usd": 100 * g["ret"].apply(lambda r: float(r[r <= 0].mean()) if (r <= 0).any() else np.nan)})
