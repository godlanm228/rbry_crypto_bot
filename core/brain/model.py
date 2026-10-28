"""Gradient boosting that ranks coins, and the calibration behind "X of 10".

What the brain predicts: how a coin will do over the next 14 days *against the
other coins on the same day*. The probes on 2019-2024 decided this:
- the whole market's next two weeks are close to unpredictable from anything
  we store (correlations change sign between 2019-22 and 2023-24), and a
  model asked for raw results learns which past months were good: it scored
  on regime features (BTC's volume share, number of listed coins) and failed
  on the next months;
- the ordering of coins on a day is steady: young, very volatile and recently
  pumped coins trail BTC, older calmer ones hold up, in both periods.
So the label is the coin's rank that day by ``excess_14d`` (its 14-day return
minus BTC's), centred on zero; a rank cannot be steered by one crash.

The raw score is only an ordering, and its scale moves from one refit to the
next. ``Calibration`` therefore works on the coin's place among that day's
coins (0 = worst, 1 = best) and maps it onto what really happened to coins in
that place on days the model never learned from (isotonic regression: a
better place never means a worse outcome). It gives every number a card
shows: the chance to beat BTC, the 14-day result per $100, the fixed trade's
chance of profit and its result in units of risk. A calibrated 6 of 10 must
have come true 6 times in 10.

Hyperparameters are fixed and conservative (big leaves, strong
regularisation); tuning them on the exam would grade the tuning. The number of
trees comes from early stopping on the newest tenth of the training rows,
then the model is refit on all of them.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MODEL_VERSION = "m1"
TARGET = "excess_14d"
MAX_ROUNDS = 800
PATIENCE = 50
SEED = 7
PARAMS = {
    "objective": "regression", "metric": "l2", "learning_rate": 0.03, "num_leaves": 31,
    "min_data_in_leaf": 400, "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 10.0, "max_bin": 63, "verbose": -1, "seed": SEED,
    "num_threads": max(1, (os.cpu_count() or 2) - 1),
}
# what a calibrated card can say, as functions of the graded rows
OUTCOMES = {
    "beat_btc": lambda r: (r["excess_14d"] > 0).astype(float),     # chance to do better than BTC
    "excess": lambda r: r["excess_14d"],                           # vs BTC, share of the position
    "hold": lambda r: r["hold_ret"],                               # bought and held 14 days, after costs
    "trade_win": lambda r: (r["r_net"] > 0).astype(float),         # the fixed trade ends in profit
    "trade_r": lambda r: r["r_net"],                               # the fixed trade, in units of risk
}


def _lgb():
    import lightgbm   # imported late: the bot itself starts without it
    return lightgbm


def day_rank(rows: pd.DataFrame, col: str = TARGET) -> pd.Series:
    """Each coin's place among that day's coins, from -0.5 (worst) to +0.5 (best)."""
    return rows.groupby("day")[col].rank(pct=True) - 0.5


# -- calibration ---------------------------------------------------------------------

@dataclass
class Calibration:
    """Raw score -> what came true, learned on out-of-sample rows."""
    knots_x: np.ndarray
    knots_y: np.ndarray
    rows: int

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(raw, float), self.knots_x, self.knots_y)


def fit_calibration(raw: np.ndarray, outcome: np.ndarray, bins: int = 400) -> Calibration:
    """Isotonic fit on score bins: the same answer as on every row, quick on a million of them."""
    raw, outcome = np.asarray(raw, float), np.asarray(outcome, float)
    ok = np.isfinite(raw) & np.isfinite(outcome)
    raw, outcome = raw[ok], outcome[ok]
    edges = np.unique(np.quantile(raw, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, raw, side="right") - 1, 0, max(0, len(edges) - 2))
    binned = pd.DataFrame({"b": which, "x": raw, "y": outcome}).groupby("b").agg(
        x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
    kx, ky = _weighted_isotonic(binned["x"].to_numpy(), binned["y"].to_numpy(), binned["n"].to_numpy())
    return Calibration(kx, ky, int(ok.sum()))


def _weighted_isotonic(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-decreasing weighted fit of y on sorted x (pool adjacent violators); knots for np.interp."""
    blocks: list[list[float]] = []          # [mean, weight, x_left, x_right]
    for xi, yi, wi in zip(x, y, w):
        blocks.append([float(yi), float(wi), float(xi), float(xi)])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, l2, _ = blocks[-2]
            m1, w1, _, r1 = blocks[-1]
            blocks[-2:] = [[(m2 * w2 + m1 * w1) / (w2 + w1), w2 + w1, l2, r1]]
    kx = np.array([v for b in blocks for v in (b[2], b[3])])
    ky = np.array([b[0] for b in blocks for _ in (0, 1)])
    return kx, ky


def day_place(frame: pd.DataFrame, score) -> pd.Series:
    """A score's place among the same day's coins: 0 = worst, 1 = best."""
    return pd.Series(np.asarray(score, float), index=frame.index).groupby(frame["day"].to_numpy()).rank(pct=True)


def calibrations(place: np.ndarray, graded: pd.DataFrame) -> dict[str, Calibration]:
    return {name: fit_calibration(place, f(graded).to_numpy(dtype=float)) for name, f in OUTCOMES.items()}


# -- the model ------------------------------------------------------------------------

@dataclass
class Brain:
    features: tuple[str, ...]
    model: object
    trained_until: str                      # last signal day in the training rows
    rows: int
    calibration: dict[str, Calibration] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(frame.loc[:, list(self.features)].to_numpy(dtype=np.float32))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score, place among the day's coins, and every calibrated outcome once attached.

        ``frame`` must hold all eligible coins of each day it covers: the place
        is relative to them.
        """
        score = self.score(frame)
        out = pd.DataFrame({"score": score, "place": day_place(frame, score)}, index=frame.index)
        for name, cal in self.calibration.items():
            out[name] = cal(out["place"].to_numpy())
        return out

    def importance(self, top: int = 15) -> list[tuple[str, float]]:
        gain = np.asarray(self.model.feature_importance("gain"), float)
        share = gain / gain.sum() if gain.sum() > 0 else gain
        return [(self.features[i], float(share[i])) for i in np.argsort(-share)[:top]]

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "Brain":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def train(rows: pd.DataFrame, features: tuple[str, ...]) -> Brain:
    """Fit on graded ``rows``; early stopping on the newest tenth, then a refit on all of them."""
    lgb = _lgb()
    rows = rows[rows[TARGET].notna()]
    x = rows.loc[:, list(features)].to_numpy(dtype=np.float32)
    y = day_rank(rows).to_numpy(dtype=float)
    days = rows["day"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    early = days < np.quantile(days, 0.9)
    probe = lgb.train(PARAMS, lgb.Dataset(x[early], y[early]), num_boost_round=MAX_ROUNDS,
                      valid_sets=[lgb.Dataset(x[~early], y[~early])],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    rounds = max(50, int((probe.best_iteration or MAX_ROUNDS) * 1.1))
    model = lgb.train(PARAMS, lgb.Dataset(x, y), num_boost_round=rounds)
    return Brain(tuple(features), model, str(pd.Timestamp(rows["day"].max()).date()), len(rows),
                 meta={"model_version": MODEL_VERSION, "trees": model.num_trees()})
