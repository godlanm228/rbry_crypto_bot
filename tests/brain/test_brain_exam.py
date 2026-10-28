"""The exam never lets the brain see a month it is graded on; calibration keeps promises honest."""

import numpy as np
import pandas as pd

from core.brain import model
from core.brain import walkforward as wf


def _rows(days=900, coins=40, seed=1):
    rng = np.random.default_rng(seed)
    day = np.repeat(pd.date_range("2019-01-01", periods=days, freq="D"), coins)
    f = rng.normal(size=len(day))
    excess = 0.02 * f + rng.normal(0, 0.05, len(day))
    return pd.DataFrame({"history": np.tile([f"C{i}" for i in range(coins)], days), "base": "X", "day": day,
                         "f": f, "excess_14d": excess, "hold_ret": excess, "r_net": excess * 10,
                         "ret_net": excess, "reason": 0.0, "hours": 336.0, "turnover_30d": 5e6})


def test_each_month_learns_only_from_trades_finished_before_it(monkeypatch):
    seen = []

    class Fake:
        meta = {"trees": 1}

        def __init__(self, learn):
            self.last = learn["day"].max()

        def score(self, frame):
            return np.zeros(len(frame))

        def importance(self, top):
            return []

    def fake_train(learn, features):
        seen.append(learn["day"].max())
        return Fake(learn)

    monkeypatch.setattr(wf, "train", fake_train)
    monkeypatch.setattr(wf, "MIN_TRAIN_ROWS", 100)
    pred = wf.run(_rows(), start="2020-01-01", end="2020-06-30", features=("f",))
    months = sorted(pred["model_month"].unique())
    assert len(months) == 6 and len(seen) == 6
    for m0, last in zip(months, seen):
        assert last <= pd.Timestamp(m0) - pd.Timedelta(days=wf.GAP_DAYS)
    assert pred["day"].min() == pd.Timestamp("2020-01-01") and pred["day"].max() == pd.Timestamp("2020-06-30")


def test_calibration_never_promises_more_for_a_lower_score():
    rng = np.random.default_rng(3)
    raw = rng.normal(size=20000)
    won = (rng.random(20000) < 1 / (1 + np.exp(-raw))).astype(float)
    cal = model.fit_calibration(raw, won)
    grid = cal(np.linspace(-3, 3, 50))
    assert np.all(np.diff(grid) >= -1e-12)
    for x in (-1.0, 0.0, 1.0):
        assert abs(cal(np.array([x]))[0] - 1 / (1 + np.exp(-x))) < 0.06


def test_the_brain_ranks_a_planted_signal():
    rows = _rows(days=600)
    brain = model.train(rows[rows["day"] < "2020-06-01"], ("f",))
    test = rows[rows["day"] >= "2020-07-01"]
    score = brain.score(test)
    assert np.corrcoef(score, test["f"])[0, 1] > 0.9


def test_a_coin_still_held_is_not_bought_again():
    days = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-15", "2024-01-16", "2024-01-03"])
    sel = pd.DataFrame({"history": ["A", "A", "A", "A", "B"], "day": days})
    kept = wf.one_at_a_time(sel)
    assert sorted(zip(kept["history"], kept["day"].dt.day)) == [("A", 1), ("A", 15), ("B", 3)]
