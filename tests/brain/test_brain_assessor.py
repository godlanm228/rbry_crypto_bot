"""The trade assessor: both sides of every coin-day, each feature once, promises graded on later months."""

import numpy as np
import pandas as pd

from core.brain import assessor as asr


def test_every_coin_day_is_a_long_and_where_shortable_a_short():
    day = pd.Timestamp("2024-01-01")
    pred = pd.DataFrame({"history": ["A", "B"], "day": day, "place": [0.9, 0.1]})
    rows = pd.DataFrame({"history": ["A", "B"], "day": day, "base": ["A", "B"], "hold_ret": [0.05, -0.02],
                         "short_liq_ret": [np.nan, 0.03], "atr": 1.0, "turnover_30d": 5e6})
    for f in asr.FEATURES:
        if f not in ("place", "side"):
            rows[f] = 0.0
    frame = asr.candidates(pred, rows)
    assert sorted(zip(frame["history"], frame["side"], frame["ret"])) == [("A", 1.0, 0.05), ("B", -1.0, 0.03),
                                                                         ("B", 1.0, -0.02)]
    assert len(asr.FEATURES) == len(set(asr.FEATURES))


def test_a_learnable_edge_is_promised_and_comes_true():
    rng = np.random.default_rng(4)
    days = np.repeat(pd.date_range("2021-01-01", "2022-06-30", freq="D"), 60)
    f = rng.normal(size=len(days))
    ret = np.where(rng.random(len(days)) < 1 / (1 + np.exp(-2 * f)), 0.05, -0.05)
    frame = pd.DataFrame({"history": "X", "base": "X", "day": days, "side": 1.0, "place": 0.5, "f": f,
                          "ret": ret, "atr": 1.0, "turnover_30d": 5e6})
    pred = asr.run(frame, start="2022-01-01", end="2022-06-30", features=("f",))
    sure = pred[pred["win"] >= 0.7]
    assert len(sure) > 100 and (sure["ret"] > 0).mean() > 0.65
