"""The selector: a forecast only from trades closed before its month, the best action of a coin-hour, a bar."""

import numpy as np
import pandas as pd

from core.brain import selector as se

DAY = 86_400_000
T0 = int(pd.Timestamp("2021-01-05").value // 10**6)


def _frame(n_days=500, seed=1):
    """Trades whose result depends on the situation: buying pays when BTC is up, shorting when it is down."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        ts = T0 + d * DAY
        up = rng.random() < 0.5
        for action, side in (("L24", 1), ("S24", -1)):
            for k in range(30):
                good = (side > 0) == up
                rows.append({"history": f"C{k}", "base": f"C{k}", "signal_ts": ts, "turnover": 1e7, "action": action,
                             "side": side, "net": (0.02 if good else -0.02) + rng.normal(0, 0.01), "held": 24,
                             "exit_ts": ts + DAY, "year": pd.Timestamp(ts, unit="ms").year,
                             "btc_sma": 0.05 if up else -0.05, "trend": "up" if up else "down", "mood": "calm"})
    f = pd.DataFrame(rows)
    for col in se.FEATURES:
        if col not in f:
            f[col] = 0.0
    return f


def test_the_model_learns_the_situation_from_closed_trades_only_and_forecasts_the_month_after():
    f = _frame()
    pred = se.walk_forward(f, start="2022-01-01")
    assert pred["signal_ts"].min() >= int(pd.Timestamp("2022-01-01").value // 10**6)
    first = pred[pred["signal_ts"] < int(pd.Timestamp("2022-02-01").value // 10**6)]
    assert len(first) and first["exit_ts"].min() > int(pd.Timestamp("2022-01-01").value // 10**6)
    best = se.choose(pred, 0.005)
    assert (best["net"] > 0).mean() > 0.95                      # it buys when BTC is up and shorts when down
    assert set(best["action"]) == {"L24", "S24"}


def test_a_coin_hour_gets_one_action_and_only_above_the_bar():
    pred = pd.DataFrame({"history": ["A", "A", "B"], "signal_ts": [1, 1, 1], "action": ["L24", "S24", "L24"],
                         "forecast": [0.012, 0.03, 0.004], "net": [0.0, 0.0, 0.0]})
    out = se.choose(pred, 0.01)
    assert out["action"].tolist() == ["S24"] and out["history"].tolist() == ["A"]


def test_the_promise_table_sets_forecasts_against_results():
    pred = pd.DataFrame({"forecast": [0.001, 0.002, 0.03, 0.04], "net": [-0.01, 0.0, 0.02, 0.05]})
    t = se.promise_table(pred)
    assert int(t["trades"].sum()) == 4 and abs(t["real_pct"].iloc[-1] - 3.5) < 1e-9


class _Model:
    """Forecasts by action: shorting pays, holding a day a little."""
    def predict(self, x):
        return np.where(x["action"].astype(str) == "S24", 0.03, np.where(x["action"].astype(str) == "L24", 0.012, -0.01))


def _cands(perp):
    c = pd.DataFrame({"history": ["A", "B"], "signal_ts": [1, 1], "perp": [perp, 1.0]})
    for col in se.FEATURES:
        if col not in c:
            c[col] = 0.0
    c["trend"], c["mood"] = "up", "calm"
    return c


def test_a_short_is_an_action_only_with_a_live_perpetual():
    x = se.expand(_cands(0.0))
    assert set(x[x["history"] == "A"]["side"]) == {1} and set(x[x["history"] == "B"]["side"]) == {1, -1}


def test_the_live_choice_takes_each_coins_best_action_above_the_bar():
    best = se.best_actions(_cands(0.0), _Model(), se.FEATURES, 0.01)
    assert dict(zip(best["history"], best["action"])) == {"A": "L24", "B": "S24"}
    assert best["history"].tolist() == ["B", "A"]                       # the best forecast first
    assert not len(se.best_actions(_cands(0.0), _Model(), se.FEATURES, 0.05))
