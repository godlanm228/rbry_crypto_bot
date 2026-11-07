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


def test_forecasts_and_the_pick_give_the_same_as_before():
    x = se.forecasts(_cands(1.0), _Model(), se.FEATURES)
    assert len(x) == 16 and set(x["action"]) == {a for a, _, _ in se.ACTIONS}
    assert se.pick_best(x, 0.01).equals(se.best_actions(_cands(1.0), _Model(), se.FEATURES, 0.01))


def test_a_forecast_taken_apart_adds_up_to_itself():
    # the reasons kept with a live decision: the model's own attribution, base plus each feature's part
    f = _frame(n_days=40)
    model = se.train(f, se.FEATURES)
    rows = f.head(6)
    split = se.contributions(model, rows, se.FEATURES)
    forecast = model.predict(se._x(rows, se.FEATURES))
    for s, y in zip(split, forecast):
        assert abs(s["base"] + sum(s["by"].values()) - y) < 1e-6
    up, down = se.top_factors(split[0], 2)
    assert all(v > 0 for _, v in up) and all(v < 0 for _, v in down) and len(up) <= 2 and len(down) <= 2


# the weekly gate (07.11): a new version goes on paper only when it passes; the one before is kept for a rollback

LIVE = se.FEATURE_SETS["live"]


def test_a_sound_new_version_passes_the_gate():
    f = _frame(n_days=260)                                                 # 10 000 trades learned before the probe weeks
    champion = se.train(f[f["exit_ts"] < int(f["signal_ts"].max()) - 30 * DAY], LIVE)
    verdict = se.gate(f, champion, se.train(f, LIVE), LIVE)
    assert verdict["promoted"] and verdict["probe"]["n"] >= se.PROBE_MIN_TRADES and verdict["probe"]["mean"] > 0
    assert verdict["agreement"] > 0.9


def test_a_version_that_turned_upside_down_or_a_recipe_that_stopped_working_stays_out():
    f = _frame(n_days=260)                                                 # 10 000 trades learned before the probe weeks
    champion = se.train(f, LIVE)
    flipped = se.train(f.assign(net=-f["net"]), LIVE)                     # a broken label: the opposite of the live one
    verdict = se.gate(f, champion, flipped, LIVE)
    assert not verdict["promoted"] and "расходится" in verdict["notes"][0]
    late = int(f["signal_ts"].max()) - 8 * 7 * DAY
    turned = f.assign(net=np.where(f["signal_ts"] >= late, -f["net"], f["net"]))   # the market turned in the last weeks
    verdict = se.gate(turned, champion, se.train(turned, LIVE), LIVE)
    assert not verdict["promoted"] and any("не заработал" in n for n in verdict["notes"])


def test_versions_are_kept_and_a_rollback_brings_the_one_before(tmp_path):
    f = _frame(n_days=60)
    live = tmp_path / "live"
    for stamp in ("2026-11-01 00:48", "2026-11-08 00:50"):
        se.promote(se.train(f, LIVE), {"trained_at": stamp, "features": list(LIVE)}, live)
    assert se.load(live)[1]["trained_at"] == "2026-11-08 00:50"
    assert se.load(tmp_path / "live-prev")[1]["trained_at"] == "2026-11-01 00:48"
    assert sorted(p.name for p in (tmp_path / "archive").iterdir()) == ["2026-11-01-0048", "2026-11-08-0050"]
    se.reject(se.train(f, LIVE), {"trained_at": "2026-11-15 00:51", "features": list(LIVE)}, live)
    assert se.load(live)[1]["trained_at"] == "2026-11-08 00:50"           # a rejected version never goes live
    assert se.rollback(live)["trained_at"] == "2026-11-01 00:48"
    assert se.load(tmp_path / "live-rejected")[1]["trained_at"] == "2026-11-08 00:50"
    assert se.rollback(tmp_path / "empty" / "live") is None                # no version before: nothing to do
