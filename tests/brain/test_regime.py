"""Market regimes and the switcher: a regime known at the close, held 6 hours before it counts, and one account."""

import numpy as np
import pandas as pd

from core.brain import regime as rg

HOUR = 3_600_000
T0 = int(pd.Timestamp("2023-01-01").value // 10**6)


def _btc(closes, start=T0):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    ts = start + np.arange(len(c), dtype=np.int64) * HOUR
    return np.column_stack([ts, o, np.maximum(o, c), np.minimum(o, c), c])


def test_a_new_regime_counts_after_six_hours_and_a_flicker_changes_nothing():
    raw = np.array(["a"] * 10 + ["b"] * 3 + ["a"] * 5 + ["b"] * 7)
    out = rg.confirmed(raw, 6)
    assert (out[:18] == "a").all()                 # three hours of b were a flicker
    assert out[18 + 4] == "a" and out[18 + 5] == "b"  # the sixth b in a row switches


def test_a_steady_rise_is_up_and_a_steady_fall_down():
    up = _btc(100 * np.exp(np.arange(3000) * 0.0005))
    r = rg.btc_regimes(up)
    assert r["regime"].iloc[-1].startswith("up|")
    down = _btc(100 * np.exp(-np.arange(3000) * 0.0005))
    assert rg.btc_regimes(down)["regime"].iloc[-1].startswith("down|")
    assert rg.btc_regimes(up)["regime"].iloc[100] == ""       # before 50 days of averages nothing is known


def _regimes(labels, start=T0):
    return pd.DataFrame({"ts": start + np.arange(len(labels), dtype=np.int64) * HOUR, "regime": labels})


def test_holding_through_a_regime_pays_a_fee_at_each_switch():
    regimes = _regimes(["x"] * 3 + ["y"] * 3 + ["x"] * 3)
    asset = _btc([100] * 9)
    curve = rg.hold_curve(regimes, asset, "x", 1, None, T0, T0 + 9 * HOUR)
    fee = 0.001 + rg.HOLD_SLIP
    # in from hour 1 (after the first x closed), out at hour 4, in again at hour 7
    assert abs(curve[-1] - 100 * (1 - fee) ** 3) < 1e-9


def test_the_switcher_holds_in_one_regime_and_takes_a_specialists_trades_in_another():
    regimes = _regimes(["hold"] * 4 + ["trade"] * 4)
    asset = _btc([100, 100, 110, 121, 121, 121, 121, 121])
    trades = {"g": pd.DataFrame({"signal_ts": [T0 + 5 * HOUR], "held": [2], "net": [0.5], "history": ["C"],
                                 "turnover": [1e8]})}
    champs = {"hold": {"kind": "hold", "asset": "BTC", "side": 1}, "trade": {"kind": "genome", "id": "g"}}
    curve, done = rg.switch_curve(regimes, champs, trades, {"BTC": asset}, {}, T0, T0 + 8 * HOUR)
    fee = 0.001 + rg.HOLD_SLIP
    held = 100 * (1 - fee) * 1.21 * (1 - fee)          # in at hour 1, +10% twice, out when "trade" begins
    assert done["trades"] == 1 and done["switches"] == 2
    assert abs(curve[-1] - (held + 0.2 * held * 0.5)) < 1e-6


def test_regime_names_in_words():
    assert rg.ru("down|nervous") == "падение · нервно" and len(rg.REGIMES) == 6
