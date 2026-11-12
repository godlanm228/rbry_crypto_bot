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


class _Market:
    """Hourly prices as the simulator's cache holds them: {name: rows of ts, open, high, low, close}."""

    def __init__(self, coins: dict):
        self.coins, self.base_of = coins, {name: name for name in coins}

    def prices(self, name):
        return self.coins[name]

    def btc(self):
        return self.coins["BTC"]

    def funding(self, name):
        return None


def test_the_live_market_state_is_historys_for_the_same_coins():
    # the watcher counted ~50 liquid coins where history counts every coin (12.11.2026): one definition now
    rng = np.random.default_rng(3)
    hours = 1000
    walk = lambda drift: 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, hours)))
    coins = {"BTC": _btc(walk(0.0002))}
    coins.update({f"C{i}": _btc(walk(0.0004 * (i % 5 - 2))) for i in range(14)})
    history = rg.market_states(_Market(coins)).iloc[-1]
    closes = [rows[:, 4] for name, rows in coins.items() if name != "BTC"]
    live = rg.states_at(closes, coins["BTC"][:, 4], [], int(coins["BTC"][-1, 0]))
    assert live["breadth"] == history["breadth"] and live["alts"] == history["alts"]
    assert (live["breadth_state"], live["alt_state"]) == (history["breadth_state"], history["alt_state"])
    # BTC itself is in the breadth (15 coins) and has nothing to beat (14 in the alts' share)
    assert abs(live["breadth"] * 15 - round(live["breadth"] * 15)) < 1e-9
    assert abs(live["alts"] * 14 - round(live["alts"] * 14)) < 1e-9
    # a coin given twice weighs twice (a replay did that to the liquid ones): the watcher passes each once
    winners = [c for c in closes if c[-1] / c[-1 - rg.ALT_HOURS] > coins["BTC"][-1, 4] / coins["BTC"][-1 - rg.ALT_HOURS, 4]]
    assert 0 < len(winners) < len(closes)
    doubled = rg.states_at(closes + winners, coins["BTC"][:, 4], [], int(coins["BTC"][-1, 0]))
    assert doubled["alts"] > live["alts"]
