"""The event library: what an hour's close shows, what followed it, and how the verdict is reached."""

import numpy as np
import pandas as pd

from core.brain import events as ev
from core.sandbox.market_store import History

HOUR, DAY = 3_600_000, 86_400_000
T0 = int(pd.Timestamp("2021-03-01").value // 10**6)


def _hourly(closes, volume=None, start=T0, taker=0.5):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    qv = np.full(len(c), 1e6) if volume is None else np.asarray(volume, float)
    return pd.DataFrame({"ts": start + np.arange(len(c), dtype=np.int64) * HOUR, "open": o,
                         "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c,
                         "volume": qv / c, "quote_volume": qv, "trades": 100, "taker_buy_quote": qv * taker})


def _history(frame, name="AAA"):
    return History(name, f"{name}USDT", name, frame, frame.iloc[:0], False)


def _wiggle(n, seed=1):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))


def test_fire_keeps_one_event_a_day_and_only_where_a_state_begins():
    mask = np.zeros(100, bool)
    mask[[5, 6, 7, 20, 40, 41]] = True
    assert ev.fire(mask).tolist() == [5, 40]                   # 6, 7, 20 within a day of 5
    state = np.zeros(100, bool)
    state[10:60] = True
    state[80:90] = True
    assert ev.fire(state, edge=True).tolist() == [10, 80]      # a long state is one event


def test_a_spike_on_heavy_volume_is_seen_at_its_close_and_nothing_earlier():
    c = _wiggle(400)
    volume = np.full(400, 1e6)
    c[300:] *= 1.06                                            # +6% in hour 300
    volume[300] = 2e7
    x = ev.features(_hourly(c, volume))
    hits = np.flatnonzero(ev.BY_ID["SPIKE"].detect(x).fillna(False).to_numpy())
    assert hits.tolist() == [300]
    assert np.isnan(x["sigma"].iloc[50])                       # not enough hours behind it yet


def test_forward_results_start_at_the_next_open_and_count_funding_after_the_entry():
    c = np.r_[np.full(10, 100.0), 110.0, np.full(100, 120.0)]
    g = _hourly(c)
    funding = pd.DataFrame({"ts": T0 + np.array([10, 12, 20], dtype=np.int64) * HOUR, "rate": [0.01, 0.02, 0.04]})
    f = ev.forward(g, funding)
    # signal at the close of hour 9: entry at hour 10's open (100), exit at hour 13's close (120)
    assert abs(f["g4_0"][9] - 0.2) < 1e-12
    # one hour late: entry at hour 11's open (110), exit at hour 14's close
    assert abs(f["g4_1"][9] - (120 / 110 - 1)) < 1e-12
    # payments after hour 10's open and up to hour 13's close: the one at hour 12
    assert abs(f["fr4_0"][9] - 0.02) < 1e-12
    assert abs(f["mfe4"][9] - (120 * 1.001 / 100 - 1)) < 1e-9


def test_costs_of_a_long_and_a_short_and_the_short_liquidated_once_the_price_doubled():
    assert abs(ev.net_long(np.array([0.0]), 0.0)[0] - (0.999 / 1.001 - 1)) < 1e-12
    r = ev.net_short(np.array([-0.1, 0.5]), 0.0, np.array([0.001, 0.0]), np.array([0.0, 1.2]))
    assert abs(r[0] - (0.9995 - 0.9 * 1.0005 + 0.001)) < 1e-12
    assert r[1] == -1.0


def _rows(n, start, excess, side_gross, hz=4, perp=True):
    ts = start + np.arange(n, dtype=np.int64) * 3 * DAY
    out = pd.DataFrame({"event": "SPIKE", "signal_ts": ts, "entry_ts": ts, "hour": 12, "weekend": False,
                        "daypart": "утро", "big": True, "btc": True, "nervous": False, "perp": perp, "slip": 0.0005})
    for h in ev.HORIZONS:
        for d in ev.DELAYS:
            out[f"g{h}_{d}"] = side_gross
            out[f"fr{h}_{d}"] = 0.0
        out[f"twin{h}"] = side_gross - excess
        out[f"mae{h}"] = -0.02
        out[f"mfe{h}"] = 0.03
        out[f"react{h}"] = np.nan
    return out


def test_the_direction_comes_from_2021_2024_and_a_fall_is_sold():
    rows = _rows(300, T0, excess=-0.02, side_gross=-0.03)
    assert ev.direction(rows, 4) == -1
    t = ev.sided(rows, 4, -1, float("nan"))
    assert (t["excess"] > 0).all() and (t["net0"] > 0).all()


def test_a_reaction_moves_a_long_and_a_short_opposite_ways():
    rows = _rows(50, T0, excess=0.0, side_gross=0.02)
    long_late = ev.sided(rows, 4, 1, 0.005)["net0"].iloc[0]
    long_now = ev.sided(rows, 4, 1, 0.0)["net0"].iloc[0]
    short_late = ev.sided(rows, 4, -1, 0.005)["net0"].iloc[0]
    short_now = ev.sided(rows, 4, -1, 0.0)["net0"].iloc[0]
    assert long_late > long_now and short_late < short_now


def test_shorts_count_only_where_a_perpetual_was_live():
    rows = pd.concat([_rows(40, T0, 0.01, 0.02, perp=True), _rows(40, T0, 0.01, 0.02, perp=False)])
    assert len(ev.sided(rows, 4, -1, 0.0)) == 40 and len(ev.sided(rows, 4, 1, 0.0)) == 80


def test_periods_go_by_when_the_outcome_is_known():
    late_2024 = int(pd.Timestamp("2024-12-31 22:00").value // 10**6)
    assert ev.period(np.array([late_2024]), 1).tolist() == ["2018–2024"]
    assert ev.period(np.array([late_2024]), 4).tolist() == ["2025"]


def test_the_bar_rises_with_tries_and_the_verdict_needs_every_period():
    assert ev.bar(1) < ev.bar(57) < ev.bar(57 * ev.COND_VALUES)
    good = {"n": 200, "t": 5.0, "excess_pct": 1.0, "usd_per_100": 0.5}
    parts = {p: dict(good) for p in ev.PERIODS}
    assert ev.verdict(parts, 3.3) == "КАНДИДАТ"
    parts["2026"] = {**good, "usd_per_100": -0.1}
    assert ev.verdict(parts, 3.3).startswith("перевес есть")
    parts["2026"] = {**good, "excess_pct": -0.3}
    assert ev.verdict(parts, 3.3).startswith("работало")
    parts = {p: dict(good) for p in ev.PERIODS}
    parts["2018–2024"]["t"] = 2.5
    assert ev.verdict(parts, 3.3).startswith("наблюдать")
    parts["2025"]["n"] = 3
    assert ev.verdict(parts, 3.3) == "мало случаев"


def test_urgency_by_what_is_left_after_waiting():
    t = pd.DataFrame({"net0": [0.02], "net1": [0.005], "net3": [0.0], "net6": [0.0]})
    assert ev.urgency(t)[0].startswith("срочное")
    t = pd.DataFrame({"net0": [0.02], "net1": [0.018], "net3": [0.015], "net6": [0.002]})
    assert ev.urgency(t)[0] == "может подождать окна до 3 ч"


def test_a_window_takes_a_signal_inside_at_once_and_one_before_it_late():
    t = pd.DataFrame({"hour": [9, 13], "month": [1, 1], "net0": [0.02, 0.03], "net1": [0.01, 0.0],
                      "net3": [0.005, 0.0], "net6": [0.0, 0.0]})
    win = ev.best_window([t], [3], years=1.0).set_index("start")["usd_per_100_trade_sum_per_year"]
    assert abs(win[12] - 100 * (0.005 + 0.03)) < 1e-9          # 9:00 is 3 h before 12: entered 3 h late
    assert abs(win[10] - 100 * 0.01) < 1e-9                    # 1 h late; 13:00 is past 10–13
    assert abs(win[9] - 100 * 0.02) < 1e-9


def test_one_coin_end_to_end_an_event_with_its_twin_and_costs():
    n = 24 * 60
    c = _wiggle(n, seed=3)
    volume = np.full(n, 1e7)                                   # $240M a day: a big, cheap coin
    c[1000:] *= 1.08
    volume[1000] = 2e8
    rows = ev.coin_rows(_history(_hourly(c, volume)))
    spike = rows[rows["event"] == "SPIKE"]
    assert len(spike) == 1
    r = spike.iloc[0]
    assert r["signal_ts"] == T0 + 1001 * HOUR and not r["perp"] and r["big"]
    assert abs(r["slip"] - 0.0005) < 1e-12
    assert np.isfinite(r["twin24"]) and np.isfinite(r["g24_0"])
