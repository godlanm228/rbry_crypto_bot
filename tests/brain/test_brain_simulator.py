"""Simulator v2: positions walk their hourly path; the portfolio is valued at market."""

import numpy as np
import pandas as pd

from core.brain import dataset as ds
from core.brain import simulator as sim

HOUR, DAY = 3_600_000, 86_400_000
D0 = pd.Timestamp("2024-01-01")
T0 = int((D0 - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1))


def _path(closes, start=T0 + DAY):
    """Hourly candles that open at the previous close and touch only their own open and close."""
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    ts = start + np.arange(len(c)) * HOUR
    return np.column_stack([ts, o, np.maximum(o, c), np.minimum(o, c), c])


def _funding(days=30):
    ts = np.arange(T0, T0 + days * DAY, 8 * HOUR, dtype=float)
    return np.column_stack([ts, np.zeros(len(ts))])


def test_a_plain_hold_matches_the_dataset_label():
    prices = _path(np.linspace(100, 110, 15 * 24))
    w = sim.walk(prices, None, D0, +1, 1.0, 0.0, sim.Exit(days=14))
    out = ds.trade_outcomes(prices[:, 0].astype(np.int64), *(prices[:, k] for k in (1, 2, 3, 4)),
                            np.array([T0]), np.array([1.0]), np.array([0.0]), False)
    assert w.reason == "time" and abs(w.ret - out["hold_ret"][0]) < 1e-12


def test_stop_and_target_match_the_fixed_trade():
    for closes in (np.r_[np.full(5, 100.0), 97.0, np.full(400, 97.0)], np.r_[np.full(5, 100.0), 105.0, np.full(400, 105.0)]):
        prices = _path(closes)
        w = sim.walk(prices, None, D0, +1, 1.0, 0.0, sim.Exit(days=14, stop_atr=2.0, target_r=2.0))
        out = ds.trade_outcomes(prices[:, 0].astype(np.int64), *(prices[:, k] for k in (1, 2, 3, 4)),
                                np.array([T0]), np.array([1.0]), np.array([0.0]), False)
        assert abs(w.ret - out["ret_net"][0]) < 1e-12 and w.reason in ("stop", "target")


def test_the_trailing_floor_keeps_seventy_percent_of_the_best_gain():
    closes = np.r_[np.linspace(100, 120, 48), np.linspace(120, 100, 48), np.full(300, 100.0)]
    w = sim.walk(_path(closes), None, D0, +1, 1.0, 0.0, sim.Exit(days=14, keep=0.7, keep_from=0.05))
    assert w.reason == "keep"
    assert 0.13 < w.ret < 0.141                      # 70% of the +20% peak, minus fees


def test_a_short_on_one_x_margin_loses_at_most_its_size_and_needs_a_live_perpetual():
    closes = np.r_[np.full(5, 100.0), np.linspace(100, 260, 50), np.full(300, 260.0)]
    w = sim.walk(_path(closes), _funding(), D0, -1, 1.0, 0.0, sim.Exit(days=14, leverage=1.0))
    assert w.reason == "liquidated" and w.ret == -1.0
    dead = _funding(1)[:1]                           # one payment, long before the entry
    dead[:, 0] -= 10 * DAY
    assert sim.walk(_path(closes), dead, D0, -1, 1.0, 0.0, sim.Exit()) is None


def test_the_portfolio_sees_a_drawdown_inside_a_trade(tmp_path):
    closes = np.r_[np.full(24, 100.0), np.full(24 * 5, 50.0), np.full(24 * 10, 100.0)]
    market = sim.Market.__new__(sim.Market)
    market._prices, market._funding, market.base_of = {"A": _path(closes)}, {}, {"A": "A"}
    orders = pd.DataFrame([{"day": D0, "history": "A", "side": 1, "priority": 1.0, "atr": 1.0, "slip_pct": 0.0,
                            "rule": sim.Exit(days=14)}])
    values, trades = sim.run(orders, pd.date_range(D0, periods=20), market)
    assert values.min() < 10_000 * 0.96               # half of a 10% position gone while it was open
    assert abs(values.iloc[-1] - 10_000 * (1 + 0.1 * trades["ret"].iloc[0])) < 1e-6


def test_a_pair_opens_both_legs_or_none():
    up = _path(np.r_[np.full(24, 100.0), np.full(24 * 14, 110.0)])
    down = _path(np.r_[np.full(24, 100.0), np.full(24 * 14, 90.0)])
    market = sim.Market.__new__(sim.Market)
    market._prices, market._funding, market.base_of = {"A": up, "B": down, "C": down}, {"B": _funding()}, \
        {"A": "A", "B": "B", "C": "C"}
    leg = {"day": D0, "priority": 1.0, "atr": 1.0, "slip_pct": 0.0, "rule": sim.Exit(days=14)}
    orders = pd.DataFrame([{**leg, "history": "A", "side": 1, "group": "p1"}, {**leg, "history": "B", "side": -1, "group": "p1"},
                           {**leg, "history": "A", "side": 1, "group": "p2", "priority": 0.5},
                           {**leg, "history": "C", "side": -1, "group": "p2", "priority": 0.5}])
    values, trades = sim.run(orders, pd.date_range(D0, periods=20), market)
    assert sorted(trades["history"]) == ["A", "B"]         # p2 is refused whole: its long A is already held
    assert trades["ret"].min() > 0                           # A rose and B fell: both legs won


def test_a_delisting_notice_ends_the_position_at_the_next_open():
    prices = _path(np.r_[np.full(30, 100.0), np.full(400, 80.0)], start=T0 - 24 * HOUR)
    notice = np.array([T0 + DAY + 5 * HOUR + 1_000], dtype=float)          # mid-hour 5 of the trade
    w = sim.walk(prices, None, D0, +1, 1.0, 0.0, sim.Exit(on_delist=True), notices=notice)
    assert w.reason == "event:delist" and w.exit_ts == T0 + DAY + 7 * HOUR   # out at hour 6's open, booked at its end
    assert sim.walk(prices, None, D0, +1, 1.0, 0.0, sim.Exit(), notices=notice).reason == "time"


def test_a_squeeze_closes_the_short_and_a_crash_the_long():
    rise = np.r_[np.full(40, 100.0), np.linspace(100, 170, 10), np.full(400, 170.0)]
    w = sim.walk(_path(rise, start=T0), _funding(), D0, -1, 1.0, 0.0, sim.Exit(squeeze=0.5))
    assert w.reason == "event:squeeze" and -0.75 < w.ret < -0.5
    fall = np.r_[np.full(40, 100.0), np.linspace(100, 60, 10), np.full(400, 60.0)]
    w = sim.walk(_path(fall, start=T0), None, D0, +1, 1.0, 0.0, sim.Exit(crash=0.3))
    assert w.reason == "event:crash" and -0.45 < w.ret < -0.3


def test_a_btc_crash_closes_longs():
    btc = _path(np.r_[np.full(40, 100.0), np.linspace(100, 85, 6), np.full(400, 85.0)], start=T0)
    coin = _path(np.full(400, 50.0), start=T0)
    w = sim.walk(coin, None, D0, +1, 1.0, 0.0, sim.Exit(btc_crash=0.10), btc=btc)
    assert w.reason == "event:btc_crash"
