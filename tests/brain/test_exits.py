"""Exits for the panic buy: one long walked hour by hour, the picks of a moment, a $100 portfolio."""

import numpy as np
import pandas as pd

from core.brain import exits as ex

HOUR = 3_600_000


def _bars(*rows):
    o, h, l, c = (np.array(x, float) for x in zip(*rows))
    return o, h, l, c


def test_a_stop_is_hit_by_the_low_first_and_an_open_below_it_exits_at_the_open():
    o, h, l, c = _bars((100, 101, 99, 100), (100, 112, 89, 110), (110, 111, 109, 110))
    exits, held, reason, worst = ex.walk_long(o, h, l, c, 100.0, 0.10, ex.Rule("x", "", 3, stop_pct=0.1, take=1))
    assert exits == [(1.0, 90.0)] and held == 2 and reason == "stop"   # the high of hour 2 came after its low
    o, h, l, c = _bars((100, 101, 99, 100), (80, 82, 79, 81))
    exits, _, reason, _ = ex.walk_long(o, h, l, c, 100.0, 0.10, ex.Rule("x", "", 2, stop_pct=0.1))
    assert exits == [(1.0, 80.0)] and reason == "stop"


def test_half_out_then_the_rest_never_loses():
    o, h, l, c = _bars((100, 101, 99, 100), (100, 106, 100, 104), (104, 105, 99, 100), (100, 101, 99, 100))
    exits, held, reason, _ = ex.walk_long(o, h, l, c, 100.0, None, ex.Rule("x", "", 4, half_pct=0.05))
    assert exits == [(0.5, 105.0), (0.5, 100.0)] and held == 3 and reason == "breakeven"


def test_a_trailing_floor_keeps_part_of_the_best_earlier_gain():
    o, h, l, c = _bars((100, 101, 99, 100), (100, 120, 100, 118), (118, 119, 108, 109))
    exits, _, reason, _ = ex.walk_long(o, h, l, c, 100.0, 0.2, ex.Rule("x", "", 3, stop_pct=0.2, keep=0.5))
    assert len(exits) == 1 and abs(exits[0][1] - 110.0) < 1e-9 and reason == "keep"   # half of hour 2's +20%


def test_a_target_and_the_time_exit():
    o, h, l, c = _bars((100, 101, 99, 100), (100, 125, 99, 120))
    exits, _, reason, _ = ex.walk_long(o, h, l, c, 100.0, 0.1, ex.Rule("x", "", 2, stop_pct=0.1, take=2))
    assert exits == [(1.0, 120.0)] and reason == "target"
    o, h, l, c = _bars((100, 101, 99, 100), (100, 103, 98, 102))
    exits, held, reason, worst = ex.walk_long(o, h, l, c, 100.0, None, ex.Rule("x", "", 2))
    assert exits == [(1.0, 102.0)] and held == 2 and reason == "time" and abs(worst + 0.02) < 1e-12


def test_costs_both_ways():
    assert abs(ex.net_of([(1.0, 100.0)], 100.0, 0.001) - (0.999 * 0.999 / (1.001 * 1.001) - 1)) < 1e-12


def test_picks_of_one_moment():
    t = pd.DataFrame({"signal_ts": [1, 1, 1, 1, 2], "fell24": [-0.3, -0.1, -0.5, -0.2, -0.1],
                      "turnover": [5, 50, 1, 20, 3]})
    assert ex.picked(t, "fell").tolist() == [True, False, True, True, True]
    assert ex.picked(t, "liquid").tolist() == [True, True, False, True, True]
    t["moment_size"] = t.groupby("signal_ts")["fell24"].transform("size")
    assert ex.picked(t, "mass4").tolist() == [True, True, False, True, False]     # a lone coin is no mass panic


def test_a_portfolio_takes_a_share_of_its_equity_and_no_more_than_max_open():
    trades = pd.DataFrame({"signal_ts": [0, HOUR, 2 * HOUR], "held": [10, 10, 1], "net": [0.5, -0.5, 0.0],
                           "history": ["A", "B", "C"]})
    book = ex.portfolio(trades, 0, 10 * HOUR, share=0.2, max_open=2)
    assert book["trades"] == 2                                   # C waited for a free slot that never came
    assert abs(book["end_usd"] - (100 + 20 * 0.5 + 20 * -0.5)) < 1e-9
    assert book["drawdown_pct"] <= 0


def test_the_usual_daily_move_needs_most_of_a_week():
    closes = 100 * np.exp(np.cumsum(np.r_[0, np.tile([0.01, -0.01], 100)]))
    assert abs(ex.daily_sigma(closes, 169) - 0.01 * np.sqrt(24)) < 1e-3
    assert np.isnan(ex.daily_sigma(closes, 40))
