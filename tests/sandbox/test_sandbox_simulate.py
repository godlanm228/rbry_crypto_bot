"""Trade resolution on hand-made candles."""

import numpy as np
import pytest

from core.sandbox.simulate import Costs, Setup, net_return, simulate

FREE = Costs(fee_pct=0.0, slippage_pct=0.0)


def bars(*rows):
    o, h, l, c = (np.array(x, dtype=float) for x in zip(*rows))
    return o, h, l, c


# signal on bar 0 (close 100); market entry at the open of bar 1 = 100;
# ATR 2 -> stop 96 (2 ATR), target 108 (2R)
BASE = (100, 101, 99, 100)
ATR_SETUP = Setup(0, stop_atr=2.0, target_r=2.0, atr=2.0, max_hold=5)


def test_target_hit_is_plus_two_r_before_costs():
    out = simulate(*bars(BASE, (100, 103, 99, 102), (102, 109, 101, 108), (108, 108, 108, 108),
                         (108, 108, 108, 108), (108, 108, 108, 108)), ATR_SETUP, FREE)
    assert (out.entry, out.stop, out.target, out.exit, out.reason) == (100, 96, 108, 108, "target")
    assert (out.entry_bar, out.exit_bar, out.bars) == (1, 2, 2)
    assert out.r_net == pytest.approx(2.0)
    assert out.mfe_r == pytest.approx(2.0) and out.mae_r == pytest.approx(0.25)   # low 99 = 1/4 R


def test_costs_are_paid_on_both_sides():
    candles = bars(BASE, (100, 103, 99, 102), (102, 109, 101, 108), *[(108,) * 4] * 3)
    out = simulate(*candles, ATR_SETUP)
    expected = 108 * 0.9995 * 0.999 / (100 * 1.0005 * 1.001) - 1
    assert net_return(100, 108, Costs()) == pytest.approx(expected)
    assert out.r_net == pytest.approx(expected / 0.04)        # risk 4 on 100 = 4%


def test_stop_wins_when_one_candle_touches_both():
    out = simulate(*bars(BASE, (100, 110, 95, 100), *[(100,) * 4] * 5), ATR_SETUP, FREE)
    assert (out.reason, out.exit) == ("stop", 96) and out.r_net == pytest.approx(-1.0)


def test_gap_through_the_stop_exits_at_the_open():
    out = simulate(*bars(BASE, (100, 101, 98, 99), (93, 94, 92, 93), *[(93,) * 4] * 4), ATR_SETUP, FREE)
    assert (out.reason, out.exit) == ("stop_gap", 93) and out.r_net == pytest.approx(-1.75)


def test_gap_through_the_target_exits_at_the_open():
    out = simulate(*bars(BASE, (100, 104, 99, 103), (110, 112, 109, 111), *[(111,) * 4] * 4), ATR_SETUP, FREE)
    assert (out.reason, out.exit) == ("target_gap", 110) and out.r_net == pytest.approx(2.5)


def test_time_exit_at_the_last_close():
    setup = Setup(0, stop_atr=2.0, target_r=2.0, atr=2.0, max_hold=2)
    out = simulate(*bars(BASE, (100, 102, 99, 101), (101, 103, 100, 102), (102,) * 4), setup, FREE)
    assert (out.reason, out.exit_bar, out.exit) == ("time", 2, 102) and out.r_net == pytest.approx(0.5)


def test_unfinished_trade_is_left_out():
    assert simulate(*bars(BASE, (100, 102, 99, 101)), ATR_SETUP, FREE) is None


def test_market_setup_that_opens_through_its_stop_never_existed():
    setup = Setup(0, stop=97.0, target=110.0, max_hold=3)
    assert simulate(*bars(BASE, (96, 98, 95, 97), *[(97,) * 4] * 3), setup, FREE) is None


def test_limit_fills_at_the_limit_and_ignores_the_target_on_the_fill_day():
    setup = Setup(0, stop=94.0, target=105.0, limit=97.0, valid_days=3, max_hold=5)
    candles = bars(BASE, (100, 101, 98, 99), (99, 110, 96.5, 100), (100, 106, 99, 104), *[(104,) * 4] * 4)
    out = simulate(*candles, setup, FREE)
    assert (out.entry_bar, out.entry) == (2, 97)
    assert (out.reason, out.exit_bar, out.exit) == ("target", 3, 105)
    assert out.r_net == pytest.approx(8 / 3)


def test_limit_fills_at_the_open_when_the_day_opens_below_it():
    setup = Setup(0, stop=94.0, target=105.0, limit=97.0, valid_days=3, max_hold=3)
    out = simulate(*bars(BASE, (95.5, 96, 95, 95.5), *[(96,) * 4] * 4), setup, FREE)
    assert (out.entry, out.reason) == (95.5, "time")


def test_limit_that_never_fills_is_no_trade():
    setup = Setup(0, stop=94.0, target=105.0, limit=97.0, valid_days=2, max_hold=3)
    assert simulate(*bars(BASE, *[(100, 101, 98, 100)] * 6), setup, FREE) is None
