"""Trade plan arithmetic on hand-computed numbers."""

import dataclasses
import math

import pandas as pd
import pytest

from core.risk import trade_plan as tp
from core.risk.service import STYLES


def test_stop_goes_under_the_nearest_support_with_a_buffer():
    # price 100, ATR 4: support 95 -> 95 - 0.5*4 = 93 (7 away, within 3 ATR)
    assert tp.place_stop(100, 4, [80, 95]) == (93, "support", 95)


def test_stop_is_never_closer_than_one_atr():
    # support 99 -> 97 would be 3 away (< 4): pushed to 100 - 4 = 96
    assert tp.place_stop(100, 4, [99]) == (96, "support", 99)


def test_far_or_missing_support_falls_back_to_two_atr():
    # support 80 -> 78 is 22 away (> 3*4): plain 2 ATR stop, support kept for info
    assert tp.place_stop(100, 4, [80]) == (92, "volatility", 80)
    assert tp.place_stop(100, 4, []) == (92, "volatility", None)


def test_target_is_the_next_resistance_at_least_one_atr_away():
    # 102 is closer than 1 ATR (104): the target is 110
    assert tp.place_target(100, 4, 93, [102, 110, 130]) == (110, "resistance")
    # nothing above: 2R = 100 + 2*7
    assert tp.place_target(100, 4, 93, [90]) == (114, "two_r")


def test_hit_chance_is_the_calibrated_driftless_formula():
    # 0.86 * ln(100/93) / ln(110/93) = 0.86 * 0.072571 / 0.167884 = 0.37175
    assert tp.hit_chance(100, 93, 110) == pytest.approx(0.37175, abs=1e-4)


def test_week_range_uses_calibrated_tails():
    # +-2% daily log moves alternating: sample sigma = 0.02*sqrt(90/89) = 0.020112,
    # weekly 0.053211 -> 100*exp(-0.89*0.053211) = 95.37, 100*exp(0.80*0.053211) = 104.35
    closes = [100.0]
    for i in range(90):
        closes.append(closes[-1] * math.exp(0.02 if i % 2 == 0 else -0.02))
    week = tp.week_range(100.0, closes)
    assert week.low == pytest.approx(95.37, abs=0.01)
    assert week.high == pytest.approx(104.35, abs=0.01)


def test_trend_needs_price_and_average_slope_to_agree():
    rising = [float(i) for i in range(1, 101)]
    assert tp.trend_of(101.0, rising) == "up"
    assert tp.trend_of(10.0, rising[::-1]) == "down"
    assert tp.trend_of(50.0, rising) == "flat"            # rising average, price below it
    assert tp.trend_of(50.0, rising[:40]) == "flat"       # too short to judge


@pytest.mark.parametrize("rr, trend, verdict", [
    (2.5, "up", "good"), (2.5, "down", "ok"), (1.6, "flat", "ok"), (1.2, "up", "weak")])
def test_verdict(rr, trend, verdict):
    assert tp.verdict_for(rr, trend) == verdict


def test_better_entry_waits_just_above_the_support():
    # support 90, ATR 4: entry 91, stop min(88, 87) = 87, R:R (110-91)/(91-87) = 4.75
    better = tp.better_entry(100, 4, 90, 110)
    assert (better.price, better.stop, better.reward_risk) == (91, 87, 4.75)
    assert better.hit_chance == pytest.approx(tp.hit_chance(91, 87, 110))


def test_no_better_entry_when_it_is_too_close_or_too_far():
    assert tp.better_entry(100, 4, 98, 110) is None       # 1 away: we are already there
    assert tp.better_entry(100, 4, 80, 110) is None       # 19% away: a different trade


def _wave(days=120, low=95.0, high=110.0, period=30, wick=2.0):
    mid, amp = (low + high) / 2, (high - low) / 2
    closes = [mid + amp * math.sin(2 * math.pi * i / period) for i in range(days)]
    return tp.Candles(tuple(c + wick for c in closes), tuple(c - wick for c in closes), tuple(closes))


def test_full_plan_on_a_range_market():
    plan = tp.build_trade_plan("sol", 100.0, _wave(), user_id=1, style=STYLES["balanced"],
                               account_value=1000.0)
    assert plan.symbol == "SOL"
    assert plan.stop_basis == "support" and plan.stop < plan.support < plan.price
    assert plan.target_basis == "resistance" and plan.target == pytest.approx(112.0, abs=0.2)
    assert plan.reward_risk == pytest.approx((plan.target - 100) / (100 - plan.stop))
    # the stop costs the style's 2% of $1000, fees included
    assert plan.sizing.approved and plan.sizing.loss_at_stop == pytest.approx(20.0, abs=0.01)
    assert 0 < plan.hit_chance < 1 and plan.week.low < 100 < plan.week.high


def test_short_history_has_no_plan():
    c = _wave(days=40)
    with pytest.raises(tp.PlanUnavailable):
        tp.build_trade_plan("SOL", 100.0, c, user_id=1, style=STYLES["balanced"], account_value=1000.0)
    with pytest.raises(tp.PlanUnavailable):
        tp.build_trade_plan("SOL", 100.0, None, user_id=1, style=STYLES["balanced"], account_value=1000.0)


def test_candles_from_frame_drops_the_open_candle_and_needs_high_low():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-10-24", "2026-10-25", "2026-10-26"]),
        "price": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5], "low": [0.5, 1.5, 2.5], "volume": [1, 1, 1]})
    c = tp.candles_from_frame(frame, today=pd.Timestamp("2026-10-26"))
    assert c.closes == (1.0, 2.0)
    assert tp.candles_from_frame(frame.drop(columns=["high", "low"])) is None


def test_candles_from_klines_drops_the_open_candle():
    rows = [{"high": 2, "low": 1, "close": 1.5}, {"high": 3, "low": 2, "close": 2.5}]
    assert tp.candles_from_klines(rows).closes == (1.5,)


def test_alternatives_are_good_setups_best_first():
    base = tp.build_trade_plan("SOL", 100.0, _wave(), user_id=1, style=STYLES["balanced"],
                               account_value=1000.0)
    plans = [dataclasses.replace(base, symbol=s, verdict=v, reward_risk=rr) for s, v, rr in
             [("ETH", "good", 2.6), ("LINK", "good", 3.1), ("DOT", "ok", 4.0), ("SOL", "good", 5.0),
              ("ADA", "good", 2.1)]]
    assert [p.symbol for p in tp.pick_alternatives(plans, "sol")] == ["LINK", "ETH"]


def test_alternatives_must_beat_the_plan_on_screen():
    # 27.10: NEAR (2.3) listed UNI (2.0) as "a better setup right now"
    base = tp.build_trade_plan("SOL", 100.0, _wave(), user_id=1, style=STYLES["balanced"],
                               account_value=1000.0)
    plans = [dataclasses.replace(base, symbol=s, verdict="good", reward_risk=rr)
             for s, rr in [("UNI", 2.0), ("DOT", 2.6)]]
    assert [p.symbol for p in tp.pick_alternatives(plans, "NEAR", beat=2.3)] == ["DOT"]
    assert tp.pick_alternatives(plans, "NEAR", beat=3.0) == []
