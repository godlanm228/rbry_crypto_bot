"""The portfolio books money as a person would: limits, one position per coin, the brake."""

import numpy as np
import pandas as pd

from core.brain import playbook
from core.brain import portfolio as pf

D = pd.Timestamp("2024-01-01")


def _order(day, history, ret, side=1, priority=1.0, hold=15):
    return {"day": D + pd.Timedelta(days=day), "history": history, "side": side, "priority": priority,
            "ret": ret, "exit_day": D + pd.Timedelta(days=day + hold)}


def test_positions_are_sized_on_value_and_booked_when_they_close():
    orders = pd.DataFrame([_order(0, "A", 0.10, priority=3), _order(0, "B", -0.05, priority=2),
                           _order(0, "A", 0.5, priority=1)])
    days = pd.date_range(D, periods=20)
    values, trades = pf.run(orders, days)
    assert len(trades) == 2                                  # the second A is refused: A is already held
    assert abs(values.iloc[-1] - (10_000 + 1_000 * 0.10 - 1_000 * 0.05)) < 1e-9
    assert values.iloc[5] == 10_000                           # open positions count at cost


def test_limits_on_positions_and_shorts():
    orders = pd.DataFrame([_order(0, f"C{i}", 0.0, side=-1) for i in range(8)])
    values, trades = pf.run(orders, pd.date_range(D, periods=20), rules=pf.Rules(max_short_share=0.3))
    assert len(trades) == 3                                  # 3 x 10% fills the 30% short limit
    orders = pd.DataFrame([_order(0, f"C{i}", 0.0) for i in range(12)])
    values, trades = pf.run(orders, pd.date_range(D, periods=20))
    assert len(trades) == pf.MAX_POSITIONS


def test_after_a_deep_fall_new_positions_are_half_size():
    orders = pd.DataFrame([_order(0, "A", -1.0, hold=2), _order(0, "B", -1.0, hold=2), _order(0, "C", -1.0, hold=2),
                           _order(5, "D", 0.1)])
    values, trades = pf.run(orders, pd.date_range(D, periods=30), rules=pf.Rules(position_share=0.1, drawdown_brake=0.2))
    d = trades.set_index("history")
    assert abs(d.loc["D", "size"] - values.iloc[4] * 0.1 * 0.5) < 1e-9   # 30% down: the brake halves it


def test_idle_money_can_sit_in_btc():
    days = pd.date_range(D, periods=3)
    values, _ = pf.run(pd.DataFrame(columns=["day", "history", "side", "priority", "ret", "exit_day"]), days,
                       btc_daily_ret=pd.Series([0.0, 0.1, -0.1], index=days),
                       idle_in_btc=pd.Series([True, True, False], index=days))
    assert np.allclose(values.to_numpy(), [10_000, 11_000, 11_000])


def test_the_playbook_picks_the_best_longs_and_the_worst_shorts():
    day = pd.Timestamp("2024-03-01")
    frame = pd.DataFrame({"day": day, "history": [f"C{i}" for i in range(10)], "base": "X",
                          "place": np.linspace(0.1, 1.0, 10), "btc_trend": 0.05, "hold_ret": 0.01, "ret_net": 0.01,
                          "hours": 30.0, "short_hold_ret": 0.02, "short_ret_net": 0.02, "short_hours": 30.0})
    frame.loc[0, "short_hold_ret"] = np.nan                  # no perpetual: cannot be shorted
    longs = playbook.BY_ID["L2"].orders(frame)
    shorts = playbook.BY_ID["N1"].orders(frame).query("side < 0")
    assert sorted(longs["history"]) == ["C7", "C8", "C9"]
    assert sorted(shorts["history"]) == ["C1", "C2", "C3"]
    assert playbook.BY_ID["S1"].orders(frame).empty          # BTC is up: no shorts
