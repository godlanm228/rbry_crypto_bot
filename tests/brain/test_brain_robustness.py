"""Many versions of one strategy: same seed, same versions; the spread reads plainly."""

import pandas as pd

from core.brain import robustness as rb
from core.brain.playbook import periods


def test_versions_are_reproducible_and_vary():
    a, b = rb.variants(20, seed=5), rb.variants(20, seed=5)
    assert a == b
    assert len({v["positions"] for v in a}) > 1 and len({v["shift"] for v in a}) > 1
    assert all(0 <= v["shift"] <= rb.MAX_START_SHIFT and v["positions"] in rb.POSITIONS for v in a)


def test_a_strategy_that_always_wins_wins_in_every_version():
    days = pd.date_range("2024-11-01", "2025-03-31", freq="D")
    orders = pd.DataFrame({"day": days, "history": [f"C{i % 40}" for i in range(len(days))], "side": 1,
                           "priority": 1.0, "ret": 0.02, "exit_day": days + pd.Timedelta(days=3)})
    btc = pd.Series(100.0, index=days)
    results = rb.run_variants(orders, days, periods, btc * 0, None, btc, rb.variants(10))
    spread = rb.spread_of(results)
    assert set(spread.index) == {"2020–2024", "2025"}
    assert spread.loc["2025", "up_of10"] == 10 and spread.loc["2025", "bad_tenth_pct"] > 0
