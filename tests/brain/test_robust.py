"""Robustness: earning when history is bent, the crises, and the chance that choosing by history is luck."""

import numpy as np
import pandas as pd

from core.brain import factory as fa
from core.brain import robust as rb

HOUR = 3_600_000


def _stats(growths, drawdown=-10.0):
    return {"learn_n": 500, "years": {str(y): {"n": 50, "growth_pct": g, "drawdown_pct": drawdown}
                                      for y, g in zip(fa.YEARS, growths)}}


def test_robust_means_earning_over_the_learning_years_and_in_both_checks():
    crash_buyer = _stats([4, -7, -22, 81, -8, 61, 9, 17, 14], -36)      # the panic, an hour late
    assert rb.holds(crash_buyer, "razgon") and not rb.strict(crash_buyer, "razgon")   # 3 losing years: not strict
    assert not rb.holds(crash_buyer, "seif")                                         # a 36% drawdown
    assert not rb.holds(_stats([4, -7, -22, 81, -8, 61, 9, -1, 14], -36), "razgon")  # 2025 lost
    assert not rb.holds(_stats([-20, -20, -20, 5, 5, 5, 5, 5, 5]), "razgon")          # loses over 2018-2024
    assert abs(rb.learn_growth(_stats([10] * 9)) - 10) < 1e-9


def test_the_crises_are_the_trades_open_during_a_shock():
    luna = int(pd.Timestamp("2022-05-08").value // 10**6)
    t = pd.DataFrame({"signal_ts": [luna - 5 * 24 * HOUR, luna - 24 * HOUR, luna + 30 * 24 * HOUR],
                      "held": [24, 72, 24], "net": [0.1, -1.0, 0.2]})
    row = {r["crisis"]: r for r in rb.crises(t)}["LUNA"]
    assert row["trades"] == 1 and row["worst_pct"] == -100.0


def test_a_choice_that_holds_on_unseen_years_is_no_luck():
    rng = np.random.default_rng(1)
    pool = {}
    for i in range(30):
        g = list(rng.normal(0, 5, 9))
        pool[f"g{i}"] = _stats(g)
    pool["star"] = _stats([30] * 9)                                     # best everywhere
    p = rb.pbo(pool, "razgon")
    assert p["splits"] == 35 and p["pbo"] == 0.0 and p["median_rank"] > 0.95
