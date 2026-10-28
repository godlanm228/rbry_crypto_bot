"""Pairs: every top coin against every shortable bottom coin; a day trades no coin twice."""

import pandas as pd

from core.brain import pairs as pr


def test_a_day_takes_the_best_pairs_without_reusing_a_coin():
    day = pd.Timestamp("2024-01-01")
    pred = pd.DataFrame({"day": day, "long_history": ["A", "A", "B", "C"], "short_history": ["X", "Y", "X", "Z"],
                         "priority": [0.9, 0.8, 0.7, 0.1]})
    chosen = pr.choose(pred, per_day=3)
    assert list(zip(chosen["long_history"], chosen["short_history"])) == [("A", "X"), ("C", "Z")]
