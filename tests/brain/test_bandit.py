"""The RL experiment: the agent learns only from trades closed before its decision and finds the paying action."""

import numpy as np
import pandas as pd

from core.brain import bandit as bd
from core.brain import selector as se

DAY = 86_400_000
T0 = int(pd.Timestamp("2018-06-01").value // 10**6)


def test_the_agent_finds_what_pays_in_a_situation_and_only_learns_after_the_close():
    rows, cands = [], []
    rng = np.random.default_rng(2)
    for d in range(400):
        ts = T0 + d * DAY
        up = d % 2 == 0
        cands.append({"history": "C", "signal_ts": ts, "trend": "up" if up else "down", "mood": "calm",
                      "breadth_state": "wide", "alt_state": "btc", **{c: 0 for c in se.EVENT_COLS}, "ev_DUMP": 1})
        for action, side in (("L24", 1), ("S24", -1)):
            good = (side > 0) == up
            rows.append({"history": "C", "signal_ts": ts, "action": action, "side": side, "held": 24,
                         "exit_ts": ts + DAY, "year": pd.Timestamp(ts, unit="ms").year, "turnover": 1e8,
                         "net": (0.03 if good else -0.03) + rng.normal(0, 0.005)})
    picks = bd.run(pd.DataFrame(rows), pd.DataFrame(cands), start="2018-06-01")
    late = picks[picks["signal_ts"] >= T0 + 200 * DAY]
    assert len(late) > 150 and (late["net"] > 0).mean() > 0.95
    first = picks[picks["signal_ts"] == T0]
    assert first.empty or abs(first["forecast"].iloc[0]) < 0.1               # it knew nothing on its first day
