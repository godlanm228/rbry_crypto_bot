"""The trade evaluator: how wild a coin's two weeks were, measured from the trade's entry, and the why in words."""

import numpy as np
import pandas as pd

from core.brain import evaluator as ev

HOUR, DAY = 3_600_000, 86_400_000
D0 = pd.Timestamp("2026-01-10")
T0 = int((D0 - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1))


def _path(closes, start):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    ts = start + np.arange(len(c), dtype=np.int64) * HOUR
    return np.column_stack([ts, o, np.maximum(o, c), np.minimum(o, c), c])


class Market:
    def __init__(self, prices, btc):
        self._p, self._btc = prices, btc

    def prices(self, history):
        return self._p.get(history, np.empty((0, 5)))

    def btc(self):
        return self._btc


def test_excursions_from_the_entry_within_two_weeks():
    squeeze = np.r_[np.full(48, 100.0), np.linspace(100, 160, 24), np.full(30 * 24, 120.0)]   # +60% on day 3
    calm = np.r_[np.full(10, 100.0), np.linspace(100, 80, 24), np.full(30 * 24, 80.0)]        # -20% at worst
    late = np.r_[np.full(20 * 24, 100.0), np.full(24, 300.0), np.full(10 * 24, 100.0)]       # after the window
    start = T0 + DAY
    market = Market({"S": _path(squeeze, start), "C": _path(calm, start), "L": _path(late, start)},
                    _path(np.full(40 * 24, 1.0), start))
    rows = pd.DataFrame({"history": ["S", "C", "L"], "day": [D0] * 3})
    out = ev.labels(ev.excursions(market, rows)).set_index("history")
    assert abs(out.loc["S", "max_up"] - 0.6) < 1e-9 and out.loc["S", "squeeze"] == 1.0
    assert abs(out.loc["C", "max_down"] + 0.2) < 1e-9 and out.loc["C", "crash"] == 0.0
    assert out.loc["L", "max_up"] == 0.0 and out.loc["L", "squeeze"] == 0.0


def test_an_unfinished_window_is_unknown_unless_the_coin_stopped():
    start = T0 + DAY
    market = Market({"NEW": _path(np.full(5 * 24, 100.0), start), "DEAD": _path(np.full(5 * 24, 100.0), start)},
                    _path(np.full(6 * 24, 1.0), start))
    rows = pd.DataFrame({"history": ["NEW"], "day": [D0]})
    assert np.isnan(ev.excursions(market, rows)["max_up"].iloc[0])        # the newest data: not over yet
    market = Market({"DEAD": _path(np.full(5 * 24, 100.0), start)}, _path(np.full(30 * 24, 1.0), start))
    rows = pd.DataFrame({"history": ["DEAD"], "day": [D0]})
    assert ev.excursions(market, rows)["max_up"].iloc[0] == 0.0             # stopped trading: over


def test_why_in_words():
    assert ev.why("atr_pct|dd_365|funding_1d") == "резкость (ATR), падение от пика за год, фандинг за день"
    assert ev.why("something_new") == "something_new"
