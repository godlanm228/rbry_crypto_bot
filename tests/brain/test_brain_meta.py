"""The meta-brain: no look-ahead in its state, only stable markers, sizes that keep the money at work."""

import numpy as np
import pandas as pd

from core.brain import meta
from core.brain.walkforward import GAP_DAYS


def _outcomes(days, spike_day=None):
    out = pd.DataFrame({b: 0.0 for b in meta.BOOKS}, index=days)
    if spike_day is not None:
        out.loc[spike_day, "long"] = 1.0
    return out


def test_the_state_sees_a_signal_day_only_once_it_is_graded():
    days = pd.date_range("2023-01-01", periods=120)
    spike = days[60]
    rows = pd.DataFrame({"day": days, "ret_7d": 0.0, "funding_7d": 0.0, **{c: 0.0 for c in meta.MARKET}})
    st = meta.state(rows, _outcomes(days, spike))
    assert st.loc[spike + pd.Timedelta(days=GAP_DAYS - 1), "recent_long"] == 0.0
    assert st.loc[spike + pd.Timedelta(days=GAP_DAYS), "recent_long"] > 0.0


def test_key_markers_keep_only_features_with_one_sign_in_every_half_year():
    rng = np.random.default_rng(1)
    days = pd.date_range("2021-01-01", "2022-12-31")
    y = pd.Series(rng.normal(size=len(days)), index=days, name="long")
    steady = y + rng.normal(scale=2.0, size=len(days))
    flips = np.where(days.month <= 6, 1.0, -1.0) * y + rng.normal(scale=0.5, size=len(days))
    x = pd.DataFrame({"steady": steady, "flips": flips, "noise": rng.normal(size=len(days))}, index=days)
    assert meta.key_markers(x, y) == {"steady": 1.0}


def test_tilted_sizes_follow_the_forecast_against_the_usual_outcome():
    days = pd.date_range("2022-01-01", periods=200)
    outcomes = pd.DataFrame({b: 0.02 for b in meta.BOOKS}, index=days)
    pred = pd.DataFrame({"pred_long": 0.02, "pred_short": -0.01, "pred_btc": 0.10}, index=days[150:])
    w = meta.tilted(pred, outcomes)
    assert np.allclose(w["long"], 1.0) and np.allclose(w["short"], 0.0) and np.allclose(w["btc"], 2.0)


def test_a_book_of_thirds():
    days = pd.date_range("2024-01-01", periods=2)
    on = pd.DataFrame({"long": [True, False], "short": [True, True], "btc": [False, False]}, index=days)
    outcomes = pd.DataFrame({"long": [0.09, 0.30], "short": [0.03, -0.06], "btc": [0.5, 0.5]}, index=days)
    assert np.allclose(meta.mix(on, outcomes), [0.04, -0.02])
