"""The meta-brain: no look-ahead in its state, only stable markers, sizes that keep the money at work."""

import numpy as np
import pandas as pd
import pytest

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


def test_short_books_hold_the_same_coins_a_few_days_with_costs():
    from core.brain.dataset import FEE_PCT, FUTURES_FEE_PCT
    days = pd.date_range("2024-03-01", periods=3)
    closes = {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 300.0], "BTC": [50.0, 55.0, 55.0]}
    rows = pd.DataFrame([{"history": h, "base": h, "day": d, "close": c[i], "turnover_30d": 1e9,
                          "funding_1d": 0.001 if h == "B" else np.nan}
                         for h, c in closes.items() for i, d in enumerate(days)])
    scored = pd.DataFrame({"history": ["A", "B"] * 2, "day": [days[0]] * 2 + [days[1]] * 2, "place": [0.9, 0.1] * 2,
                           "hold_ret": [0.0, np.nan] * 2, "short_liq_ret": [np.nan, 0.0] * 2, "slip_pct": 0.0})
    out = meta.short_books(scored, rows, hold_days=1, picks=1)
    fee, ffee = FEE_PCT / 100, FUTURES_FEE_PCT / 100
    assert out.loc[days[0], "long"] == pytest.approx(110 * (1 - fee) / (100 * (1 + fee)) - 1)
    assert out.loc[days[0], "short"] == pytest.approx((100 * (1 - ffee) - 90 * (1 + ffee)) / 100 + 0.001)
    assert out.loc[days[1], "short"] == -1.0                       # 1x margin: a short loses at most its size
    assert out.loc[days[0], "btc"] == pytest.approx(55 * (1 - 0.0005) * (1 - fee) / (50 * 1.0005 * (1 + fee)) - 1)


def test_an_outcome_counts_in_the_period_it_is_known_in():
    days = pd.DatetimeIndex(["2024-12-10", "2024-12-20", "2025-12-10", "2025-12-20"])
    out = pd.DataFrame({b: 0.0 for b in meta.BOOKS}, index=days)
    assert meta.period_of(days).tolist() == ["2022–2024", "2025", "2025", "2025"]
    assert list(meta.known_by(out).index) == list(days[:3])          # graded on 2026 prices: left out
    from core.brain import walkforward as wf
    assert wf.period(pd.Series(days)).tolist() == ["dev", "valid", "valid", "holdout"]


def test_btc_always_holds_every_day():
    from core.brain.playbook import BY_ID
    frame = pd.DataFrame({"day": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"])})
    idle = BY_ID["B0"].idle_in_btc(frame)
    assert idle.all() and list(idle.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    assert BY_ID["B0"].orders(frame).empty
