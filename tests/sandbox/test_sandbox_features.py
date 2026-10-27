"""Situation vectors: no peeking at the future, levels equal to the live detector."""

import math

import numpy as np
import pandas as pd
import pytest

from core.analysis.signal_generator import find_support_resistance
from core.sandbox.features import FEATURES, WARMUP, build_situations
from core.sandbox.levels import support_resistance


def _walk(days=330, seed=3, start="2023-01-01"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, days)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.005, days))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.03, days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.03, days))
    return pd.DataFrame({"day": pd.date_range(start, periods=days, freq="D"), "open": open_, "high": high,
                         "low": low, "close": close, "volume": rng.uniform(1e6, 5e6, days)})


def test_numpy_levels_equal_the_live_detector():
    rng = np.random.default_rng(11)
    for _ in range(100):
        n = int(rng.integers(45, 230))
        c = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))), 1)
        h, l = c * (1 + rng.uniform(0, 0.03, n)), c * (1 - rng.uniform(0, 0.03, n))
        live = find_support_resistance(pd.DataFrame({"high": h, "low": l, "close": c}))
        assert support_resistance(h, l) == (live["support"], live["resistance"])


def test_rows_do_not_change_when_the_future_is_cut_off():
    coin, btc = _walk(seed=3), _walk(seed=4)
    fng = pd.DataFrame({"day": coin["day"], "value": np.arange(len(coin)) % 100})
    full = build_situations(coin, btc, fng).set_index("day")
    numeric = [c for c in full.columns if full[c].dtype.kind in "fi"]
    for t in (WARMUP + 5, WARMUP + 60, len(coin) - 2):
        cut = build_situations(coin.iloc[:t + 1], btc.iloc[:t + 1], fng.iloc[:t + 1]).set_index("day")
        day = coin["day"].iloc[t]
        a, b = full.loc[day, numeric].astype(float), cut.loc[day, numeric].astype(float)
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), (a - b)[~np.isclose(a, b, equal_nan=True)]
        for col in ("trend", "verdict", "stop_basis"):
            assert full.loc[day, col] == cut.loc[day, col]


def test_vector_has_every_feature_after_warmup():
    s = build_situations(_walk(), _walk(seed=9), pd.DataFrame({"day": _walk()["day"], "value": 50}))
    assert len(s) == 330 - WARMUP + 1
    last = s.iloc[-1]
    missing = [f for f in FEATURES if not math.isfinite(float(last[f]))]
    assert missing == []
    assert 0 <= last["range_pos_30"] <= 1 and last["dd_90"] <= 0 and 0 <= last["rsi14"] <= 100


def test_plan_levels_match_the_card_math():
    s = build_situations(_walk(), None, None)
    row = s.dropna(subset=["plan_stop"]).iloc[-1]
    assert row["plan_stop"] < row["close"] < row["plan_target"]
    assert row["plan_rr"] == pytest.approx((row["plan_target"] - row["close"]) / (row["close"] - row["plan_stop"]))
    assert row["verdict"] in {"good", "ok", "weak"} and row["trend"] in {"up", "down", "flat"}
