"""Statistics, memory and clusters on numbers small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from core.sandbox.evaluate import (
    add_context, fit_scaler, kmeans, knn_mean, period_of, spearman, t_cluster, verdict)
from core.sandbox.replay import mark_overlaps
from core.sandbox.strategies import structure_stop


def test_t_cluster_counts_a_week_as_one_bet():
    # two weeks, each with identical results inside: the mean is 1, the
    # cluster sums of deviations are +-1 -> se = sqrt(2 * 2/1) / 4 = 0.5,
    # where four independent trades would give t = 1 / (0.577 / 2) = 3.46
    values = np.array([1.5, 1.5, 0.5, 0.5])
    assert t_cluster(values, np.array(["w1", "w1", "w2", "w2"])) == pytest.approx(2.0)


def test_spearman_is_rank_based():
    assert spearman(np.array([1, 2, 3, 4.0]), np.array([10, 20, 30, 1000.0])) == pytest.approx(1.0)
    assert spearman(np.array([1, 2, 3, 4.0]), np.array([4, 3, 2, 1.0])) == pytest.approx(-1.0)


def test_knn_mean_averages_the_nearest_rows():
    memory = np.array([[0.0], [0.1], [5.0], [5.1]], dtype=np.float32)
    values = np.array([1.0, 3.0, -1.0, -3.0])
    pred = knn_mean(memory, values, np.array([[0.05], [5.05]], dtype=np.float32), k=2)
    assert pred.tolist() == [2.0, -2.0]


def test_kmeans_finds_two_blobs():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 0.1, (50, 2)), rng.normal(5, 0.1, (50, 2))]).astype(np.float32)
    centres, labels = kmeans(x, k=2)
    assert len(set(labels[:50])) == 1 and len(set(labels[50:])) == 1 and labels[0] != labels[-1]
    assert sorted(np.round(centres[:, 0]).tolist()) == [0.0, 5.0]


def test_scaler_is_robust_and_fills_unknowns_with_zero():
    x = np.array([[1.0, np.nan], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0],
                  [5.0, 4.0], [6.0, 5.0], [7.0, 6.0], [1000.0, 7.0]])
    z = fit_scaler(x)(x)
    # median 4.5, IQR 6.25 - 2.75 = 3.5: the outlier is ~284 IQRs away -> clipped
    assert z[0, 1] == 0.0 and z[7, 0] == 4.0 and z[0, 0] == pytest.approx(-1.0)


def test_periods_are_fixed_dates():
    days = pd.Series(["2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01"])
    assert period_of(days).tolist() == ["dev", "valid", "valid", "holdout"]


def test_excess_is_the_result_over_its_random_day_twins():
    t = pd.DataFrame({
        "strategy": ["control", "oversold", "oversold"], "symbol": ["SOL"] * 3,
        "signal_day": ["2024-03-01", "2024-03-05", "2025-04-02"], "r_net": [-0.4, 1.0, 0.5],
        "twin_r": [np.nan, -0.2, 0.75], "reason": ["stop", "target", "time"], "bars": [3] * 3,
        "overlap": [0] * 3, "exit_day": ["2024-03-04"] * 3})
    ctx = add_context(t)
    assert ctx["excess_r"].tolist()[1:] == pytest.approx([1.2, -0.25])
    assert np.isnan(ctx["excess_r"].iloc[0]) and ctx["period"].tolist() == ["dev", "dev", "valid"]


def test_a_twin_keeps_the_trade_shape_in_atrs():
    from core.sandbox.replay import twin
    from core.sandbox.simulate import Setup
    # signal close 100, ATR 4: stop 93 (1.75 ATR below), target 112 (3 ATR above),
    # limit 96 (1 ATR below); on a day with close 50 and ATR 1 -> 48.25 / 53 / 49
    s = twin(Setup(10, stop=93.0, target=112.0, limit=96.0, valid_days=7, max_hold=14), 100.0, 4.0, 42, 50.0, 1.0)
    assert (s.signal_bar, s.stop, s.target, s.limit, s.valid_days, s.max_hold) == (42, 48.25, 53.0, 49.0, 7, 14)
    atr_based = twin(Setup(10, stop_atr=2.0, target_r=3.0, atr=4.0), 100.0, 4.0, 42, 50.0, 1.0)
    assert (atr_based.stop, atr_based.stop_atr, atr_based.target_r, atr_based.atr) == (None, 2.0, 3.0, 1.0)


def test_verdict_needs_both_periods():
    good = {"n": 100, "avg_r": 0.2, "avg_excess_r": 0.15, "t_excess": 2.5}
    weak = {**good, "t_excess": 1.0}
    assert verdict(good, good).startswith("КАНДИДАТ")
    assert verdict(good, weak) == "в плюсе, но не лучше случайного входа (едет на рынке)"
    assert verdict({**good, "n": 10}, good) == "мало сделок"


def test_overlapping_trades_are_flagged_per_rule_and_coin():
    t = pd.DataFrame({"strategy": ["a"] * 3 + ["b"], "symbol": ["X"] * 4,
                      "entry_bar": [1, 3, 6, 2], "exit_bar": [5, 7, 9, 4]})
    assert mark_overlaps(t).sort_values(["strategy", "entry_bar"])["overlap"].tolist() == [0, 1, 0, 0]


def test_structure_stop_follows_the_plan_rules():
    assert structure_stop(100, 4, 95) == 93          # level - 0.5 ATR
    assert structure_stop(100, 4, 99) == 96          # never closer than 1 ATR
    assert structure_stop(100, 4, 80) == 92          # further than 3 ATR -> 2 ATR


def test_memory_filter_uses_a_cut_off_fixed_before_the_holdout():
    from core.sandbox.evaluate import memory_filter
    from core.sandbox.features import FEATURES
    rng = np.random.default_rng(5)
    days = [*pd.date_range("2023-01-01", periods=400), *pd.date_range("2025-01-01", periods=200),
            *pd.date_range("2026-01-01", periods=100)]
    u = rng.uniform(0, 1, len(days))
    sit = pd.DataFrame({f: 0.0 for f in FEATURES}, index=range(len(days)))
    sit[FEATURES[0]] = u
    sit["symbol"], sit["day_str"] = "X", [d.strftime("%Y-%m-%d") for d in days]
    trades = add_context(pd.DataFrame({
        "strategy": "plan", "symbol": "X", "signal_day": sit["day_str"], "exit_day": sit["day_str"],
        "r_net": u, "twin_r": 0.0, "reason": "time", "bars": 1, "overlap": 0}))
    picked = memory_filter(trades, sit, "plan")
    # the feature is the outcome: the memory keeps the best holdout days only
    assert set(picked["period"]) == {"holdout"} and set(picked["strategy"]) == {"plan+memory"}
    assert 10 <= len(picked) <= 35 and picked["excess_r"].min() > 0.6


def test_every_look_at_the_holdout_is_logged(tmp_path):
    from core.sandbox.evaluate import holdout_report
    from core.sandbox.store import LearningStore
    store = LearningStore(tmp_path / "learning.db")
    t = add_context(pd.DataFrame({"strategy": ["plan"], "symbol": ["X"], "signal_day": ["2026-02-01"],
                                  "exit_day": ["2026-02-05"], "r_net": [1.0], "twin_r": [0.5],
                                  "reason": ["target"], "bars": [4], "overlap": [0]}))
    first = holdout_report(t, ["plan"], store, "r1")
    second = holdout_report(t, ["plan"], store, "r1")
    assert first["holdout_opened_total"].iloc[0] == 1 and second["holdout_opened_total"].iloc[0] == 2
    assert first["of10_target"].iloc[0] == 10 and store.holdout_opens() == 2
