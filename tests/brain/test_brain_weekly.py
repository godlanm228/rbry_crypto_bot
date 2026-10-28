"""The brain's week: only unseen days are graded, a failing check keeps the champion."""

import numpy as np
import pandas as pd

from core.brain import model
from core.brain import weekly as wk


def _rows(days=500, coins=40, seed=2, signal=0.02):
    rng = np.random.default_rng(seed)
    day = np.repeat(pd.date_range("2024-01-01", periods=days, freq="D"), coins)
    f = rng.normal(size=len(day))
    excess = signal * f + rng.normal(0, 0.05, len(day))
    rows = pd.DataFrame({"history": np.tile([f"C{i}" for i in range(coins)], days), "base": "X", "day": day,
                         "f": f, "excess_14d": excess, "hold_ret": excess, "r_net": excess * 10,
                         "turnover_30d": 5e6})
    rows.loc[rows["day"] > rows["day"].max() - pd.Timedelta(days=14), ["excess_14d", "hold_ret", "r_net"]] = np.nan
    return rows


def test_the_champion_is_graded_only_on_days_it_never_learned_from():
    rows = _rows()
    champion = model.train(wk.graded(rows[rows["day"] <= "2024-12-31"]), ("f",))
    added = wk.grade_champion(champion, rows, pd.DataFrame(columns=wk.RECORD_COLUMNS), "m1-test")
    assert added["day"].min() > pd.Timestamp(champion.trained_until)
    assert added["excess_14d"].notna().all() and set(added["model"]) == {"m1-test"}
    again = wk.grade_champion(champion, rows, added, "m1-test")
    assert again.empty


def test_a_working_recipe_replaces_the_champion(monkeypatch):
    monkeypatch.setattr(wk, "RANKING_FEATURES", ("f",))
    rows = _rows()
    champion = model.train(wk.graded(rows[rows["day"] <= "2024-10-31"]), ("f",))
    record = wk.grade_champion(champion, rows[rows["day"] <= "2025-01-31"], pd.DataFrame(columns=wk.RECORD_COLUMNS),
                               "old")
    result = wk.week(rows, champion, record, "old", "new", features=("f",))
    assert result.promoted and result.champion.meta["model_id"] == "new"
    assert result.probe["spread_usd"] > 0 and result.agreement > 0.9
    assert "beat_btc" in result.champion.calibration and len(result.record) > len(record)


def test_a_recipe_that_stopped_working_keeps_the_champion(monkeypatch):
    rows = _rows()
    champion = model.train(wk.graded(rows[rows["day"] <= "2024-10-31"]), ("f",))
    monkeypatch.setattr(wk, "probe", lambda rows, features: {"from": "x", "to": "y", "days": 50,
                                                             "spread_usd": -1.5, "t": -2.5})
    result = wk.week(rows, champion, pd.DataFrame(columns=wk.RECORD_COLUMNS), "old", "new", features=("f",))
    assert not result.promoted and result.champion is champion and result.champion.meta["model_id"] == "old"
    assert any("не отличил" in n for n in result.notes)


def test_the_audit_speaks_plainly():
    rows = _rows(days=120)
    record = rows[rows["excess_14d"].notna()].copy()
    record["place"] = record.groupby("day")["f"].rank(pct=True)
    text = wk.audit_text(record, pd.Period("2024-03", "M"))
    assert "Аудит мозга за 03.2024" in text and "лучшие 10% по мозгу" in text and "месяц назад" in text
    assert "нет закрытых дней" in wk.audit_text(record, pd.Period("2030-01", "M")).lower()
