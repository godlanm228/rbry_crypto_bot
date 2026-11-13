"""«Разгон v2»: a version's trade with and without leverage, its picks, the rule of a step, the ladder."""

import numpy as np
import pandas as pd
import pytest

from core.brain import exits as ex
from core.brain import razgon as rz
from core.brain.journal import Journal

HOUR = 3_600_000
T0 = int(pd.Timestamp("2021-03-01").value // 10**6)


def _prices(closes, start=T0, wick=0.0):
    """Hourly rows of ts, open, high, low, close: each hour opens at the close before it."""
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    ts = start + np.arange(len(c), dtype=np.int64) * HOUR
    return np.column_stack([ts, o, np.maximum(o, c) * (1 + wick), np.minimum(o, c) * (1 - wick), c])


class _Market:
    def __init__(self, prices: dict, funding: dict = None):
        self._prices, self._funding = prices, funding or {}
        self.base_of = {name: name for name in prices}

    def prices(self, name):
        return self._prices[name]

    def funding(self, name):
        return self._funding.get(name)

    def btc(self):
        return self._prices["BTC"]


def _walk(closes, v, levered, funding=None, slip=0.0):
    p = _prices(closes)
    return rz.walk(p, funding, T0, v, 0.0, slip, int(p[-1, 0]), levered)


def test_the_first_version_is_the_paper_books_rule():
    rule = rz.V1.rule()
    paper = ex.BY_RULE["H72-HALF5"]
    assert (rule.hours, rule.half_pct, rule.stop, rule.stop_pct, rule.take, rule.keep) == (
        paper.hours, paper.half_pct, paper.stop, paper.stop_pct, paper.take, paper.keep)
    assert (rz.V1.picks, rz.V1.share, rz.V1.max_open, rz.V1.leverage) == (ex.PICKS, ex.SHARE, ex.MAX_OPEN, 1.0)


def test_leverage_multiplies_the_result_and_costs_of_a_trade():
    flat_then_up = [100.0] * 2 + [103.0] * 80                  # +3% and held to the end: no half, closed by time
    spot = _walk(flat_then_up, rz.V1, False)
    two = _walk(flat_then_up, rz.Version(leverage=2.0), True)
    assert spot["reason"] == two["reason"] == "time" and spot["lev"] == 1.0 and two["lev"] == 2.0
    assert spot["net"] == pytest.approx(1.03 * 0.999 / 1.001 - 1)             # 0.1% a side on spot
    assert two["net"] == pytest.approx(2 * (1.03 * 0.9995 / 1.0005 - 1))      # 0.05% a side on the position, twice it


def test_a_fall_of_half_takes_a_2x_margin_and_a_third_takes_a_3x_one():
    crash = [100.0, 100.0, 60.0] + [95.0] * 80                 # -40% and back
    assert _walk(crash, rz.Version(leverage=2.0), True)["reason"] == "time"            # 2x lives through -40%
    three = _walk(crash, rz.Version(leverage=3.0), True)
    assert (three["reason"], three["net"]) == ("liquidated", -1.0)                     # 3x is gone at -31.3%
    gone = _walk([100.0, 100.0, 45.0] + [200.0] * 80, rz.Version(leverage=2.0), True)
    assert (gone["reason"], gone["net"]) == ("liquidated", -1.0)                       # and never sees the recovery
    assert _walk([100.0, 100.0, 45.0] + [200.0] * 80, rz.V1, False)["net"] > 0         # spot does
    assert rz.liquidation_fall(2.0) == pytest.approx(0.48) and rz.liquidation_fall(3.0) == pytest.approx(1 / 3 - 0.02)


def test_a_levered_long_pays_the_funding_of_its_hours_on_the_whole_position():
    closes = [100.0] * 82
    funding = np.array([[T0 + 8 * HOUR, 0.001], [T0 + 16 * HOUR, 0.001], [T0 + 500 * HOUR, 0.001]])
    free = _walk(closes, rz.Version(leverage=2.0), True)
    paid = _walk(closes, rz.Version(leverage=2.0), True, funding)
    assert free["net"] - paid["net"] == pytest.approx(2 * 0.002)       # two payments inside 72 hours, times the leverage


def test_the_rest_trails_its_best_gain_instead_of_closing_after_three_days():
    run = [100.0, 100.0, 106.0, 120.0, 130.0, 130.0, 110.0] + [108.0] * 200
    v1 = _walk(run, rz.V1, False)
    hold = _walk(run, rz.Version(hours=168, keep=0.5), False)
    assert v1["reason"] == "time" and v1["held"] == 72
    assert hold["reason"] == "keep" and hold["held"] < 10      # half the best gain (+30% -> +15%) is kept: out at 115
    assert hold["net"] > v1["net"]


def _table(rows):
    return pd.DataFrame(rows, columns=["history", "signal_ts", "turnover", "moment_size", "perp", "delta", "slip",
                                       "year"])


def test_a_version_takes_the_most_liquid_coins_of_a_mass_moment_only():
    t = _table([(f"C{i}", T0, 10.0 * i, 6, True, 0.0, 0.0, 2021) for i in range(1, 7)]
               + [(f"L{i}", T0 + HOUR, 5.0 * i, 3, True, 0.0, 0.0, 2021) for i in range(1, 4)])     # 3 coins: no mass
    assert list(rz.picked(t, rz.V1)["history"]) == ["C4", "C5", "C6"]
    assert sorted(rz.picked(t, rz.Version(picks=5))["history"]) == ["C2", "C3", "C4", "C5", "C6"]


def test_a_coin_without_a_perpetual_is_bought_on_spot_under_a_levered_version():
    closes = [100.0] * 2 + [103.0] * 80
    market = _Market({"A": _prices(closes), "B": _prices(closes), "BTC": _prices(closes)},
                     {"A": np.array([[T0 - HOUR, 0.0]])})
    t = _table([("A", T0, 9.0, 4, True, 0.0, 0.0, 2021), ("B", T0, 8.0, 4, False, 0.0, 0.0, 2021)])
    tr = rz.trades(t, rz.Version(leverage=2.0), market, int(market.btc()[-1, 0])).set_index("history")
    assert tr.loc["A", "lev"] == 2.0 and tr.loc["B", "lev"] == 1.0
    assert tr.loc["A", "net"] == pytest.approx(2 * tr.loc["B", "net"], rel=0.05)


def _exam(yearly, y2025, y2026, drawdown=-30.0, bent_ok=True, holds=True, liquidations=0):
    year = lambda g: {"growth_pct": g, "drawdown_pct": drawdown, "n": 50}
    years = {str(y): year(yearly) for y in range(2018, 2025)} | {"2025": year(y2025), "2026": year(y2026)}
    bent = {name: {"holds": bent_ok} for name in ("late", "costs", "missed", "no_best")}
    return {"learn_yearly_pct": yearly, "holds": holds, "bent": bent, "liquidations": liquidations, "trades": 100, **years}


def test_a_step_becomes_a_version_only_when_it_is_faster_not_lower_later_and_no_more_fragile():
    current = _exam(30.0, 15.0, 16.0)
    ok, why = rz.decide(_exam(45.0, 20.0, 16.0), current)
    assert ok and "быстрее" in why[0] and any("не хуже текущей" in w for w in why)
    assert not rz.decide(_exam(25.0, 20.0, 20.0), current)[0]                       # slower over 2018-2024
    ok, why = rz.decide(_exam(45.0, 10.0, 20.0), current)                           # 2025 ends lower
    assert not ok and any("2025" in w and "ниже" in w for w in why)
    ok, why = rz.decide(_exam(60.0, 30.0, 30.0, drawdown=-62.0), current)           # past the owner's 50%
    assert not ok and any("глубже допустимых -50%" in w for w in why)
    ok, why = rz.decide(_exam(45.0, 20.0, 20.0, bent_ok=False), current)            # breaks where the current holds
    assert not ok and any("стала хрупче" in w for w in why)
    # ... but a bent history the current version does not hold either is not asked of the step
    assert rz.decide(_exam(45.0, 20.0, 20.0, bent_ok=False), _exam(30.0, 15.0, 16.0, bent_ok=False))[0]


def _rising_market():
    """Six coins in a mass moment every 10 days of 2018-2026: each rises 3% after its signal."""
    start = int(pd.Timestamp("2018-01-01").value // 10**6)
    hours = int((pd.Timestamp("2026-06-01") - pd.Timestamp("2018-01-01")).total_seconds() // 3600)
    closes = np.full(hours, 100.0)
    signals = [int(s) for s in np.arange(240, hours - 400, 240, dtype=np.int64)]
    for s in signals:
        closes[s + 1:s + 100] = 103.0
    prices = _prices(closes, start=start)
    market = _Market({f"C{i}": prices for i in range(6)} | {"BTC": prices},
                     {f"C{i}": np.array([[start, 0.0]]) for i in range(6)})
    rows = [(f"C{i}", int(start + s * HOUR), 10.0 + i, 6, True, 0.0, 0.0, int(pd.Timestamp(start + s * HOUR, unit="ms").year))
            for s in signals for i in range(6)]
    return _table(rows), market, int(prices[-1, 0])


def test_the_ladder_examines_each_step_once_against_the_version_current_at_its_turn(tmp_path):
    t, market, newest = _rising_market()
    journal = Journal(tmp_path / "journal.jsonl")
    steps = (rz.Step("lev2", "плечо 2x", "почему", {"leverage": 2.0}), rz.Step("slow", "1 монета", "почему", {"picks": 1}))
    ladder = rz.climb(rz.load(tmp_path / "none.json"), t, market, newest, steps, journal, "events-test", "2026-11-13")
    assert [v["name"] for v in ladder["versions"]] == ["v1", "v2"] and ladder["current"] == "v2"
    lev, slow = ladder["steps"]
    assert lev["accepted"] and lev["on"] == "v1" and lev["became"] == "v2"          # every trade gains: 2x is faster
    assert not slow["accepted"] and slow["on"] == "v2"                              # judged against v2, not v1
    assert ladder["versions"][1]["exam"]["version"]["leverage"] == 2.0
    ideas = [e for e in journal.entries() if e["event"] == "idea"]
    results = [e for e in journal.entries() if e["event"] == "result"]
    assert [e["id"] for e in ideas] == ["RZ2-lev2", "RZ2-slow"] and len(results) == 6
    first_result = next(i for i, e in enumerate(journal.entries()) if e["event"] == "result")
    assert journal.entries()[first_result - 1]["id"] == "RZ2-lev2"                  # the idea is written before its result
    rz.save(ladder, tmp_path / "versions.json")
    again = rz.climb(rz.load(tmp_path / "versions.json"), t, market, newest, steps, journal, "events-test")
    assert len(again["steps"]) == 2 and len(again["versions"]) == 2                 # nothing is decided twice
    assert rz.current_version(again) == rz.Version(leverage=2.0)


def test_leverage_only_while_btc_is_not_deep_under_its_average():
    closes = [100.0] * 2 + [103.0] * 80
    market = _Market({"A": _prices(closes), "B": _prices(closes), "BTC": _prices(closes)},
                     {name: np.array([[T0 - HOUR, 0.0]]) for name in ("A", "B")})
    t = _table([("A", T0, 9.0, 4, True, 0.0, 0.0, 2021), ("B", T0, 8.0, 4, True, 0.0, 0.0, 2021)])
    t["btc_sma"] = [-0.03, -0.15]                      # A's moment: BTC 3% under its average; B's: 15% under
    v = rz.Version(leverage=2.0, lever_above=-0.08)
    tr = rz.trades(t, v, market, int(market.btc()[-1, 0])).set_index("history")
    assert tr.loc["A", "lev"] == 2.0 and tr.loc["B", "lev"] == 1.0
    assert "только пока BTC не ниже -8%" in v.words() and "только пока" not in rz.Version(leverage=2.0).words()
