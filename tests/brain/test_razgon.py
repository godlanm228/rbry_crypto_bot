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


# the second engine: free money in BTC by its trend

DAY = 24 * HOUR
COST = rz.FEE_PCT / 100.0 + rz.BTC_SLIP


def _closed(rows):
    """Finished trades of a version: (coin, signal, turnover, net, hours held)."""
    return pd.DataFrame([(h, int(ts), float(turn), 2021, 0.0, float(net), int(held), "time", 0.0, 1.0)
                         for h, ts, turn, net, held in rows],
                        columns=["history", "signal_ts", "turnover", "year", "slip", "net", "held", "reason", "worst", "lev"])


def test_btc_holds_the_free_money_from_the_day_after_it_closes_above_its_average():
    closes = [100.0] * (10 * 24) + [100.0] * 23 + [110.0] + [110.0] * 48      # day 11 closes above the 5-day average
    plan = rz.idle_plan(_prices(closes), 5)
    assert not plan["on"][:11 * 24].any() and plan["on"][11 * 24:].all()      # decided at the close, held from 00:00
    late = rz.idle_plan(_prices(closes), 5, delay=1)
    assert not late["on"][11 * 24] and late["on"][11 * 24 + 1:].all()
    assert not rz.idle_plan(_prices([100.0] * 400), 5)["on"].any()            # never above: the money stays in dollars


def test_a_plan_that_is_never_on_books_exactly_as_the_portfolio():
    rng = np.random.default_rng(3)
    rows = [(f"C{rng.integers(6)}", T0 + int(rng.integers(0, 600)) * HOUR, rng.random(), rng.normal(0.01, 0.08),
             int(rng.integers(1, 72))) for _ in range(120)]
    t = _closed(rows)
    plan = rz.idle_plan(_prices([100.0] * 900), 5)
    mine = rz.book(t, T0, T0 + 700 * HOUR, rz.V1, plan)
    theirs = ex.portfolio(t, T0, T0 + 700 * HOUR, share=rz.V1.share, max_open=rz.V1.max_open)
    assert mine["trades"] == theirs["trades"] > 20
    assert mine["end_usd"] == pytest.approx(theirs["end_usd"], rel=1e-12)
    assert mine["drawdown_pct"] == pytest.approx(theirs["drawdown_pct"], rel=1e-12)
    assert mine["btc_pct"] == 0.0


def test_idle_money_in_btc_earns_its_rise_and_carries_its_fall():
    up = list(np.linspace(100.0, 160.0, 20 * 24))                 # 20 days up, a fall of a quarter, back up
    closes = up + [120.0] * 24 + list(np.linspace(120.0, 180.0, 10 * 24))
    prices = _prices(closes)
    plan = rz.idle_plan(prices, 3)
    start = T0 + 5 * DAY                                          # the plan is on by then
    assert plan["on"][5 * 24]
    b = rz.book(_closed([]), start, T0 + len(closes) * HOUR, rz.V1, plan)
    bought = prices[5 * 24, 1]
    # the fall day closes under the 3-day average: out at the next day's open (120), back in once a close is above it
    flips = [h for h in range(5 * 24 + 1, len(closes)) if plan["on"][h] != plan["on"][h - 1]]
    assert len(flips) == 2 and flips[0] == 21 * 24 and flips[1] % 24 == 0
    out_at, back_at = prices[flips[0], 1], prices[flips[1], 1]
    want = 100 * (1 - COST) * out_at / bought * (1 - COST) * (1 - COST) * closes[-1] / back_at
    assert b["end_usd"] == pytest.approx(want, rel=1e-9)
    assert b["drawdown_pct"] == pytest.approx(100 * (120.0 / 160.0 * (1 - COST) - 1), abs=0.3)   # the fall is in the book
    assert 85 < b["btc_pct"] < 100 and b["trades"] == 0
    assert rz.book(_closed([]), start, T0 + len(closes) * HOUR, rz.V1, plan, cost_mult=2.0)["end_usd"] < b["end_usd"]


def test_a_panic_trade_is_paid_for_with_btc_and_its_proceeds_go_back():
    prices = _prices([100.0] * 24 + [101.0] * (30 * 24))          # day 1 closes above its 1-day... a flat BTC after
    plan = rz.idle_plan(prices, 1)
    plan = {**plan, "on": np.ones(len(plan["on"]), dtype=bool)}   # free money always in BTC, at a flat 101
    start = T0 + 2 * DAY
    t = _closed([("A", start + 48 * HOUR, 5.0, 0.10, 24)])
    b = rz.book(t, start, T0 + 20 * DAY, rz.V1, plan)
    equity = 100 * (1 - COST)                                     # after the first buy of BTC
    size = rz.V1.share * equity
    want = equity - size / (1 - COST) + size * 1.10 * (1 - COST)  # sold for the trade, bought back with its proceeds
    assert b["trades"] == 1 and b["end_usd"] == pytest.approx(want, rel=1e-9)
    flat = rz.book(t, start, T0 + 20 * DAY, rz.V1, rz.idle_plan(_prices([100.0] * 900), 5))
    assert flat["end_usd"] == pytest.approx(100 + rz.V1.share * 100 * 0.10)   # the same trade from dollars


def test_a_version_with_the_second_engine_is_examined_with_it():
    closes = list(np.linspace(100.0, 200.0, 400 * 24))
    market = _Market({"BTC": _prices(closes, start=int(pd.Timestamp("2020-12-01").value // 10**6))})
    plan, late = rz.plans(rz.Version(idle_sma=5), market)
    assert plan["on"].sum() > late["on"].sum() - 2 and rz.plans(rz.V1, market) == (None, None)
    newest = int(market.btc()[-1, 0])
    with_btc = rz.books(_closed([]), rz.Version(idle_sma=5), newest, plan)
    assert with_btc["years"]["2021"]["growth_pct"] > 30 and with_btc["years"]["2021"]["btc_pct"] > 90
    assert rz.books(_closed([]), rz.V1, newest)["years"]["2021"]["growth_pct"] == 0.0
    assert "в BTC, пока он выше своей 5-дневной средней" in rz.Version(idle_sma=5).words()
    assert "BTC" not in rz.V1.words()


def test_a_part_of_the_free_money_in_btc_pays_a_trade_from_the_dollars_first():
    plan = rz.idle_plan(_prices([100.0] * 24 + [101.0] * (30 * 24)), 1)
    plan = {**plan, "on": np.ones(len(plan["on"]), dtype=bool)}
    start = T0 + 2 * DAY
    half = rz.Version(idle_sma=50, idle_share=0.5)
    t = _closed([("A", start + 48 * HOUR, 5.0, 0.10, 24)])
    b = rz.book(t, start, T0 + 20 * DAY, half, plan)
    btc = 50 * (1 - COST)                                         # half of $100 bought, the other half waits
    size = half.share * (50 + btc)                                # paid from the dollars: no BTC is sold
    cash = 50 - size + size * 1.10
    refill = 0.5 * (cash + btc) - btc                             # proceeds top BTC up to half of the free money
    assert b["end_usd"] == pytest.approx(cash - refill + btc + refill * (1 - COST), rel=1e-9)
    assert 45 < b["btc_pct"] < 55
    full = rz.book(t, start, T0 + 20 * DAY, rz.Version(idle_sma=50), plan)
    assert b["end_usd"] > full["end_usd"]                         # at a flat BTC the part pays less in costs
    assert "50% свободных денег — в BTC" in half.words()
