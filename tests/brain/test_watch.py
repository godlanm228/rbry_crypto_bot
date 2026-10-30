"""The mass-panic watcher: deciding a moment, following a paper trade, the book, the messages."""

import numpy as np
import pandas as pd

from core.brain import watch

HOUR, DAY = 3_600_000, 86_400_000
T0 = int(pd.Timestamp("2026-11-02 13:00").value // 10**6)        # a moment: the hour after the panic


def _seen(n, nervous=True):
    panic = [{"base": f"C{i}", "symbol": f"C{i}USDT", "events": ["DUMP"], "turnover": 1e7 * (i + 1), "r1": -0.05,
              "r24": -0.1, "close": 10.0 + i, "sigma_day": 0.05} for i in range(n)]
    return watch.Seen(T0 - HOUR, nervous, -0.02, 50, panic)


def test_a_mass_moment_needs_four_coins_and_a_nervous_market_and_takes_the_three_most_liquid():
    assert watch.decide(_seen(3), T0) is None
    assert watch.decide(_seen(6, nervous=False), T0) is None
    d = watch.decide(_seen(6), T0 + 60_000)
    assert d["moment"] == T0 and d["size"] == 6
    assert [p["base"] for p in d["picks"]] == ["C5", "C4", "C3"]


def _prices(path, start=T0):
    c = np.asarray(path, float)
    o = np.r_[c[0], c[:-1]]
    ts = start + np.arange(len(c), dtype=np.int64) * HOUR
    return np.column_stack([ts, o, np.maximum(o, c), np.minimum(o, c), c])


def test_a_trade_still_open_is_marked_and_a_sold_half_is_dated():
    prices = _prices([100, 106, 104, 103])
    f = watch.follow(prices, T0, 100.0, 0.05, watch.VARIANTS["nostop"], T0 + 3 * HOUR, 0.0)
    assert f["status"] == "open" and f["half_ts"] == T0 + 2 * HOUR
    assert abs(f["net"] - (0.5 * 105 * 0.999 + 0.5 * 103 * 0.999) / (100 * 1.001) + 1) < 1e-9


def test_the_rest_closes_at_the_entry_after_a_half():
    prices = _prices([100, 106, 99, 103])
    f = watch.follow(prices, T0, 100.0, 0.05, watch.VARIANTS["nostop"], T0 + 3 * HOUR, 0.0)
    assert f["status"] == "breakeven" and f["exit_ts"] == T0 + 3 * HOUR


def test_three_days_close_it_and_the_disaster_stop_only_in_its_variant():
    path = [100, 98] + [97] * 80
    f = watch.follow(_prices(path), T0, 100.0, 0.05, watch.VARIANTS["nostop"], T0 + 90 * HOUR, 0.0)
    assert f["status"] == "time" and f["exit_ts"] == T0 + 72 * HOUR
    crash = _prices([100, 60, 50, 50])
    assert watch.follow(crash, T0, 100.0, 0.05, watch.VARIANTS["stop30"], T0 + 3 * HOUR, 0.0)["status"] == "stop"
    assert watch.follow(crash, T0, 100.0, 0.05, watch.VARIANTS["nostop"], T0 + 3 * HOUR, 0.0)["status"] == "open"


def test_the_book_takes_a_fifth_of_its_equity_a_pick_and_at_most_five():
    picks = [{"base": f"C{i}", "symbol": f"C{i}USDT", "turnover": 1e8, "close": 100.0, "sigma_day": 0.05}
             for i in range(3)]
    decisions = [{"moment": T0, "picks": picks}, {"moment": T0 + HOUR, "picks": [dict(p, base=p["base"] + "x")
                                                                              for p in picks]}]
    flat = _prices([100] * 10, start=T0)
    b = watch.book(decisions, lambda s, f: flat[flat[:, 0] >= f], {}, "nostop", T0 + 5 * HOUR)
    assert len(b.positions) == 5 and b.skipped == 1
    assert abs(b.positions[0].size - 20.0) < 1e-9
    assert all(p.status == "open" for p in b.positions)


def test_news_tells_each_half_and_close_once():
    prices = _prices([100, 106, 99, 103])
    picks = [{"base": "SOL", "symbol": "SOLUSDT", "turnover": 1e8, "close": 100.0, "sigma_day": 0.05}]
    books = {v: watch.book([{"moment": T0, "picks": picks}], lambda s, f: prices, {}, v, T0 + 3 * HOUR)
             for v in watch.VARIANTS}
    lines, keys = watch.news(books, set())
    assert sum(x.startswith("✂️") for x in lines) == 1 and sum(x.startswith("🏁") for x in lines) == 2
    assert watch.news(books, keys) == ([], set())


def test_quiet_at_night_in_berlin():
    assert watch.quiet(int(pd.Timestamp("2026-11-02 02:30").value // 10**6))       # 03:30 Berlin
    assert not watch.quiet(int(pd.Timestamp("2026-11-02 08:30").value // 10**6))   # 09:30 Berlin


def test_the_card_says_what_to_do_in_prices():
    text = watch.card(watch.decide(_seen(5), T0))
    assert "C4 ≈ $14.000" in text and "$14.700 (+5%)" in text and "$9.800 (−30%)" in text
    assert "бумага" in text


def test_turnover_is_the_median_of_finished_days():
    ts = T0 - 40 * DAY + np.arange(40 * 24, dtype=np.int64) * HOUR
    frame = pd.DataFrame({"ts": ts, "quote_volume": np.r_[np.full(30 * 24, 1e6), np.full(10 * 24, 2e6)]})
    assert abs(watch.turnover(frame) - 24e6) < 1e-6                              # the last 30 days, 20 at 1e6/h


def _strategy(**genes):
    from core.brain import factory as fa
    g = fa.normal(fa.Genome(**{"events": ("DUMP",), "hours": 72, "half_pct": 0.05, **genes}))
    return {"id": g.id, "goal": "razgon", "genome": g.to_dict(), "title": fa.describe(g),
            "years": {"2025": {"growth_pct": 10.0, "drawdown_pct": -12.0}, "2026": {"growth_pct": 5.0,
                                                                                  "drawdown_pct": -8.0}}}


def test_a_factory_candidate_takes_what_its_genome_takes_this_hour():
    seen = _seen(6)
    for c in seen.coins:
        c.update(big=c["turnover"] >= 5e7, perp=True, r4=-0.06)
    seen.btc_up = False
    d = watch.strategy_decide(seen, _strategy(mass=4, pick="liquid", k=2, nervous=True), T0 + 60_000)
    assert [p["base"] for p in d["picks"]] == ["C5", "C4"] and d["size"] == 6 and d["strategy"]
    assert watch.strategy_decide(seen, _strategy(mass=4, btc=True), T0) is None              # BTC is falling
    assert watch.strategy_decide(seen, _strategy(events=("SPIKE",)), T0) is None              # no such event


def test_a_paper_short_gains_as_the_price_falls_and_is_gone_when_it_doubles():
    rule = watch.fa.Genome(events=("SPIKE",), side=-1, hours=3, half_pct=0.05).rule()
    f = watch.follow(_prices([100, 94, 101, 97]), T0, 100.0, 0.05, rule, T0 + 3 * HOUR, 0.0, side=-1)
    assert f["status"] == "breakeven" and f["half_ts"] == T0 + 2 * HOUR and f["net"] > 0
    up = watch.follow(_prices([100, 150, 210]), T0, 100.0, 0.05, rule, T0 + 2 * HOUR, 0.0, side=-1)
    assert up["status"] == "liquidated" and up["net"] == -1.0


def test_a_factory_card_says_the_side_and_the_prices():
    s = _strategy(mass=4, pick="liquid", k=2, nervous=True, stop=("pct", 0.3))
    seen = _seen(6)
    for c in seen.coins:
        c.update(big=True, perp=True)
    text = watch.strategy_card(watch.strategy_decide(seen, s, T0), s)
    assert "купить (спот)" in text and "C5 ≈ $15.000" in text and "$15.750 (+5%)" in text and "стоп $10.500" in text
