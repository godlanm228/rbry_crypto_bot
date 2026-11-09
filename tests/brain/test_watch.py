"""The mass-panic watcher: deciding a moment, following a paper trade, the book, the messages."""

import numpy as np
import pandas as pd

from core.brain import exits as ex
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
    assert watch.follow(crash, T0, 100.0, 0.05, ex.BY_RULE["H72-HALF5-S30"], T0 + 3 * HOUR, 0.0)["status"] == "stop"
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
    assert sum(x.startswith("✂️") for x in lines) == 1 and sum(x.startswith("🏁") for x in lines) == 1   # one book
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


def test_the_account_takes_the_panic_first_and_each_pick_with_its_own_side_and_exit():
    from core.brain import selector as se
    pick = lambda b, **kw: {"base": b, "symbol": f"{b}USDT", "turnover": 1e8, "close": 100.0, "sigma_day": 0.05, **kw}
    panic = [{"moment": T0, "picks": [pick("C0"), pick("C1"), pick("C2")]}, {"moment": T0 - DAY, "picks": [pick("OLD")]}]
    model = [{"moment": T0, "picks": [pick("C2", action="L24", forecast=0.05), pick("C9", action="S24", forecast=0.02)]}]
    flat = _prices([100] * 10, start=T0)
    b = watch.account_book([("паника", panic, lambda p: (watch.VARIANTS["nostop"], 1)),
                            ("модель выбора", model, lambda p: se.RULES[p["action"]])],
                           lambda s, f: flat[flat[:, 0] >= f], {}, T0 + 5 * HOUR, start_ms=T0)
    assert [(p.base, p.source, p.side) for p in b.positions] == [("C0", "паника", 1), ("C1", "паника", 1),
                                                                 ("C2", "паника", 1), ("C9", "модель выбора", -1)]
    assert b.positions[3].rule.hours == 24 and abs(b.value - 100) < 1.0          # before the account: ignored
    lines, keys = watch.entry_lines(b, set())
    assert len(lines) == 4 and "шорт" in lines[3] and "половину закрыть на $105.00" in lines[0]
    assert watch.entry_lines(b, keys) == ([], set())


def test_a_day_of_the_account_in_words():
    b = watch.Book(watch.ACCOUNT, 100.0, 100.0, label=watch.ACCOUNT_LABEL)
    text = watch.day_text(b, {watch.ACCOUNT: b}, [{"books": {watch.ACCOUNT: 100.0}}], T0, T0)
    assert "Баланс $100.00" in text and "Сделок пока не было" in text
    assert watch.spark([1, 2, 3]) == "▁▄█"


def test_the_selector_explores_with_three_positions_of_the_account_at_most_and_each_skip_says_why():
    from core.brain import selector as se
    pick = lambda b, **kw: {"base": b, "symbol": f"{b}USDT", "turnover": 1e8, "close": 100.0, "sigma_day": 0.05, **kw}
    model = [{"moment": T0, "picks": [pick(f"M{i}", action="L24", forecast=0.05 - i / 1000) for i in range(4)]},
             {"moment": T0 + HOUR, "picks": [pick("M0", action="L24", forecast=0.05)]}]
    flat = _prices([100] * 10, start=T0)
    b = watch.account_book([("модель выбора", model, lambda p: se.RULES[p["action"]])],
                           lambda s, f: flat[flat[:, 0] >= f], {}, T0 + 5 * HOUR, start_ms=T0)
    assert [p.base for p in b.positions] == ["M0", "M1", "M2"] and b.skipped == 2
    assert [(s["base"], s["reason"]) for s in b.skips] == [("M3", "cap"), ("M0", "held")]
    assert b.skips[0]["holding"] == ["M0 (модель выбора)", "M1 (модель выбора)", "M2 (модель выбора)"]


def test_a_torn_last_line_is_left_out_and_cut_before_the_next_append(tmp_path):
    path = tmp_path / "decisions.jsonl"
    watch.append(path, {"moment": 1})
    with path.open("ab") as fh:                         # a run killed in the middle of its line
        fh.write(b'{"moment": 2, "pi')
    assert watch.load(path) == [{"moment": 1}]
    watch.append(path, {"moment": 3})
    assert watch.load(path) == [{"moment": 1}, {"moment": 3}]
    assert path.read_bytes().count(b"\n") == 2


def test_a_broken_line_in_the_middle_still_raises(tmp_path):
    path = tmp_path / "decisions.jsonl"
    path.write_text('{"moment": 1}\nnot json\n{"moment": 3}\n', encoding="utf-8")
    try:
        watch.load(path)
    except ValueError:
        return
    raise AssertionError("a damaged journal must not pass silently")


def test_a_file_that_is_only_a_torn_line_starts_afresh(tmp_path):
    path = tmp_path / "seen.jsonl"
    path.write_bytes(b'{"hour": 1')
    assert watch.load(path) == []
    watch.append(path, {"hour": 2})
    assert watch.load(path) == [{"hour": 2}]


def test_a_coin_market_db_lacks_gets_its_hours_from_the_api_without_a_concat_warning():
    import warnings
    cols = ["ts", "open", "high", "low", "close"]
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
    fresh = pd.DataFrame([[T0 + i * HOUR, 1.0, 1.0, 1.0, 1.0] for i in range(-2, 2)], columns=cols)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        got = watch.with_tail(empty, "NEWUSDT", T0, 1, lambda s, since: fresh)
        same = watch.with_tail(got, "NEWUSDT", T0, 1, lambda s, since: fresh.iloc[:0])
    assert list(got["ts"]) == [T0 - 2 * HOUR, T0 - HOUR, T0] and len(same) == 3


def test_no_book_trades_a_tokenized_share():
    # 05.11 15:01 UTC: four shares fell at the US open and made a "mass panic"; the decision stays as written
    d = {"moment": 1, "picks": [{"base": "SOXLB"}, {"base": "CRCLB"}, {"base": "NVDAB"}]}
    other = {"moment": 2, "picks": [{"base": "TRUMP"}, {"base": "SOXLB"}]}
    assert watch.tradable([d, other], {"SOXLB", "CRCLB", "NVDAB"}) == [{"moment": 1, "picks": []},
                                                                       {"moment": 2, "picks": [{"base": "TRUMP"}]}]
    assert d["picks"][0] == {"base": "SOXLB"}                       # the written decision is untouched


# what a decision saw, the trade journal, the panic's pulse (Vlad, 05.11: a trade without the market around it
# and the reasons for it can neither be audited nor learned from)

def _btc_frame(closes, end=T0 - HOUR):
    c = np.asarray(closes, float)
    ts = end - np.arange(len(c))[::-1].astype(np.int64) * HOUR
    return pd.DataFrame({"ts": ts, "close": c})


def test_btc_numbers_at_the_hours_close():
    closes = np.r_[np.full(1300, 100.0), 101.0]
    n = watch.btc_numbers(_btc_frame(closes), T0 - HOUR)
    assert n["close"] == 101.0 and abs(n["r1"] - 0.01) < 1e-9 and abs(n["r24"] - 0.01) < 1e-9
    assert abs(n["vs_sma50"] - (101 / ((1199 * 100 + 101) / 1200) - 1)) < 1e-6
    assert watch.btc_numbers(_btc_frame(closes), T0) == {}          # the hour asked for is not there: nothing


def test_clean_keeps_json_strict():
    out = watch.clean({"a": float("nan"), "b": np.float32(0.1234567), "c": [np.int64(3), np.bool_(True)],
                       "d": {"e": float("inf")}, "f": "x"})
    assert out == {"a": None, "b": 0.123457, "c": [3, True], "d": {"e": None}, "f": "x"}
    import json
    json.loads(json.dumps(out, allow_nan=False))


def test_a_decision_carries_the_market_and_each_coins_numbers_as_seen():
    seen = _seen(5)
    seen.regime, seen.btc = "down|nervous", {"close": 77000.0, "r24": -0.031}
    seen.states = {"breadth": 0.2, "breadth_state": "narrow"}
    seen.context = {"fear_greed": {"day": "2026-11-02", "value": 21.0}, "urgent": [{"story": "Binance"}]}
    for c in seen.coins:
        c.update(r4=-0.06, usual=True, big=False, perp=True)
    d = watch.decide(seen, T0 + 60_000)
    m = d["market"]
    assert m["regime"] == "down|nervous" and m["btc"]["r24"] == -0.031 and m["states"]["breadth_state"] == "narrow"
    assert m["panic"] == ["C0", "C1", "C2", "C3", "C4"] and m["fear_greed"]["value"] == 21.0
    assert m["urgent"] == [{"story": "Binance"}] and m["checked"] == 50
    pick = d["picks"][0]
    assert {k: pick[k] for k in ("base", "symbol", "turnover", "close", "sigma_day")} == \
        {"base": "C4", "symbol": "C4USDT", "turnover": 5e7, "close": 14.0, "sigma_day": 0.05}
    assert pick["coin"]["events"] == ["DUMP"] and pick["coin"]["r4"] == -0.06 and pick["coin"]["perp"] is True


def test_the_journal_has_each_trade_with_its_decision_its_excursions_and_the_skips():
    seen = _seen(4)
    seen.regime = "flat|nervous"
    d = watch.decide(seen, T0 + 60_000)
    path = _prices([100, 104, 97, 103, 103], start=T0)
    btc = _prices([50_000, 50_500, 51_000, 50_000, 50_000], start=T0)
    prices = lambda s, f: (btc if s == "BTCUSDT" else path)[(btc if s == "BTCUSDT" else path)[:, 0] >= f]
    b = watch.book([d, {**d, "moment": T0 + HOUR}], prices, {}, "nostop", T0 + 4 * HOUR)
    out = watch.ledger({"nostop": b}, {watch.VARIANT_RU["nostop"]: [d]}, prices, T0 + 4 * HOUR)
    trades = out["books"]["nostop"]["trades"]
    assert [t["base"] for t in trades] == ["C3", "C2", "C1"]
    t = trades[0]
    assert t["decision"]["market"]["regime"] == "flat|nervous" and t["decision"]["panic"] == ["C0", "C1", "C2", "C3"]
    assert t["pick"]["coin"]["events"] == ["DUMP"] and t["rule"] == "H72-HALF5"
    assert abs(t["held"]["best"] - 0.04) < 1e-9 and abs(t["held"]["worst"] + 0.03) < 1e-9 and t["held"]["hours"] == 5
    assert abs(t["held"]["btc"]) < 1e-9                       # BTC opened at 50 000 and is there again
    skips = out["books"]["nostop"]["skips"]
    assert {(s["base"], s["reason"], s["why"]) for s in skips} == {(b_, "held", "эта монета уже в счёте")
                                                                   for b_ in ("C3", "C2", "C1")}


def test_the_panics_pulse_in_one_line():
    day0 = T0 - 13 * HOUR                                     # 2026-11-02 00:00 UTC
    hours = [{"hour": day0 + i * HOUR, "nervous": i in (13, 14), "panic": []} for i in range(16)]
    hours[13]["panic"], hours[14]["panic"] = ["ZEC", "XPL"], ["SOXLB", "NVDAB", "CRCLB", "KORUB", "TIA"]
    line = watch.pulse(hours, day0, day0 + 16 * HOUR, stocks={"SOXLB", "NVDAB", "CRCLB", "KORUB"})
    assert line.startswith("🌊 Паника за день: рынок нервный 2 ч из 16")
    assert "больше всего монет в панике за час — 2 из нужных 4 (14:00–15:00: ZEC, XPL)" in line
    assert line.endswith("массового момента не было.")
    assert watch.pulse([], day0, day0 + HOUR) == ""
    calm = [{"hour": day0, "nervous": False, "panic": []}]
    assert "ни одной монеты в панике" in watch.pulse(calm, day0, day0 + HOUR, moments=0)


def test_the_day_says_what_was_skipped_and_the_places():
    b = watch.Book(watch.ACCOUNT, 100.0, 100.0, label=watch.ACCOUNT_LABEL)
    b.skips = [{"moment": T0, "base": "XPL", "source": "модель выбора", "reason": "cap", "holding": ["TIA (модель выбора)"]},
               {"moment": T0, "base": "SOL", "source": "паника", "reason": "no_price", "holding": []}]
    places = [("паника", 5, 0.02), ("модель выбора", 3, 0.018)]
    text = watch.day_text(b, {watch.ACCOUNT: b}, [], T0 + 5 * HOUR, T0 - DAY, places, "🌊 Паника за день: …")
    assert "↷ Пропущено: XPL (модель выбора, 14:00) — места этой стратегии заняты: TIA (модель выбора)" in text
    assert "SOL" not in text and "🌊 Паника за день" in text
    b.skips = [{"moment": T0, "base": f"X{i}", "source": "модель выбора", "reason": "full", "holding": []} for i in range(8)]
    text = watch.day_text(b, {watch.ACCOUNT: b}, [], T0 + 5 * HOUR, T0 - DAY, places)
    assert text.count("↷ Пропущено:") == 5 and "↷ … и ещё 3 пропущено" in text
    assert "Места в счёте: паника — до 5, модель выбора — до 3; вместе не больше 5 сделок." in text


def test_tail_reads_the_last_lines_only(tmp_path):
    path = tmp_path / "h.jsonl"
    for i in range(50):
        watch.append(path, {"i": i, "pad": "x" * 100})
    assert [r["i"] for r in watch.tail(path, 3)] == [47, 48, 49]
    assert [r["i"] for r in watch.tail(path, 3, chunk=500)] == [47, 48, 49]   # a chunk cut mid-line
    assert watch.tail(tmp_path / "none.jsonl", 3) == []


def test_each_decision_is_booked_with_the_places_it_had_then():
    # the selector got 3 places on 06.11: picks made before keep the one place they had, no hindsight
    from core.brain import selector as se
    pick = lambda b: {"base": b, "symbol": f"{b}USDT", "turnover": 1e8, "close": 100.0, "sigma_day": 0.05,
                      "action": "L24", "forecast": 0.05}
    model = [{"moment": T0, "picks": [pick("M0"), pick("M1"), pick("M2")]},
             {"moment": T0 + 2 * HOUR, "picks": [pick("N0"), pick("N1"), pick("N2")]}]
    flat = _prices([100] * 10, start=T0)
    history = [{"from": T0 - DAY, "places": {"модель выбора": 1}},
               {"from": T0 + 2 * HOUR, "places": {"модель выбора": 3}}]
    b = watch.account_book([("модель выбора", model, lambda p: se.RULES[p["action"]])],
                           lambda s, f: flat[flat[:, 0] >= f], {}, T0 + 5 * HOUR, start_ms=T0,
                           caps=watch.caps_at(history))
    assert [p.base for p in b.positions] == ["M0", "N0", "N1"]
    assert [(s["base"], s["reason"]) for s in b.skips] == [("M1", "cap"), ("M2", "cap"), ("N2", "cap")]
    assert watch.caps_at([])(T0) == watch.ACCOUNT_CAPS and watch.caps_at(history)(T0 - 2 * DAY) == {"модель выбора": 1}


# the review of 06.11

def test_a_torn_line_longer_than_the_tail_read_is_cut_not_kept(tmp_path):
    path = tmp_path / "d.jsonl"
    watch.append(path, {"i": 0})
    with path.open("ab") as fh:
        fh.write(b'{"i": 1, "pad": "' + b"x" * 200_000)          # a big decision killed mid-write
    watch.append(path, {"i": 2})
    assert watch.load(path) == [{"i": 0}, {"i": 2}]


def test_the_pulse_counts_an_hour_once_and_names_a_dozen_coins_at_most():
    day0 = T0 - 13 * HOUR
    hours = [{"hour": day0, "nervous": True, "panic": [f"C{i}" for i in range(30)]},
             {"hour": day0, "nervous": True, "panic": [f"C{i}" for i in range(30)]},       # a rerun of the hour
             {"hour": day0 + HOUR, "nervous": False, "panic": []}]
    line = watch.pulse(hours, day0, day0 + 2 * HOUR)
    assert "рынок нервный 1 ч из 2" in line and "C11 и ещё 18" in line and "C12" not in line


def test_the_past_keeps_its_own_order_of_sources_and_a_label_out_of_an_entry_has_no_place():
    from core.brain import selector as se
    pick = lambda b, **kw: {"base": b, "symbol": f"{b}USDT", "turnover": 1e8, "close": 100.0, "sigma_day": 0.05, **kw}
    panic = [{"moment": T0, "picks": [pick(f"P{i}") for i in range(3)]}]
    model = [{"moment": T0, "picks": [pick(f"S{i}", action="L24", forecast=0.05) for i in range(3)]}]
    flat = _prices([100] * 10, start=T0)
    sources = [("модель выбора", model, lambda p: se.RULES[p["action"]]),          # today the selector ranks first
               ("паника", panic, lambda p: (watch.VARIANTS["nostop"], 1))]
    history = [{"from": T0 - DAY, "places": {"паника": 5, "модель выбора": 3}, "order": ["паника", "модель выбора"]},
               {"from": T0 + DAY, "places": {"паника": 5, "модель выбора": 3}, "order": ["модель выбора", "паника"]}]
    b = watch.account_book(sources, lambda s, f: flat[flat[:, 0] >= f], {}, T0 + 5 * HOUR, T0,
                           caps=watch.caps_at(history), order=watch.order_at(history))
    assert [p.base for p in b.positions] == ["P0", "P1", "P2", "S0", "S1"]       # the order of T0, not today's
    late = [{"from": T0 - DAY, "places": {"паника": 5}}, {"from": T0 + DAY, "places": {"паника": 5, "фабрика": 1}}]
    assert watch.caps_at(late)(T0)["фабрика"] == 0 and watch.caps_at(late)(T0 + DAY)["фабрика"] == 1


def _journal():
    sel = {"moment": T0, "base": "TRUMP", "side": -1, "source": "модель выбора", "status": "open", "net_pct": 4.1,
           "pnl_usd": 0.8, "entry": 1.276, "size": 20.0, "rule": "S72S2T2", "held": {"best": 0.06, "worst": -0.01},
           "decision": {"market": {"regime": "flat|nervous", "nervous": True, "btc": {"r24": -0.02},
                                   "states": {"breadth_state": "narrow"}, "fear_greed": {"value": 30.0},
                                   "urgent": [{"story": "Binance"}], "panic": ["A", "B"]}},
           "pick": {"coin": {"events": ["DUMP"]}, "action": "S72S2T2", "forecast": 0.012,
                    "why": {"forecasts": {"S72S2T2": 0.012}, "features": {"mood": "nervous", "fell24": -0.1},
                            "base": -0.003, "by": {"mood": 0.006, "fell24": 0.004, "hour": -0.002}}}}
    old = {**sel, "moment": T0 - 9 * DAY, "base": "SOXLB", "status": "time", "exit_ts": T0 - 8 * DAY}
    return {"books": {watch.ACCOUNT: {"trades": [old, sel], "skips": [
                {"moment": T0, "base": "XPL", "source": "модель выбора", "reason": "cap", "holding": ["TIA"]}]},
            "selector": {"trades": [sel]}}}


def test_the_digest_holds_the_window_only_and_stays_within_its_size():
    hours = [{"hour": T0 - HOUR, "nervous": True, "regime": "flat|nervous", "btc": {"r1": -0.01, "r24": -0.02},
              "panic": ["A"], "taken": {"panic": [], "selector": ["TRUMP"]},
              "selector_best": [{"base": "TRUMP", "forecast": 0.012}]}]
    c1 = {"day": "2026-11-01", "btc_up": False, "coins": 80, "market": {"regime": "down|calm"},
          "orders": [{"base": "NIL", "side": -1, "place": 0.01, "why": {"features": {"ret_7d": 0.4}, "base": 0.0,
                                                                         "by": {"ret_7d": -0.2}}}]}
    text = watch.digest(_journal(), hours, c1, T0 - DAY)
    assert "SOXLB" not in text and text.count('"base": "TRUMP"') == 2 and '"same_trade_as_account": true' in text
    assert '"factors": {"up": [["mood", "nervous", 0.006], ["fell24", -0.1, 0.004]], "down": [["hour", null, -0.002]]}' in text
    assert '"why": "места этой стратегии заняты"' in text and "## C1 newest decision" in text
    assert '"down": [["ret_7d", 0.4, -0.2]]' in text
    small = watch.digest(_journal(), hours, c1, T0 - DAY, max_chars=300)
    assert len(small) < 420 and "… cut:" in small


def test_a_selector_version_falling_short_on_paper_hands_over_to_the_one_before_or_stops():
    from core.brain import selector as se
    flat = _prices([100] * 200, start=T0)
    picks = lambda i: [{"base": f"M{i}", "symbol": f"M{i}USDT", "turnover": 1e8, "close": 100.0, "sigma_day": 0.05,
                        "action": "L4", "forecast": 0.03}]
    decisions = [{"moment": T0 + i * 10 * HOUR, "model": "v2", "picks": picks(i)} for i in range(16)]
    book = watch.selector_book(decisions, lambda s, f: flat[flat[:, 0] >= f], {}, T0 + 199 * HOUR)
    record = watch.selector_record(decisions, book)
    assert record["v2"]["n"] == 16 and record["v2"]["mean_net"] < 0 and abs(record["v2"]["mean_forecast"] - 0.03) < 1e-12
    assert watch.falls_short(record["v2"]) and not watch.falls_short({**record["v2"], "n": 14})
    live, prev = (object(), {"trained_at": "v2"}), (object(), {"trained_at": "v1"})
    assert watch.choose_selector(live, prev, record) == (prev, "rollback")
    assert watch.choose_selector(live, prev, {**record, "v1": record["v2"]}) == (None, "suspended")
    assert watch.choose_selector(live, None, record) == (None, "suspended")
    assert watch.choose_selector(live, prev, {}) == (live, "")


def test_an_edge_version_is_judged_by_its_loss_not_by_a_forecast_of_another_thing():
    # the night researcher, 08.11: an "edge" forecast is the edge over a random entry, not the trade's result
    losing_a_bit = {"n": 20, "mean_net": -0.01, "mean_forecast": 0.02}
    assert watch.falls_short(losing_a_bit, "net") and not watch.falls_short(losing_a_bit, "edge")
    assert watch.falls_short({**losing_a_bit, "mean_net": -0.03}, "edge")
