"""The paper journal: live positions stay open, decisions are kept as made, the book and its words."""

import numpy as np
import pandas as pd
import pytest

from core.brain import paper
from core.brain import simulator as sim
from core.brain.playbook import EXITS

HOUR, DAY = 3_600_000, 86_400_000
D0 = pd.Timestamp("2026-11-01")
T0 = int((D0 - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1))


def _path(closes, start=T0 + DAY):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    ts = start + np.arange(len(c)) * HOUR
    return np.column_stack([ts, o, np.maximum(o, c), np.minimum(o, c), c])


def _funding(until, start=T0 - 30 * DAY):
    ts = np.arange(start, until, 8 * HOUR, dtype=float)
    return np.column_stack([ts, np.full(len(ts), 0.0001)])


def test_a_live_position_stays_open_and_is_marked_at_the_newest_close():
    prices = _path(np.linspace(100, 104, 5 * 24))                 # 5 days of a 14-day hold so far
    now = int(prices[-1, 0]) + HOUR
    w = sim.walk(prices, None, D0, +1, 1.0, 0.0, sim.Exit(days=14), live_until=now)
    assert w.reason == "open" and w.exit_ts == sim.STILL_OPEN
    assert max(w.marks) == pd.Timestamp("2026-11-06") and w.ret == pytest.approx(w.marks[max(w.marks)])
    assert sim.walk(prices, None, D0, +1, 1.0, 0.0, sim.Exit(days=14)).reason == "ended"   # history: it stopped
    stale = sim.walk(prices, None, D0, +1, 1.0, 0.0, sim.Exit(days=14), live_until=now + 2 * DAY)
    assert stale.reason == "ended"                                # no prices for two days: the coin stopped


def test_a_live_short_needs_a_perpetual_that_pays_now_and_is_not_settled_by_the_edge_of_the_data():
    prices = _path(np.full(3 * 24, 100.0))
    now = int(prices[-1, 0]) + HOUR
    w = sim.walk(prices, _funding(now), D0, -1, 1.0, 0.0, EXITS["liq"], live_until=now)
    assert w.reason == "open" and w.ret > w.marks[max(w.marks)]   # the funding received is in
    assert sim.walk(prices, _funding(now - 3 * DAY), D0, -1, 1.0, 0.0, EXITS["liq"], live_until=now) is None


def test_an_order_enters_after_its_decision():
    prices = _path(np.r_[100.0, np.full(400, 110.0)])            # 00:00 opens at 100, 02:00 at 110
    decided = T0 + DAY + 90 * 60 * 1000                          # 01:30 after the close
    w = sim.walk(prices, None, D0, +1, 1.0, 0.0, sim.Exit(days=14), enter_from=decided)
    assert w.entry_ts == T0 + DAY + 2 * HOUR                      # the 02:00 open, not the 00:00 one
    assert w.ret == pytest.approx(0.0, abs=0.003)                 # bought at 110, not at 100


class FakeMarket:
    def __init__(self, prices, funding, btc):
        self._p, self._f, self._btc = prices, funding, btc

    def prices(self, history):
        return self._p.get(history, np.empty((0, 5)))

    def funding(self, history):
        return self._f.get(history)

    def notices(self, history):
        return np.empty(0)

    def btc(self):
        return self._btc

    def btc_closes(self):
        p = self._btc
        last = pd.DataFrame({"day": p[:, 0] // DAY, "ts": p[:, 0], "close": p[:, 4]}).groupby("day").last()
        last = last[last["ts"] >= last.index * DAY + DAY - HOUR]
        return pd.Series(last["close"].to_numpy(), index=pd.to_datetime(last.index * DAY, unit="ms"))


class FakeBrain:
    meta = {"model_id": "test"}

    def score(self, frame):
        return frame["ret_7d"].to_numpy()


def _rows(day):
    coins = [f"C{i}" for i in range(8)]
    return pd.DataFrame({"history": coins + ["BTC"], "base": coins + ["BTC"], "day": day,
                         "turnover_30d": 5e7, "ret_7d": np.r_[np.arange(8.0), 4.5],
                         "perp_listed": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                         "funding_1d": [np.nan, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
                         "atr": 1.0, "atr_pct": 0.05, "close": [10.0] * 8 + [110.0], "sma50": [9.0] * 8 + [100.0]})


def test_the_decision_is_n5s_picks_with_shorts_only_on_live_perpetuals():
    d = paper.decide(_rows(D0), FakeBrain(), T0 + DAY + HOUR)
    longs = [o["base"] for o in d["orders"] if o["side"] > 0]
    shorts = [o["base"] for o in d["orders"] if o["side"] < 0]
    assert sorted(longs) == ["C5", "C6", "C7"]
    assert sorted(shorts) == ["C1", "C2", "C4"]                   # C0: no funding that day, C3: no perpetual
    assert d["btc_up"] and d["day"] == "2026-11-01"
    assert sim.Exit(**d["orders"][0]["exit"]) == EXITS["hold"]       # kept as made, read back as the same rule


def test_the_book_and_its_words(tmp_path):
    decision = paper.decide(_rows(D0), FakeBrain(), T0 + DAY + 30 * 60 * 1000)
    path = tmp_path / "decisions.jsonl"
    paper.append(path, decision)
    decisions = paper.load(path)
    assert decisions == [decision]
    hours = 10 * 24
    up, flat, down = np.linspace(10, 12, hours), np.full(hours, 10.0), np.linspace(10, 9, hours)
    prices = {**{c: _path(up) for c in ("C5", "C6", "C7")}, **{c: _path(down) for c in ("C1", "C2", "C4")}}
    btc = _path(np.r_[np.full(24, 110.0), np.linspace(110, 121, hours)], start=T0)
    market = FakeMarket(prices, {c: _funding(T0 + 20 * DAY) for c in ("C1", "C2", "C4")}, btc)
    now = int(T0 + DAY + hours * HOUR)
    b = paper.book(decisions, market, now)
    assert len(b.still_open) == 6 and b.closed.empty              # nothing is over after 10 days
    assert b.values.iloc[-1] > b.values.iloc[0] and b.btc_half.iloc[-1] > b.btc_half.iloc[0]
    text = paper.report_text(b, decisions)
    assert text.startswith("📒 Бумажный C1") and "BTC за то же время" in text and "держит BTC" in text
    assert "лонг: C7 +19,6%" in text and "шорт: C1" in text and "Неделя:" in text and "Дней без решения: 10" in text
    assert ("Первое решение (закрытие 01.11, монет в оценке: 9): купить C7, C6, C5; шорт C1, C2, C4"
            in paper.start_text(decision))
    assert paper.state(b, decisions)["decisions"] == 1


def test_a_week_against_its_usual_range():
    days = pd.date_range("2026-11-01", periods=9)
    fine = pd.Series(np.linspace(10_000, 10_100, 9), index=days)
    assert "в обычных пределах" in paper.window_check(fine, 1)[1]
    bad = pd.Series(np.r_[np.full(2, 10_000.0), np.linspace(10_000, 9_000, 7)], index=days)
    assert "хуже 9 из 10" in paper.window_check(bad, 1)[1]
    assert paper.window_check(fine, 4) is None
