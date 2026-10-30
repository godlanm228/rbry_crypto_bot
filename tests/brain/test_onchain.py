"""On-chain events: Binance's reserves a day, known the morning after, and a wallet's tracking change is no flow."""

import numpy as np
import pandas as pd

from core.brain import events as ev
from core.brain import onchain
from core.sandbox.context_sources import parse_cex_reserves
from core.sandbox.market_store import History, MarketStore

HOUR, DAY = 3_600_000, 86_400_000
D0 = pd.Timestamp("2024-03-01")
MS0 = int(D0.value // 10**6)


def _series(values, start=D0):
    return pd.DataFrame({"day": pd.date_range(start, periods=len(values), freq="D"), "value": values})


def _days(values, start=D0):
    return [(d.strftime("%Y-%m-%d"), v) for d, v in zip(pd.date_range(start, periods=len(values), freq="D"), values)]


class _Market:
    def __init__(self, bases, price=1.0):
        self.base_of = {b: b for b in bases}
        ts = MS0 - 10 * DAY + np.arange(24 * 40, dtype=np.int64) * HOUR
        self._p = np.column_stack([ts, *[np.full(len(ts), price)] * 4])

    def prices(self, base):
        return self._p


def test_the_unfinished_days_point_is_left_out():
    payload = {"tokens": [{"date": MS0 // 1000, "tokens": {"usdt": 5.0}},
                          {"date": MS0 // 1000 + DAY // 1000, "tokens": {"usdt": 6.0, "BTC": 1.0}},
                          {"date": MS0 // 1000 + DAY // 1000 + 3600 * 16, "tokens": {"usdt": 9.0}}]}
    out = parse_cex_reserves(payload)
    assert out["USDT"] == [("2024-03-01", 5.0), ("2024-03-02", 6.0)] and out["BTC"] == [("2024-03-02", 1.0)]


def test_a_tokens_inflow_counts_from_the_next_morning_and_only_if_it_is_worth_a_million():
    s = _series([10e6, 10e6, 11.5e6, 11.5e6, 10e6])
    events = onchain.token_events(s, lambda ms: 1.0)
    # day 2 closes 15% up ($1.5M): known at 03:00 of day 3; day 4 is 13% down
    assert events == [("CEX-IN", MS0 + 3 * DAY + 3 * HOUR), ("CEX-OUT", MS0 + 5 * DAY + 3 * HOUR)]
    assert onchain.token_events(s, lambda ms: 0.5) == []       # $0.75M moves: too small
    gap = pd.DataFrame({"day": pd.to_datetime(["2024-03-01", "2024-03-03"]), "value": [10e6, 20e6]})
    assert onchain.token_events(gap, lambda ms: 1.0) == []      # a missing day is no change


def test_a_day_when_many_coins_move_together_is_a_tracking_change(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    coins = [f"C{i}" for i in range(onchain.GLITCH_COINS + 1)]
    for i, c in enumerate(coins):
        jump = [10e6, 10e6, 12e6] if i else [10e6, 12e6, 12e6]  # C0 alone a day earlier
        store.save_series(f"cex:binance:{c}", _days(jump))
    store.save_series("cex:binance:USDT", _days([1e9, 1e9, 0.9e9]))
    store.save_series("cex:binance:USDC", _days([1e9, 1e9, 1e9]))
    store.save_series("stablecoins_usd", _days([100e9, 101e9, 101e9]))
    out = onchain.all_events(store, _Market(coins + ["BTC", "ETH"]))
    assert out["C0"] == [("CEX-IN", MS0 + 2 * DAY + 3 * HOUR)]
    assert all(c not in out for c in coins[1:])                # six coins on one day: not money
    # the stablecoins on Binance: 2.0B -> 1.9B (-5%); every stablecoin +1% a day earlier
    assert sorted(out["BTC"]) == [("STABLE-MINT", MS0 + 2 * DAY + 3 * HOUR), ("STABLE-OUT", MS0 + 3 * DAY + 3 * HOUR)]
    assert out["ETH"] == out["BTC"] and "USDT" not in out


def test_the_library_books_an_outside_event_at_the_hour_it_is_known():
    n = 24 * 60
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    t0 = int(pd.Timestamp("2023-03-01").value // 10**6)
    qv = np.full(n, 1e7)
    g = pd.DataFrame({"ts": t0 + np.arange(n, dtype=np.int64) * HOUR, "open": np.r_[c[0], c[:-1]], "high": c * 1.002,
                      "low": c * 0.998, "close": c, "volume": qv / c, "quote_volume": qv, "trades": 100,
                      "taker_buy_quote": qv / 2})
    h = History("AAA", "AAAUSDT", "AAA", g, g.iloc[:0], False)
    signal = t0 + 1001 * HOUR
    rows = ev.coin_rows(h, external=[("CEX-IN", signal), ("CEX-OUT", t0 + 5 * HOUR), ("CEX-OUT", signal + 1)])
    booked = rows[rows["event"].isin(ev.EXTERNAL)]
    assert booked["event"].tolist() == ["CEX-IN"]             # hour 5 is before the warm-up, +1 ms is no hour's close
    assert int(booked["signal_ts"].iloc[0]) == signal and np.isfinite(booked["twin24"].iloc[0])
