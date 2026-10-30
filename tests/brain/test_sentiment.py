"""Who is buying: Coinbase's and Korea's premiums and the options' fear, each known at its hour's close."""

import numpy as np
import pandas as pd

from core.brain import sentiment as se
from core.sandbox import context_sources as cs
from core.sandbox.market_store import MarketStore

HOUR, DAY = 3_600_000, 86_400_000
T0 = int(pd.Timestamp("2022-01-01").value // 10**6)


class _Market:
    def __init__(self, closes):
        ts = T0 + np.arange(len(closes), dtype=np.int64) * HOUR
        c = np.asarray(closes, float)
        self._p = np.column_stack([ts, c, c, c, c])
        self.base_of = {"BTC": "BTC"}

    def prices(self, base):
        return self._p


def test_the_sources_parse_to_hour_starts_and_closes():
    assert cs.parse_coinbase([[1505300400, 1, 2, 1.5, 1.8, 9], [1505296800, 1, 2, 1.5, 1.7, 9]]) == \
        [(1505296800000, 1.7), (1505300400000, 1.8)]
    assert cs.parse_upbit([{"candle_date_time_utc": "2017-09-30T23:00:00", "trade_price": 4895000.0}]) == \
        [(int(pd.Timestamp("2017-09-30 23:00").value // 10**6), 4895000.0)]
    assert cs.parse_dvol({"result": {"data": [[1000, 50, 51, 49, 50.5]], "continuation": 900}}) == ([(1000, 50.5)], 900)
    csv = ("KEY,FREQ,CURRENCY,TIME_PERIOD,OBS_VALUE\nX,D,KRW,2024-01-02,1400.0\nX,D,USD,2024-01-02,1.1\n"
           "X,D,KRW,2024-01-03,1430.0\n")
    assert cs.parse_ecb_krw(csv) == [("2024-01-02", 1400.0 / 1.1)]      # a day without the dollar's rate is out


def test_the_wons_rate_counts_only_once_its_day_is_over():
    fx = pd.DataFrame({"day": pd.to_datetime(["2022-01-03", "2022-01-04"]), "value": [1200.0, 1300.0]})
    d3 = int(pd.Timestamp("2022-01-03").value // 10**6)
    rate = se.won_per_dollar(fx, np.array([d3 + 23 * HOUR, d3 + DAY, d3 + 2 * DAY - HOUR, d3 + 2 * DAY]))
    assert np.isnan(rate[0]) and rate[1:].tolist() == [1200.0, 1200.0, 1300.0]


def test_an_extreme_is_against_the_month_before_it_and_fires_once():
    rng = np.random.default_rng(2)
    v = rng.normal(0, 1, 2000)
    v[1500:1510] = 10.0
    hi, lo = se.extremes(v)
    assert hi[1500] and not hi[: 720 * 3 // 4].any()                   # nothing before 3 weeks of history
    from core.brain.events import fire
    assert 1500 in fire(hi, edge=True).tolist() and 1501 not in fire(hi, edge=True).tolist()


def test_a_coinbase_premium_jump_is_a_signal_at_that_hours_close(tmp_path):
    n = 24 * 40
    store = MarketStore(tmp_path / "m.db")
    rng = np.random.default_rng(4)
    binance = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    premium = rng.normal(0, 0.0003, n)
    premium[900] = 0.01                                               # Coinbase 1% dearer for one hour
    ts = T0 + np.arange(n, dtype=np.int64) * HOUR
    store.save_hourly("coinbase:BTC", zip(ts.tolist(), (binance * (1 + premium)).tolist()))
    out = se.all_events(store, _Market(binance))
    cb = [s for e, s in out["BTC"] if e == "CB-HI"]
    assert T0 + 901 * HOUR in cb and all(s > T0 + 540 * HOUR for s in cb)
    assert not any(e.startswith(("KR", "FEAR")) for e, _ in out["BTC"])  # no Korean or options data stored
