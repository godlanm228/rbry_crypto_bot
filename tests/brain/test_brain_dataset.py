"""The brain's textbook: the fixed trade on hourly candles, no peeking, breadth."""

import numpy as np
import pandas as pd

from core.brain import dataset as ds
from core.sandbox.market_store import History

HOUR = 3_600_000
DAY = 24 * HOUR


def _hours(prices, start=0):
    """Hourly candles from a list of (open, high, low, close)."""
    o, h, l, c = (np.array([p[i] for p in prices], dtype=float) for i in range(4))
    return np.arange(len(prices), dtype=np.int64) * HOUR + start, o, h, l, c


def _flat(n, price=100.0):
    return [(price, price + 0.5, price - 0.5, price)] * n


def _outcome(prices, atr=1.0, ended=False, slippage=0.0):
    ts, o, h, l, c = _hours(prices, start=DAY)          # the signal day is day 0, trading starts on day 1
    out = ds.trade_outcomes(ts, o, h, l, c, np.array([0]), np.array([atr]), np.array([slippage]), ended)
    return {k: v[0] for k, v in out.items()}


def test_target_stop_and_time_exits():
    span = ds.TRADE_DAYS * 24
    # stop = 98, target = 104 with ATR 1
    up = _flat(5) + [(100, 104.5, 99.5, 104)] + _flat(span)
    res = _outcome(up)
    assert res["reason"] == 1 and res["hours"] == 6 and abs(res["r_net"] - 1.896) < 0.01   # 2R minus fees
    down = _flat(3) + [(100, 100.2, 97.5, 98)] + _flat(span)
    assert _outcome(down)["reason"] == -1 and _outcome(down)["r_net"] < -1
    res = _outcome(_flat(span + 10))
    assert res["reason"] == 0 and res["hours"] == span and -0.2 < res["r_net"] < 0


def test_one_hour_through_both_levels_is_a_stop_and_gaps_exit_at_the_open():
    span = ds.TRADE_DAYS * 24
    both = _flat(2) + [(100, 105, 97, 101)] + _flat(span)
    assert _outcome(both)["reason"] == -1
    gap_down = _flat(2) + [(90, 91, 89, 90)] + _flat(span)
    res = _outcome(gap_down)
    assert res["reason"] == -1 and res["r_net"] < -4.9          # exited at 90, not at the stop of 98
    gap_up = _flat(2) + [(110, 111, 109, 110)] + _flat(span)
    assert _outcome(gap_up)["reason"] == 1 and _outcome(gap_up)["r_net"] > 4.8


def test_running_trades_have_no_label_and_dead_coins_close_at_their_last_price():
    short = _flat(30) + [(100, 100.5, 99, 99)] * 3
    assert np.isnan(_outcome(short)["r_net"])
    res = _outcome(short, ended=True)
    assert res["reason"] == 2 and res["hours"] == 33


def test_no_trade_when_the_coin_does_not_open_after_the_signal():
    ts, o, h, l, c = _hours(_flat(400), start=DAY + 7 * HOUR)
    out = ds.trade_outcomes(ts, o, h, l, c, np.array([0]), np.array([1.0]), np.array([0.0]), False)
    assert np.isnan(out["r_net"][0])


def test_slippage_grows_as_turnover_shrinks():
    assert ds.slippage_pct(np.array([50e6, 8e6, 2e6, 3e5, np.nan])).tolist() == [0.05, 0.10, 0.20, 0.40, 0.40]


def _history(days=420, seed=5, ended=False, name="AAA"):
    rng = np.random.default_rng(seed)
    n = days * 24
    close = 10 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    hourly = pd.DataFrame({"ts": np.arange(n, dtype=np.int64) * HOUR + 1577836800000, "open": open_,
                           "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
                           "close": close, "volume": 1.0, "quote_volume": rng.uniform(5e4, 2e5, n),
                           "trades": rng.integers(10, 100, n), "taker_buy_quote": 4e4})
    from core.sandbox.market_store import resample
    return History(name, f"{name}USDT", name, hourly, resample(hourly, "1d"), ended)


def test_features_do_not_change_when_the_future_is_cut_off():
    full = _history()
    btc = ds._daily_frame(_history(seed=9, name="BTC").daily)
    funding = pd.DataFrame({"ts": np.arange(0, 400 * DAY, 8 * HOUR) + 1577836800000 + 30 * DAY, "rate": 0.0001})
    notices = [1577836800000 + 300 * DAY + 5 * HOUR]
    rows = ds.coin_rows(full, btc, None, funding, notices)
    cols = list(ds.SITUATION_FEATURES + ds.COIN_FEATURES)
    for keep_days in (260, 330, 400):
        cut = History(full.name, full.symbol, full.base, full.hourly.iloc[:keep_days * 24],
                      full.daily.iloc[:keep_days], False)
        part = ds.coin_rows(cut, btc, None, funding, notices)
        day = part["day"].iloc[-1]
        a = rows.set_index("day").loc[day, cols].astype(float).to_numpy()
        b = part.set_index("day").loc[day, cols].astype(float).to_numpy()
        assert np.allclose(a, b, equal_nan=True, rtol=1e-5)
    assert rows.set_index("day").loc[pd.Timestamp("2020-10-27"), "delist_notice"] == 1.0
    assert rows.set_index("day").loc[pd.Timestamp("2020-10-26"), "delist_notice"] == 0.0


def test_breadth_counts_coins_ahead_of_btc():
    days = pd.date_range("2024-01-01", periods=3, freq="D")
    light = [pd.DataFrame({"day": days, "base": "BTC", "ret_1d": 0.01, "ret_7d": 0.0, "ret_30d": 0.0,
                           "above_sma50": 1.0, "turnover": 1e9})]
    for i in range(30):
        light.append(pd.DataFrame({"day": days, "base": f"C{i}", "ret_1d": 0.02 if i < 6 else 0.0,
                                   "ret_7d": 0.1 if i < 15 else -0.1, "ret_30d": np.nan, "above_sma50": float(i % 2),
                                   "turnover": 1e6 * (10 if i < 15 else 1)}))
    b = ds.breadth(pd.concat(light, ignore_index=True)).iloc[0]
    assert abs(b["beat_btc_1d"] - 0.2) < 1e-9 and abs(b["beat_btc_7d"] - 0.5) < 1e-9
    assert abs(b["beat_btc_7d_vw"] - 150 / 165) < 1e-9 and np.isnan(b["beat_btc_30d"])
    assert abs(b["above_sma50"] - 0.5) < 1e-9


def test_the_short_mirrors_the_long_and_collects_funding():
    span = ds.TRADE_DAYS * 24
    ts, o, h, l, c = _hours(_flat(3) + [(100, 100.2, 95.5, 96)] + _flat(span, price=96), start=DAY)
    funding_ts = np.arange(DAY, DAY + 20 * DAY, 8 * HOUR, dtype=np.int64)
    out = ds.trade_outcomes(ts, o, h, l, c, np.array([0]), np.array([1.0]), np.array([0.0]), False,
                            funding_ts, np.full(len(funding_ts), 0.001))
    # short target = 100 - 4 = 96 hit in hour 4; one funding payment of 0.1% fell inside those hours
    assert out["short_reason"][0] == 1 and out["short_hours"][0] == 4
    expected = ds.short_return(100.0, 96.0, 0.0, 0.001)
    assert abs(out["short_ret_net"][0] - expected) < 1e-12 and abs(out["short_r_net"][0] - expected / 0.02) < 1e-9
    assert out["reason"][0] == -1                       # the same hours stopped the long out
    assert abs(out["funding_hold"][0] - 0.001 * 42) < 1e-12
    no_perp = ds.trade_outcomes(ts, o, h, l, c, np.array([0]), np.array([1.0]), np.array([0.0]), False)
    assert np.isnan(no_perp["short_r_net"][0]) and np.isfinite(no_perp["r_net"][0])


def test_a_short_squeeze_is_a_stop_above():
    span = ds.TRADE_DAYS * 24
    ts, o, h, l, c = _hours(_flat(2) + [(100, 103, 99.8, 102.5)] + _flat(span, price=102.5), start=DAY)
    funding_ts = np.array([0], dtype=np.int64)
    out = ds.trade_outcomes(ts, o, h, l, c, np.array([0]), np.array([1.0]), np.array([0.0]), False,
                            funding_ts, np.array([0.0]))
    assert out["short_reason"][0] == -1 and out["short_r_net"][0] < -1
