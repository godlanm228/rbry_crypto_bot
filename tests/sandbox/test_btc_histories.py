"""BTC-quoted pairs in dollars, for the hours before a coin's USDT pair (plan item 5)."""

import numpy as np
import pandas as pd

from core.sandbox.market_store import HOUR_MS, KLINE_FIELDS, MarketStore

T0 = int(pd.Timestamp("2019-01-01").value // 10**6)


def _frame(n, price, start=T0, volume=1.0, step=0.0):
    ts = start + np.arange(n, dtype=np.int64) * HOUR_MS
    c = price * (1 + step) ** np.arange(n)
    f = pd.DataFrame({"ts": ts, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 10.0,
                      "quote_volume": volume, "trades": 5, "taker_buy_quote": volume / 2})
    return f[["ts", *KLINE_FIELDS]]


def test_a_btc_pair_becomes_dollars_until_its_usdt_pair_begins(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    n = 24 * 90
    store.save_klines("BTCUSDT", "1h", _frame(n, 4000.0))
    store.save_klines("ADABTC", "1h", _frame(n, 0.00001, step=0.001, volume=2.0))
    store.save_klines("ADAUSDT", "1h", _frame(24 * 30, 0.05, start=T0 + 24 * 70 * HOUR_MS))
    store.mark_fetched("data/spot/monthly/klines/ADABTC/1h/ADABTC-1h-2019-01.zip", "ok", 1)
    assert store.btc_pairs() == ["ADABTC"]
    (h,) = list(store.btc_histories(min_days=30))
    assert h.name == "ADA~BTC" and h.base == "ADA" and h.ended
    assert int(h.hourly["ts"].max()) < T0 + 24 * 70 * HOUR_MS                    # only before the USDT pair
    assert abs(h.hourly["close"].iloc[0] - 0.04) < 1e-12                        # 0.00001 BTC x $4000
    assert abs(h.hourly["quote_volume"].iloc[0] - 8000.0) < 1e-9                 # 2 BTC of turnover
