"""The listing radar: a first day booked with what followed, a ridge logistic fit, and the daily warning."""

import numpy as np
import pandas as pd

from core.brain import radar as rd

H, D = 3_600_000, 86_400_000
T0 = int(pd.Timestamp("2024-03-01").value // 10**6)


def _hourly(closes):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"ts": T0 + np.arange(len(c), dtype=np.int64) * H, "open": o, "high": np.maximum(o, c),
                         "low": np.minimum(o, c), "close": c, "quote_volume": 1e6})


def test_a_listing_is_bought_at_hour_25_and_a_dead_coin_ends_at_its_last_close():
    path = [10.0] * 24 + [8.0] * (35 * 24)
    states = pd.DataFrame({"ts": [T0 + 23 * H], "breadth": [0.4], "alts": [0.2], "funding": [0.0]})
    btc = pd.DataFrame({"ts": T0 - np.arange(800, 0, -1) * H, "close": 100.0})
    r = rd.listing_row(_hourly(path), "NEW", T0 - D, None, states, btc)
    assert abs(r["day_ret"]) < 1e-12 and r["breadth"] == 0.4 and r["perp"] == 0.0
    assert -0.21 < r["ret7"] < -0.19 and np.isnan(r["ret90"])                     # still trading: 90 days unknown
    dead = rd.listing_row(_hourly([10.0] * 24 + [2.0] * 100), "DEAD", T0 - D, None, states, btc, ended=True)
    assert dead["ret90"] < -0.79                                                  # out at its last close


def test_the_ridge_logistic_learns_a_clear_sign():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 2))
    y = (x[:, 0] + 0.3 * rng.normal(size=400) > 0).astype(float)
    w = rd.fit(x, y, ridge=1.0)
    assert w[1] > 1.0 and abs(w[2]) < 0.5


def test_the_daily_line_warns_with_the_base_rate():
    model = {"base_up": 0.23, "listings": 252}
    line = rd.day_line(["Binance Will List Foo (FOO) and Bar (BAR)"], model)
    assert "BAR, FOO" in line and "~2 из 10" in line
    assert "нет" in rd.day_line([], model)
