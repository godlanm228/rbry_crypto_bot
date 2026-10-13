"""RSI edge cases, warm-up honesty and support/resistance after a decline."""

import numpy as np
import pandas as pd

from core.analysis.signal_generator import find_support_resistance
from core.analysis.technical_analysis import rsi


def test_rsi_pure_rise_is_100_not_nan():
    assert rsi(pd.Series(np.linspace(100, 130, 40))).iloc[-1] == 100.0


def test_rsi_flat_series_is_50():
    assert rsi(pd.Series([10.0] * 30)).iloc[-1] == 50.0


def test_rsi_warmup_is_not_backfilled_from_the_future():
    values = rsi(pd.Series(np.linspace(100, 90, 30) + np.sin(np.arange(30))))
    assert values.iloc[:13].isna().all()


def test_rsi_on_200_points_converges_to_long_history():
    rng = np.random.default_rng(7)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 1200))))
    long_value = rsi(prices).iloc[-1]
    assert abs(rsi(prices.tail(201)).iloc[-1] - long_value) < 0.5
    # The old 31-point window is what produced multi-point errors.
    assert abs(rsi(prices.tail(31)).iloc[-1] - long_value) > abs(rsi(prices.tail(201)).iloc[-1] - long_value)


def test_supports_survive_a_decline():
    # Rally with tested lows near 100 and 110, then a drop to ~105: the 100
    # support lies below the price and must still be reported.
    up = [100, 104, 101, 106, 100.5, 108, 110, 114, 110.5, 118, 110.2, 122, 126, 130,
          127, 132, 128, 134, 131, 136]
    path = up * 3 + list(np.linspace(136, 105, 20))
    close = pd.Series(path, dtype=float)
    frame = pd.DataFrame({"close": close, "high": close * 1.003, "low": close * 0.997})
    levels = find_support_resistance(frame)
    price = float(close.iloc[-1])
    assert any(level < price for level in levels["support"])
