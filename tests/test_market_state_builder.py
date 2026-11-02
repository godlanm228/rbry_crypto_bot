"""MarketStateBuilder v2: explicit data quality, no neutral fills."""

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.data_collectors.market_state_builder import MarketStateBuilder

NOW_MS = int(time.time() * 1000)


class _StubFearGreed:
    def __init__(self, value=None):
        self.value = value

    async def get_current_index(self):
        if self.value is None:
            raise RuntimeError("api down")
        return self.value


def _snapshot_service(markets):
    snapshot = SimpleNamespace(markets=markets, ts=time.time())
    return SimpleNamespace(get_usable=lambda max_age: snapshot)


def _history_service(by_coin):
    snapshot = SimpleNamespace(by_coin=by_coin, ts=time.time())
    return SimpleNamespace(get_usable=lambda max_age: snapshot)


def _builder(markets=None, history=None, fear_greed=None):
    builder = MarketStateBuilder()
    builder.fear_greed_service = _StubFearGreed(fear_greed)
    if markets is not None:
        builder.set_market_snapshot(_snapshot_service(markets))
    if history is not None:
        builder.set_history_snapshot(_history_service(history))
    return builder


def _history_df(prices, volumes=None):
    df = pd.DataFrame({"price": prices})
    df["close"] = df["price"]
    if volumes is not None:
        df["volume"] = volumes
    return df


def _ohlcv(closes, volumes=None, last_open=False):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    hour_ms = 3_600_000
    close_times = [NOW_MS - (n - 1 - i) * hour_ms for i in range(n)]
    if last_open:
        close_times[-1] = NOW_MS + hour_ms
    return pd.DataFrame({
        "open": closes * 0.999,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": volumes if volumes is not None else [100.0] * n,
        "close_time": close_times,
    })


MARKETS = {
    "testcoin": {
        "current_price": 100.0,
        "symbol": "TST",
        "price_change_percentage_24h_in_currency": 4.0,
        "price_change_percentage_7d_in_currency": 9.0,
    }
}


class TestHonestGaps:
    def test_no_market_data_returns_none(self):
        builder = _builder(markets={})
        state = asyncio.run(builder.build_market_state("testcoin"))
        assert state is None

    def test_missing_history_is_explicit_not_default_filled(self):
        builder = _builder(markets=MARKETS)
        state = asyncio.run(builder.build_market_state("testcoin"))
        assert state is not None
        # The old builder returned rsi=50, volatility=0.5, news=0.0 here.
        assert state.technical.rsi is None
        assert state.technical.volatility_annual_pct is None
        assert state.sentiment.news_sentiment is None
        assert "no_price_history_available" in state.quality.block_reasons
        assert "history" in state.quality.missing_sources

    def test_news_is_always_unavailable_until_a2(self):
        history = {"testcoin": _history_df([100.0 + i for i in range(30)])}
        builder = _builder(markets=MARKETS, history=history)
        state = asyncio.run(builder.build_market_state("testcoin"))
        assert state.sentiment.news_sentiment is None
        assert "news" in state.quality.missing_sources

    def test_missing_momentum_field_stays_none(self):
        markets = {"testcoin": {"current_price": 100.0, "symbol": "TST"}}
        builder = _builder(markets=markets)
        state = asyncio.run(builder.build_market_state("testcoin"))
        assert state.technical.price_momentum_24h_pct is None
        assert state.context.trend is None

    def test_fear_greed_failure_is_marked_unavailable(self):
        builder = _builder(markets=MARKETS, fear_greed=None)
        state = asyncio.run(builder.build_market_state("testcoin"))
        assert state.sentiment.fear_greed_index is None
        assert "fear_greed" in state.quality.missing_sources


class TestRealComputation:
    def test_rsi_and_volatility_from_history(self):
        # Uptrend with periodic pullbacks: a strictly monotonic series has
        # zero average loss and mathematically undefined RSI.
        prices = [100.0]
        for i in range(29):
            step = 0.995 if i % 3 == 2 else 1.02
            prices.append(prices[-1] * step)
        history = {"testcoin": _history_df(prices)}
        builder = _builder(markets=MARKETS, history=history, fear_greed=50.0)
        state = asyncio.run(builder.build_market_state("testcoin"))
        assert state.technical.rsi is not None
        assert state.technical.rsi > 70  # monotonic rise -> overbought RSI
        assert state.technical.volatility_annual_pct is not None
        assert state.context.rsi_zone == "overbought"
        assert state.context.trend == "bull"

    def test_volume_ratio_excludes_signal_candle_from_baseline(self):
        # 21 candles: 20 at volume 100, signal candle 200 -> ratio must be
        # exactly 2.0 (baseline excludes the signal candle).
        closes = [100.0] * 21
        volumes = [100.0] * 20 + [200.0]
        builder = _builder(markets=MARKETS, fear_greed=50.0)
        state = asyncio.run(builder.build_market_state(
            "testcoin", ohlcv=_ohlcv(closes, volumes)
        ))
        assert state.technical.volume_ratio == pytest.approx(2.0)

    def test_open_candle_excluded_everywhere(self):
        # Same fixture but the spike candle is still open -> the last CLOSED
        # candle (volume 100) becomes the signal candle -> ratio 1.0.
        closes = [100.0] * 22
        volumes = [100.0] * 21 + [999.0]
        builder = _builder(markets=MARKETS, fear_greed=50.0)
        state = asyncio.run(builder.build_market_state(
            "testcoin", ohlcv=_ohlcv(closes, volumes, last_open=True)
        ))
        assert state.technical.volume_ratio == pytest.approx(1.0)

    def test_support_resistance_from_supplied_candles(self):
        # Oscillating tape between ~90 and ~110, current price 100:
        # a support below and a resistance above must both be found.
        closes = [100.0 + 10.0 * np.sin(i / 3.0) for i in range(80)]
        builder = _builder(markets=MARKETS, fear_greed=50.0)
        state = asyncio.run(builder.build_market_state(
            "testcoin", ohlcv=_ohlcv(closes)
        ))
        assert state.technical.support_level is not None
        assert state.technical.support_level < 100.0
        assert state.technical.resistance_level is not None
        assert state.technical.resistance_level > 100.0
        assert state.technical.support_distance_pct is not None
        assert state.technical.support_distance_pct >= 0

    def test_patterns_source_marked_available_with_candles(self):
        closes = [100.0] * 40 + [115.0]
        volumes = [100.0] * 40 + [300.0]
        builder = _builder(markets=MARKETS, fear_greed=50.0)
        state = asyncio.run(builder.build_market_state(
            "testcoin", ohlcv=_ohlcv(closes, volumes)
        ))
        assert state.quality.sources["patterns"].available
        assert any(
            p["type"] == "resistance_breakout"
            for p in state.patterns.detected_patterns
        )

    def test_serialization_roundtrip_v2(self):
        from core.synthesis.market_state_vector import MarketStateVector
        history = {"testcoin": _history_df([100.0 + i for i in range(30)])}
        builder = _builder(markets=MARKETS, history=history, fear_greed=42.0)
        state = asyncio.run(builder.build_market_state("testcoin"))
        restored = MarketStateVector.from_json(state.to_json())
        assert restored.technical.rsi == pytest.approx(state.technical.rsi)
        assert restored.quality.missing_sources == state.quality.missing_sources
        assert restored.to_dict()["schema_version"] == 2
