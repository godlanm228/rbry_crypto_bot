"""Detector v2 regressions.

The central regression is the preserved false ADA alert of 2026-07-17: the
old detector treated "price within 2% BELOW a rolling high that included
the signal candle, on an unclosed candle" as a resistance breakout.
"""

import numpy as np
import pandas as pd
import pytest

from core.analysis.pattern_registry import direction_of, resolve
from core.analysis.signal_generator import (
    detect_patterns,
    drop_open_candle,
)

NOW_MS = 1_784_300_000_000  # fixed "now" so tests never depend on wall clock


def _frame(closes, highs=None, lows=None, volumes=None, last_open=False):
    """Build an hourly OHLCV frame with explicit close_time stamps."""
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs if highs is not None else closes * 1.001, dtype=float)
    lows = np.asarray(lows if lows is not None else closes * 0.999, dtype=float)
    volumes = np.asarray(volumes if volumes is not None else [100.0] * n, dtype=float)

    hour_ms = 3_600_000
    # Last candle closes exactly at NOW_MS (closed); if last_open, in the future.
    close_times = [NOW_MS - (n - 1 - i) * hour_ms for i in range(n)]
    if last_open:
        close_times[-1] = NOW_MS + hour_ms

    index = pd.to_datetime([ct - hour_ms for ct in close_times], unit="ms")
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "close_time": close_times,
        },
        index=index,
    )


def _flat_series(n, price=0.16):
    """A quiet tape: tiny alternating moves, constant volume."""
    return [price * (1 + 0.001 * ((-1) ** i)) for i in range(n)]


class TestAdaRegression:
    """The preserved bad ADA delivery must never fire again."""

    def test_near_resistance_below_prior_high_is_not_a_breakout(self):
        # Rolling high 0.180 set mid-window; last close 0.177 is within 2%
        # of it (0.177 > 0.98 * 0.180) — the old rule fired here.
        closes = _flat_series(40, 0.170)
        closes[25] = 0.180  # the prior high
        closes.append(0.177)  # signal candle BELOW the high
        highs = [c * 1.001 for c in closes]
        df = _frame(closes, highs=highs)

        found = detect_patterns(df, window=20, now_ms=NOW_MS)
        assert not any(p["type"] == "resistance_breakout" for p in found)

    def test_unclosed_candle_cannot_emit_close_confirmed_signal(self):
        # Genuine breakout shape — but the signal candle has not closed.
        closes = _flat_series(40, 0.170) + [0.190]
        volumes = [100.0] * 40 + [400.0]
        df = _frame(closes, volumes=volumes, last_open=True)

        found = detect_patterns(df, window=20, now_ms=NOW_MS)
        assert not any(p["type"] == "resistance_breakout" for p in found)


class TestBreakoutV2:
    def test_confirmed_breakout_with_volume_is_detected(self):
        closes = _flat_series(40, 0.170) + [0.190]  # ~11% above prior highs
        volumes = [100.0] * 40 + [400.0]            # 4x prior average volume
        df = _frame(closes, volumes=volumes)

        found = detect_patterns(df, window=20, now_ms=NOW_MS)
        breakouts = [p for p in found if p["type"] == "resistance_breakout"]
        assert len(breakouts) == 1
        assert breakouts[0]["direction"] == "bullish"
        assert breakouts[0]["strength"] == "strong"
        assert breakouts[0]["level"] == pytest.approx(max(closes[:-1]) * 1.001)
        assert "target" not in breakouts[0]

    def test_close_marginally_above_level_within_buffer_is_not_breakout(self):
        # Close 0.05% above the prior high: below any volatility buffer.
        closes = _flat_series(40, 0.170)
        prior_high = max(c * 1.001 for c in closes)
        closes.append(prior_high * 1.0005)
        df = _frame(closes, highs=[c * 1.001 for c in closes])

        found = detect_patterns(df, window=20, now_ms=NOW_MS)
        assert not any(p["type"] == "resistance_breakout" for p in found)

    def test_signal_candle_never_defines_its_own_level(self):
        # The signal candle spikes to a new high but CLOSES below the prior
        # high — with the old "include the signal candle in the rolling
        # high" bug this shape polluted the level; it must not fire.
        closes = _flat_series(40, 0.170) + [0.169]
        highs = [c * 1.001 for c in closes]
        highs[-1] = 0.200  # intracandle spike, no close confirmation
        df = _frame(closes, highs=highs)

        found = detect_patterns(df, window=20, now_ms=NOW_MS)
        assert not any(p["type"] == "resistance_breakout" for p in found)

    def test_breakdown_mirror(self):
        closes = _flat_series(40, 0.170) + [0.150]
        volumes = [100.0] * 40 + [250.0]
        df = _frame(closes, volumes=volumes)

        found = detect_patterns(df, window=20, now_ms=NOW_MS)
        breakdowns = [p for p in found if p["type"] == "support_breakdown"]
        assert len(breakdowns) == 1
        assert breakdowns[0]["direction"] == "bearish"
        assert breakdowns[0]["strength"] == "strong"


class TestDropOpenCandle:
    def test_open_candle_removed(self):
        df = _frame(_flat_series(5), last_open=True)
        closed = drop_open_candle(df, now_ms=NOW_MS)
        assert len(closed) == 4

    def test_closed_frame_untouched(self):
        df = _frame(_flat_series(5))
        assert len(drop_open_candle(df, now_ms=NOW_MS)) == 5

    def test_frame_without_close_time_passes_through(self):
        df = pd.DataFrame({"close": [1.0, 2.0]})
        assert len(drop_open_candle(df, now_ms=NOW_MS)) == 2


class TestRegistry:
    def test_alias_naming_mismatch_resolved(self):
        assert resolve("head_shoulders").canonical == "head_and_shoulders"
        assert resolve("head_and_shoulders").canonical == "head_and_shoulders"
        assert resolve("inverse_head_shoulders").canonical == "inverse_head_and_shoulders"

    def test_directions(self):
        assert direction_of("double_bottom") == "bullish"
        assert direction_of("double_top") == "bearish"
        assert direction_of("doji") == "neutral"
        assert direction_of("unknown_thing") == "neutral"

    def test_detected_patterns_carry_canonical_metadata(self):
        closes = _flat_series(40, 0.170) + [0.190]
        df = _frame(closes, volumes=[100.0] * 40 + [400.0])
        found = detect_patterns(df, window=20, now_ms=NOW_MS)
        for p in found:
            assert resolve(p["type"]) is not None
            assert p["direction"] in ("bullish", "bearish", "neutral")
            assert p["detector_version"]
