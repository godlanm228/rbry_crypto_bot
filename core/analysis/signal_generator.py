"""Technical signal and pattern detection over OHLCV candles.

Detector contract (DETECTOR_VERSION):

- Rules that require candle closure operate on COMPLETED candles only.
  ``drop_open_candle()`` removes the still-open candle using its
  ``close_time`` column; data without ``close_time`` (e.g. daily history
  rows) is assumed closed and documented as such by the caller.
- A breakout/breakdown level is computed from candles BEFORE the signal
  candle — the signal candle never participates in its own trigger level.
- Breakout requires the completed candle to CLOSE beyond the level plus a
  volatility-aware buffer. "Near the level" is not a breakout.
- Volume confirmation compares the signal candle to prior completed candles
  only, never to a window that includes the signal candle.
- Detected patterns carry canonical names and directions from
  ``core.analysis.pattern_registry`` and never carry price targets:
  arbitrary target formulas were removed deliberately.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .pattern_registry import resolve as resolve_pattern

DETECTOR_VERSION = "2.0.0"

# Breakout buffer: max(floor, fraction of average true range).
_BREAKOUT_BUFFER_FLOOR_PCT = 0.1     # 0.1% minimum buffer
_BREAKOUT_BUFFER_ATR_FRACTION = 0.25
_VOLUME_STRONG_RATIO = 2.0
_VOLUME_MEDIUM_RATIO = 1.3


def drop_open_candle(ohlcv: pd.DataFrame, now_ms: Optional[int] = None) -> pd.DataFrame:
    """Remove candles that have not closed yet, using their ``close_time``.

    Binance klines carry ``close_time`` in milliseconds; the currently open
    candle has a close_time in the future. Frames without a ``close_time``
    column are returned unchanged — the caller asserts their data is closed
    (e.g. finished daily rows).
    """
    if 'close_time' not in ohlcv.columns or ohlcv.empty:
        return ohlcv
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    closed = ohlcv[ohlcv['close_time'] <= now_ms]
    return closed


def _pattern_dict(name: str, strength: str, level: Optional[float] = None,
                  detail: Optional[str] = None) -> Dict:
    meta = resolve_pattern(name)
    direction = meta.direction if meta else "neutral"
    kind = meta.kind if meta else "unknown"
    result = {
        'type': meta.canonical if meta else name,
        'direction': direction,
        'kind': kind,
        # Legacy consumers read a combined 'signal' string.
        'signal': f"{direction}_{kind}" if direction != "neutral" else kind,
        'strength': strength,
        'detector_version': DETECTOR_VERSION,
    }
    if level is not None:
        result['level'] = float(level)
    if detail:
        result['detail'] = detail
    return result


def _volume_strength(volume: Optional[np.ndarray], baseline_window: int) -> str:
    """Signal-candle volume vs mean of PRIOR completed candles."""
    if volume is None or len(volume) < baseline_window + 1:
        return "weak"
    baseline = float(np.mean(volume[-(baseline_window + 1):-1]))
    if baseline <= 0:
        return "weak"
    ratio = float(volume[-1]) / baseline
    if ratio >= _VOLUME_STRONG_RATIO:
        return "strong"
    if ratio >= _VOLUME_MEDIUM_RATIO:
        return "medium"
    return "weak"


def _breakout_buffer_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         window: int) -> float:
    """Volatility-aware buffer as percent of the level (versioned policy)."""
    n = min(window, len(close) - 1)
    if n < 2:
        return _BREAKOUT_BUFFER_FLOOR_PCT
    prev_close = close[-(n + 1):-1]
    tr = np.maximum(
        high[-n:] - low[-n:],
        np.maximum(np.abs(high[-n:] - prev_close), np.abs(low[-n:] - prev_close)),
    )
    atr_pct = float(np.mean(tr) / np.mean(prev_close)) * 100.0
    return max(_BREAKOUT_BUFFER_FLOOR_PCT, atr_pct * _BREAKOUT_BUFFER_ATR_FRACTION)


def detect_patterns(ohlcv: pd.DataFrame, window: int = 20,
                    now_ms: Optional[int] = None) -> List[Dict]:
    """Chart patterns on completed candles: breakout/breakdown, double
    top/bottom, head & shoulders. Emits canonical, target-free patterns."""
    df = drop_open_candle(ohlcv, now_ms=now_ms)
    if len(df) < window + 1:
        return []

    patterns: List[Dict] = []
    close = df['close'].to_numpy(dtype=float)
    high = (df['high'] if 'high' in df.columns else df['close']).to_numpy(dtype=float)
    low = (df['low'] if 'low' in df.columns else df['close']).to_numpy(dtype=float)
    volume = df['volume'].to_numpy(dtype=float) if 'volume' in df.columns else None

    # --- Resistance breakout / support breakdown -------------------------
    # Level from candles BEFORE the signal candle; close must clear it by a
    # volatility-aware buffer. Proximity alone must never fire (that exact
    # bug produced the false ADA alert on 2026-07-17).
    prior_high = float(np.max(high[-(window + 1):-1]))
    prior_low = float(np.min(low[-(window + 1):-1]))
    buffer_pct = _breakout_buffer_pct(high, low, close, window)
    signal_close = float(close[-1])

    if signal_close > prior_high * (1.0 + buffer_pct / 100.0):
        patterns.append(_pattern_dict(
            "resistance_breakout",
            _volume_strength(volume, window),
            level=prior_high,
            detail=f"close {signal_close:.6g} > prior {window}-bar high {prior_high:.6g} "
                   f"(+{buffer_pct:.2f}% buffer)",
        ))
    if signal_close < prior_low * (1.0 - buffer_pct / 100.0):
        patterns.append(_pattern_dict(
            "support_breakdown",
            _volume_strength(volume, window),
            level=prior_low,
            detail=f"close {signal_close:.6g} < prior {window}-bar low {prior_low:.6g} "
                   f"(-{buffer_pct:.2f}% buffer)",
        ))

    # --- Local extrema for reversal structures ---------------------------
    def _find_extrema(values: np.ndarray, order: int = 3):
        peaks, troughs = [], []
        for i in range(order, len(values) - order):
            seg = values[i - order:i + order + 1]
            if values[i] == seg.max():
                peaks.append((i, float(values[i])))
            if values[i] == seg.min():
                troughs.append((i, float(values[i])))
        return peaks, troughs

    peaks, troughs = _find_extrema(close, order=3)
    recent_cutoff = len(close) - window  # formation must touch the last window

    # Double bottom: two similar troughs, second one recent, close confirmed
    # above the neckline (the high between the troughs).
    for a in range(len(troughs) - 1):
        i1, low1 = troughs[a]
        i2, low2 = troughs[a + 1]
        if i2 < recent_cutoff or i2 - i1 < 3:
            continue
        if abs(low2 - low1) / max(low1, 1e-12) >= 0.02:
            continue
        neckline = float(np.max(close[i1:i2 + 1]))
        if signal_close > neckline:
            patterns.append(_pattern_dict(
                "double_bottom", _volume_strength(volume, window),
                level=min(low1, low2),
                detail=f"lows {low1:.6g}/{low2:.6g}, neckline {neckline:.6g} confirmed",
            ))
            break

    # Double top: mirrored, confirmed below the neckline.
    for a in range(len(peaks) - 1):
        i1, high1 = peaks[a]
        i2, high2 = peaks[a + 1]
        if i2 < recent_cutoff or i2 - i1 < 3:
            continue
        if abs(high2 - high1) / max(high1, 1e-12) >= 0.02:
            continue
        neckline = float(np.min(close[i1:i2 + 1]))
        if signal_close < neckline:
            patterns.append(_pattern_dict(
                "double_top", _volume_strength(volume, window),
                level=max(high1, high2),
                detail=f"highs {high1:.6g}/{high2:.6g}, neckline {neckline:.6g} confirmed",
            ))
            break

    # Head & shoulders: three recent peaks, similar shoulders, higher head,
    # close below the trough-based neckline.
    if len(peaks) >= 3:
        left, head, right = peaks[-3], peaks[-2], peaks[-1]
        if right[0] >= recent_cutoff:
            shoulder_diff = abs(left[1] - right[1]) / max(left[1], right[1])
            head_up = (head[1] - max(left[1], right[1])) / head[1]
            if shoulder_diff < 0.03 and head_up > 0.03:
                neckline = float(np.min(close[left[0]:right[0] + 1]))
                if signal_close < neckline:
                    patterns.append(_pattern_dict(
                        "head_and_shoulders",
                        'strong' if head_up > 0.06 else 'medium',
                        level=head[1],
                        detail=f"neckline {neckline:.6g} broken",
                    ))
    if len(troughs) >= 3:
        left, head, right = troughs[-3], troughs[-2], troughs[-1]
        if right[0] >= recent_cutoff:
            shoulder_diff = abs(left[1] - right[1]) / max(left[1], right[1])
            head_down = (min(left[1], right[1]) - head[1]) / min(left[1], right[1])
            if shoulder_diff < 0.03 and head_down > 0.03:
                neckline = float(np.max(close[left[0]:right[0] + 1]))
                if signal_close > neckline:
                    patterns.append(_pattern_dict(
                        "inverse_head_and_shoulders",
                        'strong' if head_down > 0.06 else 'medium',
                        level=head[1],
                        detail=f"neckline {neckline:.6g} broken",
                    ))

    return patterns


def detect_candlestick_patterns(ohlcv: pd.DataFrame,
                                now_ms: Optional[int] = None) -> List[Dict]:
    """Japanese candlestick patterns on completed candles."""
    df = drop_open_candle(ohlcv, now_ms=now_ms)
    if len(df) < 5 or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        return []

    patterns: List[Dict] = []
    o = df['open'].to_numpy(dtype=float)
    h = df['high'].to_numpy(dtype=float)
    l = df['low'].to_numpy(dtype=float)
    c = df['close'].to_numpy(dtype=float)

    body = abs(c[-1] - o[-1])
    lower_shadow = min(o[-1], c[-1]) - l[-1]
    upper_shadow = h[-1] - max(o[-1], c[-1])

    if lower_shadow > body * 2 and upper_shadow < body * 0.3 and c[-2] < o[-2]:
        patterns.append(_pattern_dict("hammer", "medium"))

    curr_body = c[-1] - o[-1]
    prev_body = c[-2] - o[-2]
    if prev_body < 0 and curr_body > 0 and o[-1] < c[-2] and c[-1] > o[-2]:
        patterns.append(_pattern_dict("bullish_engulfing", "strong"))
    elif prev_body > 0 and curr_body < 0 and o[-1] > c[-2] and c[-1] < o[-2]:
        patterns.append(_pattern_dict("bearish_engulfing", "strong"))

    if body < (h[-1] - l[-1]) * 0.1:
        patterns.append(_pattern_dict("doji", "weak"))

    if upper_shadow > body * 2 and lower_shadow < body * 0.3 and c[-2] > o[-2]:
        patterns.append(_pattern_dict("shooting_star", "medium"))

    prev_body_abs = abs(c[-2] - o[-2])
    curr_body_abs = abs(c[-1] - o[-1])
    if curr_body_abs < prev_body_abs * 0.6:
        if c[-2] < o[-2] and c[-1] > o[-1] and c[-1] < o[-2] and o[-1] > c[-2]:
            patterns.append(_pattern_dict("bullish_harami", "medium"))
        if c[-2] > o[-2] and c[-1] < o[-1] and c[-1] > o[-2] and o[-1] < c[-2]:
            patterns.append(_pattern_dict("bearish_harami", "medium"))

    a_open, a_close = o[-3], c[-3]
    b_open, b_close = o[-2], c[-2]
    c_open, c_close = o[-1], c[-1]
    body1 = abs(a_close - a_open)
    body2 = abs(b_close - b_open)
    if a_close < a_open and body2 < body1 * 0.5 and c_close > c_open and c_close > (a_open - body1 * 0.5):
        patterns.append(_pattern_dict("morning_star", "strong"))
    if a_close > a_open and body2 < body1 * 0.5 and c_close < c_open and c_close < (a_open + body1 * 0.5):
        patterns.append(_pattern_dict("evening_star", "strong"))

    prev_bear = c[-2] < o[-2]
    prev_bull = c[-2] > o[-2]
    mid_prev = (o[-2] + c[-2]) / 2
    if prev_bear and c[-1] > o[-1] and o[-1] < l[-2] and c[-1] > mid_prev:
        patterns.append(_pattern_dict("piercing_line", "strong"))
    if prev_bull and c[-1] < o[-1] and o[-1] > h[-2] and c[-1] < mid_prev:
        patterns.append(_pattern_dict("dark_cloud_cover", "strong"))

    bullish3 = (c[-3] > o[-3]) and (c[-2] > o[-2]) and (c[-1] > o[-1])
    bearish3 = (c[-3] < o[-3]) and (c[-2] < o[-2]) and (c[-1] < o[-1])
    if bullish3 and c[-1] > c[-2] > c[-3]:
        patterns.append(_pattern_dict("three_white_soldiers", "strong"))
    if bearish3 and c[-1] < c[-2] < c[-3]:
        patterns.append(_pattern_dict("three_black_crows", "strong"))

    return patterns


def detect_chart_formations(ohlcv: pd.DataFrame,
                            now_ms: Optional[int] = None) -> List[Dict]:
    """Triangles, wedges, rectangles, flags on completed candles."""
    df = drop_open_candle(ohlcv, now_ms=now_ms)
    if len(df) < 30:
        return []

    patterns: List[Dict] = []
    close = df['close'].to_numpy(dtype=float)
    high = (df['high'] if 'high' in df.columns else df['close']).to_numpy(dtype=float)
    low = (df['low'] if 'low' in df.columns else df['close']).to_numpy(dtype=float)

    window = min(30, len(df))
    recent_highs = high[-window:]
    recent_lows = low[-window:]

    high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
    low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]

    if abs(high_slope) < (recent_highs.std() * 0.01) and low_slope > 0:
        patterns.append(_pattern_dict("ascending_triangle", "strong",
                                      level=float(recent_highs.max())))
    elif abs(low_slope) < (recent_lows.std() * 0.01) and high_slope < 0:
        patterns.append(_pattern_dict("descending_triangle", "strong",
                                      level=float(recent_lows.min())))
    elif high_slope < 0 and low_slope > 0:
        patterns.append(_pattern_dict("symmetrical_triangle", "medium"))

    range_pct = (recent_highs.max() - recent_lows.min()) / max(recent_lows.min(), 1e-9)
    if abs(high_slope) < (recent_highs.std() * 0.01) and abs(low_slope) < (recent_lows.std() * 0.01) and range_pct < 0.08:
        patterns.append(_pattern_dict("rectangle", "medium"))

    if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
        patterns.append(_pattern_dict("rising_wedge", "medium"))
    if high_slope < 0 and low_slope < 0 and high_slope < low_slope:
        patterns.append(_pattern_dict("falling_wedge", "medium"))

    if len(close) >= 40:
        pre_move = close[-40] - close[-20]
        post_move = close[-1] - close[-20]
        vol = np.std(close[-20:]) / max(close[-20:].mean(), 1e-9)
        if pre_move > 0 and post_move < 0 and vol < 0.02:
            patterns.append(_pattern_dict("bull_flag", "medium",
                                          level=float(recent_highs.max())))
        if pre_move < 0 and post_move > 0 and vol < 0.02:
            patterns.append(_pattern_dict("bear_flag", "medium",
                                          level=float(recent_lows.min())))

    return patterns


def find_support_resistance(ohlcv: pd.DataFrame, sensitivity: float = 0.02) -> Dict[str, List[float]]:
    """Key support/resistance levels from repeatedly tested pivots."""
    if len(ohlcv) < 50:
        return {'support': [], 'resistance': []}

    high = ohlcv['high'] if 'high' in ohlcv.columns else ohlcv['close']
    low = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    resistance_levels = []
    support_levels = []

    for i in range(10, len(high) - 10):
        if high.iloc[i] == high.iloc[i - 10:i + 10].max():
            level = high.iloc[i]
            touches = ((high - level).abs() / level < sensitivity).sum()
            if touches >= 2:
                resistance_levels.append(float(level))
        if low.iloc[i] == low.iloc[i - 10:i + 10].min():
            level = low.iloc[i]
            touches = ((low - level).abs() / level < sensitivity).sum()
            if touches >= 2:
                support_levels.append(float(level))

    def dedupe_levels(levels):
        if not levels:
            return []
        sorted_levels = sorted(set(levels))
        deduped = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            if (level - deduped[-1]) / deduped[-1] > sensitivity:
                deduped.append(level)
        # All levels: callers pick the nearest one on each side of the price.
        # Keeping only the 3 highest dropped every support after a decline.
        return deduped

    return {
        'support': dedupe_levels(support_levels),
        'resistance': dedupe_levels(resistance_levels),
    }

