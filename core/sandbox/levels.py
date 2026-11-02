"""Support/resistance for replay: the live detector, vectorised.

``core.analysis.signal_generator.find_support_resistance`` costs ~35 ms per
200-candle window; a replay calls it once per coin per day (~90k times).
This is the same rule in numpy and must stay equal to it - the test suite
compares both on random windows.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MIN_CANDLES = 50
PIVOT_HALF = 10
SENSITIVITY = 0.02


def _pivot_levels(values: np.ndarray, is_high: bool, sensitivity: float) -> list[float]:
    n = len(values)
    # pivot i: equal to the max/min of values[i-10 : i+10] (i+10 excluded)
    windows = sliding_window_view(values, 2 * PIVOT_HALF)[: n - 2 * PIVOT_HALF]
    centre = values[PIVOT_HALF: n - PIVOT_HALF]
    extreme = windows.max(axis=1) if is_high else windows.min(axis=1)
    pivots = centre[centre == extreme]
    if pivots.size == 0:
        return []
    touches = (np.abs(values[None, :] - pivots[:, None]) / pivots[:, None] < sensitivity).sum(axis=1)
    return [float(v) for v in pivots[touches >= 2]]


def _dedupe(levels: list[float], sensitivity: float) -> list[float]:
    if not levels:
        return []
    ordered = sorted(set(levels))
    kept = [ordered[0]]
    for level in ordered[1:]:
        if (level - kept[-1]) / kept[-1] > sensitivity:
            kept.append(level)
    return kept


def support_resistance(highs: np.ndarray, lows: np.ndarray,
                       sensitivity: float = SENSITIVITY) -> tuple[list[float], list[float]]:
    """(supports, resistances) exactly as find_support_resistance returns them."""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    if len(highs) < MIN_CANDLES:
        return [], []
    return (_dedupe(_pivot_levels(lows, False, sensitivity), sensitivity),
            _dedupe(_pivot_levels(highs, True, sensitivity), sensitivity))
