"""Central pattern metadata registry.

One canonical name per pattern. Detectors, scorers, trackers, and renderers
must resolve names through this registry instead of keeping their own lists -
that is how `head_and_shoulders` vs `head_shoulders` mismatches happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


REGISTRY_VERSION = "1.0.0"

Direction = str  # "bullish" | "bearish" | "neutral"
Kind = str       # "reversal" | "continuation" | "breakout" | "indecision" | "range"


@dataclass(frozen=True)
class PatternMeta:
    canonical: str
    direction: Direction
    kind: Kind
    translation_key: str


_P = PatternMeta

_REGISTRY = {
    # Chart patterns
    "double_bottom": _P("double_bottom", "bullish", "reversal", "patterns.double_bottom"),
    "double_top": _P("double_top", "bearish", "reversal", "patterns.double_top"),
    "head_and_shoulders": _P("head_and_shoulders", "bearish", "reversal", "patterns.head_and_shoulders"),
    "inverse_head_and_shoulders": _P("inverse_head_and_shoulders", "bullish", "reversal", "patterns.inverse_head_and_shoulders"),
    "resistance_breakout": _P("resistance_breakout", "bullish", "breakout", "patterns.resistance_breakout"),
    "support_breakdown": _P("support_breakdown", "bearish", "breakout", "patterns.support_breakdown"),
    # Formations
    "ascending_triangle": _P("ascending_triangle", "bullish", "continuation", "patterns.ascending_triangle"),
    "descending_triangle": _P("descending_triangle", "bearish", "continuation", "patterns.descending_triangle"),
    "symmetrical_triangle": _P("symmetrical_triangle", "neutral", "range", "patterns.symmetrical_triangle"),
    "rectangle": _P("rectangle", "neutral", "range", "patterns.rectangle"),
    "rising_wedge": _P("rising_wedge", "bearish", "reversal", "patterns.rising_wedge"),
    "falling_wedge": _P("falling_wedge", "bullish", "reversal", "patterns.falling_wedge"),
    "bull_flag": _P("bull_flag", "bullish", "continuation", "patterns.bull_flag"),
    "bear_flag": _P("bear_flag", "bearish", "continuation", "patterns.bear_flag"),
    # Candlesticks
    "hammer": _P("hammer", "bullish", "reversal", "patterns.hammer"),
    "shooting_star": _P("shooting_star", "bearish", "reversal", "patterns.shooting_star"),
    "bullish_engulfing": _P("bullish_engulfing", "bullish", "reversal", "patterns.bullish_engulfing"),
    "bearish_engulfing": _P("bearish_engulfing", "bearish", "reversal", "patterns.bearish_engulfing"),
    "bullish_harami": _P("bullish_harami", "bullish", "reversal", "patterns.bullish_harami"),
    "bearish_harami": _P("bearish_harami", "bearish", "reversal", "patterns.bearish_harami"),
    "morning_star": _P("morning_star", "bullish", "reversal", "patterns.morning_star"),
    "evening_star": _P("evening_star", "bearish", "reversal", "patterns.evening_star"),
    "piercing_line": _P("piercing_line", "bullish", "reversal", "patterns.piercing_line"),
    "dark_cloud_cover": _P("dark_cloud_cover", "bearish", "reversal", "patterns.dark_cloud_cover"),
    "three_white_soldiers": _P("three_white_soldiers", "bullish", "continuation", "patterns.three_white_soldiers"),
    "three_black_crows": _P("three_black_crows", "bearish", "continuation", "patterns.three_black_crows"),
    "doji": _P("doji", "neutral", "indecision", "patterns.doji"),
}

# Historic spellings seen across the codebase and stored data.
_ALIASES = {
    "head_shoulders": "head_and_shoulders",
    "inverse_head_shoulders": "inverse_head_and_shoulders",
    "inv_head_and_shoulders": "inverse_head_and_shoulders",
    "cup_handle": "double_bottom",  # closest honest mapping for legacy name
}


def resolve(name: str) -> Optional[PatternMeta]:
    """Resolve any known name or alias to canonical metadata; None if unknown."""
    if not name:
        return None
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    return _REGISTRY.get(key)


def direction_of(name: str) -> Direction:
    meta = resolve(name)
    return meta.direction if meta else "neutral"


def all_canonical_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
