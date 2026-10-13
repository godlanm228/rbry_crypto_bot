"""Transparent price-momentum classification used by public market cards."""

import math
from dataclasses import dataclass
from typing import Literal


Direction = Literal["up", "flat", "down", "unavailable"]
Strength = Literal["strong", "moderate", "limited", "unavailable"]


@dataclass(frozen=True)
class RuleSignal:
    translation_key: str
    direction: Direction
    strength: Strength


def classify_price_momentum(change_24h, change_7d) -> RuleSignal:
    """Describe recent price movement; this is a description, not advice.

    Symmetric rules: when the day and the week point in opposite directions
    (e.g. +2.5% today after -20% this week) the card says "mixed" instead of
    "rising" — the old ordering checked "up" before "down".
    """
    if not isinstance(change_24h, (int, float)) or not isinstance(change_7d, (int, float)):
        return RuleSignal("analysis.signal_unavailable", "unavailable", "unavailable")
    if not (math.isfinite(change_24h) and math.isfinite(change_7d)):
        return RuleSignal("analysis.signal_unavailable", "unavailable", "unavailable")

    day, week = change_24h, change_7d
    if day > 5 and week > 10:
        return RuleSignal("analysis.signal_strong_buy", "up", "strong")
    if day < -5 and week < -10:
        return RuleSignal("analysis.signal_strong_sell", "down", "strong")
    rising = day > 2 or week > 5
    falling = day < -2 or week < -5
    if rising and not falling:
        return RuleSignal("analysis.signal_buy", "up", "moderate")
    if falling and not rising:
        return RuleSignal("analysis.signal_sell", "down", "moderate")
    return RuleSignal("analysis.signal_hold", "flat", "limited")
