"""Transparent price-momentum classification used by public market cards."""

from dataclasses import dataclass
from typing import Literal


Direction = Literal["up", "flat", "down", "unavailable"]
Strength = Literal["strong", "moderate", "limited", "unavailable"]


@dataclass(frozen=True)
class RuleSignal:
    translation_key: str
    direction: Direction
    strength: Strength


_RULES = (
    (lambda day, week: day > 5 and week > 10, "analysis.signal_strong_buy", "up", "strong"),
    (lambda day, week: day < -5 and week < -10, "analysis.signal_strong_sell", "down", "strong"),
    (lambda day, week: day > 2 or week > 5, "analysis.signal_buy", "up", "moderate"),
    (lambda day, week: day < -2 or week < -5, "analysis.signal_sell", "down", "moderate"),
)


def classify_price_momentum(change_24h, change_7d) -> RuleSignal:
    if not isinstance(change_24h, (int, float)) or not isinstance(change_7d, (int, float)):
        return RuleSignal("analysis.signal_unavailable", "unavailable", "unavailable")

    for predicate, key, direction, strength in _RULES:
        if predicate(change_24h, change_7d):
            return RuleSignal(key, direction, strength)

    return RuleSignal("analysis.signal_hold", "flat", "limited")

