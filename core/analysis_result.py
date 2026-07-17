"""Public analysis contract with honest, explicit semantics.

The bot may combine deterministic rules, sentiment classification and an
evidence-synthesis engine. None of those outputs is a calibrated market
probability, so this contract intentionally has no ``confidence`` or
``target_price`` fields, and position sizes belong to the risk engine, not
to market analysis.

v2 adds the explicit decision fields required by the shared pipeline:
direction, horizon, evidence strength, data-quality summary, invalidation,
and market regime. ``scenario`` remains the legacy presentation field and is
always derivable from ``direction`` (up→positive, down→defensive,
sideways→neutral, unavailable→unavailable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


Scenario = Literal["positive", "neutral", "defensive", "unavailable"]
RiskLevel = Literal["low", "medium", "high", "unknown"]
AnalysisMethod = Literal["multi_factor_synthesis", "rule_based"]
Direction = Literal["up", "down", "sideways", "unavailable"]
EvidenceStrength = Literal["strong", "moderate", "limited", "unavailable"]

_DIRECTIONS = {"up", "down", "sideways", "unavailable"}
_STRENGTHS = {"strong", "moderate", "limited", "unavailable"}

DIRECTION_TO_SCENARIO = {
    "up": "positive",
    "down": "defensive",
    "sideways": "neutral",
    "unavailable": "unavailable",
}


@dataclass(frozen=True)
class Invalidation:
    """The condition that would change the stated view.

    ``condition`` is machine-readable (e.g. ``close_below:support``);
    ``level`` is the raw price when one applies. Renderers translate this
    into user language — the contract never stores display prose.
    """

    condition: str
    level: Optional[float] = None
    timeframe: Optional[str] = None


@dataclass(frozen=True)
class DataQualitySummary:
    """How much real evidence stands behind the result."""

    coverage: float = 0.0
    available_sources: tuple[str, ...] = ()
    missing_sources: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    block_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisResult:
    scenario: Scenario
    risk_level: RiskLevel
    method: AnalysisMethod
    rule_score: Optional[int] = None
    sentiment: Optional[str] = None
    evidence: tuple[str, ...] = ()

    # v2 decision fields (defaults keep legacy rule-based construction valid)
    direction: Direction = "unavailable"
    horizon: Optional[str] = None
    evidence_strength: EvidenceStrength = "unavailable"
    quality: Optional[DataQualitySummary] = None
    invalidation: Optional[Invalidation] = None
    market_regime: Optional[str] = None

    raw_analysis: Any = field(default=None, repr=False, compare=False)
    market_state: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.rule_score is not None and not 0 <= self.rule_score <= 10:
            raise ValueError("rule_score must be between 0 and 10")
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"invalid direction: {self.direction!r}")
        if self.evidence_strength not in _STRENGTHS:
            raise ValueError(f"invalid evidence_strength: {self.evidence_strength!r}")

    @property
    def is_available(self) -> bool:
        return self.scenario != "unavailable"
