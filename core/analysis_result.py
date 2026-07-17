"""Public analysis contract with honest, explicit semantics.

The bot may combine deterministic rules, sentiment classification and an
experimental synthesis engine. None of those outputs is a calibrated market
probability, so this contract intentionally has no ``confidence`` or
``target_price`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


Scenario = Literal["positive", "neutral", "defensive", "unavailable"]
RiskLevel = Literal["low", "medium", "high", "unknown"]
AnalysisMethod = Literal["multi_factor_synthesis", "rule_based"]


@dataclass(frozen=True)
class AnalysisResult:
    scenario: Scenario
    risk_level: RiskLevel
    method: AnalysisMethod
    rule_score: Optional[int] = None
    sentiment: Optional[str] = None
    evidence: tuple[str, ...] = ()
    raw_analysis: Any = field(default=None, repr=False, compare=False)
    market_state: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.rule_score is not None and not 0 <= self.rule_score <= 10:
            raise ValueError("rule_score must be between 0 and 10")

    @property
    def is_available(self) -> bool:
        return self.scenario != "unavailable"

