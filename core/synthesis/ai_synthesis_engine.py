"""Synthesis engine - turns a MarketStateVector into the canonical
AnalysisResult through the evidence scorer.

Deliberately removed from the previous version (do not reintroduce):

- ``calculate_targets`` - target/stop prices came from arbitrary
  percentage formulas scaled by an uncalibrated confidence; position
  levels belong to the risk engine with verified invalidation;
- FinBERT inference over self-generated technical prose - classifying our
  own sentence about RSI is not news sentiment;
- confidence adjustment from the feedback loop - the learning ledger has
  zero validated events; adjustments were noise (Package A3 reintroduces
  learning on real data).
"""

from __future__ import annotations

import logging
from typing import Optional

from core.analysis_result import (
    AnalysisResult,
    DIRECTION_TO_SCENARIO,
    DataQualitySummary,
    Invalidation,
)
from .evidence_scorer import EvidenceScore, score_market_state
from .market_state_vector import MarketStateVector

logger = logging.getLogger(__name__)

# The evidence mix (1h candles, 24h/7d momentum, daily history) supports a
# short multi-hour to multi-day view; this is a versioned config statement,
# not a promise of timing precision.
SYNTHESIS_HORIZON = "6h-3d"
ENGINE_VERSION = "2.0.0"


def _risk_level(state: MarketStateVector, evidence: EvidenceScore) -> str:
    # This label is the 14-day realized-volatility regime and is shown as
    # such; it does not rank future drawdowns (audit 2026-10-14), so it must
    # not be presented as a general "risk level".
    regime = state.context.volatility_regime
    if regime == "high":
        return "high"
    if regime == "medium":
        return "medium"
    if regime == "low":
        return "low"
    return "unknown"


def _sentiment_label(state: MarketStateVector) -> Optional[str]:
    sentiment = state.sentiment.news_sentiment
    if sentiment is None:
        return None
    if sentiment > 0.2:
        return "positive"
    if sentiment < -0.2:
        return "negative"
    return "neutral"


def _invalidation(state: MarketStateVector, evidence: EvidenceScore) -> Optional[Invalidation]:
    if evidence.direction == "up" and state.technical.support_level is not None:
        return Invalidation(
            condition="close_below_support",
            level=state.technical.support_level,
            timeframe="1h",
        )
    if evidence.direction == "down" and state.technical.resistance_level is not None:
        return Invalidation(
            condition="close_above_resistance",
            level=state.technical.resistance_level,
            timeframe="1h",
        )
    return None


class AISynthesisEngine:
    """Deterministic evidence synthesis. The LLM explanation layer sits
    elsewhere and never contributes to this result."""

    async def synthesize(self, market_state: MarketStateVector) -> AnalysisResult:
        evidence = score_market_state(market_state)

        quality = DataQualitySummary(
            coverage=evidence.coverage,
            available_sources=evidence.available_components,
            missing_sources=evidence.missing_components,
            conflicts=evidence.conflicts,
            block_reasons=evidence.block_reasons,
        )

        evidence_notes = tuple(
            f"{c.note}[{c.value:+.2f}]" if c.available else c.note
            for c in evidence.components
        )

        result = AnalysisResult(
            scenario=DIRECTION_TO_SCENARIO[evidence.direction],
            risk_level=_risk_level(market_state, evidence),
            method="multi_factor_synthesis",
            sentiment=_sentiment_label(market_state),
            evidence=evidence_notes,
            direction=evidence.direction,
            horizon=SYNTHESIS_HORIZON if evidence.direction != "unavailable" else None,
            evidence_strength=evidence.strength,
            quality=quality,
            invalidation=_invalidation(market_state, evidence),
            market_regime=market_state.context.trend,
            raw_analysis=evidence,
            market_state=market_state,
        )

        logger.info(
            "SYNTHESIS coin=%s direction=%s strength=%s score=%+.3f coverage=%.0f%% "
            "conflicts=%s engine=%s scorer=%s",
            market_state.coin_id,
            result.direction,
            result.evidence_strength,
            evidence.score,
            evidence.coverage * 100,
            len(evidence.conflicts),
            ENGINE_VERSION,
            evidence.config_version,
        )
        return result


_ai_synthesis_engine: Optional[AISynthesisEngine] = None


def get_ai_synthesis_engine() -> AISynthesisEngine:
    global _ai_synthesis_engine
    if _ai_synthesis_engine is None:
        _ai_synthesis_engine = AISynthesisEngine()
    return _ai_synthesis_engine
