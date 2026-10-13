"""Evidence scorer — aggregates directional market evidence honestly.

Replaces the former ``ProbabilisticScorer``, which was a fixed weighted
heuristic presented as a probability and silently scored missing inputs as
neutral. Documented rules of this scorer (CONFIG_VERSION):

- every component maps to a directional value on one common scale
  [-1.0 bearish .. +1.0 bullish];
- only AVAILABLE components are weighted; the weight of missing components
  is renormalized away, never replaced by a neutral 0.5;
- coverage below MIN_COVERAGE blocks a directional call entirely;
- |score| below MIN_MARGIN yields "sideways", not a forced buy/sell;
- strong bullish AND bearish pattern evidence at once is a CONFLICT and
  blocks a directional call;
- volume NEVER chooses direction on its own (it only strengthens patterns
  through their strength labels);
- Fear & Greed is regime context, not directional evidence — using it
  identically for buy and sell scores was a known defect of the old scorer;
- the output is an evidence score and strength label, NOT a probability.
  Calibrated probabilities require the prediction ledger (Package A3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .market_state_vector import MarketStateVector

CONFIG_VERSION = "2.0.0"

# patterns: 0 — a walk-forward backtest of every detector (20 coins, 1h and
#   4h, two years, 318k trades, fees and slippage included; audit
#   2026-10-14) found no edge over random entries, while patterns fired on
#   85% of bars and their conflicts blocked the whole verdict. They stay in
#   the evidence notes as chart description, not as directional evidence.
# news: 0 until the news snapshot service (package A2) exists; counting a
#   component that is never available capped coverage and made every coin
#   without Binance candles "unavailable" forever.
WEIGHTS = {
    "rsi": 0.18,
    "momentum": 0.27,
    "patterns": 0.0,
    "news": 0.0,
    "levels": 0.10,
}

MIN_COVERAGE = 0.5     # at least half the evidence weight must be real
MIN_MARGIN = 0.15      # |score| below this is sideways, not a direction
# Momentum scale: a combined move of this size (percent) is full evidence.
MOMENTUM_FULL_SCALE_PCT = 10.0
# Support/resistance proximity that counts as "near the level".
LEVEL_NEAR_PCT = 3.0
PATTERN_CONFLICT_THRESHOLD = 1.0  # strength-weight per side


@dataclass(frozen=True)
class ComponentEvidence:
    name: str
    available: bool
    value: float          # -1..+1, 0.0 when unavailable
    weight: float         # configured weight (before renormalization)
    note: str


@dataclass(frozen=True)
class EvidenceScore:
    direction: str        # 'up' | 'down' | 'sideways' | 'unavailable'
    strength: str         # 'strong' | 'moderate' | 'limited' | 'unavailable'
    score: float          # -1..+1 aggregated over available evidence
    coverage: float       # 0..1 share of evidence weight that was available
    components: tuple[ComponentEvidence, ...]
    conflicts: tuple[str, ...]
    block_reasons: tuple[str, ...]
    config_version: str = CONFIG_VERSION

    @property
    def available_components(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components if c.available)

    @property
    def missing_components(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components if not c.available)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _score_rsi(state: MarketStateVector) -> ComponentEvidence:
    rsi = state.technical.rsi
    if rsi is None:
        return ComponentEvidence("rsi", False, 0.0, WEIGHTS["rsi"], "rsi_unavailable")
    if rsi <= 30:
        value = _clamp((30.0 - rsi) / 15.0, 0.0, 1.0)
        return ComponentEvidence("rsi", True, value, WEIGHTS["rsi"], f"rsi_oversold({rsi:.1f})")
    if rsi >= 70:
        value = -_clamp((rsi - 70.0) / 15.0, 0.0, 1.0)
        return ComponentEvidence("rsi", True, value, WEIGHTS["rsi"], f"rsi_overbought({rsi:.1f})")
    return ComponentEvidence("rsi", True, 0.0, WEIGHTS["rsi"], f"rsi_neutral({rsi:.1f})")


def _score_momentum(state: MarketStateVector) -> ComponentEvidence:
    m24 = state.technical.price_momentum_24h_pct
    m7d = state.technical.price_momentum_7d_pct
    if m24 is None and m7d is None:
        return ComponentEvidence("momentum", False, 0.0, WEIGHTS["momentum"], "momentum_unavailable")
    if m24 is not None and m7d is not None:
        combined = 0.6 * m24 + 0.4 * m7d
        note = f"momentum(24h={m24:+.1f}%,7d={m7d:+.1f}%)"
    elif m24 is not None:
        combined = m24
        note = f"momentum(24h={m24:+.1f}%,7d=missing)"
    else:
        combined = m7d
        note = f"momentum(24h=missing,7d={m7d:+.1f}%)"
    value = _clamp(combined / MOMENTUM_FULL_SCALE_PCT)
    return ComponentEvidence("momentum", True, value, WEIGHTS["momentum"], note)


def _score_patterns(state: MarketStateVector) -> tuple[ComponentEvidence, Optional[str]]:
    patterns_source = state.quality.sources.get("patterns")
    if patterns_source is not None and not patterns_source.available:
        return (
            ComponentEvidence("patterns", False, 0.0, WEIGHTS["patterns"], "patterns_unavailable"),
            None,
        )
    if patterns_source is None and not state.patterns.detected_patterns:
        # No detector ran and nothing was recorded — that is absence of
        # evidence, not evidence of calm.
        return (
            ComponentEvidence("patterns", False, 0.0, WEIGHTS["patterns"], "patterns_unavailable"),
            None,
        )

    bull = state.patterns.bullish_weight()
    bear = state.patterns.bearish_weight()
    conflict = None
    if bull >= PATTERN_CONFLICT_THRESHOLD and bear >= PATTERN_CONFLICT_THRESHOLD:
        conflict = f"patterns_conflict(bullish={bull:.1f},bearish={bear:.1f})"
        return (
            ComponentEvidence("patterns", True, 0.0, WEIGHTS["patterns"], conflict),
            conflict,
        )
    value = _clamp((bull - bear) / 3.0)
    note = f"patterns(bullish={bull:.1f},bearish={bear:.1f})"
    return ComponentEvidence("patterns", True, value, WEIGHTS["patterns"], note), None


def _score_news(state: MarketStateVector) -> ComponentEvidence:
    sentiment = state.sentiment.news_sentiment
    if sentiment is None:
        # The single most important honesty rule: no news data is NOT
        # neutral news. It is missing evidence and reduces coverage.
        return ComponentEvidence("news", False, 0.0, WEIGHTS["news"], "news_unavailable")
    confidence = state.sentiment.news_confidence
    if confidence is None:
        confidence = 0.5
    value = _clamp(sentiment) * _clamp(confidence, 0.0, 1.0)
    note = f"news(sentiment={sentiment:+.2f},articles={state.sentiment.news_article_count})"
    return ComponentEvidence("news", True, value, WEIGHTS["news"], note)


def _score_levels(state: MarketStateVector) -> ComponentEvidence:
    support = state.technical.support_distance_pct
    resistance = state.technical.resistance_distance_pct
    if support is None and resistance is None:
        return ComponentEvidence("levels", False, 0.0, WEIGHTS["levels"], "levels_unavailable")

    value = 0.0
    notes = []
    if support is not None and 0.0 <= support <= LEVEL_NEAR_PCT:
        value += 0.4
        notes.append(f"near_support({support:.1f}%)")
    if resistance is not None and 0.0 <= resistance <= LEVEL_NEAR_PCT:
        value -= 0.4
        notes.append(f"near_resistance({resistance:.1f}%)")
    if not notes:
        notes.append("between_levels")
    return ComponentEvidence("levels", True, _clamp(value), WEIGHTS["levels"], ",".join(notes))


def score_market_state(state: MarketStateVector) -> EvidenceScore:
    """Aggregate all available evidence into one auditable directional score."""
    pattern_component, pattern_conflict = _score_patterns(state)
    components = (
        _score_rsi(state),
        _score_momentum(state),
        pattern_component,
        _score_news(state),
        _score_levels(state),
    )

    # A conflict can only block the verdict for a component that carries weight.
    conflicts = tuple(c for c in (pattern_conflict,) if c and WEIGHTS["patterns"] > 0)
    block_reasons = list(state.quality.block_reasons)

    available = [c for c in components if c.available]
    available_weight = sum(c.weight for c in available)
    coverage = available_weight / sum(WEIGHTS.values())

    if coverage < MIN_COVERAGE:
        block_reasons.append(
            f"insufficient_evidence_coverage({coverage:.0%}<{MIN_COVERAGE:.0%})"
        )
        return EvidenceScore(
            direction="unavailable",
            strength="unavailable",
            score=0.0,
            coverage=coverage,
            components=components,
            conflicts=conflicts,
            block_reasons=tuple(block_reasons),
        )

    if conflicts:
        block_reasons.append("conflicting_directional_evidence")
        return EvidenceScore(
            direction="unavailable",
            strength="unavailable",
            score=0.0,
            coverage=coverage,
            components=components,
            conflicts=conflicts,
            block_reasons=tuple(block_reasons),
        )

    # Renormalize over available weight only.
    score = sum(c.value * c.weight for c in available) / available_weight

    if abs(score) < MIN_MARGIN:
        return EvidenceScore(
            direction="sideways",
            strength="limited",
            score=score,
            coverage=coverage,
            components=components,
            conflicts=conflicts,
            block_reasons=tuple(block_reasons),
        )

    direction = "up" if score > 0 else "down"
    magnitude = abs(score)
    if magnitude >= 0.5:
        strength = "strong"
    elif magnitude >= 0.3:
        strength = "moderate"
    else:
        strength = "limited"

    return EvidenceScore(
        direction=direction,
        strength=strength,
        score=score,
        coverage=coverage,
        components=components,
        conflicts=conflicts,
        block_reasons=tuple(block_reasons),
    )
