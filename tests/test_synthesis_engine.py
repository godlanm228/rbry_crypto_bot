"""Synthesis engine v2: canonical results, no targets, honest degradation."""

import asyncio

from core.analysis_result import AnalysisResult
from core.synthesis.ai_synthesis_engine import AISynthesisEngine
from core.synthesis.market_state_vector import (
    DataQuality,
    MarketContext,
    MarketStateVector,
    PatternIndicators,
    SentimentIndicators,
    TechnicalIndicators,
)


def _state(technical=None, sentiment=None, patterns=None, context=None, quality=None):
    return MarketStateVector(
        coin_id="testcoin",
        timestamp=1_784_300_000.0,
        current_price=100.0,
        technical=technical or TechnicalIndicators(),
        sentiment=sentiment or SentimentIndicators(),
        patterns=patterns or PatternIndicators(),
        context=context or MarketContext(),
        quality=quality or DataQuality(),
    )


def _bullish_state():
    quality = DataQuality()
    quality.mark("patterns", True)
    return _state(
        technical=TechnicalIndicators(
            rsi=25.0,
            price_momentum_24h_pct=6.0,
            price_momentum_7d_pct=12.0,
            support_level=95.0,
            support_distance_pct=5.0,
            volatility_annual_pct=40.0,
        ),
        patterns=PatternIndicators(detected_patterns=[
            {"type": "double_bottom", "direction": "bullish",
             "kind": "reversal", "strength": "strong"},
        ]),
        context=MarketContext(trend="bull", volatility_regime="medium"),
        quality=quality,
    )


def _synthesize(state) -> AnalysisResult:
    return asyncio.run(AISynthesisEngine().synthesize(state))


class TestCanonicalResult:
    def test_returns_analysis_result_with_direction_and_quality(self):
        result = _synthesize(_bullish_state())
        assert isinstance(result, AnalysisResult)
        assert result.method == "multi_factor_synthesis"
        assert result.direction == "up"
        assert result.scenario == "positive"  # derived mapping
        assert result.evidence_strength in ("moderate", "strong")
        assert result.horizon is not None
        assert result.quality is not None
        assert result.quality.coverage > 0
        assert "news" in result.quality.missing_sources

    def test_no_targets_no_confidence_anywhere(self):
        result = _synthesize(_bullish_state())
        assert not hasattr(result, "target_price")
        assert not hasattr(result, "stop_loss")
        assert not hasattr(result, "confidence")

    def test_invalidation_uses_verified_support_level(self):
        result = _synthesize(_bullish_state())
        assert result.invalidation is not None
        assert result.invalidation.condition == "close_below_support"
        assert result.invalidation.level == 95.0

    def test_risk_level_from_volatility_regime_not_invented(self):
        result = _synthesize(_bullish_state())
        assert result.risk_level == "medium"
        state = _bullish_state()
        state.context.volatility_regime = None
        assert _synthesize(state).risk_level == "unknown"


class TestHonestDegradation:
    def test_insufficient_evidence_is_unavailable_not_forced_hold(self):
        result = _synthesize(_state(
            technical=TechnicalIndicators(rsi=25.0),  # only RSI available
        ))
        assert result.scenario == "unavailable"
        assert result.direction == "unavailable"
        assert result.horizon is None
        assert not result.is_available

    def test_conflicting_patterns_block_directional_result(self, monkeypatch):
        import core.synthesis.evidence_scorer as evidence_scorer
        # Mechanism test with a weighted pattern component.
        monkeypatch.setitem(evidence_scorer.WEIGHTS, "patterns", 0.27)
        quality = DataQuality()
        quality.mark("patterns", True)
        result = _synthesize(_state(
            technical=TechnicalIndicators(
                rsi=50.0,
                price_momentum_24h_pct=1.0,
                price_momentum_7d_pct=1.0,
            ),
            patterns=PatternIndicators(detected_patterns=[
                {"type": "double_top", "direction": "bearish",
                 "kind": "reversal", "strength": "strong"},
                {"type": "double_bottom", "direction": "bullish",
                 "kind": "reversal", "strength": "strong"},
            ]),
            quality=quality,
        ))
        assert result.direction == "unavailable"
        assert result.quality.conflicts

    def test_missing_news_never_becomes_neutral_sentiment(self):
        result = _synthesize(_bullish_state())
        assert result.sentiment is None  # no news data → no sentiment claim
