"""Evidence scorer: honest aggregation, coverage gates, conflicts."""

import pytest

from core.synthesis.evidence_scorer import (
    MIN_COVERAGE,
    score_market_state,
)
from core.synthesis.market_state_vector import (
    DataQuality,
    MarketContext,
    MarketStateVector,
    PatternIndicators,
    SentimentIndicators,
    TechnicalIndicators,
)


def _state(technical=None, sentiment=None, patterns=None, quality=None):
    return MarketStateVector(
        coin_id="testcoin",
        timestamp=1_784_300_000.0,
        current_price=100.0,
        technical=technical or TechnicalIndicators(),
        sentiment=sentiment or SentimentIndicators(),
        patterns=patterns or PatternIndicators(),
        context=MarketContext(),
        quality=quality or DataQuality(),
    )


def _pattern(name, direction, strength):
    return {"type": name, "direction": direction, "kind": "reversal", "strength": strength}


def _rich_bullish_state(news_sentiment=None):
    quality = DataQuality()
    quality.mark("patterns", True)
    return _state(
        technical=TechnicalIndicators(
            rsi=25.0,
            price_momentum_24h_pct=6.0,
            price_momentum_7d_pct=12.0,
            support_distance_pct=1.5,
            resistance_distance_pct=20.0,
        ),
        sentiment=SentimentIndicators(
            news_sentiment=news_sentiment,
            news_confidence=0.9 if news_sentiment is not None else None,
            fear_greed_index=50.0,
        ),
        patterns=PatternIndicators(detected_patterns=[
            _pattern("double_bottom", "bullish", "strong"),
        ]),
        quality=quality,
    )


class TestMissingIsNotNeutral:
    def test_missing_news_is_reported_missing_and_excluded(self):
        result = score_market_state(_rich_bullish_state(news_sentiment=None))
        assert "news" in result.missing_components
        assert "news" not in result.available_components
        assert all("news_unavailable" in c.note for c in result.components if c.name == "news")

    def test_present_news_changes_score_missing_news_does_not_fake_it(self):
        without_news = score_market_state(_rich_bullish_state(news_sentiment=None))
        with_news = score_market_state(_rich_bullish_state(news_sentiment=0.8))
        assert with_news.coverage > without_news.coverage
        # Positive news must add to the bullish score; its absence must not
        # be scored as a neutral 0.5 like the old scorer did.
        assert with_news.score > without_news.score

    def test_missing_levels_add_no_weight(self):
        state = _rich_bullish_state()
        state.technical.support_distance_pct = None
        state.technical.resistance_distance_pct = None
        result = score_market_state(state)
        assert "levels" in result.missing_components

    def test_coverage_below_minimum_blocks_direction(self):
        # Only RSI available: coverage 0.18 < 0.5.
        result = score_market_state(_state(
            technical=TechnicalIndicators(rsi=25.0),
        ))
        assert result.direction == "unavailable"
        assert result.strength == "unavailable"
        assert result.coverage < MIN_COVERAGE
        assert any("insufficient_evidence_coverage" in r for r in result.block_reasons)


class TestConflicts:
    def test_strong_double_top_and_double_bottom_blocks_directional_call(self):
        quality = DataQuality()
        quality.mark("patterns", True)
        state = _state(
            technical=TechnicalIndicators(
                rsi=50.0,
                price_momentum_24h_pct=1.0,
                price_momentum_7d_pct=1.0,
            ),
            patterns=PatternIndicators(detected_patterns=[
                _pattern("double_top", "bearish", "strong"),
                _pattern("double_bottom", "bullish", "strong"),
            ]),
            quality=quality,
        )
        result = score_market_state(state)
        assert result.direction == "unavailable"
        assert result.conflicts
        assert "conflicting_directional_evidence" in result.block_reasons


class TestDirectionAndStrength:
    def test_bullish_confluence(self):
        result = score_market_state(_rich_bullish_state(news_sentiment=0.8))
        assert result.direction == "up"
        assert result.strength in ("moderate", "strong")
        assert result.score > 0

    def test_bearish_confluence(self):
        quality = DataQuality()
        quality.mark("patterns", True)
        result = score_market_state(_state(
            technical=TechnicalIndicators(
                rsi=80.0,
                price_momentum_24h_pct=-8.0,
                price_momentum_7d_pct=-15.0,
                resistance_distance_pct=1.0,
                support_distance_pct=25.0,
            ),
            patterns=PatternIndicators(detected_patterns=[
                _pattern("double_top", "bearish", "strong"),
            ]),
            quality=quality,
        ))
        assert result.direction == "down"
        assert result.score < 0

    def test_tiny_margin_is_sideways_not_forced_direction(self):
        quality = DataQuality()
        quality.mark("patterns", True)
        result = score_market_state(_state(
            technical=TechnicalIndicators(
                rsi=50.0,
                price_momentum_24h_pct=0.2,
                price_momentum_7d_pct=-0.1,
            ),
            patterns=PatternIndicators(detected_patterns=[]),
            quality=quality,
        ))
        assert result.direction == "sideways"
        assert result.strength == "limited"


class TestNonDirectionalInputs:
    def test_volume_alone_never_chooses_direction(self):
        quality = DataQuality()
        quality.mark("patterns", True)
        result = score_market_state(_state(
            technical=TechnicalIndicators(
                rsi=50.0,
                volume_ratio=10.0,  # huge spike
                price_momentum_24h_pct=0.0,
                price_momentum_7d_pct=0.0,
            ),
            patterns=PatternIndicators(detected_patterns=[]),
            quality=quality,
        ))
        assert result.direction == "sideways"

    def test_fear_greed_never_enters_the_directional_score(self):
        base = _rich_bullish_state(news_sentiment=0.5)
        greedy = _rich_bullish_state(news_sentiment=0.5)
        greedy.sentiment.fear_greed_index = 95.0
        fearful = _rich_bullish_state(news_sentiment=0.5)
        fearful.sentiment.fear_greed_index = 5.0
        assert score_market_state(base).score == pytest.approx(
            score_market_state(greedy).score
        )
        assert score_market_state(base).score == pytest.approx(
            score_market_state(fearful).score
        )


class TestOutputContract:
    def test_no_probability_language_in_output(self):
        result = score_market_state(_rich_bullish_state())
        assert not hasattr(result, "confidence")
        assert not hasattr(result, "probability")
        assert result.config_version
