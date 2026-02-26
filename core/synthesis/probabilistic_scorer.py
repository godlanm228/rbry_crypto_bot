"""
Probabilistic Scorer - Replaces hardcoded thresholds with continuous scoring

This module implements probabilistic scoring using sigmoid functions instead of
binary thresholds, producing continuous confidence scores (0-1) for all components.
"""

import numpy as np
import math
try:
    from scipy.special import expit as _expit  # Sigmoid function
    _EXPIT_BACKEND = "scipy"
except Exception:
    def _expit(x):
        try:
            return 1.0 / (1.0 + np.exp(-x))
        except Exception:
            return 1.0 / (1.0 + math.exp(-x))
    _EXPIT_BACKEND = "numpy/math"
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

from .market_state_vector import MarketStateVector, PatternIndicators

logger = logging.getLogger(__name__)
logger.info(f"Probabilistic scorer expit backend: {_EXPIT_BACKEND}")


@dataclass
class ProbabilisticSignal:
    """
    Probabilistic trading signal with continuous confidence

    Attributes:
        signal_type: 'buy', 'sell', or 'hold'
        confidence: Overall confidence (0-1)
        component_scores: Dict of individual component scores
        reasoning: Text explanation of the signal
    """
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1 continuous
    component_scores: Dict[str, float]
    reasoning: str = ""


class ProbabilisticScorer:
    """
    Probabilistic scoring system using sigmoid functions

    Replaces ALL hardcoded thresholds with continuous probability curves:
    - OLD: if RSI < 35: score += 2 (binary)
    - NEW: oversold_score = sigmoid(-(rsi - 30) / 5) (continuous 0-1)

    Component weights (initial, will be adapted by learning):
    - RSI: 15%
    - Volume: 12%
    - Momentum: 18%
    - Patterns: 15%
    - Sentiment: 20%
    - Fear & Greed: 10%
    - Support/Resistance: 10%
    """

    # Component weights (will be overridden by adaptive weights manager)
    WEIGHTS = {
        'rsi': 0.15,
        'volume': 0.12,
        'momentum': 0.18,
        'patterns': 0.15,
        'sentiment': 0.20,
        'fear_greed': 0.10,
        'support_resistance': 0.10
    }

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize probabilistic scorer

        Args:
            custom_weights: Optional dict to override default weights
        """
        if custom_weights:
            self.WEIGHTS = custom_weights.copy()
            # Normalize to ensure sum = 1.0
            total = sum(self.WEIGHTS.values())
            self.WEIGHTS = {k: v / total for k, v in self.WEIGHTS.items()}

    def score_rsi(self, rsi_normalized: float) -> Dict[str, float]:
        """
        Score RSI with continuous oversold/overbought probabilities

        Args:
            rsi_normalized: RSI value normalized to 0-1 scale (divide by 100)

        Returns:
            Dict with 'oversold' and 'overbought' scores (0-1)
        """
        # Convert back to 0-100 scale for sigmoid centering
        rsi = rsi_normalized * 100

        # Oversold: sigmoid centered at RSI=30, steepness=5
        # When RSI=30, score=0.5; RSI<30 increases score toward 1.0
        oversold_score = _expit(-(rsi - 30) / 5)

        # Overbought: sigmoid centered at RSI=70, steepness=5
        # When RSI=70, score=0.5; RSI>70 increases score toward 1.0
        overbought_score = _expit((rsi - 70) / 5)

        return {
            'oversold': float(oversold_score),
            'overbought': float(overbought_score)
        }

    def score_volume(self, volume_ratio: float) -> float:
        """
        Score volume with continuous probability

        Args:
            volume_ratio: Volume / average volume (e.g., 1.5 = 150% of average)

        Returns:
            Score 0-1, where higher volume → higher score
        """
        # Sigmoid threshold at 1.5x average volume
        # volume_ratio=1.5 → score=0.5
        # volume_ratio>1.5 → score increases toward 1.0
        score = _expit((volume_ratio - 1.5) / 0.5)
        return float(score)

    def score_price_momentum(self, momentum_24h: float, momentum_7d: float) -> Dict[str, float]:
        """
        Score price momentum across timeframes

        Args:
            momentum_24h: 24h price change (-1 to 1 normalized)
            momentum_7d: 7d price change (-1 to 1 normalized)

        Returns:
            Dict with 'short_term', 'long_term', and 'combined' scores
        """
        # Short-term (24h): sigmoid centered at 0, steepness=0.1
        # Positive momentum → bullish score, negative → bearish score
        short_term_score = _expit(momentum_24h / 0.1)

        # Long-term (7d): sigmoid centered at 0, steepness=0.15
        long_term_score = _expit(momentum_7d / 0.15)

        # Combined: weighted average (70% short-term, 30% long-term)
        combined_score = 0.7 * short_term_score + 0.3 * long_term_score

        return {
            'short_term': float(short_term_score),
            'long_term': float(long_term_score),
            'combined': float(combined_score)
        }

    def score_patterns(self, patterns: PatternIndicators) -> Dict[str, float]:
        """
        Score detected chart patterns

        Args:
            patterns: PatternIndicators with detected patterns

        Returns:
            Dict with 'bullish', 'bearish', and 'net' scores
        """
        # Pattern strength mapping
        STRENGTH_WEIGHTS = {
            'weak': 0.3,
            'medium': 0.6,
            'strong': 1.0
        }

        bullish_score = 0.0
        bearish_score = 0.0

        bullish_patterns = ['double_bottom', 'inverse_head_shoulders', 'bull_flag',
                           'ascending_triangle', 'cup_handle', 'hammer', 'bullish_engulfing']
        bearish_patterns = ['double_top', 'head_shoulders', 'bear_flag',
                           'descending_triangle', 'shooting_star', 'bearish_engulfing']

        # Aggregate pattern scores
        for pattern_dict in patterns.detected_patterns:
            pattern_name = pattern_dict.get('pattern', '')
            strength = pattern_dict.get('strength', 'medium')
            weight = STRENGTH_WEIGHTS.get(strength, 0.6)

            if pattern_name in bullish_patterns:
                bullish_score += weight
            elif pattern_name in bearish_patterns:
                bearish_score += weight

        # Normalize scores (cap at 5 patterns max for 1.0 score)
        bullish_normalized = min(bullish_score / 3.0, 1.0)
        bearish_normalized = min(bearish_score / 3.0, 1.0)

        # Net score: difference between bullish and bearish, mapped to 0-1
        # (bullish - bearish) ranges from -1 to 1, map to 0 to 1
        net_score = (bullish_normalized - bearish_normalized + 1.0) / 2.0

        return {
            'bullish': float(bullish_normalized),
            'bearish': float(bearish_normalized),
            'net': float(net_score)
        }

    def score_sentiment(self, news_sentiment: float, news_confidence: float,
                       fear_greed: Optional[float]) -> Dict[str, float]:
        """
        Score sentiment from news and Fear & Greed

        Args:
            news_sentiment: News sentiment (-1 to 1)
            news_confidence: Confidence in news sentiment (0-1)
            fear_greed: Fear & Greed Index (0-100) or None

        Returns:
            Dict with 'news', 'fear_greed', and 'combined' scores
        """
        # News sentiment: map -1 to 1 → 0 to 1, weighted by confidence
        news_score = (news_sentiment + 1.0) / 2.0  # -1..1 → 0..1
        news_weighted = news_score * news_confidence  # Weight by confidence

        # Fear & Greed: map 0-100 to 0-1
        if fear_greed is not None:
            fg_score = fear_greed / 100.0
        else:
            fg_score = 0.5  # Neutral if unavailable

        # Combined: 60% news (if confident), 40% fear & greed
        combined_score = 0.6 * news_weighted + 0.4 * fg_score

        return {
            'news': float(news_weighted),
            'fear_greed': float(fg_score),
            'combined': float(combined_score)
        }

    def score_support_resistance(self, support_distance: Optional[float],
                                 resistance_distance: Optional[float]) -> Dict[str, float]:
        """
        Score proximity to support/resistance levels

        Args:
            support_distance: Normalized distance to support (negative = below)
            resistance_distance: Normalized distance to resistance (positive = below)

        Returns:
            Dict with 'near_support' and 'near_resistance' scores
        """
        # Near support: closer to support → higher buy signal
        if support_distance is not None:
            # Sigmoid: distance=0 (at support) → score=0.8, distance=-0.1 (below) → score→1.0
            near_support_score = _expit(-support_distance / 0.05)
        else:
            near_support_score = 0.5  # Neutral if unavailable

        # Near resistance: closer to resistance → higher sell signal
        if resistance_distance is not None:
            # Sigmoid: distance=0 (at resistance) → score=0.8, distance=+0.1 (above) → score→1.0
            near_resistance_score = _expit(resistance_distance / 0.05)
        else:
            near_resistance_score = 0.5  # Neutral if unavailable

        return {
            'near_support': float(near_support_score),
            'near_resistance': float(near_resistance_score)
        }

    def synthesize_signal(self, market_state: MarketStateVector) -> ProbabilisticSignal:
        """
        Synthesize overall signal from all components

        Args:
            market_state: Complete market state vector

        Returns:
            ProbabilisticSignal with signal type, confidence, and component breakdown
        """
        # Score all components
        rsi_scores = self.score_rsi(market_state.technical.rsi_normalized)
        volume_score = self.score_volume(market_state.technical.volume_ratio)
        momentum_scores = self.score_price_momentum(
            market_state.technical.price_momentum_24h,
            market_state.technical.price_momentum_7d
        )
        pattern_scores = self.score_patterns(market_state.patterns)
        sentiment_scores = self.score_sentiment(
            market_state.sentiment.news_sentiment,
            market_state.sentiment.news_confidence,
            market_state.sentiment.fear_greed_index
        )
        sr_scores = self.score_support_resistance(
            market_state.technical.support_distance,
            market_state.technical.resistance_distance
        )

        # Aggregate component scores for BUY signal
        buy_components = {
            'rsi': rsi_scores['oversold'],  # Oversold → BUY
            'volume': volume_score,  # High volume confirms
            'momentum': momentum_scores['combined'],  # Positive momentum
            'patterns': pattern_scores['net'],  # Bullish patterns
            'sentiment': sentiment_scores['combined'],  # Positive sentiment
            'fear_greed': sentiment_scores['fear_greed'],  # Greed/Fear
            'support_resistance': sr_scores['near_support']  # Near support → BUY
        }

        # Aggregate for SELL signal
        sell_components = {
            'rsi': rsi_scores['overbought'],  # Overbought → SELL
            'volume': volume_score,  # High volume confirms
            'momentum': 1.0 - momentum_scores['combined'],  # Negative momentum
            'patterns': 1.0 - pattern_scores['net'],  # Bearish patterns
            'sentiment': 1.0 - sentiment_scores['combined'],  # Negative sentiment
            'fear_greed': sentiment_scores['fear_greed'],  # Could go either way
            'support_resistance': sr_scores['near_resistance']  # Near resistance → SELL
        }

        # Weighted average for BUY signal
        buy_score = sum(buy_components[comp] * self.WEIGHTS[comp]
                       for comp in buy_components.keys())

        # Weighted average for SELL signal
        sell_score = sum(sell_components[comp] * self.WEIGHTS[comp]
                        for comp in sell_components.keys())

        # Determine signal type and confidence
        if buy_score > sell_score:
            signal_type = 'buy'
            confidence = buy_score
            active_components = buy_components
        elif sell_score > buy_score:
            signal_type = 'sell'
            confidence = sell_score
            active_components = sell_components
        else:
            signal_type = 'hold'
            confidence = 0.5
            active_components = {}

        # Generate reasoning
        reasoning = self._generate_reasoning(signal_type, confidence, active_components)

        return ProbabilisticSignal(
            signal_type=signal_type,
            confidence=float(confidence),
            component_scores=active_components,
            reasoning=reasoning
        )

    def _generate_reasoning(self, signal_type: str, confidence: float,
                           components: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the signal"""
        # Sort components by contribution
        sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_components[:3]

        reasoning_parts = []
        for comp_name, score in top_3:
            if score > 0.6:  # Only mention components that significantly contribute
                comp_weight = self.WEIGHTS.get(comp_name, 0)
                contribution = score * comp_weight
                reasoning_parts.append(f"{comp_name}({score:.2f})")

        if reasoning_parts:
            reasoning = f"{signal_type.upper()} signal driven by: {', '.join(reasoning_parts)}"
        else:
            reasoning = f"{signal_type.upper()} signal with mixed indicators"

        return reasoning


# Global singleton (can be overridden with adaptive weights)
_probabilistic_scorer: Optional[ProbabilisticScorer] = None


def get_probabilistic_scorer(custom_weights: Optional[Dict[str, float]] = None) -> ProbabilisticScorer:
    """Get global probabilistic scorer instance"""
    global _probabilistic_scorer
    if _probabilistic_scorer is None or custom_weights is not None:
        _probabilistic_scorer = ProbabilisticScorer(custom_weights)
    return _probabilistic_scorer
