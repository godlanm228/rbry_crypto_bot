"""
Core AI Synthesis Engine Components

This package contains the multi-modal AI synthesis engine that combines
technical indicators, sentiment data, and pattern recognition into unified
probabilistic trading signals.
"""

from .market_state_vector import (
    MarketStateVector,
    TechnicalIndicators,
    SentimentIndicators,
    PatternIndicators,
    MarketContext,
    create_empty_market_state
)

from .probabilistic_scorer import (
    ProbabilisticScorer,
    ProbabilisticSignal,
    get_probabilistic_scorer
)

from .feedback_loop import (
    FeedbackLoop,
    HistoricalContext,
    get_feedback_loop
)

from .ai_synthesis_engine import (
    AISynthesisEngine,
    SynthesizedAnalysis,
    get_ai_synthesis_engine
)

__all__ = [
    'MarketStateVector',
    'TechnicalIndicators',
    'SentimentIndicators',
    'PatternIndicators',
    'MarketContext',
    'create_empty_market_state',
    'ProbabilisticScorer',
    'ProbabilisticSignal',
    'get_probabilistic_scorer',
    'FeedbackLoop',
    'HistoricalContext',
    'get_feedback_loop',
    'AISynthesisEngine',
    'SynthesizedAnalysis',
    'get_ai_synthesis_engine'
]
