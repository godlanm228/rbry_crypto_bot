"""Evidence synthesis components.

The former ProbabilisticScorer (uncalibrated heuristic presented as a
probability) was replaced by the evidence scorer; see evidence_scorer.py
for the documented scoring rules.
"""

from .market_state_vector import (
    DataQuality,
    MarketContext,
    MarketStateVector,
    PatternIndicators,
    SentimentIndicators,
    SourceStatus,
    TechnicalIndicators,
)

from .evidence_scorer import (
    ComponentEvidence,
    EvidenceScore,
    score_market_state,
)

from .feedback_loop import (
    FeedbackLoop,
    HistoricalContext,
    get_feedback_loop,
)

from .ai_synthesis_engine import (
    AISynthesisEngine,
    get_ai_synthesis_engine,
)

__all__ = [
    'AISynthesisEngine',
    'ComponentEvidence',
    'DataQuality',
    'EvidenceScore',
    'FeedbackLoop',
    'HistoricalContext',
    'MarketContext',
    'MarketStateVector',
    'PatternIndicators',
    'SentimentIndicators',
    'SourceStatus',
    'TechnicalIndicators',
    'get_ai_synthesis_engine',
    'get_feedback_loop',
    'score_market_state',
]
