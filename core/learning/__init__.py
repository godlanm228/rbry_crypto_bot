"""
Self-Learning System Components

This module contains the self-learning feedback loop system that enables
the bot to improve predictions over time by learning from past outcomes.
"""

from .quality_metrics import (
    QualityMetrics,
    QualityMetricsCalculator,
    get_quality_calculator
)

from .training_database import (
    TrainingDatabase,
    get_training_database,
    cleanup_training_database
)

from .adaptive_weights import (
    AdaptiveWeightManager,
    get_adaptive_weights_manager,
    get_current_adaptive_weights
)

__all__ = [
    'QualityMetrics',
    'QualityMetricsCalculator',
    'get_quality_calculator',
    'TrainingDatabase',
    'get_training_database',
    'cleanup_training_database',
    'AdaptiveWeightManager',
    'get_adaptive_weights_manager',
    'get_current_adaptive_weights',
]
