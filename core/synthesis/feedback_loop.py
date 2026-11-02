"""
Feedback Loop - Self-learning system that queries historical performance

This module implements the feedback loop that queries the training database
BEFORE final verdict to adjust confidence based on historical outcomes of
similar patterns in similar market conditions.

Key features:
- Context-aware similarity matching (RSI=30 in bull vs bear is different)
- Gradual quality scoring (not binary win/loss)
- Multi-source training data (automatic, manual, predictions)
- Recency weighting (30-day half-life)
- Adaptive component weight integration
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..learning.training_database import get_training_database
from ..learning.adaptive_weights import get_adaptive_weights_manager
from .market_state_vector import MarketStateVector

logger = logging.getLogger(__name__)


@dataclass
class HistoricalContext:
    """
    Historical context for confidence adjustment

    Attributes:
        similar_signals_count: Number of similar historical signals found
        avg_quality_score: Average quality of similar signals
        confidence_adjustment: Adjustment to apply to raw confidence (-0.25 to +0.25)
        recency_weighted_quality: Quality score weighted by recency
        context_match_quality: How well contexts matched (0-1)
    """
    similar_signals_count: int
    avg_quality_score: float
    confidence_adjustment: float
    recency_weighted_quality: float
    context_match_quality: float
    win_rate: float = 0.0  # For backward compatibility with alerts.py


class FeedbackLoop:
    """
    Self-learning feedback loop

    Queries training database for similar patterns with similar market context
    BEFORE final verdict, adjusts confidence based on historical win rates.

    Problems solved:
    1. No self-burial: Gradual quality scoring (not binary)
    2. Multiple learning sources: Combines alerts, manual validations, price predictions
    3. Context awareness: Bull+high_volume vs bear+low_volume treated differently
    4. Granular metrics: Target accuracy, timing, risk management
    """

    # Configuration
    DEFAULT_LOOKBACK_DAYS = 90
    MIN_SIMILAR_SIGNALS = 5
    RECENCY_HALF_LIFE_DAYS = 30
    MAX_CONFIDENCE_ADJUSTMENT = 0.25
    MIN_CONTEXT_SIMILARITY = 0.4  # 40% context match required

    def __init__(self):
        """Initialize feedback loop"""
        self.training_db = get_training_database()
        self.adaptive_weights = get_adaptive_weights_manager()

    def find_similar_with_context(
        self,
        coin_id: str,
        market_state: MarketStateVector,
        signal_type: str,
        lookback_days: Optional[int] = None
    ) -> List[Dict]:
        """
        Find similar historical signals with context-aware matching

        Args:
            coin_id: Coin identifier
            market_state: Current market state with context
            signal_type: 'buy', 'sell', or 'hold'
            lookback_days: Days to look back (default: 90)

        Returns:
            List of similar prediction dicts with context similarity scores

        Context matching considers:
        - Market trend (bull/bear/sideways)
        - Volatility regime (low/medium/high)
        - Volume regime (low/medium/high)
        - Fear & Greed zone
        - RSI zone
        """
        if lookback_days is None:
            lookback_days = self.DEFAULT_LOOKBACK_DAYS

        # Build market context dict from market_state
        market_context = market_state.context.to_dict()

        # Query database for similar patterns
        similar_predictions = self.training_db.query_similar(
            coin_id=coin_id,
            signal_type=signal_type,
            market_context=market_context,
            lookback_days=lookback_days,
            min_quality=0.0,  # Don't filter by quality here (we need all data)
            limit=50
        )

        logger.debug(
            f"Found {len(similar_predictions)} similar {signal_type} signals for {coin_id} "
            f"in context: {market_context['trend']}/{market_context['volatility_regime']}"
        )

        return similar_predictions

    def compute_confidence_adjustment(
        self,
        similar_examples: List[Dict],
        current_timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute confidence adjustment from similar historical examples

        Args:
            similar_examples: List of similar prediction dicts
            current_timestamp: Current time (default: now)

        Returns:
            Tuple of (confidence_adjustment, weighted_avg_quality)

        Formula:
        - weighted_quality = Σ(quality * recency_weight * source_weight * context_similarity) / Σ(weights)
        - confidence_adjustment = (weighted_quality - 0.5) * 0.3  # Scale: -0.15 to +0.15
        - Clamp to [-0.25, +0.25]
        """
        if not similar_examples:
            return 0.0, 0.5  # Neutral if no data

        if current_timestamp is None:
            current_timestamp = datetime.now().timestamp()

        total_weighted_quality = 0.0
        total_weight = 0.0

        for example in similar_examples:
            quality_score = example['quality_score']
            context_similarity = example.get('context_similarity', 0.5)
            source_weight = example['weight']  # From database (automatic=1.0, manual=3.0, etc.)

            # Calculate recency weight (exponential decay with 30-day half-life)
            age_days = (current_timestamp - example['timestamp']) / 86400
            recency_weight = np.exp(-age_days / self.RECENCY_HALF_LIFE_DAYS)

            # Combined weight
            combined_weight = recency_weight * source_weight * context_similarity

            total_weighted_quality += quality_score * combined_weight
            total_weight += combined_weight

        if total_weight == 0:
            return 0.0, 0.5

        # Calculate weighted average quality
        weighted_avg_quality = total_weighted_quality / total_weight

        # Convert quality score to confidence adjustment
        # Quality 0.8 -> +0.15, Quality 0.5 -> 0.0, Quality 0.2 -> -0.15
        confidence_adjustment = (weighted_avg_quality - 0.5) * 0.3

        # Clamp to prevent extreme adjustments
        confidence_adjustment = np.clip(
            confidence_adjustment,
            -self.MAX_CONFIDENCE_ADJUSTMENT,
            self.MAX_CONFIDENCE_ADJUSTMENT
        )

        return float(confidence_adjustment), float(weighted_avg_quality)

    def get_historical_context(
        self,
        coin_id: str,
        market_state: MarketStateVector,
        signal_type: str,
        lookback_days: Optional[int] = None
    ) -> Optional[HistoricalContext]:
        """
        Get historical context for confidence adjustment

        This is the main entry point called by AI synthesis engine BEFORE final verdict.

        Args:
            coin_id: Coin identifier
            market_state: Current market state
            signal_type: Predicted signal type
            lookback_days: Days to look back

        Returns:
            HistoricalContext with adjustment info, or None if insufficient data
        """
        # Find similar examples
        similar_examples = self.find_similar_with_context(
            coin_id, market_state, signal_type, lookback_days
        )

        if len(similar_examples) < self.MIN_SIMILAR_SIGNALS:
            logger.debug(
                f"Insufficient historical data for {coin_id} {signal_type}: "
                f"{len(similar_examples)} examples (min: {self.MIN_SIMILAR_SIGNALS})"
            )
            return None

        # Compute confidence adjustment
        confidence_adjustment, weighted_quality = self.compute_confidence_adjustment(
            similar_examples
        )

        # Calculate average context match quality
        avg_context_similarity = np.mean([
            ex.get('context_similarity', 0.5) for ex in similar_examples
        ])

        # Calculate simple win rate for backward compatibility
        high_quality_count = sum(1 for ex in similar_examples if ex['quality_score'] >= 0.6)
        win_rate = high_quality_count / len(similar_examples)

        logger.info(
            f"Historical context for {coin_id} {signal_type}: "
            f"{len(similar_examples)} signals, "
            f"avg quality: {weighted_quality:.3f}, "
            f"adjustment: {confidence_adjustment:+.3f}"
        )

        return HistoricalContext(
            similar_signals_count=len(similar_examples),
            avg_quality_score=weighted_quality,
            confidence_adjustment=confidence_adjustment,
            recency_weighted_quality=weighted_quality,
            context_match_quality=float(avg_context_similarity),
            win_rate=win_rate
        )

    def update_component_weights(
        self,
        component_scores: Dict[str, float],
        actual_quality: float
    ):
        """
        Update adaptive component weights based on prediction outcome

        This should be called after each prediction validation.

        Args:
            component_scores: Component scores at prediction time
            actual_quality: Actual quality score achieved
        """
        self.adaptive_weights.update_weights(
            component_scores=component_scores,
            actual_quality=actual_quality
        )

        logger.debug(f"Updated component weights based on quality {actual_quality:.3f}")

    def get_current_adaptive_weights(self) -> Dict[str, float]:
        """Get current adaptive component weights"""
        return self.adaptive_weights.get_current_weights()

    def save_prediction(
        self,
        coin_id: str,
        market_state: MarketStateVector,
        signal_type: str,
        confidence: float,
        target_price: float,
        stop_loss: float,
        component_scores: Dict[str, float],
        source: str = 'automatic_alert'
    ) -> int:
        """
        Save prediction to training database for future learning

        Args:
            coin_id: Coin identifier
            market_state: Market state at prediction time
            signal_type: Predicted signal type
            confidence: Predicted confidence
            target_price: Target price
            stop_loss: Stop loss price
            component_scores: Component scores
            source: Training source type

        Returns:
            prediction_id: Database ID
        """
        prediction_id = self.training_db.save_prediction(
            coin_id=coin_id,
            timestamp=market_state.timestamp,
            signal_type=signal_type,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            entry_price=market_state.current_price,
            market_context=market_state.context.to_dict(),
            component_scores=component_scores,
            source=source
        )

        logger.debug(f"Saved prediction {prediction_id} for {coin_id}: {signal_type} @ {market_state.current_price}")
        return prediction_id

    def validate_prediction(
        self,
        prediction_id: int,
        max_price: float,
        min_price: float,
        final_price: float,
        time_to_result_hours: float,
        quality_metrics: Dict[str, float]
    ):
        """
        Validate prediction with actual results

        This should be called when validating a completed prediction,
        either automatically or manually.

        Args:
            prediction_id: Database ID of prediction
            max_price: Maximum price reached
            min_price: Minimum price reached
            final_price: Final price at validation
            time_to_result_hours: Hours elapsed
            quality_metrics: Dict of quality metrics
        """
        overall_quality = quality_metrics.get('overall_quality', 0.5)

        self.training_db.update_result(
            prediction_id=prediction_id,
            max_price=max_price,
            min_price=min_price,
            final_price=final_price,
            time_to_result_hours=time_to_result_hours,
            quality_score=overall_quality,
            quality_metrics=quality_metrics
        )

        logger.info(f"Validated prediction {prediction_id} with quality {overall_quality:.3f}")

    def get_statistics(self, coin_id: Optional[str] = None, days: int = 30) -> Dict:
        """
        Get training database statistics

        Args:
            coin_id: Optional coin filter
            days: Lookback period

        Returns:
            Dict with statistics
        """
        return self.training_db.get_statistics(coin_id, days)


# Global singleton
_feedback_loop: Optional[FeedbackLoop] = None


def get_feedback_loop() -> FeedbackLoop:
    """Get global feedback loop instance"""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop
