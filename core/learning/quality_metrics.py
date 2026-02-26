"""
Quality Metrics Calculator

Calculates granular quality scores for predictions (not binary win/loss).
This prevents the "self-burial" problem where early bad predictions permanently
penalize patterns. Instead, partial success still contributes positively.

Quality metrics (all 0-1 scale):
- Direction correctness: 1.0 if direction correct, 0.0 if wrong, 0.5 if neutral
- Target accuracy: How close price got to target (0-100% of target, continuous)
- Timing accuracy: Exponential decay (faster = better, capped at 7 days)
- Risk management: Based on drawdown control
- Confidence calibration: Predicted confidence vs actual outcome (Brier score)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """
    Granular quality metrics for a prediction

    All metrics are on 0-1 scale (higher = better)
    """
    direction_correctness: float  # 0-1: Did price move in predicted direction?
    target_accuracy: float  # 0-1: How close did price get to target?
    timing_accuracy: float  # 0-1: How quickly was target reached?
    risk_management: float  # 0-1: How well was risk controlled?
    confidence_calibration: float  # 0-1: Was confidence well-calibrated?
    overall_quality: float  # 0-1: Weighted combination of all metrics

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for storage"""
        return {
            'direction_correctness': self.direction_correctness,
            'target_accuracy': self.target_accuracy,
            'timing_accuracy': self.timing_accuracy,
            'risk_management': self.risk_management,
            'confidence_calibration': self.confidence_calibration,
            'overall_quality': self.overall_quality
        }


class QualityMetricsCalculator:
    """
    Calculates quality scores for completed predictions

    Metrics are granular (0-1 continuous) instead of binary (win/loss).
    This allows the system to learn from partial successes and recover
    from early failures.
    """

    # Weights for overall quality score
    METRIC_WEIGHTS = {
        'direction_correctness': 0.30,
        'target_accuracy': 0.25,
        'timing_accuracy': 0.20,
        'risk_management': 0.15,
        'confidence_calibration': 0.10
    }

    # Timing parameters
    OPTIMAL_TIME_DAYS = 7  # Optimal timeframe for target achievement
    TIMEOUT_DAYS = 30  # Maximum time before giving up

    def calculate_direction_correctness(
        self,
        signal_type: str,
        entry_price: float,
        max_price: float,
        min_price: float,
        final_price: float
    ) -> float:
        """
        Calculate direction correctness score

        Args:
            signal_type: 'buy', 'sell', or 'hold'
            entry_price: Entry price of the prediction
            max_price: Maximum price reached after prediction
            min_price: Minimum price reached after prediction
            final_price: Final price at validation time

        Returns:
            Score 0-1:
            - 1.0: Price moved significantly in predicted direction
            - 0.5: Neutral movement (< 2% change)
            - 0.0: Price moved significantly in opposite direction
        """
        if signal_type == 'hold':
            # For hold signal, neutral movement is good
            price_change_pct = abs(final_price - entry_price) / entry_price
            if price_change_pct < 0.02:  # < 2% change
                return 1.0
            elif price_change_pct < 0.05:  # 2-5% change
                return 0.7
            else:
                return 0.4  # > 5% change (hold failed)

        elif signal_type == 'buy':
            # For buy signal, higher price is good
            price_change_pct = (final_price - entry_price) / entry_price

            if price_change_pct > 0.05:  # > 5% gain
                return 1.0
            elif price_change_pct > 0.02:  # 2-5% gain
                return 0.8
            elif price_change_pct > -0.02:  # -2% to 2% (neutral)
                return 0.5
            elif price_change_pct > -0.05:  # -5% to -2% loss
                return 0.2
            else:  # > 5% loss
                return 0.0

        elif signal_type == 'sell':
            # For sell signal, lower price is good (or avoided loss)
            price_change_pct = (final_price - entry_price) / entry_price

            if price_change_pct < -0.05:  # > 5% decline (good sell)
                return 1.0
            elif price_change_pct < -0.02:  # 2-5% decline
                return 0.8
            elif price_change_pct < 0.02:  # -2% to 2% (neutral)
                return 0.5
            elif price_change_pct < 0.05:  # 2-5% gain (missed opportunity)
                return 0.2
            else:  # > 5% gain (bad sell)
                return 0.0

        return 0.5  # Default neutral

    def calculate_target_accuracy(
        self,
        signal_type: str,
        entry_price: float,
        target_price: float,
        max_price: float,
        min_price: float
    ) -> float:
        """
        Calculate target accuracy score

        Args:
            signal_type: 'buy', 'sell', or 'hold'
            entry_price: Entry price
            target_price: Predicted target price
            max_price: Maximum price reached
            min_price: Minimum price reached

        Returns:
            Score 0-1 based on how close price got to target:
            - 1.0: Target fully hit
            - 0.5: Got halfway to target
            - 0.0: Moved in opposite direction
        """
        if signal_type == 'buy':
            # For buy, check if max_price approached target
            target_distance = target_price - entry_price
            if target_distance <= 0:
                return 0.5  # Invalid target

            actual_distance = max_price - entry_price
            accuracy = min(actual_distance / target_distance, 1.0)

            # Clamp to 0-1
            return max(0.0, accuracy)

        elif signal_type == 'sell':
            # For sell, check if min_price approached target (downward)
            target_distance = entry_price - target_price
            if target_distance <= 0:
                return 0.5  # Invalid target

            actual_distance = entry_price - min_price
            accuracy = min(actual_distance / target_distance, 1.0)

            # Clamp to 0-1
            return max(0.0, accuracy)

        else:  # hold
            # For hold, neutral movement is good
            price_moved = max(abs(max_price - entry_price), abs(min_price - entry_price))
            volatility = price_moved / entry_price

            if volatility < 0.02:  # < 2% movement
                return 1.0
            elif volatility < 0.05:  # 2-5% movement
                return 0.7
            else:  # > 5% movement
                return 0.4

    def calculate_timing_accuracy(
        self,
        time_to_target_days: float,
        optimal_time_days: float = None
    ) -> float:
        """
        Calculate timing accuracy with exponential decay

        Args:
            time_to_target_days: Days taken to reach target (or final validation time)
            optimal_time_days: Optimal timeframe (default: 7 days)

        Returns:
            Score 0-1:
            - 1.0: Reached target within optimal time
            - Exponential decay after optimal time
            - ~0.1: Reached at timeout (30 days)
            - 0.0: Timeout exceeded
        """
        if optimal_time_days is None:
            optimal_time_days = self.OPTIMAL_TIME_DAYS

        if time_to_target_days <= 0:
            return 1.0  # Immediate success

        if time_to_target_days <= optimal_time_days:
            # Linear interpolation from 1.0 to 0.8 in optimal window
            return 1.0 - 0.2 * (time_to_target_days / optimal_time_days)

        # Exponential decay after optimal time
        # At 30 days: score ≈ 0.1
        # At 60 days: score ≈ 0.0
        decay_rate = 0.1  # Decay per day
        time_over = time_to_target_days - optimal_time_days
        score = 0.8 * np.exp(-decay_rate * time_over)

        return float(max(0.0, score))

    def calculate_risk_management(
        self,
        entry_price: float,
        stop_loss: float,
        min_price: float,
        signal_type: str
    ) -> float:
        """
        Calculate risk management score

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            min_price: Minimum price reached
            signal_type: 'buy', 'sell', or 'hold'

        Returns:
            Score 0-1:
            - 1.0: Never breached stop-loss
            - 0.5: Breached stop but recovered
            - 0.0: Exceeded stop by >50%
        """
        if signal_type != 'buy':
            # For sell/hold, risk management is less critical
            return 0.8  # Default good score

        # For buy signal, check drawdown vs stop-loss
        stop_loss_pct = (stop_loss - entry_price) / entry_price
        actual_drawdown_pct = (min_price - entry_price) / entry_price

        if actual_drawdown_pct >= 0:
            # No drawdown
            return 1.0

        if actual_drawdown_pct >= stop_loss_pct:
            # Within stop-loss
            # Score based on how close to stop-loss
            distance_ratio = actual_drawdown_pct / stop_loss_pct
            return 1.0 - 0.3 * distance_ratio

        # Breached stop-loss
        breach_amount = abs(actual_drawdown_pct) - abs(stop_loss_pct)
        stop_loss_size = abs(stop_loss_pct)

        if breach_amount < stop_loss_size * 0.2:  # Breached by < 20%
            return 0.5
        elif breach_amount < stop_loss_size * 0.5:  # Breached by 20-50%
            return 0.3
        else:  # Breached by > 50%
            return 0.0

    def calculate_confidence_calibration(
        self,
        predicted_confidence: float,
        actual_quality: float
    ) -> float:
        """
        Calculate confidence calibration score using Brier score

        Args:
            predicted_confidence: Predicted confidence (0-1)
            actual_quality: Actual outcome quality (0-1)

        Returns:
            Score 0-1 based on calibration quality:
            - 1.0: Perfect calibration
            - 0.5: Moderate calibration error
            - 0.0: Completely miscalibrated
        """
        # Brier score: (predicted - actual)^2
        # Lower is better, range is 0 to 1
        brier_score = (predicted_confidence - actual_quality) ** 2

        # Convert to 0-1 scale where 1.0 is best
        # Brier=0 → calibration=1.0, Brier=1 → calibration=0.0
        calibration_score = 1.0 - brier_score

        return float(calibration_score)

    def calculate_overall_quality(
        self,
        metrics: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted overall quality score

        Args:
            metrics: Dict of individual metric scores
            weights: Optional custom weights (defaults to METRIC_WEIGHTS)

        Returns:
            Weighted average quality score (0-1)
        """
        if weights is None:
            weights = self.METRIC_WEIGHTS

        # Calculate weighted sum
        total_score = sum(
            metrics[metric] * weights[metric]
            for metric in weights.keys()
            if metric in metrics
        )

        return float(total_score)

    def calculate_all_metrics(
        self,
        signal_type: str,
        predicted_confidence: float,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        max_price: float,
        min_price: float,
        final_price: float,
        time_to_result_days: float
    ) -> QualityMetrics:
        """
        Calculate all quality metrics for a completed prediction

        Args:
            signal_type: 'buy', 'sell', or 'hold'
            predicted_confidence: Original predicted confidence (0-1)
            entry_price: Entry price
            target_price: Predicted target price
            stop_loss: Stop loss price
            max_price: Maximum price reached
            min_price: Minimum price reached
            final_price: Final price at validation
            time_to_result_days: Days elapsed

        Returns:
            QualityMetrics object with all calculated metrics
        """
        # Calculate individual metrics
        direction = self.calculate_direction_correctness(
            signal_type, entry_price, max_price, min_price, final_price
        )

        target = self.calculate_target_accuracy(
            signal_type, entry_price, target_price, max_price, min_price
        )

        timing = self.calculate_timing_accuracy(time_to_result_days)

        risk = self.calculate_risk_management(
            entry_price, stop_loss, min_price, signal_type
        )

        # For calibration, use direction correctness as proxy for actual quality
        # (In practice, could use target accuracy or overall quality)
        calibration = self.calculate_confidence_calibration(
            predicted_confidence, direction
        )

        # Calculate overall quality
        metrics_dict = {
            'direction_correctness': direction,
            'target_accuracy': target,
            'timing_accuracy': timing,
            'risk_management': risk,
            'confidence_calibration': calibration
        }

        overall = self.calculate_overall_quality(metrics_dict)

        return QualityMetrics(
            direction_correctness=direction,
            target_accuracy=target,
            timing_accuracy=timing,
            risk_management=risk,
            confidence_calibration=calibration,
            overall_quality=overall
        )


# Global singleton
_quality_calculator: Optional[QualityMetricsCalculator] = None


def get_quality_calculator() -> QualityMetricsCalculator:
    """Get global quality metrics calculator instance"""
    global _quality_calculator
    if _quality_calculator is None:
        _quality_calculator = QualityMetricsCalculator()
    return _quality_calculator
