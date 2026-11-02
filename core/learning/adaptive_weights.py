"""
Adaptive Weight Manager

Dynamically adjusts component weights based on predictive performance.
Learns which indicators (RSI, volume, momentum, etc.) are most useful
for predictions by analyzing correlation between scores and outcomes.

Uses exponential moving average (EMA) with safety constraints to
prevent extreme weight changes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveWeightManager:
    """
    Adaptive component weight management

    Initial weights: RSI 15%, Volume 12%, Momentum 18%, Patterns 15%,
                    Sentiment 20%, Fear & Greed 10%, Support/Resistance 10%

    Learning process:
    1. After each prediction validation, analyze correlation between
       component scores and quality outcome
    2. Compute component contribution: How much did each component
       "push" toward the correct decision?
    3. Adjust weights using exponential moving average (EMA) with learning rate 0.05
    4. Normalize to ensure weights sum to 1.0

    Safety constraints:
    - Min weight: 0.05 (5%)
    - Max weight: 0.40 (40%)
    - Revert if quality drops >10% over 30-day window
    """

    # Initial default weights
    DEFAULT_WEIGHTS = {
        'rsi': 0.15,
        'volume': 0.12,
        'momentum': 0.18,
        'patterns': 0.15,
        'sentiment': 0.20,
        'fear_greed': 0.10,
        'support_resistance': 0.10
    }

    # Safety constraints
    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 0.40
    LEARNING_RATE = 0.05
    QUALITY_THRESHOLD_DROP = 0.10  # 10% drop triggers revert

    def __init__(self, weights_file: Optional[Path] = None):
        """
        Initialize adaptive weight manager

        Args:
            weights_file: Path to weights JSON file (default: data/adaptive_weights.json)
        """
        self.weights_file = weights_file or Path("data/adaptive_weights.json")
        self.weights_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing weights or use defaults
        self.weights = self._load_weights()
        self.weight_history: List[Dict] = []
        self.baseline_quality: Optional[float] = None

    def _load_weights(self) -> Dict[str, float]:
        """Load weights from file or return defaults"""
        if self.weights_file.exists():
            try:
                with open(self.weights_file, 'r') as f:
                    data = json.load(f)
                    weights = data.get('current_weights', self.DEFAULT_WEIGHTS.copy())
                    self.weight_history = data.get('history', [])
                    self.baseline_quality = data.get('baseline_quality')

                    logger.info(f"Loaded adaptive weights from {self.weights_file}")
                    return weights
            except Exception as e:
                logger.warning(f"Error loading weights file: {e!r}, using defaults")

        return self.DEFAULT_WEIGHTS.copy()

    def _save_weights(self):
        """Save current weights to file"""
        try:
            data = {
                'current_weights': self.weights,
                'history': self.weight_history[-100:],  # Keep last 100 entries
                'baseline_quality': self.baseline_quality,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.weights_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved adaptive weights to {self.weights_file}")

        except Exception as e:
            logger.error(f"Error saving weights: {e!r}")

    def update_weights(
        self,
        component_scores: Dict[str, float],
        actual_quality: float,
        timestamp: Optional[float] = None
    ):
        """
        Update weights based on a single prediction validation

        Args:
            component_scores: Component scores at prediction time
            actual_quality: Actual quality score achieved (0-1)
            timestamp: Optional timestamp for this update
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # Calculate component contributions
        # Components that scored high AND quality was high -> positive contribution
        # Components that scored high BUT quality was low -> negative contribution
        adjustments = {}

        for component, score in component_scores.items():
            if component not in self.weights:
                continue

            # Contribution: correlation between component score and quality
            # Range: -1 to +1
            contribution = (score - 0.5) * (actual_quality - 0.5) * 2

            # Apply learning rate
            adjustment = contribution * self.LEARNING_RATE

            # Update weight with EMA
            current_weight = self.weights[component]
            new_weight = current_weight + adjustment

            # Apply constraints
            new_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))

            adjustments[component] = new_weight - current_weight
            self.weights[component] = new_weight

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # Record in history
        self.weight_history.append({
            'timestamp': timestamp,
            'weights': self.weights.copy(),
            'quality': actual_quality,
            'adjustments': adjustments
        })

        # Save to file
        self._save_weights()

        logger.debug(f"Updated adaptive weights: {self.weights}")

    def batch_update_weights(self, training_examples: List[Dict]):
        """
        Batch update weights from multiple training examples

        Args:
            training_examples: List of dicts with 'component_scores', 'quality_score', 'timestamp'
        """
        if not training_examples:
            logger.warning("No training examples provided for batch update")
            return

        # Calculate average component contributions across all examples
        total_contributions = {comp: 0.0 for comp in self.weights.keys()}
        total_count = len(training_examples)

        for example in training_examples:
            component_scores = example.get('component_scores', {})
            actual_quality = example.get('quality_score', 0.5)

            for component, score in component_scores.items():
                if component not in total_contributions:
                    continue

                contribution = (score - 0.5) * (actual_quality - 0.5) * 2
                total_contributions[component] += contribution

        # Average contributions
        avg_contributions = {comp: total / total_count
                           for comp, total in total_contributions.items()}

        # Apply adjustments
        for component, contribution in avg_contributions.items():
            adjustment = contribution * self.LEARNING_RATE
            current_weight = self.weights[component]
            new_weight = current_weight + adjustment

            # Apply constraints
            new_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))
            self.weights[component] = new_weight

        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # Record in history
        avg_quality = sum(ex.get('quality_score', 0.5) for ex in training_examples) / total_count
        self.weight_history.append({
            'timestamp': datetime.now().timestamp(),
            'weights': self.weights.copy(),
            'quality': avg_quality,
            'batch_size': total_count,
            'avg_contributions': avg_contributions
        })

        self._save_weights()
        logger.info(f"Batch updated weights from {total_count} examples")

    def get_current_weights(self) -> Dict[str, float]:
        """Get current component weights"""
        return self.weights.copy()

    def get_weight_history(self, days: int = 30) -> List[Dict]:
        """
        Get weight evolution history

        Args:
            days: Number of days to look back

        Returns:
            List of weight history entries
        """
        if not self.weight_history:
            return []

        cutoff_timestamp = datetime.now().timestamp() - (days * 86400)

        return [
            entry for entry in self.weight_history
            if entry.get('timestamp', 0) >= cutoff_timestamp
        ]

    def reset_to_defaults(self):
        """Reset weights to defaults (useful if performance degrades)"""
        logger.warning("Resetting adaptive weights to defaults")

        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.weight_history.append({
            'timestamp': datetime.now().timestamp(),
            'weights': self.weights.copy(),
            'action': 'reset_to_defaults',
            'reason': 'Performance degradation or manual reset'
        })

        self._save_weights()

    def validate_performance(self, recent_quality_avg: float) -> bool:
        """
        Validate that adaptive weights are improving performance

        Args:
            recent_quality_avg: Average quality over recent period (e.g., 30 days)

        Returns:
            True if performance is acceptable, False if should reset
        """
        if self.baseline_quality is None:
            # Set baseline on first validation
            self.baseline_quality = recent_quality_avg
            logger.info(f"Set baseline quality: {self.baseline_quality:.3f}")
            return True

        # Check if quality has dropped significantly
        quality_drop = self.baseline_quality - recent_quality_avg

        if quality_drop > self.QUALITY_THRESHOLD_DROP:
            logger.warning(
                f"Quality dropped by {quality_drop:.1%} "
                f"(baseline: {self.baseline_quality:.3f}, current: {recent_quality_avg:.3f})"
            )
            return False

        # Update baseline if quality improved
        if recent_quality_avg > self.baseline_quality:
            self.baseline_quality = recent_quality_avg
            self._save_weights()
            logger.info(f"Updated baseline quality to {self.baseline_quality:.3f}")

        return True

    def get_component_effectiveness(self, component_name: str, days: int = 30) -> Dict:
        """
        Analyze effectiveness of a specific component

        Args:
            component_name: Component to analyze
            days: Lookback period

        Returns:
            Dict with effectiveness metrics
        """
        if not self.weight_history:
            return {
                'component': component_name,
                'current_weight': self.weights.get(component_name, 0.0),
                'data_points': 0
            }

        cutoff_timestamp = datetime.now().timestamp() - (days * 86400)

        # Filter recent history
        recent_history = [
            entry for entry in self.weight_history
            if entry.get('timestamp', 0) >= cutoff_timestamp
        ]

        if not recent_history:
            return {
                'component': component_name,
                'current_weight': self.weights.get(component_name, 0.0),
                'data_points': 0
            }

        # Calculate statistics
        weights = [entry['weights'].get(component_name, 0.0) for entry in recent_history]
        qualities = [entry.get('quality', 0.5) for entry in recent_history]

        return {
            'component': component_name,
            'current_weight': self.weights.get(component_name, 0.0),
            'initial_weight': weights[0] if weights else 0.0,
            'min_weight': min(weights) if weights else 0.0,
            'max_weight': max(weights) if weights else 0.0,
            'avg_weight': np.mean(weights) if weights else 0.0,
            'weight_trend': 'increasing' if len(weights) > 1 and weights[-1] > weights[0] else 'decreasing',
            'avg_quality_when_updated': np.mean(qualities) if qualities else 0.0,
            'data_points': len(recent_history)
        }


# Global singleton
_adaptive_weights_manager: Optional[AdaptiveWeightManager] = None


def get_adaptive_weights_manager(weights_file: Optional[Path] = None) -> AdaptiveWeightManager:
    """Get global adaptive weights manager instance"""
    global _adaptive_weights_manager
    if _adaptive_weights_manager is None:
        _adaptive_weights_manager = AdaptiveWeightManager(weights_file)
    return _adaptive_weights_manager


def get_current_adaptive_weights() -> Dict[str, float]:
    """Get current adaptive weights (convenience function)"""
    manager = get_adaptive_weights_manager()
    return manager.get_current_weights()
