"""
Training Database - Persistent storage for self-learning system

SQLite database for storing prediction examples with rich metadata:
- Basic: coin_id, timestamp, prediction details
- Market context: trend, volatility regime, volume regime, Fear & Greed
- Component scores: All probabilistic scores at prediction time
- Outcome: actual results, quality metrics
- Metadata: source (alert/manual/backtest), weight, validation timestamp

Supports context-aware similarity search for feedback loop.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class TrainingDatabase:
    """
    Persistent training database with context-aware indexing

    Schema:
        predictions table:
            - id: Primary key
            - coin_id: Coin identifier
            - timestamp: Prediction timestamp
            - signal_type: 'buy', 'sell', 'hold'
            - confidence: Predicted confidence (0-1)
            - target_price: Predicted target
            - stop_loss: Stop loss price
            - entry_price: Entry price
            - market_context: JSON of market regime
            - component_scores: JSON of all component scores
            - source: 'alert', 'manual', 'price_prediction', 'pattern'
            - weight: Training weight (1.0-3.0)
            - validated: Boolean (has result data)
            - validation_timestamp: When result was recorded
            - max_price: Maximum price reached
            - min_price: Minimum price reached
            - final_price: Final price at validation
            - time_to_result_hours: Hours to result
            - quality_score: Overall quality metric (0-1)
            - quality_metrics: JSON of detailed quality metrics
    """

    DEFAULT_DB_PATH = Path("data/training_examples.db")

    # Training source weights
    SOURCE_WEIGHTS = {
        'automatic_alert': 1.0,
        'manual_validation': 3.0,
        'price_prediction': 2.0,
        'pattern_validation': 1.5
    }

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize training database

        Args:
            db_path: Path to SQLite database (default: data/training_examples.db)
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local connection storage
        self._thread_local = threading.local()

        # Initialize database schema
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._thread_local, 'connection'):
            self._thread_local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._thread_local.connection.row_factory = sqlite3.Row
        return self._thread_local.connection

    def _init_database(self):
        """Initialize database schema and indexes"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                target_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                entry_price REAL NOT NULL,
                market_context TEXT NOT NULL,
                component_scores TEXT NOT NULL,
                source TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                validated INTEGER NOT NULL DEFAULT 0,
                validation_timestamp REAL,
                max_price REAL,
                min_price REAL,
                final_price REAL,
                time_to_result_hours REAL,
                quality_score REAL,
                quality_metrics TEXT
            )
        """)

        # Create indexes for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_coin_timestamp
            ON predictions(coin_id, timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_validated_quality
            ON predictions(validated, quality_score DESC, timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_type
            ON predictions(signal_type, validated, timestamp DESC)
        """)

        conn.commit()
        logger.info(f"Training database initialized at {self.db_path}")

    def save_prediction(
        self,
        coin_id: str,
        timestamp: float,
        signal_type: str,
        confidence: float,
        target_price: float,
        stop_loss: float,
        entry_price: float,
        market_context: Dict,
        component_scores: Dict[str, float],
        source: str = 'automatic_alert'
    ) -> int:
        """
        Save new prediction to database

        Args:
            coin_id: Coin identifier
            timestamp: Prediction timestamp
            signal_type: 'buy', 'sell', 'hold'
            confidence: Predicted confidence
            target_price: Target price
            stop_loss: Stop loss price
            entry_price: Entry price
            market_context: Market regime context dict
            component_scores: Component scores dict
            source: Training source type

        Returns:
            prediction_id: Database ID of saved prediction
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get source weight
        weight = self.SOURCE_WEIGHTS.get(source, 1.0)

        cursor.execute("""
            INSERT INTO predictions (
                coin_id, timestamp, signal_type, confidence,
                target_price, stop_loss, entry_price,
                market_context, component_scores,
                source, weight, validated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            coin_id,
            timestamp,
            signal_type,
            confidence,
            target_price,
            stop_loss,
            entry_price,
            json.dumps(market_context),
            json.dumps(component_scores),
            source,
            weight
        ))

        conn.commit()
        prediction_id = cursor.lastrowid

        logger.info(f"Saved prediction {prediction_id} for {coin_id}: {signal_type} @ {entry_price}")
        return prediction_id

    def update_result(
        self,
        prediction_id: int,
        max_price: float,
        min_price: float,
        final_price: float,
        time_to_result_hours: float,
        quality_score: float,
        quality_metrics: Dict[str, float]
    ):
        """
        Update prediction with actual results and quality metrics

        Args:
            prediction_id: Database ID of prediction
            max_price: Maximum price reached
            min_price: Minimum price reached
            final_price: Final price at validation
            time_to_result_hours: Hours elapsed
            quality_score: Overall quality metric
            quality_metrics: Detailed quality metrics dict
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        validation_timestamp = datetime.now().timestamp()

        cursor.execute("""
            UPDATE predictions SET
                validated = 1,
                validation_timestamp = ?,
                max_price = ?,
                min_price = ?,
                final_price = ?,
                time_to_result_hours = ?,
                quality_score = ?,
                quality_metrics = ?
            WHERE id = ?
        """, (
            validation_timestamp,
            max_price,
            min_price,
            final_price,
            time_to_result_hours,
            quality_score,
            json.dumps(quality_metrics),
            prediction_id
        ))

        conn.commit()
        logger.info(f"Updated prediction {prediction_id} with quality score {quality_score:.3f}")

    def query_similar(
        self,
        coin_id: str,
        signal_type: str,
        market_context: Dict,
        lookback_days: int = 90,
        min_quality: float = 0.0,
        limit: int = 50
    ) -> List[Dict]:
        """
        Query similar predictions with context-aware filtering

        Args:
            coin_id: Coin identifier
            signal_type: 'buy', 'sell', or 'hold'
            market_context: Market regime context for similarity matching
            lookback_days: Days to look back
            min_quality: Minimum quality score filter
            limit: Maximum results

        Returns:
            List of similar prediction dicts with similarity scores
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Calculate timestamp threshold
        lookback_seconds = lookback_days * 86400
        timestamp_threshold = datetime.now().timestamp() - lookback_seconds

        # Query validated predictions
        cursor.execute("""
            SELECT * FROM predictions
            WHERE coin_id = ?
            AND signal_type = ?
            AND validated = 1
            AND timestamp >= ?
            AND quality_score >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (coin_id, signal_type, timestamp_threshold, min_quality, limit * 2))

        rows = cursor.fetchall()

        # Calculate context similarity and filter
        similar_predictions = []

        for row in rows:
            pred_context = json.loads(row['market_context'])

            # Calculate context similarity (0-1)
            similarity = self._calculate_context_similarity(market_context, pred_context)

            if similarity >= 0.4:  # Minimum 40% context match
                pred_dict = dict(row)
                pred_dict['market_context'] = pred_context
                pred_dict['component_scores'] = json.loads(row['component_scores'])
                if row['quality_metrics']:
                    pred_dict['quality_metrics'] = json.loads(row['quality_metrics'])
                pred_dict['context_similarity'] = similarity

                similar_predictions.append(pred_dict)

        # Sort by similarity * quality (most relevant first)
        similar_predictions.sort(
            key=lambda x: x['context_similarity'] * x['quality_score'],
            reverse=True
        )

        return similar_predictions[:limit]

    def get_pending_predictions(self, min_age_hours: float = 6, limit: int = 100) -> List[Dict]:
        """
        Return unvalidated predictions older than min_age_hours.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        ts_threshold = datetime.now().timestamp() - (min_age_hours * 3600)
        cursor.execute(
            """
            SELECT * FROM predictions
            WHERE validated = 0 AND timestamp <= ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (ts_threshold, limit),
        )
        rows = cursor.fetchall()
        pending = []
        for row in rows:
            d = dict(row)
            try:
                d['market_context'] = json.loads(d.get('market_context') or "{}")
            except Exception:
                d['market_context'] = {}
            try:
                d['component_scores'] = json.loads(d.get('component_scores') or "{}")
            except Exception:
                d['component_scores'] = {}
            pending.append(d)
        return pending

    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """
        Calculate similarity between two market contexts (0-1)

        Considers:
        - Trend match (bull/bear/sideways)
        - Volatility regime match
        - Volume regime match
        - Fear & Greed zone proximity
        - RSI zone match
        """
        similarity_scores = []

        # Trend match (40% weight)
        if context1.get('trend') == context2.get('trend'):
            similarity_scores.append((1.0, 0.4))
        else:
            similarity_scores.append((0.0, 0.4))

        # Volatility regime (20% weight)
        if context1.get('volatility_regime') == context2.get('volatility_regime'):
            similarity_scores.append((1.0, 0.2))
        elif self._adjacent_regime(context1.get('volatility_regime'), context2.get('volatility_regime')):
            similarity_scores.append((0.5, 0.2))
        else:
            similarity_scores.append((0.0, 0.2))

        # Volume regime (15% weight)
        if context1.get('volume_regime') == context2.get('volume_regime'):
            similarity_scores.append((1.0, 0.15))
        elif self._adjacent_regime(context1.get('volume_regime'), context2.get('volume_regime')):
            similarity_scores.append((0.5, 0.15))
        else:
            similarity_scores.append((0.0, 0.15))

        # Fear & Greed zone (15% weight)
        if context1.get('fear_greed_zone') == context2.get('fear_greed_zone'):
            similarity_scores.append((1.0, 0.15))
        elif self._adjacent_fear_greed(context1.get('fear_greed_zone'), context2.get('fear_greed_zone')):
            similarity_scores.append((0.5, 0.15))
        else:
            similarity_scores.append((0.0, 0.15))

        # RSI zone (10% weight)
        if context1.get('rsi_zone') == context2.get('rsi_zone'):
            similarity_scores.append((1.0, 0.1))
        else:
            similarity_scores.append((0.5, 0.1))

        # Weighted average
        total_similarity = sum(score * weight for score, weight in similarity_scores)
        return total_similarity

    def _adjacent_regime(self, regime1: Optional[str], regime2: Optional[str]) -> bool:
        """Check if two regimes are adjacent (e.g., low-medium, medium-high)"""
        if not regime1 or not regime2:
            return False

        regimes = ['low', 'medium', 'high']
        try:
            idx1 = regimes.index(regime1)
            idx2 = regimes.index(regime2)
            return abs(idx1 - idx2) == 1
        except ValueError:
            return False

    def _adjacent_fear_greed(self, zone1: Optional[str], zone2: Optional[str]) -> bool:
        """Check if two fear/greed zones are adjacent"""
        if not zone1 or not zone2:
            return False

        zones = ['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
        try:
            idx1 = zones.index(zone1)
            idx2 = zones.index(zone2)
            return abs(idx1 - idx2) == 1
        except ValueError:
            return False

    def get_statistics(self, coin_id: Optional[str] = None, days: int = 30) -> Dict:
        """
        Get training database statistics

        Args:
            coin_id: Optional coin filter
            days: Lookback period

        Returns:
            Dict with statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        timestamp_threshold = datetime.now().timestamp() - (days * 86400)

        where_clause = "WHERE timestamp >= ?"
        params = [timestamp_threshold]

        if coin_id:
            where_clause += " AND coin_id = ?"
            params.append(coin_id)

        # Total predictions
        cursor.execute(f"SELECT COUNT(*) as count FROM predictions {where_clause}", params)
        total = cursor.fetchone()['count']

        # Validated predictions
        cursor.execute(
            f"SELECT COUNT(*) as count FROM predictions {where_clause} AND validated = 1",
            params
        )
        validated = cursor.fetchone()['count']

        # Average quality
        cursor.execute(
            f"SELECT AVG(quality_score) as avg_quality FROM predictions {where_clause} AND validated = 1",
            params
        )
        avg_quality = cursor.fetchone()['avg_quality'] or 0.0

        # By signal type
        cursor.execute(
            f"""
            SELECT signal_type, COUNT(*) as count, AVG(quality_score) as avg_quality
            FROM predictions {where_clause} AND validated = 1
            GROUP BY signal_type
            """,
            params
        )
        by_signal = {row['signal_type']: {'count': row['count'], 'avg_quality': row['avg_quality']}
                    for row in cursor.fetchall()}

        return {
            'total_predictions': total,
            'validated_predictions': validated,
            'validation_rate': validated / total if total > 0 else 0,
            'avg_quality_score': avg_quality,
            'by_signal_type': by_signal,
            'lookback_days': days
        }

    def close(self):
        """Close database connection"""
        if hasattr(self._thread_local, 'connection'):
            self._thread_local.connection.close()
            delattr(self._thread_local, 'connection')


# Global singleton
_training_db: Optional[TrainingDatabase] = None


def get_training_database(db_path: Optional[Path] = None) -> TrainingDatabase:
    """Get global training database instance"""
    global _training_db
    if _training_db is None:
        _training_db = TrainingDatabase(db_path)
    return _training_db


def cleanup_training_database():
    """Cleanup global training database (call on shutdown)"""
    global _training_db
    if _training_db:
        _training_db.close()
        _training_db = None
