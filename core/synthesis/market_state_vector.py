"""
Market State Vector - Unified data structure for multi-modal AI synthesis

This module defines the core data structures that combine technical indicators,
sentiment data, and pattern information into a single unified vector for AI analysis.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators normalized to comparable scales"""

    # RSI normalized to 0-1 scale (original 0-100 divided by 100)
    rsi_normalized: float

    # Volume ratio relative to average (e.g., 1.5 = 150% of average volume)
    volume_ratio: float

    # Price momentum over different timeframes (-1 to 1 normalized)
    price_momentum_1h: float
    price_momentum_24h: float
    price_momentum_7d: float

    # Volatility metric (0-1 scale, higher = more volatile)
    volatility: float

    # Distance to support/resistance (normalized, negative = below, positive = above)
    support_distance: Optional[float] = None
    resistance_distance: Optional[float] = None

    # Raw RSI value for display purposes
    rsi_raw: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class SentimentIndicators:
    """Sentiment analysis from news and market psychology"""

    # News sentiment score (-1 to 1: negative to positive)
    news_sentiment: float

    # Confidence in news sentiment (0 to 1)
    news_confidence: float

    # Crypto Fear & Greed Index (0-100, or None if unavailable)
    fear_greed_index: Optional[float] = None

    # Number of news articles analyzed
    news_article_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class PatternIndicators:
    """Detected chart patterns and candlestick signals"""

    # List of detected patterns with their types and strengths
    # Format: [{'pattern': 'double_bottom', 'strength': 'strong', 'score': 0.85}, ...]
    detected_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Candlestick signals (e.g., 'hammer', 'doji', 'engulfing')
    candlestick_signals: List[str] = field(default_factory=list)

    # Chart formations (e.g., 'head_and_shoulders', 'triangle', 'wedge')
    chart_formations: List[str] = field(default_factory=list)

    def get_bullish_count(self) -> int:
        """Count bullish patterns"""
        bullish_patterns = ['double_bottom', 'inverse_head_shoulders', 'bull_flag',
                           'ascending_triangle', 'cup_handle', 'hammer', 'bullish_engulfing']
        return sum(1 for p in self.detected_patterns if p.get('pattern') in bullish_patterns)

    def get_bearish_count(self) -> int:
        """Count bearish patterns"""
        bearish_patterns = ['double_top', 'head_shoulders', 'bear_flag',
                           'descending_triangle', 'shooting_star', 'bearish_engulfing']
        return sum(1 for p in self.detected_patterns if p.get('pattern') in bearish_patterns)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class MarketContext:
    """Market regime and environmental context for pattern matching"""

    # Overall market trend: 'bull', 'bear', 'sideways'
    trend: str = 'sideways'

    # Volatility regime: 'low', 'medium', 'high'
    volatility_regime: str = 'medium'

    # Volume regime: 'low', 'medium', 'high'
    volume_regime: str = 'medium'

    # Fear & Greed zone: 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
    fear_greed_zone: str = 'neutral'

    # RSI zone: 'oversold', 'neutral', 'overbought'
    rsi_zone: str = 'neutral'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class MarketStateVector:
    """
    Unified market state vector combining all data sources

    This is the central data structure that feeds into the AI synthesis engine.
    It combines technical indicators, sentiment data, pattern information, and
    market context into a single unified representation.
    """

    # Basic identification
    coin_id: str
    timestamp: float
    current_price: float

    # Component vectors
    technical: TechnicalIndicators
    sentiment: SentimentIndicators
    patterns: PatternIndicators
    context: MarketContext

    # Additional metadata
    exchange: str = 'binance'
    symbol: str = ''

    def to_feature_vector(self) -> List[float]:
        """
        Convert to flat feature vector for ML models

        Returns normalized features in consistent order:
        [rsi, volume_ratio, momentum_1h, momentum_24h, momentum_7d, volatility,
         news_sentiment, news_confidence, fear_greed_normalized,
         bullish_pattern_count, bearish_pattern_count]
        """
        features = [
            self.technical.rsi_normalized,
            min(self.technical.volume_ratio / 3.0, 1.0),  # Cap at 3x for normalization
            (self.technical.price_momentum_1h + 1.0) / 2.0,  # -1 to 1 → 0 to 1
            (self.technical.price_momentum_24h + 1.0) / 2.0,
            (self.technical.price_momentum_7d + 1.0) / 2.0,
            self.technical.volatility,
            (self.sentiment.news_sentiment + 1.0) / 2.0,  # -1 to 1 → 0 to 1
            self.sentiment.news_confidence,
            (self.sentiment.fear_greed_index or 50) / 100.0,  # 0-100 → 0-1
            min(self.patterns.get_bullish_count() / 5.0, 1.0),  # Normalize pattern counts
            min(self.patterns.get_bearish_count() / 5.0, 1.0),
        ]
        return features

    def get_context_text(self) -> str:
        """
        Convert to text context for FinBERT analysis

        Returns a human-readable summary suitable for sentiment analysis models.
        """
        # Build technical summary
        rsi_desc = "oversold" if self.technical.rsi_normalized < 0.35 else \
                   "overbought" if self.technical.rsi_normalized > 0.70 else "neutral"

        vol_desc = "high" if self.technical.volume_ratio > 1.5 else \
                   "low" if self.technical.volume_ratio < 0.8 else "average"

        momentum_desc = "strongly bullish" if self.technical.price_momentum_24h > 0.10 else \
                       "bullish" if self.technical.price_momentum_24h > 0.03 else \
                       "strongly bearish" if self.technical.price_momentum_24h < -0.10 else \
                       "bearish" if self.technical.price_momentum_24h < -0.03 else "neutral"

        # Build sentiment summary
        fear_greed_val = self.sentiment.fear_greed_index
        if fear_greed_val is not None:
            if fear_greed_val < 25:
                fg_desc = "extreme fear"
            elif fear_greed_val < 45:
                fg_desc = "fear"
            elif fear_greed_val < 55:
                fg_desc = "neutral"
            elif fear_greed_val < 75:
                fg_desc = "greed"
            else:
                fg_desc = "extreme greed"
        else:
            fg_desc = "unknown"

        news_desc = "positive" if self.sentiment.news_sentiment > 0.2 else \
                   "negative" if self.sentiment.news_sentiment < -0.2 else "neutral"

        # Build pattern summary
        bullish_patterns = self.patterns.get_bullish_count()
        bearish_patterns = self.patterns.get_bearish_count()

        if bullish_patterns > bearish_patterns:
            pattern_desc = f"{bullish_patterns} bullish patterns detected"
        elif bearish_patterns > bullish_patterns:
            pattern_desc = f"{bearish_patterns} bearish patterns detected"
        else:
            pattern_desc = "mixed pattern signals"

        # Combine into context
        context = f"""
{self.coin_id.upper()} market analysis at ${self.current_price:.2f}:

Technical indicators show {rsi_desc} RSI conditions with {vol_desc} volume.
Price momentum is {momentum_desc} over the past 24 hours.

Market sentiment is {news_desc} based on recent news coverage.
The crypto fear and greed index indicates {fg_desc}.

Chart analysis reveals {pattern_desc}.

Overall market context: {self.context.trend} trend with {self.context.volatility_regime} volatility.
""".strip()

        return context

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire state to dictionary for serialization"""
        return {
            'coin_id': self.coin_id,
            'timestamp': self.timestamp,
            'current_price': self.current_price,
            'technical': self.technical.to_dict(),
            'sentiment': self.sentiment.to_dict(),
            'patterns': self.patterns.to_dict(),
            'context': self.context.to_dict(),
            'exchange': self.exchange,
            'symbol': self.symbol
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketStateVector':
        """Create MarketStateVector from dictionary"""
        return cls(
            coin_id=data['coin_id'],
            timestamp=data['timestamp'],
            current_price=data['current_price'],
            technical=TechnicalIndicators(**data['technical']),
            sentiment=SentimentIndicators(**data['sentiment']),
            patterns=PatternIndicators(**data['patterns']),
            context=MarketContext(**data['context']),
            exchange=data.get('exchange', 'binance'),
            symbol=data.get('symbol', '')
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'MarketStateVector':
        """Create MarketStateVector from JSON string"""
        return cls.from_dict(json.loads(json_str))


def create_empty_market_state(coin_id: str, current_price: float) -> MarketStateVector:
    """
    Create an empty MarketStateVector with default values

    Useful for testing or when some data sources are unavailable.
    """
    return MarketStateVector(
        coin_id=coin_id,
        timestamp=datetime.now().timestamp(),
        current_price=current_price,
        technical=TechnicalIndicators(
            rsi_normalized=0.5,
            volume_ratio=1.0,
            price_momentum_1h=0.0,
            price_momentum_24h=0.0,
            price_momentum_7d=0.0,
            volatility=0.5,
            rsi_raw=50.0
        ),
        sentiment=SentimentIndicators(
            news_sentiment=0.0,
            news_confidence=0.5,
            fear_greed_index=50.0
        ),
        patterns=PatternIndicators(),
        context=MarketContext()
    )
