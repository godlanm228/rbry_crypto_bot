"""Market State Vector - unified evidence structure for one coin.

Schema v2 rules:

- A metric that could not be computed is ``None``, never a neutral filler.
  Missing news is missing news; missing levels are missing levels.
- ``DataQuality`` records, per evidence source, whether it was available,
  the as-of time of its newest data point, and its timeframe.
- Serialization carries ``schema_version``; ``from_dict`` migrates v1 blobs
  (which had neutral fillers) by marking their quality unknown.
- There is deliberately no "empty state" factory for production paths: if a
  test needs a synthetic state, it builds one in its own fixtures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.analysis.pattern_registry import direction_of

SCHEMA_VERSION = 2


@dataclass
class SourceStatus:
    """Availability and freshness of one evidence source."""

    available: bool
    as_of: Optional[float] = None      # unix ts of newest underlying data point
    timeframe: Optional[str] = None    # e.g. "1h", "1d"
    detail: Optional[str] = None       # e.g. "binance klines", "history_snapshot"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataQuality:
    """Explicit evidence coverage; blocks conclusions that lack data."""

    sources: Dict[str, SourceStatus] = field(default_factory=dict)
    block_reasons: List[str] = field(default_factory=list)

    @property
    def available_sources(self) -> List[str]:
        return sorted(name for name, s in self.sources.items() if s.available)

    @property
    def missing_sources(self) -> List[str]:
        return sorted(name for name, s in self.sources.items() if not s.available)

    def mark(self, name: str, available: bool, *, as_of: Optional[float] = None,
             timeframe: Optional[str] = None, detail: Optional[str] = None) -> None:
        self.sources[name] = SourceStatus(available, as_of, timeframe, detail)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sources': {k: v.to_dict() for k, v in self.sources.items()},
            'block_reasons': list(self.block_reasons),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataQuality':
        quality = cls()
        for name, raw in (data.get('sources') or {}).items():
            quality.sources[name] = SourceStatus(**raw)
        quality.block_reasons = list(data.get('block_reasons') or [])
        return quality


@dataclass
class TechnicalIndicators:
    """Raw-unit technical metrics; unavailable metrics stay None."""

    rsi: Optional[float] = None                     # 0-100
    volume_ratio: Optional[float] = None            # closed candle vs prior closed mean
    price_momentum_1h_pct: Optional[float] = None   # percent
    price_momentum_24h_pct: Optional[float] = None
    price_momentum_7d_pct: Optional[float] = None
    volatility_annual_pct: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    support_distance_pct: Optional[float] = None    # (price-support)/price*100
    resistance_distance_pct: Optional[float] = None  # (resistance-price)/price*100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SentimentIndicators:
    """News and market-psychology evidence; None means unavailable."""

    news_sentiment: Optional[float] = None      # -1..1, only from real coin news
    news_confidence: Optional[float] = None     # 0..1
    news_article_count: int = 0
    fear_greed_index: Optional[float] = None    # 0-100 market-wide context

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternIndicators:
    """Detected patterns with canonical names and directions."""

    # [{'type': 'double_bottom', 'direction': 'bullish', 'kind': 'reversal',
    #   'strength': 'strong', 'detector_version': '2.0.0', ...}, ...]
    detected_patterns: List[Dict[str, Any]] = field(default_factory=list)

    _STRENGTH_WEIGHT = {'weak': 0.3, 'medium': 0.6, 'strong': 1.0}

    def _directional_weight(self, direction: str) -> float:
        total = 0.0
        for p in self.detected_patterns:
            p_direction = p.get('direction') or direction_of(p.get('type', ''))
            if p_direction == direction:
                total += self._STRENGTH_WEIGHT.get(p.get('strength'), 0.6)
        return total

    def bullish_weight(self) -> float:
        return self._directional_weight('bullish')

    def bearish_weight(self) -> float:
        return self._directional_weight('bearish')

    def to_dict(self) -> Dict[str, Any]:
        return {'detected_patterns': list(self.detected_patterns)}


@dataclass
class MarketContext:
    """Market regime; None means genuinely unknown, not 'assume neutral'."""

    trend: Optional[str] = None              # 'bull' | 'bear' | 'sideways'
    volatility_regime: Optional[str] = None  # 'low' | 'medium' | 'high'
    volume_regime: Optional[str] = None
    fear_greed_zone: Optional[str] = None
    rsi_zone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketStateVector:
    """Unified market evidence for one coin at one moment."""

    coin_id: str
    timestamp: float
    current_price: float

    technical: TechnicalIndicators
    sentiment: SentimentIndicators
    patterns: PatternIndicators
    context: MarketContext
    quality: DataQuality = field(default_factory=DataQuality)

    exchange: str = 'binance'
    symbol: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_version': SCHEMA_VERSION,
            'coin_id': self.coin_id,
            'timestamp': self.timestamp,
            'current_price': self.current_price,
            'technical': self.technical.to_dict(),
            'sentiment': self.sentiment.to_dict(),
            'patterns': self.patterns.to_dict(),
            'context': self.context.to_dict(),
            'quality': self.quality.to_dict(),
            'exchange': self.exchange,
            'symbol': self.symbol,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketStateVector':
        version = data.get('schema_version', 1)
        if version >= 2:
            return cls(
                coin_id=data['coin_id'],
                timestamp=data['timestamp'],
                current_price=data['current_price'],
                technical=TechnicalIndicators(**data.get('technical', {})),
                sentiment=SentimentIndicators(**data.get('sentiment', {})),
                patterns=PatternIndicators(**data.get('patterns', {})),
                context=MarketContext(**data.get('context', {})),
                quality=DataQuality.from_dict(data.get('quality', {})),
                exchange=data.get('exchange', 'binance'),
                symbol=data.get('symbol', ''),
            )
        return cls._from_v1(data)

    @classmethod
    def _from_v1(cls, data: Dict[str, Any]) -> 'MarketStateVector':
        """Migrate a v1 blob. v1 stored neutral fillers for missing values,
        so migrated states are marked with unknown quality rather than
        pretending their fillers were measurements."""
        tech_v1 = data.get('technical', {}) or {}
        sent_v1 = data.get('sentiment', {}) or {}
        patterns_v1 = data.get('patterns', {}) or {}
        context_v1 = data.get('context', {}) or {}

        def _pct(value):
            return value * 100.0 if isinstance(value, (int, float)) else None

        rsi_raw = tech_v1.get('rsi_raw')
        if rsi_raw is None and isinstance(tech_v1.get('rsi_normalized'), (int, float)):
            rsi_raw = tech_v1['rsi_normalized'] * 100.0

        technical = TechnicalIndicators(
            rsi=rsi_raw,
            volume_ratio=tech_v1.get('volume_ratio'),
            price_momentum_1h_pct=_pct(tech_v1.get('price_momentum_1h')),
            price_momentum_24h_pct=_pct(tech_v1.get('price_momentum_24h')),
            price_momentum_7d_pct=_pct(tech_v1.get('price_momentum_7d')),
            volatility_annual_pct=_pct(tech_v1.get('volatility')),
            support_level=None,
            resistance_level=None,
        )
        sentiment = SentimentIndicators(
            # v1 hard-coded news_sentiment=0.0 as a placeholder - that is not
            # evidence, so migrated news stays unavailable.
            news_sentiment=None,
            news_confidence=None,
            news_article_count=int(sent_v1.get('news_article_count') or 0),
            fear_greed_index=sent_v1.get('fear_greed_index'),
        )
        detected = []
        for p in patterns_v1.get('detected_patterns', []) or []:
            name = p.get('pattern') or p.get('type') or ''
            detected.append({
                'type': name,
                'direction': p.get('direction') or direction_of(name),
                'kind': p.get('kind', 'unknown'),
                'strength': p.get('strength', 'medium'),
                'detector_version': 'v1-migrated',
            })
        quality = DataQuality(block_reasons=['migrated_from_schema_v1_without_quality'])

        return cls(
            coin_id=data['coin_id'],
            timestamp=data['timestamp'],
            current_price=data['current_price'],
            technical=technical,
            sentiment=sentiment,
            patterns=PatternIndicators(detected_patterns=detected),
            context=MarketContext(
                trend=context_v1.get('trend'),
                volatility_regime=context_v1.get('volatility_regime'),
                volume_regime=context_v1.get('volume_regime'),
                fear_greed_zone=context_v1.get('fear_greed_zone'),
                rsi_zone=context_v1.get('rsi_zone'),
            ),
            quality=quality,
            exchange=data.get('exchange', 'binance'),
            symbol=data.get('symbol', ''),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'MarketStateVector':
        return cls.from_dict(json.loads(json_str))
