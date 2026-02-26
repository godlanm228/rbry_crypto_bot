"""
Market State Builder - Orchestrates data collection from all sources

Builds complete MarketStateVector by collecting data from:
- MarketSnapshotService → current price, volume, market cap
- Binance/historical data → OHLCV for technical indicators
- FearGreedService → market sentiment index
- News aggregator → news sentiment
- Pattern detection → chart patterns

Also determines market context (trend, volatility regime, volume regime).
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from core.synthesis.market_state_vector import (
    MarketStateVector,
    TechnicalIndicators,
    SentimentIndicators,
    PatternIndicators,
    MarketContext
)
from core.data_collectors.fear_greed_service import get_fear_greed_service
from core.analysis.technical_analysis import rsi, sma, realized_volatility
from core.analysis.signal_generator import generate_signals

logger = logging.getLogger(__name__)


class MarketStateBuilder:
    """
    Orchestrates data collection to build complete MarketStateVector

    Usage:
        builder = MarketStateBuilder()
        builder.set_market_snapshot(snapshot_service)
        builder.set_history_snapshot(history_snapshot)
        market_state = await builder.build_market_state(coin_id)
    """

    def __init__(self):
        self.fear_greed_service = get_fear_greed_service()
        self.market_snapshot_service = None
        self.history_snapshot = None

    def set_market_snapshot(self, snapshot_service):
        """Set MarketSnapshotService reference"""
        self.market_snapshot_service = snapshot_service

    def set_history_snapshot(self, history_snapshot):
        """Set HistorySnapshot reference"""
        self.history_snapshot = history_snapshot

    async def build_market_state(self, coin_id: str) -> Optional[MarketStateVector]:
        """
        Build complete MarketStateVector for a coin

        Args:
            coin_id: Coin identifier (e.g., 'bitcoin')

        Returns:
            MarketStateVector or None if data unavailable
        """
        try:
            # Get current price and market data
            current_price, market_data = self._get_current_market_data(coin_id)

            if current_price is None:
                logger.warning(f"No market data available for {coin_id}")
                return None

            # Get historical OHLCV data
            ohlcv_df = self._get_historical_data(coin_id)

            if ohlcv_df is None or len(ohlcv_df) < 14:
                logger.warning(f"Insufficient historical data for {coin_id}")
                # Return minimal market state with defaults
                return self._create_minimal_market_state(coin_id, current_price, market_data)

            # Build components
            technical = self._build_technical_indicators(ohlcv_df, current_price, market_data)
            sentiment = await self._build_sentiment_indicators(coin_id)
            patterns = self._build_pattern_indicators(ohlcv_df)
            context = self._build_market_context(technical, sentiment, ohlcv_df)

            # Create MarketStateVector
            market_state = MarketStateVector(
                coin_id=coin_id,
                timestamp=datetime.now().timestamp(),
                current_price=current_price,
                technical=technical,
                sentiment=sentiment,
                patterns=patterns,
                context=context,
                symbol=market_data.get('symbol', coin_id.upper())
            )

            logger.debug(f"Built market state for {coin_id}: {context.trend}/{context.volatility_regime}")
            return market_state

        except Exception as e:
            logger.error(f"Error building market state for {coin_id}: {e}", exc_info=True)
            return None

    def _get_current_market_data(self, coin_id: str) -> Tuple[Optional[float], Dict]:
        """Get current price and market data from snapshot"""
        if not self.market_snapshot_service:
            return None, {}

        snapshot = self.market_snapshot_service.get_latest()
        if not snapshot or coin_id not in snapshot.markets:
            return None, {}

        market_data = snapshot.markets[coin_id]
        current_price = market_data.get('current_price')

        return current_price, market_data

    def _get_historical_data(self, coin_id: str) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data"""
        if not self.history_snapshot:
            return None

        latest = self.history_snapshot.get_latest()
        if not latest or coin_id not in latest.by_coin:
            return None

        df = latest.by_coin[coin_id]

        # Ensure required columns
        if 'price' in df.columns and 'close' not in df.columns:
            df['close'] = df['price']

        return df

    def _build_technical_indicators(
        self,
        ohlcv_df: pd.DataFrame,
        current_price: float,
        market_data: Dict
    ) -> TechnicalIndicators:
        """
        Build technical indicators from OHLCV data

        Returns TechnicalIndicators with normalized values
        """
        df = ohlcv_df.copy()

        # RSI calculation
        if 'close' in df.columns and len(df) >= 14:
            rsi_series = rsi(df['close'], period=14)
            rsi_val = rsi_series.iloc[-1] if not rsi_series.empty else 50.0
        else:
            rsi_val = 50.0  # Default neutral

        rsi_normalized = rsi_val / 100.0

        # Volume ratio (current vs 20-day average)
        volume_ratio = 1.0
        if 'volume' in df.columns and len(df) >= 20:
            avg_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume

        # Price momentum (percentage changes)
        momentum_1h = 0.0
        momentum_24h = 0.0
        momentum_7d = 0.0

        # 1h momentum
        ch_1h = market_data.get('price_change_percentage_1h_in_currency')
        if ch_1h is not None:
            momentum_1h = ch_1h / 100.0  # Convert to -1 to 1 scale

        # 24h momentum
        ch_24h = market_data.get('price_change_percentage_24h_in_currency') or \
                market_data.get('price_change_percentage_24h', 0)
        if ch_24h:
            momentum_24h = ch_24h / 100.0

        # 7d momentum
        ch_7d = market_data.get('price_change_percentage_7d_in_currency') or \
               market_data.get('price_change_percentage_7d', 0)
        if ch_7d:
            momentum_7d = ch_7d / 100.0

        # Volatility (realized volatility)
        volatility = 0.5  # Default
        if 'close' in df.columns and len(df) >= 14:
            try:
                vol_annual = realized_volatility(df['close'], window=14)
                if not np.isnan(vol_annual):
                    # Normalize: 0.5 = 50% annual volatility (typical crypto range 30-100%)
                    volatility = np.clip(vol_annual / 100.0, 0.0, 1.0)
            except Exception as e:
                logger.debug(f"Volatility calculation error: {e}")

        # Support/Resistance distance (optional, set to None for now)
        support_distance = None
        resistance_distance = None

        return TechnicalIndicators(
            rsi_normalized=rsi_normalized,
            volume_ratio=volume_ratio,
            price_momentum_1h=momentum_1h,
            price_momentum_24h=momentum_24h,
            price_momentum_7d=momentum_7d,
            volatility=volatility,
            support_distance=support_distance,
            resistance_distance=resistance_distance,
            rsi_raw=rsi_val
        )

    async def _build_sentiment_indicators(self, coin_id: str) -> SentimentIndicators:
        """
        Build sentiment indicators from Fear & Greed and news

        Returns SentimentIndicators with sentiment scores
        """
        # Get Fear & Greed Index
        fear_greed_index = await self.fear_greed_service.get_current_index()

        # News sentiment (placeholder - would integrate with news analyzer)
        # For now, use neutral sentiment
        news_sentiment = 0.0  # -1 to 1 scale
        news_confidence = 0.5  # 0 to 1 scale
        news_article_count = 0

        # TODO: Integrate with actual news analyzer
        # from core.analysis.news_analyzer import get_news_sentiment
        # news_sentiment, news_confidence, count = await get_news_sentiment(coin_id)

        return SentimentIndicators(
            news_sentiment=news_sentiment,
            news_confidence=news_confidence,
            fear_greed_index=fear_greed_index,
            news_article_count=news_article_count
        )

    def _build_pattern_indicators(self, ohlcv_df: pd.DataFrame) -> PatternIndicators:
        """
        Build pattern indicators from OHLCV data

        Uses existing signal_generator pattern detection
        """
        detected_patterns = []

        if len(ohlcv_df) >= 20:
            try:
                # Use existing signal generator for pattern detection
                signals = generate_signals(ohlcv_df)

                for signal in signals:
                    # Map signal kinds to pattern format
                    pattern_name = signal.kind
                    strength = signal.strength

                    # Determine if bullish or bearish
                    if signal.kind in ['buy_candidate', 'trend_bullish']:
                        direction = 'bullish'
                    elif signal.kind in ['risk_sell', 'trend_bearish']:
                        direction = 'bearish'
                    else:
                        direction = 'neutral'

                    # Convert strength to score
                    strength_scores = {'weak': 0.3, 'medium': 0.6, 'strong': 0.9}
                    score = strength_scores.get(strength, 0.5)

                    detected_patterns.append({
                        'pattern': pattern_name,
                        'strength': strength,
                        'score': score,
                        'direction': direction,
                        'detail': signal.detail
                    })

            except Exception as e:
                logger.debug(f"Pattern detection error: {e}")

        return PatternIndicators(
            detected_patterns=detected_patterns,
            candlestick_signals=[],
            chart_formations=[]
        )

    def _build_market_context(
        self,
        technical: TechnicalIndicators,
        sentiment: SentimentIndicators,
        ohlcv_df: pd.DataFrame
    ) -> MarketContext:
        """
        Determine market regime and context

        Returns MarketContext with trend, volatility, and volume regimes
        """
        # Determine trend (bull/bear/sideways) using momentum
        momentum_24h = technical.price_momentum_24h
        momentum_7d = technical.price_momentum_7d

        # Simple trend classification
        if momentum_7d > 0.05 and momentum_24h > 0.02:
            trend = 'bull'
        elif momentum_7d < -0.05 and momentum_24h < -0.02:
            trend = 'bear'
        else:
            trend = 'sideways'

        # Determine volatility regime
        volatility = technical.volatility
        if volatility < 0.3:
            volatility_regime = 'low'
        elif volatility < 0.7:
            volatility_regime = 'medium'
        else:
            volatility_regime = 'high'

        # Determine volume regime
        volume_ratio = technical.volume_ratio
        if volume_ratio < 0.8:
            volume_regime = 'low'
        elif volume_ratio < 1.5:
            volume_regime = 'medium'
        else:
            volume_regime = 'high'

        # Determine Fear & Greed zone
        fear_greed = sentiment.fear_greed_index
        if fear_greed is not None:
            if fear_greed < 25:
                fear_greed_zone = 'extreme_fear'
            elif fear_greed < 45:
                fear_greed_zone = 'fear'
            elif fear_greed < 55:
                fear_greed_zone = 'neutral'
            elif fear_greed < 75:
                fear_greed_zone = 'greed'
            else:
                fear_greed_zone = 'extreme_greed'
        else:
            fear_greed_zone = 'neutral'

        # Determine RSI zone
        rsi_val = technical.rsi_raw or 50.0
        if rsi_val < 35:
            rsi_zone = 'oversold'
        elif rsi_val > 65:
            rsi_zone = 'overbought'
        else:
            rsi_zone = 'neutral'

        return MarketContext(
            trend=trend,
            volatility_regime=volatility_regime,
            volume_regime=volume_regime,
            fear_greed_zone=fear_greed_zone,
            rsi_zone=rsi_zone
        )

    def _create_minimal_market_state(
        self,
        coin_id: str,
        current_price: float,
        market_data: Dict
    ) -> MarketStateVector:
        """Create minimal market state when historical data unavailable"""
        from core.synthesis.market_state_vector import create_empty_market_state

        market_state = create_empty_market_state(coin_id, current_price)

        # Add available momentum data
        if market_data:
            ch_24h = market_data.get('price_change_percentage_24h_in_currency') or \
                    market_data.get('price_change_percentage_24h', 0)
            if ch_24h:
                market_state.technical.price_momentum_24h = ch_24h / 100.0

        return market_state


# Global singleton
_market_state_builder: Optional[MarketStateBuilder] = None


def get_market_state_builder() -> MarketStateBuilder:
    """Get global market state builder instance"""
    global _market_state_builder
    if _market_state_builder is None:
        _market_state_builder = MarketStateBuilder()
    return _market_state_builder
