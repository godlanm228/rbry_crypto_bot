"""Market State Builder - collects evidence into a MarketStateVector.

Honesty rules (v2):

- a metric that cannot be computed is None and its source is marked
  unavailable in DataQuality - nothing defaults to neutral;
- rules that need candle closure see only completed candles
  (``drop_open_candle`` on the OHLCV frame's ``close_time``);
- the volume ratio compares the last CLOSED candle against the mean of the
  prior closed candles, excluding the signal candle from its own baseline;
- callers may pass already-fetched OHLCV so a 20-coin scan performs one
  batch fetch instead of 20 (the builder itself never calls the network
  for candles);
- news evidence stays unavailable until the news snapshot service exists
  (Package A2). It is NOT filled with a neutral placeholder;
- there is no "minimal default state": when history is missing the state
  says so explicitly through quality.block_reasons.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config

from core.analysis.signal_generator import (
    detect_candlestick_patterns,
    detect_chart_formations,
    detect_patterns,
    drop_open_candle,
    find_support_resistance,
)
from core.analysis.technical_analysis import realized_volatility, rsi
from core.data_collectors.fear_greed_service import get_fear_greed_service
from core.synthesis.market_state_vector import (
    DataQuality,
    MarketContext,
    MarketStateVector,
    PatternIndicators,
    SentimentIndicators,
    TechnicalIndicators,
)

logger = logging.getLogger(__name__)

MIN_RSI_ROWS = 15
MIN_VOLUME_BASELINE = 20
MIN_LEVEL_ROWS = 50
MIN_LEVEL_GAP = 0.01  # nearest level must be at least 1% away from the price


class MarketStateBuilder:
    """Builds MarketStateVector from snapshots and optional OHLCV."""

    def __init__(self):
        self.fear_greed_service = get_fear_greed_service()
        self.market_snapshot_service = None
        self.history_snapshot = None

    def set_market_snapshot(self, snapshot_service):
        self.market_snapshot_service = snapshot_service

    def set_history_snapshot(self, history_snapshot):
        self.history_snapshot = history_snapshot

    async def build_market_state(
        self,
        coin_id: str,
        *,
        ohlcv: Optional[pd.DataFrame] = None,
    ) -> Optional[MarketStateVector]:
        """Build the market state; None only when there is no current price.

        ``ohlcv`` is an already-fetched candle frame (e.g. Binance 1h klines
        with a ``close_time`` column). Insufficient evidence never produces
        fake values - it produces explicit gaps in DataQuality.
        """
        quality = DataQuality()

        current_price, market_data = self._get_current_market_data(coin_id, quality)
        if current_price is None:
            logger.warning("No market data available for %s", coin_id)
            return None

        history_df = self._get_historical_data(coin_id, quality)
        closed_ohlcv = self._prepare_ohlcv(ohlcv, quality)

        technical = self._build_technical_indicators(
            history_df, closed_ohlcv, current_price, market_data, quality
        )
        sentiment = await self._build_sentiment_indicators(coin_id, quality)
        patterns = self._build_pattern_indicators(closed_ohlcv, quality)
        context = self._build_market_context(technical, sentiment)

        if history_df is None and closed_ohlcv is None:
            quality.block_reasons.append("no_price_history_available")

        return MarketStateVector(
            coin_id=coin_id,
            timestamp=datetime.now().timestamp(),
            current_price=current_price,
            technical=technical,
            sentiment=sentiment,
            patterns=patterns,
            context=context,
            quality=quality,
            symbol=market_data.get('symbol', coin_id.upper()),
        )

    def _get_current_market_data(
        self, coin_id: str, quality: DataQuality
    ) -> Tuple[Optional[float], Dict]:
        if not self.market_snapshot_service:
            quality.mark("market_snapshot", False, detail="service_not_configured")
            return None, {}

        max_age = max(120, config.SNAPSHOT_REFRESH_SECONDS * 3)
        if hasattr(self.market_snapshot_service, "get_usable"):
            snapshot = self.market_snapshot_service.get_usable(max_age)
        else:
            snapshot = self.market_snapshot_service.get_latest()
            if snapshot and time.time() - getattr(snapshot, "ts", 0) > max_age:
                snapshot = None
        if not snapshot or coin_id not in snapshot.markets:
            quality.mark("market_snapshot", False, detail="stale_or_missing")
            return None, {}

        market_data = snapshot.markets[coin_id]
        quality.mark(
            "market_snapshot", True,
            as_of=getattr(snapshot, "ts", None),
            timeframe="spot",
            detail="coingecko_markets",
        )
        return market_data.get('current_price'), market_data

    def _get_historical_data(
        self, coin_id: str, quality: DataQuality
    ) -> Optional[pd.DataFrame]:
        if not self.history_snapshot:
            quality.mark("history", False, detail="service_not_configured")
            return None

        max_age = max(7200, config.HISTORY_REFRESH_SECONDS * 3)
        if hasattr(self.history_snapshot, "get_usable"):
            latest = self.history_snapshot.get_usable(max_age)
        else:
            latest = self.history_snapshot.get_latest()
            if latest and time.time() - getattr(latest, "ts", 0) > max_age:
                latest = None
        if not latest or coin_id not in latest.by_coin:
            quality.mark("history", False, detail="stale_or_missing")
            return None

        df = latest.by_coin[coin_id].copy()
        if 'price' in df.columns and 'close' not in df.columns:
            df['close'] = df['price']
        quality.mark(
            "history", True,
            as_of=getattr(latest, "ts", None),
            timeframe="1d",
            detail=f"history_snapshot({len(df)} rows)",
        )
        return df

    def _prepare_ohlcv(
        self, ohlcv: Optional[pd.DataFrame], quality: DataQuality
    ) -> Optional[pd.DataFrame]:
        if ohlcv is None or len(ohlcv) == 0:
            quality.mark("ohlcv", False, detail="not_supplied")
            return None
        closed = drop_open_candle(ohlcv)
        if len(closed) == 0:
            quality.mark("ohlcv", False, detail="no_closed_candles")
            return None
        as_of = None
        if 'close_time' in closed.columns:
            as_of = float(closed['close_time'].iloc[-1]) / 1000.0
        quality.mark(
            "ohlcv", True,
            as_of=as_of,
            timeframe="1h",
            detail=f"closed_candles({len(closed)})",
        )
        return closed

    def _build_technical_indicators(
        self,
        history_df: Optional[pd.DataFrame],
        closed_ohlcv: Optional[pd.DataFrame],
        current_price: float,
        market_data: Dict,
        quality: DataQuality,
    ) -> TechnicalIndicators:
        rsi_val = None
        if history_df is not None and 'close' in history_df.columns and len(history_df) >= MIN_RSI_ROWS:
            series = rsi(history_df['close'], period=14)
            if not series.empty and not np.isnan(series.iloc[-1]):
                rsi_val = float(series.iloc[-1])

        volume_ratio = self._closed_volume_ratio(closed_ohlcv, history_df)

        def _pct(*keys):
            for key in keys:
                value = market_data.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            return None

        momentum_1h = _pct('price_change_percentage_1h_in_currency')
        momentum_24h = _pct('price_change_percentage_24h_in_currency',
                            'price_change_percentage_24h')
        momentum_7d = _pct('price_change_percentage_7d_in_currency',
                           'price_change_percentage_7d')

        volatility = None
        if history_df is not None and 'close' in history_df.columns and len(history_df) >= MIN_RSI_ROWS:
            try:
                vol_annual = realized_volatility(history_df['close'], window=14)
                if vol_annual and not np.isnan(vol_annual):
                    volatility = float(vol_annual)
            except Exception as exc:
                logger.debug("Volatility calculation error: %s", exc)

        support_level = resistance_level = None
        support_distance = resistance_distance = None
        if closed_ohlcv is not None and len(closed_ohlcv) >= MIN_LEVEL_ROWS:
            levels = find_support_resistance(closed_ohlcv)
            # Pivots within 1% of the price on 1h candles are noise, not levels.
            supports = [s for s in levels['support'] if s < current_price * (1 - MIN_LEVEL_GAP)]
            resistances = [r for r in levels['resistance'] if r > current_price * (1 + MIN_LEVEL_GAP)]
            if supports:
                support_level = max(supports)
                support_distance = (current_price - support_level) / current_price * 100.0
            if resistances:
                resistance_level = min(resistances)
                resistance_distance = (resistance_level - current_price) / current_price * 100.0
            quality.mark(
                "levels", bool(supports or resistances),
                timeframe="1h",
                detail=f"support={len(supports)},resistance={len(resistances)}",
            )
        else:
            quality.mark("levels", False, detail="insufficient_candles")

        return TechnicalIndicators(
            rsi=rsi_val,
            volume_ratio=volume_ratio,
            price_momentum_1h_pct=momentum_1h,
            price_momentum_24h_pct=momentum_24h,
            price_momentum_7d_pct=momentum_7d,
            volatility_annual_pct=volatility,
            support_level=support_level,
            resistance_level=resistance_level,
            support_distance_pct=support_distance,
            resistance_distance_pct=resistance_distance,
        )

    @staticmethod
    def _closed_volume_ratio(
        closed_ohlcv: Optional[pd.DataFrame],
        history_df: Optional[pd.DataFrame],
    ) -> Optional[float]:
        """Signal candle volume vs mean of prior closed candles.

        The signal candle is excluded from its own baseline - including it
        dilutes genuine spikes and inflates quiet candles.
        """
        for frame in (closed_ohlcv, history_df):
            if frame is None or 'volume' not in frame.columns:
                continue
            if len(frame) < MIN_VOLUME_BASELINE + 1:
                continue
            volumes = frame['volume'].to_numpy(dtype=float)
            baseline = float(np.mean(volumes[-(MIN_VOLUME_BASELINE + 1):-1]))
            if baseline > 0:
                return float(volumes[-1]) / baseline
        return None

    async def _build_sentiment_indicators(
        self, coin_id: str, quality: DataQuality
    ) -> SentimentIndicators:
        fear_greed_index = None
        try:
            fear_greed_index = await self.fear_greed_service.get_current_index()
        except Exception as exc:
            logger.debug("Fear&Greed unavailable: %s", exc)
        quality.mark(
            "fear_greed", fear_greed_index is not None,
            timeframe="1d", detail="alternative.me",
        )

        # Coin-specific news intelligence arrives with the news snapshot
        # service (Package A2). Until then news evidence is explicitly
        # unavailable - never a neutral placeholder.
        quality.mark("news", False, detail="news_snapshot_service_not_implemented")

        return SentimentIndicators(
            news_sentiment=None,
            news_confidence=None,
            news_article_count=0,
            fear_greed_index=fear_greed_index,
        )

    def _build_pattern_indicators(
        self, closed_ohlcv: Optional[pd.DataFrame], quality: DataQuality
    ) -> PatternIndicators:
        if closed_ohlcv is None or len(closed_ohlcv) < 21:
            quality.mark("patterns", False, detail="insufficient_closed_candles")
            return PatternIndicators()

        detected = []
        try:
            # Full detector registry: chart patterns, candlesticks, formations.
            detected = (
                detect_patterns(closed_ohlcv)
                + detect_candlestick_patterns(closed_ohlcv)
                + detect_chart_formations(closed_ohlcv)
            )
        except Exception as exc:
            logger.warning("Pattern detection error: %r", exc)
            quality.mark("patterns", False, detail=f"detector_error:{type(exc).__name__}")
            return PatternIndicators()

        as_of = None
        if 'close_time' in closed_ohlcv.columns:
            as_of = float(closed_ohlcv['close_time'].iloc[-1]) / 1000.0
        quality.mark(
            "patterns", True,
            as_of=as_of, timeframe="1h",
            detail=f"detected={len(detected)}",
        )
        return PatternIndicators(detected_patterns=detected)

    def _build_market_context(
        self,
        technical: TechnicalIndicators,
        sentiment: SentimentIndicators,
    ) -> MarketContext:
        trend = None
        m24 = technical.price_momentum_24h_pct
        m7d = technical.price_momentum_7d_pct
        if m24 is not None and m7d is not None:
            if m7d > 5.0 and m24 > 2.0:
                trend = 'bull'
            elif m7d < -5.0 and m24 < -2.0:
                trend = 'bear'
            else:
                trend = 'sideways'

        volatility_regime = None
        if technical.volatility_annual_pct is not None:
            if technical.volatility_annual_pct < 30.0:
                volatility_regime = 'low'
            elif technical.volatility_annual_pct < 70.0:
                volatility_regime = 'medium'
            else:
                volatility_regime = 'high'

        volume_regime = None
        if technical.volume_ratio is not None:
            if technical.volume_ratio < 0.8:
                volume_regime = 'low'
            elif technical.volume_ratio < 1.5:
                volume_regime = 'medium'
            else:
                volume_regime = 'high'

        fear_greed_zone = None
        if sentiment.fear_greed_index is not None:
            fg = sentiment.fear_greed_index
            if fg < 25:
                fear_greed_zone = 'extreme_fear'
            elif fg < 45:
                fear_greed_zone = 'fear'
            elif fg < 55:
                fear_greed_zone = 'neutral'
            elif fg < 75:
                fear_greed_zone = 'greed'
            else:
                fear_greed_zone = 'extreme_greed'

        rsi_zone = None
        if technical.rsi is not None:
            if technical.rsi < 35:
                rsi_zone = 'oversold'
            elif technical.rsi > 65:
                rsi_zone = 'overbought'
            else:
                rsi_zone = 'neutral'

        return MarketContext(
            trend=trend,
            volatility_regime=volatility_regime,
            volume_regime=volume_regime,
            fear_greed_zone=fear_greed_zone,
            rsi_zone=rsi_zone,
        )


_market_state_builder: Optional[MarketStateBuilder] = None


def get_market_state_builder() -> MarketStateBuilder:
    global _market_state_builder
    if _market_state_builder is None:
        _market_state_builder = MarketStateBuilder()
    return _market_state_builder
