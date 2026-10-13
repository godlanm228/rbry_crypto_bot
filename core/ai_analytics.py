import logging
from typing import Optional

from core.analysis_result import AnalysisResult
from core.api.binance_client import get_batch_ohlcv
from core.finbert_client import get_finbert_client

logger = logging.getLogger(__name__)

# Import new AI synthesis components
_SYNTHESIS_AVAILABLE = False
try:
    from core.data_collectors.market_state_builder import get_market_state_builder
    from core.synthesis.ai_synthesis_engine import get_ai_synthesis_engine
    _SYNTHESIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI synthesis engine not available: {e}")
    _SYNTHESIS_AVAILABLE = False


class RealAIAnalyzer:
    def __init__(self):
        self.finbert = get_finbert_client()

        # Initialize synthesis engine if available
        self.synthesis_engine = None
        self.market_state_builder = None
        if _SYNTHESIS_AVAILABLE:
            try:
                self.synthesis_engine = get_ai_synthesis_engine()
                self.market_state_builder = get_market_state_builder()
                logger.info("AI synthesis engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize synthesis engine: {e}")
                self.synthesis_engine = None

    def set_data_sources(self, market_snapshot=None, history_snapshot=None):
        """
        Set data source references for market state builder

        This should be called from main.py or handlers after services are initialized.
        """
        if self.market_state_builder:
            if market_snapshot:
                self.market_state_builder.set_market_snapshot(market_snapshot)
            if history_snapshot:
                self.market_state_builder.set_history_snapshot(history_snapshot)
            logger.info("Data sources set for AI analyzer")

    async def assess_growth_potential_with_synthesis(self, coin_id: str) -> Optional[AnalysisResult]:
        """
        Multi-factor assessment through the shared evidence pipeline.

        The engine returns the canonical AnalysisResult directly; this facade
        only supplies data sources. The evidence score is deliberately not a
        market probability: it has not been calibrated for that claim.
        """
        if not self.synthesis_engine or not self.market_state_builder:
            logger.info(
                f"SYNTHESIS unavailable for {coin_id}: "
                f"engine={'ok' if self.synthesis_engine else 'missing'} "
                f"builder={'ok' if self.market_state_builder else 'missing'}"
            )
            return None

        try:
            # One batch OHLCV fetch; the builder itself never hits the network
            # for candles. A failed fetch degrades to explicit missing data.
            ohlcv = None
            try:
                ohlcv_map = await get_batch_ohlcv([coin_id], interval="1h")
                ohlcv = ohlcv_map.get(coin_id)
            except Exception as exc:
                logger.warning(f"OHLCV fetch failed for {coin_id}: {exc}")

            market_state = await self.market_state_builder.build_market_state(
                coin_id, ohlcv=ohlcv
            )
            if market_state is None:
                logger.warning(f"SYNTHESIS skipped for {coin_id}: market_state_missing")
                return None

            result = await self.synthesis_engine.synthesize(market_state)

            logger.info(
                "Synthesis assessment for %s: direction=%s strength=%s scenario=%s risk=%s",
                coin_id,
                result.direction,
                result.evidence_strength,
                result.scenario,
                result.risk_level,
            )
            return result

        except Exception as e:
            logger.error(f"Error in synthesis assessment for {coin_id}: {e}", exc_info=True)
            return None

    async def assess_growth_potential(
        self,
        coin_id,
        current_price,
        tech_indicators,
        market_data,
        *,
        allow_synthesis: bool = True,
    ):
        """
        Structured market assessment based on the available evidence.

        NEW: Tries AI synthesis engine first, falls back to legacy method if unavailable.
        """
        # Try new synthesis engine first
        if allow_synthesis and self.synthesis_engine:
            synthesis_result = await self.assess_growth_potential_with_synthesis(coin_id)
            if synthesis_result is not None:
                return synthesis_result

        # FALLBACK: Legacy rule-based assessment
        logger.debug(f"Using legacy assessment for {coin_id}")

        # Normalize the 24h field without replacing missing data with zero.
        ch24 = market_data.get("price_change_percentage_24h_in_currency")
        if ch24 is None:
            ch24 = market_data.get("price_change_percentage_24h")

        rsi_val = tech_indicators.get("rsi")
        ctx = (
            f"{coin_id} price ${current_price}, 24h change {ch24}%, "
            f"RSI {rsi_val if rsi_val is not None else 'n/a'}, "
            f"volume {market_data.get('total_volume')}, mcap {market_data.get('market_cap')}."
        )

        base_score = 5
        evidence = []
        if rsi_val is not None:
            if rsi_val < 30:
                base_score += 2
                evidence.append("rsi_oversold")
            elif rsi_val > 70:
                base_score -= 2
                evidence.append("rsi_overbought")
            else:
                evidence.append("rsi_neutral")
        if isinstance(ch24, (int, float)):
            if ch24 > 10:
                base_score += 1
                evidence.append("strong_positive_24h_move")
            elif ch24 < -10:
                base_score -= 1
                evidence.append("strong_negative_24h_move")
            else:
                evidence.append("moderate_24h_move")
        pattern_boost = 0
        if 'patterns' in market_data:
            for pattern in market_data.get('patterns', []):
                if pattern.get('signal') == 'bullish_reversal':
                    pattern_boost += 2
                    evidence.append("bullish_reversal")
                elif pattern.get('signal') == 'bullish_continuation':
                    pattern_boost += 1
                    evidence.append("bullish_continuation")
                elif pattern.get('signal') == 'bearish_reversal':
                    pattern_boost -= 2
                    evidence.append("bearish_reversal")
                elif pattern.get('signal') == 'bearish_continuation':
                    pattern_boost -= 1
                    evidence.append("bearish_continuation")

        base_score += pattern_boost
        # No text sentiment here: classifying the bot's own generated sentence
        # is not evidence, and without a backend it fabricated "neutral".
        sentiment = None

        if not evidence:
            return AnalysisResult(
                scenario="unavailable",
                risk_level="unknown",
                method="rule_based",
                sentiment=None,
                evidence=(),
            )

        score = max(0, min(10, base_score))
        if score >= 7:
            scenario = "positive"
            risk = "medium"
        elif score >= 5:
            scenario = "neutral"
            risk = "medium"
        else:
            scenario = "defensive"
            risk = "high"

        return AnalysisResult(
            scenario=scenario,
            risk_level=risk,
            method="rule_based",
            rule_score=score,
            sentiment=sentiment,
            evidence=tuple(evidence),
        )

    def analyze_news(self, text):
        """'positive' / 'negative' / 'neutral', or None when no backend can tell.

        Never invents a tone: without an available sentiment backend the
        caller must show nothing rather than "neutral".
        """
        if not self.finbert or not self.finbert.status.get("available"):
            return None
        try:
            label, score, _backend = self.finbert.analyze_sync(text)
        except Exception as e:
            logger.error(f"News sentiment failed: {e}")
            return None
        if label == "positive" and score > 0.7:
            return "positive"
        if label == "negative" and score > 0.7:
            return "negative"
        return "neutral"

    @property
    def finbert_status(self):
        if not self.finbert:
            return {"enabled": False, "available": False, "backend": "none", "reason": "not_initialized"}
        return self.finbert.status


ai_analyzer = RealAIAnalyzer()
