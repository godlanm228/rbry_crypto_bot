import logging
from typing import Optional

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

    def analyze_news_impact(self, title, summary):
        """Детальный анализ влияния новости на рынок"""

        # Определяем затронутые монеты
        affected_coins = self._extract_coins(title + " " + summary)

        # Оцениваем влияние
        impact = {
            'coins': affected_coins,
            'direction': 'neutral',  # positive/negative/neutral
            'strength': 0,  # 0-10
            'timeframe': 'short',  # short/medium/long
            'action': ''  # что делать трейдеру
        }

        # Анализ ключевых слов для определения влияния
        text = (title + " " + summary).lower()

        if any(w in text for w in ['ban', 'hack', 'lawsuit', 'crash']):
            impact['direction'] = 'negative'
            impact['strength'] = 8
            impact['action'] = "Рассмотрите фиксацию прибыли или хеджирование позиций"
        elif any(w in text for w in ['etf', 'adoption', 'partnership', 'institutional']):
            impact['direction'] = 'positive'
            impact['strength'] = 7
            impact['action'] = "Возможность для входа или увеличения позиции"
        elif any(w in text for w in ['regulation', 'investigation']):
            impact['direction'] = 'neutral'
            impact['strength'] = 5
            impact['action'] = "Следите за развитием ситуации, не спешите с решениями"

        return impact

    def _extract_coins(self, text):
        """Извлекает упомянутые криптовалюты"""
        coins = []
        coin_patterns = {
            'bitcoin': 'BTC', 'btc': 'BTC',
            'ethereum': 'ETH', 'eth': 'ETH',
            'solana': 'SOL', 'sol': 'SOL',
            'ripple': 'XRP', 'xrp': 'XRP'
        }

        text_lower = text.lower()
        for pattern, coin in coin_patterns.items():
            if pattern in text_lower:
                coins.append(coin)

        return coins if coins else ['Весь рынок']

    async def assess_growth_potential_with_synthesis(self, coin_id: str) -> Optional[dict]:
        """
        NEW: AI-driven assessment using synthesis engine

        Returns dict compatible with old format:
        {
            "score": 0-10,
            "recommendation": "ПОКУПАТЬ"/"ДЕРЖАТЬ"/"ПРОДАВАТЬ",
            "target_price": float,
            "risk_level": "Низкий"/"Средний"/"Высокий",
            "sentiment": "positive"/"negative"/"neutral",
            "confidence": 0-1
        }
        """
        if not self.synthesis_engine or not self.market_state_builder:
            logger.info(
                f"SYNTHESIS unavailable for {coin_id}: "
                f"engine={'ok' if self.synthesis_engine else 'missing'} "
                f"builder={'ok' if self.market_state_builder else 'missing'}"
            )
            return None

        try:
            # Build market state
            market_state = await self.market_state_builder.build_market_state(coin_id)

            if market_state is None:
                logger.warning(f"SYNTHESIS skipped for {coin_id}: market_state_missing")
                return None

            # Run synthesis
            analysis = await self.synthesis_engine.synthesize(market_state)

            # Convert to old format for backward compatibility
            # Map signal_type to Russian recommendation
            rec_map = {
                'buy': 'ПОКУПАТЬ',
                'hold': 'ДЕРЖАТЬ',
                'sell': 'ПРОДАВАТЬ'
            }
            recommendation = rec_map.get(analysis.signal_type, 'ДЕРЖАТЬ')

            # Map risk_level to Russian
            risk_map = {
                'Low': 'Низкий',
                'Medium': 'Средний',
                'High': 'Высокий'
            }
            risk_level = risk_map.get(analysis.risk_level, 'Средний')

            # Convert confidence (0-1) to score (0-10)
            score = int(analysis.confidence * 10)

            # Determine sentiment from component scores
            sentiment_score = analysis.component_scores.get('sentiment', 0.5)
            if sentiment_score > 0.6:
                sentiment = 'positive'
            elif sentiment_score < 0.4:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            result = {
                "score": score,
                "recommendation": recommendation,
                "target_price": analysis.target_price,
                "risk_level": risk_level,
                "sentiment": sentiment,
                "confidence": analysis.confidence,
                # Add new fields for extended info
                "_synthesis_analysis": analysis,  # Full analysis object
                "_market_state": market_state,
                "_using_synthesis": True
            }

            logger.info(
                f"Synthesis assessment for {coin_id}: {recommendation} "
                f"(confidence: {analysis.confidence:.1%}, score: {score}/10)"
            )

            return result

        except Exception as e:
            logger.error(f"Error in synthesis assessment for {coin_id}: {e}", exc_info=True)
            return None

    async def assess_growth_potential(self, coin_id, current_price, tech_indicators, market_data):
        """
        AI-оценка на базе FinBERT + техусловий. Возвращает dict со score/rec/target/risk.

        NEW: Tries AI synthesis engine first, falls back to legacy method if unavailable.
        """
        # Try new synthesis engine first
        if self.synthesis_engine:
            synthesis_result = await self.assess_growth_potential_with_synthesis(coin_id)
            if synthesis_result is not None:
                return synthesis_result

        # FALLBACK: Legacy rule-based assessment
        logger.debug(f"Using legacy assessment for {coin_id}")

        # нормализуем ключ на 24h
        ch24 = market_data.get("price_change_percentage_24h_in_currency")
        if ch24 is None:
            ch24 = market_data.get("price_change_percentage_24h", 0)

        rsi_val = tech_indicators.get("rsi")
        ctx = (
            f"{coin_id} price ${current_price}, 24h change {ch24}%, "
            f"RSI {rsi_val if rsi_val is not None else 'n/a'}, "
            f"volume {market_data.get('total_volume')}, mcap {market_data.get('market_cap')}."
        )

        base_score = 5
        if rsi_val is not None:
            if rsi_val < 30:
                base_score += 2
            elif rsi_val > 70:
                base_score -= 2
        if isinstance(ch24, (int, float)):
            if ch24 > 10:
                base_score += 1
            elif ch24 < -10:
                base_score -= 1
        pattern_boost = 0
        if 'patterns' in market_data:
            for pattern in market_data.get('patterns', []):
                if pattern.get('signal') == 'bullish_reversal':
                    pattern_boost += 2
                elif pattern.get('signal') == 'bullish_continuation':
                    pattern_boost += 1
                elif pattern.get('signal') == 'bearish_reversal':
                    pattern_boost -= 2
                elif pattern.get('signal') == 'bearish_continuation':
                    pattern_boost -= 1

        base_score += pattern_boost
        sentiment = "neutral"
        conf = 0.0
        if self.finbert and self.finbert.status.get("enabled"):
            try:
                sentiment, conf, _backend = await self.finbert.analyze_async(ctx)
                if sentiment.startswith("pos"):
                    base_score += 1
                elif sentiment.startswith("neg"):
                    base_score -= 1
            except Exception as e:
                logger.warning(f"FinBERT inference error: {e}")

        # капрингуем
        score = max(0, min(10, base_score))
        if score >= 7:
            rec = "ПОКУПАТЬ"
            target = float(current_price) * 1.15
            risk = "Низкий"
        elif score >= 5:
            rec = "ДЕРЖАТЬ"
            target = float(current_price) * 1.08
            risk = "Средний"
        else:
            rec = "ПРОДАВАТЬ"
            target = float(current_price) * 0.95
            risk = "Высокий"

        return {
            "score": score,
            "recommendation": rec,
            "target_price": target,
            "risk_level": risk,
            "sentiment": sentiment,
            "confidence": conf,
            "_using_synthesis": False  # Mark as legacy
        }

    def analyze_news(self, text):
        if not self.finbert or not self.finbert.status.get("enabled"):
            return "AI-анализ недоступен"
        try:
            label, score, _backend = self.finbert.analyze_sync(text)
            if label == "positive" and score > 0.7:
                return "✅ Позитивный настрой новости."
            elif label == "negative" and score > 0.7:
                return "⚠️ Негативный настрой новости."
            else:
                return "ℹ️ Нейтральный настрой."
        except Exception as e:
            logger.error(f"Ошибка при анализе новости: {e}")
            return "AI-анализ неуспешен"

    @property
    def finbert_status(self):
        if not self.finbert:
            return {"enabled": False, "available": False, "backend": "none", "reason": "not_initialized"}
        return self.finbert.status


ai_analyzer = RealAIAnalyzer()
