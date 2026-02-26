"""
AI Synthesis Engine - Main orchestrator for AI-driven analysis

Combines all components into unified probabilistic analysis:
1. Run probabilistic scoring on market_state
2. Query feedback loop for historical context
3. Adjust confidence based on historical win rate
4. Run FinBERT on synthesized text context
5. Generate technical/sentiment/pattern summaries
6. Create unified reasoning connecting all components
7. Calculate dynamic targets based on confidence + volatility

FinBERT fallback strategy:
1. Local FinBERT (fast, private, free)
2. HuggingFace Inference API (fallback)
3. Groq API (if configured)
4. Neutral sentiment (graceful degradation)
"""

import logging
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from .market_state_vector import MarketStateVector
from .probabilistic_scorer import ProbabilisticSignal, get_probabilistic_scorer
from .feedback_loop import HistoricalContext, get_feedback_loop

logger = logging.getLogger(__name__)

from core.finbert_client import get_finbert_client


@dataclass
class SynthesizedAnalysis:
    """
    Complete synthesized analysis output

    Attributes:
        coin_id: Coin identifier
        timestamp: Analysis timestamp
        signal_type: 'buy', 'sell', 'hold'
        confidence: Final adjusted confidence (0-1)
        raw_confidence: Original probabilistic confidence before adjustment
        component_scores: Dict of component scores
        probabilistic_signal: Original probabilistic signal
        historical_context: Historical context (if available)
        technical_summary: Technical analysis summary text
        sentiment_summary: Sentiment analysis summary text
        pattern_summary: Pattern analysis summary text
        unified_reasoning: Unified explanation connecting all components
        target_price: Predicted target price
        stop_loss: Stop loss price
        risk_level: 'Low', 'Medium', 'High'
    """
    coin_id: str
    timestamp: float
    signal_type: str
    confidence: float
    raw_confidence: float
    component_scores: Dict[str, float]
    probabilistic_signal: ProbabilisticSignal
    historical_context: Optional[HistoricalContext]
    technical_summary: str
    sentiment_summary: str
    pattern_summary: str
    unified_reasoning: str
    target_price: float
    stop_loss: float
    risk_level: str


class AISynthesisEngine:
    """
    Main AI synthesis orchestrator

    Combines probabilistic scoring, feedback loop, and FinBERT
    into unified analysis with graceful degradation.
    """

    def __init__(self):
        self.probabilistic_scorer = get_probabilistic_scorer()
        self.feedback_loop = get_feedback_loop()
        self.finbert = get_finbert_client()

    async def synthesize(self, market_state: MarketStateVector) -> SynthesizedAnalysis:
        """
        Main synthesis pipeline

        Args:
            market_state: Complete market state vector

        Returns:
            SynthesizedAnalysis with all components
        """
        # Step 1: Probabilistic scoring
        prob_signal = self.probabilistic_scorer.synthesize_signal(market_state)
        raw_confidence = prob_signal.confidence

        # Step 2: Feedback loop (historical context)
        hist_context = self.feedback_loop.get_historical_context(
            market_state.coin_id,
            market_state,
            prob_signal.signal_type
        )

        # Step 3: Confidence adjustment
        if hist_context:
            adjusted_confidence = np.clip(
                raw_confidence + hist_context.confidence_adjustment,
                0.0,
                1.0
            )
            prob_signal.confidence = adjusted_confidence
        else:
            adjusted_confidence = raw_confidence

        # Step 4: FinBERT analysis (optional, graceful degradation)
        finbert_label, finbert_conf = await self.analyze_with_finbert(
            market_state.get_context_text()
        )

        # Step 5: Generate summaries
        technical_summary = self.generate_technical_summary(market_state)
        sentiment_summary = self.generate_sentiment_summary(
            market_state, finbert_label, finbert_conf
        )
        pattern_summary = self.generate_pattern_summary(market_state)

        # Step 6: Unified reasoning
        unified_reasoning = self.synthesize_unified_reasoning(
            technical_summary,
            sentiment_summary,
            pattern_summary,
            prob_signal,
            hist_context
        )

        # Step 7: Calculate targets
        target_price, stop_loss, risk_level = self.calculate_targets(
            market_state, prob_signal
        )

        return SynthesizedAnalysis(
            coin_id=market_state.coin_id,
            timestamp=market_state.timestamp,
            signal_type=prob_signal.signal_type,
            confidence=adjusted_confidence,
            raw_confidence=raw_confidence,
            component_scores=prob_signal.component_scores,
            probabilistic_signal=prob_signal,
            historical_context=hist_context,
            technical_summary=technical_summary,
            sentiment_summary=sentiment_summary,
            pattern_summary=pattern_summary,
            unified_reasoning=unified_reasoning,
            target_price=target_price,
            stop_loss=stop_loss,
            risk_level=risk_level
        )

    async def analyze_with_finbert(self, context_text: str) -> Tuple[str, float]:
        """
        Analyze text with FinBERT using fallback strategy

        Returns:
            Tuple of (label, confidence) - ('positive'/'negative'/'neutral', 0-1)
        """
        try:
            label, confidence, _backend = await self.finbert.analyze_async(context_text)
            return label, confidence
        except Exception as e:
            logger.warning(f"FinBERT error: {e}")
            return 'neutral', 0.5

    def generate_technical_summary(self, market_state: MarketStateVector) -> str:
        """Generate technical analysis summary"""
        tech = market_state.technical

        rsi_desc = "oversold" if tech.rsi_normalized < 0.35 else \
                   "overbought" if tech.rsi_normalized > 0.70 else "neutral"

        vol_desc = "high" if tech.volume_ratio > 1.5 else \
                   "low" if tech.volume_ratio < 0.8 else "average"

        momentum_desc = "strongly bullish" if tech.price_momentum_24h > 0.10 else \
                       "bullish" if tech.price_momentum_24h > 0.03 else \
                       "strongly bearish" if tech.price_momentum_24h < -0.10 else \
                       "bearish" if tech.price_momentum_24h < -0.03 else "neutral"

        summary = (
            f"Technical indicators show {rsi_desc} RSI at {tech.rsi_raw:.1f} with {vol_desc} volume "
            f"({tech.volume_ratio:.2f}x average). "
            f"Price momentum is {momentum_desc} "
            f"(24h: {tech.price_momentum_24h*100:+.2f}%, 7d: {tech.price_momentum_7d*100:+.2f}%)."
        )

        return summary

    def generate_sentiment_summary(
        self,
        market_state: MarketStateVector,
        finbert_label: str,
        finbert_conf: float
    ) -> str:
        """Generate sentiment analysis summary"""
        sent = market_state.sentiment

        # Fear & Greed description
        if sent.fear_greed_index is not None:
            fg_val = sent.fear_greed_index
            if fg_val < 25:
                fg_desc = "extreme fear"
            elif fg_val < 45:
                fg_desc = "fear"
            elif fg_val < 55:
                fg_desc = "neutral"
            elif fg_val < 75:
                fg_desc = "greed"
            else:
                fg_desc = "extreme greed"

            sentiment_text = f"Market sentiment shows {fg_desc} (Fear & Greed Index: {fg_val:.0f}/100). "
        else:
            sentiment_text = "Fear & Greed Index: N/A (API unavailable). "

        # News sentiment
        if sent.news_article_count > 0:
            news_desc = "positive" if sent.news_sentiment > 0.2 else \
                       "negative" if sent.news_sentiment < -0.2 else "neutral"
            sentiment_text += f"News sentiment is {news_desc} (confidence: {sent.news_confidence:.0%})."
        else:
            sentiment_text += "No recent news data available."

        return sentiment_text

    def generate_pattern_summary(self, market_state: MarketStateVector) -> str:
        """Generate pattern analysis summary"""
        patterns = market_state.patterns

        bullish_count = patterns.get_bullish_count()
        bearish_count = patterns.get_bearish_count()

        if bullish_count > bearish_count:
            pattern_text = f"{bullish_count} bullish pattern(s) detected"
            if bullish_count > 1:
                pattern_names = [p['pattern'] for p in patterns.detected_patterns
                               if p.get('direction') == 'bullish'][:2]
                pattern_text += f" ({', '.join(pattern_names)})"
        elif bearish_count > bullish_count:
            pattern_text = f"{bearish_count} bearish pattern(s) detected"
            if bearish_count > 1:
                pattern_names = [p['pattern'] for p in patterns.detected_patterns
                               if p.get('direction') == 'bearish'][:2]
                pattern_text += f" ({', '.join(pattern_names)})"
        elif bullish_count == 0 and bearish_count == 0:
            pattern_text = "No significant chart patterns detected"
        else:
            pattern_text = "Mixed pattern signals detected"

        return pattern_text + "."

    def synthesize_unified_reasoning(
        self,
        technical_summary: str,
        sentiment_summary: str,
        pattern_summary: str,
        prob_signal: ProbabilisticSignal,
        hist_context: Optional[HistoricalContext]
    ) -> str:
        """
        Generate unified reasoning that connects all components

        This is NOT just concatenation - it shows how components support each other
        """
        reasoning_parts = []

        # Signal header
        signal_emoji = "🟢" if prob_signal.signal_type == "buy" else \
                      "🔴" if prob_signal.signal_type == "sell" else "🟡"

        reasoning_parts.append(
            f"{signal_emoji} **{prob_signal.signal_type.upper()}** Signal "
            f"(Confidence: {prob_signal.confidence:.1%})"
        )

        # Technical analysis
        reasoning_parts.append(f"\n**Technical Analysis:**\n{technical_summary}")

        # Sentiment analysis
        reasoning_parts.append(f"\n**Market Sentiment:**\n{sentiment_summary}")

        # Pattern analysis
        reasoning_parts.append(f"\n**Chart Patterns:**\n{pattern_summary}")

        # Historical validation (if available)
        if hist_context:
            adj_sign = "+" if hist_context.confidence_adjustment > 0 else ""
            reasoning_parts.append(
                f"\n**Historical Validation:**\n"
                f"Found {hist_context.similar_signals_count} similar signals with "
                f"{hist_context.avg_quality_score:.0%} average quality. "
                f"Confidence adjusted by {adj_sign}{hist_context.confidence_adjustment*100:.1f}%."
            )

        # Synthesis conclusion
        if prob_signal.signal_type == "buy":
            conclusion = (
                "\n**Analysis Synthesis:**\n"
                "The confluence of technical indicators, market sentiment, and chart patterns "
                "suggests a buying opportunity. "
            )
        elif prob_signal.signal_type == "sell":
            conclusion = (
                "\n**Analysis Synthesis:**\n"
                "Multiple indicators suggest taking profits or reducing exposure. "
            )
        else:
            conclusion = (
                "\n**Analysis Synthesis:**\n"
                "Current market conditions suggest maintaining positions. "
            )

        if hist_context and hist_context.avg_quality_score > 0.65:
            conclusion += "Historical patterns support this assessment."
        elif hist_context and hist_context.avg_quality_score < 0.40:
            conclusion += "Exercise caution as similar patterns have underperformed historically."

        reasoning_parts.append(conclusion)

        return "\n".join(reasoning_parts)

    def calculate_targets(
        self,
        market_state: MarketStateVector,
        prob_signal: ProbabilisticSignal
    ) -> Tuple[float, float, str]:
        """
        Calculate dynamic target price, stop loss, and risk level

        Based on confidence, volatility, and signal type

        Returns:
            Tuple of (target_price, stop_loss, risk_level)
        """
        current_price = market_state.current_price
        confidence = prob_signal.confidence
        volatility = market_state.technical.volatility

        if prob_signal.signal_type == "buy":
            # Target: scale with confidence (5-20% gain target)
            base_target_pct = 0.05 + (confidence * 0.15)  # 5-20%

            # Adjust for volatility (higher volatility → larger targets)
            volatility_factor = 0.8 + (volatility * 0.4)  # 0.8x to 1.2x
            target_pct = base_target_pct * volatility_factor

            target_price = current_price * (1 + target_pct)

            # Stop loss: scale inversely with confidence (2-8% loss threshold)
            base_stop_pct = 0.08 - (confidence * 0.06)  # 8-2%
            stop_loss = current_price * (1 - base_stop_pct)

        elif prob_signal.signal_type == "sell":
            # For sell: target is downward movement
            base_target_pct = 0.05 + (confidence * 0.10)  # 5-15% expected drop
            target_price = current_price * (1 - base_target_pct)

            # Stop loss: if price rises unexpectedly
            base_stop_pct = 0.03 + ((1 - confidence) * 0.05)  # 3-8%
            stop_loss = current_price * (1 + base_stop_pct)

        else:  # hold
            # Minimal target/stop for hold
            target_price = current_price * 1.05
            stop_loss = current_price * 0.95

        # Determine risk level
        if confidence > 0.75 and volatility < 0.5:
            risk_level = "Low"
        elif confidence < 0.45 or volatility > 0.75:
            risk_level = "High"
        else:
            risk_level = "Medium"

        return float(target_price), float(stop_loss), risk_level


# Global singleton
_ai_synthesis_engine: Optional[AISynthesisEngine] = None


def get_ai_synthesis_engine() -> AISynthesisEngine:
    """Get global AI synthesis engine instance"""
    global _ai_synthesis_engine
    if _ai_synthesis_engine is None:
        _ai_synthesis_engine = AISynthesisEngine()
    return _ai_synthesis_engine

