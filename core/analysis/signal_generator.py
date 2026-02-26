from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from .technical_analysis import rsi, sma, ema

@dataclass
class Signal:
    kind: str
    strength: str
    detail: str

def _label_strength(value: float, low: float, high: float) -> str:
    if value <= low:
        return "weak"
    if value >= high:
        return "strong"
    return "medium"

def generate_signals(ohlcv: pd.DataFrame, rsi_period: int = 14) -> List[Signal]:
    """Expect DataFrame with columns: ['close', 'volume'] at minimum; optional 'high','low'.
    Index should be datetime ascending.
    Rules:
      - RSI oversold (<35) + volume uptick vs 20d avg -> BUY consideration
      - RSI overbought (>70) -> SELL/RISK warning
      - SMA50 cross SMA200 -> TREND change
    """
    assert 'close' in ohlcv.columns, "need close column"
    df = ohlcv.copy()
    df['rsi'] = rsi(df['close'], period=rsi_period)
    df['sma50'] = sma(df['close'], 50)
    df['sma200'] = sma(df['close'], 200)
    if 'volume' in df.columns:
        df['vol_avg20'] = df['volume'].rolling(20, min_periods=5).mean()
        df['vol_uptick'] = df['volume'] > 1.2 * df['vol_avg20']
    last = df.iloc[-1]
    signals: List[Signal] = []

    # RSI oversold with volume uptick
    if last['rsi'] < 35 and (last.get('vol_uptick') == True):
        strength = _label_strength(35 - float(last['rsi']), 2, 10)
        signals.append(Signal(kind="buy_candidate", strength=strength,
                              detail=f"RSI={last['rsi']:.1f} (<35){' + vol↑' if 'vol_uptick' in df.columns and bool(last['vol_uptick']) else ''}"))

    # RSI overbought
    if last['rsi'] > 70:
        strength = _label_strength(float(last['rsi']) - 70, 2, 10)
        signals.append(Signal(kind="risk_sell", strength=strength,
                              detail=f"RSI={last['rsi']:.1f} (>70)"))

    # SMA50 cross SMA200
    if len(df) >= 201 and pd.notna(df['sma50'].iloc[-2]) and pd.notna(df['sma200'].iloc[-2]):
        prev_up = df['sma50'].iloc[-2] < df['sma200'].iloc[-2]
        now_up = last['sma50'] >= last['sma200']
        prev_down = df['sma50'].iloc[-2] > df['sma200'].iloc[-2]
        now_down = last['sma50'] <= last['sma200']
        if prev_down and now_up:
            signals.append(Signal(kind="trend_bullish", strength="medium",
                                  detail="SMA50 crossed above SMA200 (golden cross)"))
        if prev_up and now_down:
            signals.append(Signal(kind="trend_bearish", strength="medium",
                                  detail="SMA50 crossed below SMA200 (death cross)"))

    return signals

def format_signals_for_telegram(coin_symbol: str, signals: List[Signal]) -> str:
    if not signals:
        return f"{coin_symbol}: сигналов нет."
    lines = [f"💡 {coin_symbol}:"]
    for s in signals:
        emoji = {"buy_candidate": "🟢", "risk_sell": "🟠", "trend_bullish": "📈", "trend_bearish": "📉"}.get(s.kind, "•")
        lines.append(f"{emoji} {s.kind} [{s.strength}] — {s.detail}")
    return "\n".join(lines)

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Возвращает Series с RSI(period) по ряду закрытия.
    Использует rsi() из technical_analysis.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    return rsi(close.astype(float), period=period)


def detect_patterns(ohlcv: pd.DataFrame, window: int = 20) -> List[Dict[str, any]]:
    """Detect chart patterns: double bottom/top, head&shoulders, breakouts."""
    if len(ohlcv) < window * 2:
        return []

    patterns = []
    close = ohlcv['close'].values
    volume = ohlcv['volume'].values if 'volume' in ohlcv.columns else None

    def _find_extrema(values, order: int = 3):
        peaks = []
        troughs = []
        for i in range(order, len(values) - order):
            seg = values[i - order:i + order + 1]
            if values[i] == seg.max():
                peaks.append((i, values[i]))
            if values[i] == seg.min():
                troughs.append((i, values[i]))
        return peaks, troughs

    # Double bottom detection
    lows = ohlcv['close'].rolling(window=5).min()
    for i in range(window, len(lows) - window):
        if pd.notna(lows.iloc[i]) and pd.notna(lows.iloc[i-window:i-window+5]).any():
            prev_low = lows.iloc[i-window:i-window+5].min()
            curr_low = lows.iloc[i]
            if abs(curr_low - prev_low) / prev_low < 0.02:  # Within 2%
                if close[-1] > curr_low * 1.05:  # Breaking above
                    patterns.append({
                        'type': 'double_bottom',
                        'strength': 'strong' if volume is not None and volume[-1] > volume[-20:].mean() * 1.5 else 'medium',
                        'level': curr_low,
                        'target': curr_low * 1.15
                    })
                    break

    # Double top detection
    highs = ohlcv['close'].rolling(window=5).max()
    for i in range(window, len(highs) - window):
        if pd.notna(highs.iloc[i]) and pd.notna(highs.iloc[i-window:i-window+5]).any():
            prev_high = highs.iloc[i-window:i-window+5].max()
            curr_high = highs.iloc[i]
            if abs(curr_high - prev_high) / prev_high < 0.02:
                if close[-1] < curr_high * 0.95:
                    patterns.append({
                        'type': 'double_top',
                        'strength': 'strong' if volume is not None and volume[-1] > volume[-20:].mean() * 1.5 else 'medium',
                        'level': curr_high,
                        'target': curr_high * 0.90
                    })
                    break

    # Head and shoulders / inverse head and shoulders
    peaks, troughs = _find_extrema(close, order=3)
    if len(peaks) >= 3:
        left, head, right = peaks[-3], peaks[-2], peaks[-1]
        shoulder_diff = abs(left[1] - right[1]) / max(left[1], right[1])
        head_up = (head[1] - max(left[1], right[1])) / head[1]
        if shoulder_diff < 0.03 and head_up > 0.03:
            patterns.append({
                'type': 'head_and_shoulders',
                'strength': 'strong' if head_up > 0.06 else 'medium',
                'level': head[1],
                'target': head[1] * 0.90
            })
    if len(troughs) >= 3:
        left, head, right = troughs[-3], troughs[-2], troughs[-1]
        shoulder_diff = abs(left[1] - right[1]) / max(left[1], right[1])
        head_down = (min(left[1], right[1]) - head[1]) / min(left[1], right[1])
        if shoulder_diff < 0.03 and head_down > 0.03:
            patterns.append({
                'type': 'inverse_head_and_shoulders',
                'strength': 'strong' if head_down > 0.06 else 'medium',
                'level': head[1],
                'target': head[1] * 1.10
            })

    # Breakout detection
    recent_high = ohlcv['close'].iloc[-window:].max()
    recent_low = ohlcv['close'].iloc[-window:].min()
    current = close[-1]

    if current > recent_high * 0.98:  # Near or above resistance
        vol_surge = volume[-1] > volume[-20:].mean() * 2 if volume is not None else False
        patterns.append({
            'type': 'resistance_breakout',
            'strength': 'strong' if vol_surge else 'weak',
            'level': recent_high,
            'target': recent_high * 1.08
        })

    return patterns
def detect_candlestick_patterns(ohlcv: pd.DataFrame) -> List[Dict[str, any]]:
    """Detect Japanese candlestick patterns"""
    if len(ohlcv) < 5 or not all(col in ohlcv.columns for col in ['open', 'high', 'low', 'close']):
        return []

    patterns = []
    o = ohlcv['open'].values
    h = ohlcv['high'].values
    l = ohlcv['low'].values
    c = ohlcv['close'].values

    # Last 3 candles for pattern detection
    if len(ohlcv) >= 3:
        # Hammer (bullish reversal)
        body = abs(c[-1] - o[-1])
        lower_shadow = min(o[-1], c[-1]) - l[-1]
        upper_shadow = h[-1] - max(o[-1], c[-1])

        if lower_shadow > body * 2 and upper_shadow < body * 0.3:
            if c[-2] < o[-2]:  # Previous was bearish
                patterns.append({
                    'type': 'hammer',
                    'signal': 'bullish_reversal',
                    'strength': 'medium',
                    'price': float(c[-1])
                })

        # Engulfing
        if len(ohlcv) >= 2:
            curr_body = c[-1] - o[-1]
            prev_body = c[-2] - o[-2]

            # Bullish engulfing
            if prev_body < 0 and curr_body > 0:
                if o[-1] < c[-2] and c[-1] > o[-2]:
                    patterns.append({
                        'type': 'bullish_engulfing',
                        'signal': 'bullish_reversal',
                        'strength': 'strong',
                        'price': float(c[-1])
                    })

            # Bearish engulfing
            elif prev_body > 0 and curr_body < 0:
                if o[-1] > c[-2] and c[-1] < o[-2]:
                    patterns.append({
                        'type': 'bearish_engulfing',
                        'signal': 'bearish_reversal',
                        'strength': 'strong',
                        'price': float(c[-1])
                    })

        # Doji (indecision)
        if body < (h[-1] - l[-1]) * 0.1:
            patterns.append({
                'type': 'doji',
                'signal': 'indecision',
                'strength': 'weak',
                'price': float(c[-1])
            })

        # Shooting star (bearish reversal)
        if upper_shadow > body * 2 and lower_shadow < body * 0.3 and c[-2] > o[-2]:
            patterns.append({
                'type': 'shooting_star',
                'signal': 'bearish_reversal',
                'strength': 'medium',
                'price': float(c[-1])
            })

        # Harami
        if len(ohlcv) >= 2:
            prev_body = abs(c[-2] - o[-2])
            curr_body = abs(c[-1] - o[-1])
            if curr_body < prev_body * 0.6:
                if c[-2] < o[-2] and c[-1] > o[-1] and c[-1] < o[-2] and o[-1] > c[-2]:
                    patterns.append({
                        'type': 'bullish_harami',
                        'signal': 'bullish_reversal',
                        'strength': 'medium',
                        'price': float(c[-1])
                    })
                if c[-2] > o[-2] and c[-1] < o[-1] and c[-1] > o[-2] and o[-1] < c[-2]:
                    patterns.append({
                        'type': 'bearish_harami',
                        'signal': 'bearish_reversal',
                        'strength': 'medium',
                        'price': float(c[-1])
                    })

        # Morning/Evening star
        if len(ohlcv) >= 3:
            a_open, a_close = o[-3], c[-3]
            b_open, b_close = o[-2], c[-2]
            c_open, c_close = o[-1], c[-1]
            body1 = abs(a_close - a_open)
            body2 = abs(b_close - b_open)
            body3 = abs(c_close - c_open)
            if a_close < a_open and body2 < body1 * 0.5 and c_close > c_open and c_close > (a_open - body1 * 0.5):
                patterns.append({
                    'type': 'morning_star',
                    'signal': 'bullish_reversal',
                    'strength': 'strong',
                    'price': float(c[-1])
                })
            if a_close > a_open and body2 < body1 * 0.5 and c_close < c_open and c_close < (a_open + body1 * 0.5):
                patterns.append({
                    'type': 'evening_star',
                    'signal': 'bearish_reversal',
                    'strength': 'strong',
                    'price': float(c[-1])
                })

        # Piercing line / Dark cloud cover
        if len(ohlcv) >= 2:
            prev_bear = c[-2] < o[-2]
            prev_bull = c[-2] > o[-2]
            mid_prev = (o[-2] + c[-2]) / 2
            if prev_bear and c[-1] > o[-1] and o[-1] < l[-2] and c[-1] > mid_prev:
                patterns.append({
                    'type': 'piercing_line',
                    'signal': 'bullish_reversal',
                    'strength': 'strong',
                    'price': float(c[-1])
                })
            if prev_bull and c[-1] < o[-1] and o[-1] > h[-2] and c[-1] < mid_prev:
                patterns.append({
                    'type': 'dark_cloud_cover',
                    'signal': 'bearish_reversal',
                    'strength': 'strong',
                    'price': float(c[-1])
                })

        # Three white soldiers / three black crows
        if len(ohlcv) >= 3:
            bullish = (c[-3] > o[-3]) and (c[-2] > o[-2]) and (c[-1] > o[-1])
            bearish = (c[-3] < o[-3]) and (c[-2] < o[-2]) and (c[-1] < o[-1])
            if bullish and c[-1] > c[-2] > c[-3]:
                patterns.append({
                    'type': 'three_white_soldiers',
                    'signal': 'bullish_continuation',
                    'strength': 'strong',
                    'price': float(c[-1])
                })
            if bearish and c[-1] < c[-2] < c[-3]:
                patterns.append({
                    'type': 'three_black_crows',
                    'signal': 'bearish_continuation',
                    'strength': 'strong',
                    'price': float(c[-1])
                })

    return patterns

def detect_chart_formations(ohlcv: pd.DataFrame, min_points: int = 5) -> List[Dict[str, any]]:
    """Detect chart formations: triangles, flags, wedges, rectangles."""
    if len(ohlcv) < 30:
        return []

    patterns = []
    close = ohlcv['close'].values
    high = ohlcv['high'].values if 'high' in ohlcv.columns else close
    low = ohlcv['low'].values if 'low' in ohlcv.columns else close

    # Triangle detection (last 20-30 candles)
    window = min(30, len(ohlcv))
    recent_highs = high[-window:]
    recent_lows = low[-window:]

    # Check for converging highs and lows
    high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
    low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]

    # Ascending triangle: flat resistance, rising support
    if abs(high_slope) < (recent_highs.std() * 0.01) and low_slope > 0:
        patterns.append({
            'type': 'ascending_triangle',
            'signal': 'bullish_continuation',
            'strength': 'strong',
            'breakout_level': float(recent_highs.max()),
            'target': float(recent_highs.max() * 1.05)
        })

    # Descending triangle: flat support, falling resistance
    elif abs(low_slope) < (recent_lows.std() * 0.01) and high_slope < 0:
        patterns.append({
            'type': 'descending_triangle',
            'signal': 'bearish_continuation',
            'strength': 'strong',
            'breakdown_level': float(recent_lows.min()),
            'target': float(recent_lows.min() * 0.95)
        })

    # Symmetrical triangle: both converging
    elif high_slope < 0 and low_slope > 0:
        patterns.append({
            'type': 'symmetrical_triangle',
            'signal': 'neutral_breakout',
            'strength': 'medium',
            'upper_level': float(recent_highs[-1]),
            'lower_level': float(recent_lows[-1])
        })

    # Rectangle (range-bound)
    range_pct = (recent_highs.max() - recent_lows.min()) / max(recent_lows.min(), 1e-9)
    if abs(high_slope) < (recent_highs.std() * 0.01) and abs(low_slope) < (recent_lows.std() * 0.01) and range_pct < 0.08:
        patterns.append({
            'type': 'rectangle',
            'signal': 'neutral_range',
            'strength': 'medium',
            'upper_level': float(recent_highs.max()),
            'lower_level': float(recent_lows.min())
        })

    # Wedges
    if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
        patterns.append({
            'type': 'rising_wedge',
            'signal': 'bearish_reversal',
            'strength': 'medium',
            'upper_level': float(recent_highs[-1]),
            'lower_level': float(recent_lows[-1])
        })
    if high_slope < 0 and low_slope < 0 and high_slope < low_slope:
        patterns.append({
            'type': 'falling_wedge',
            'signal': 'bullish_reversal',
            'strength': 'medium',
            'upper_level': float(recent_highs[-1]),
            'lower_level': float(recent_lows[-1])
        })

    # Flags / pennants (simple heuristic on last 20 candles)
    if len(close) >= 40:
        pre_move = close[-40] - close[-20]
        post_move = close[-1] - close[-20]
        vol = np.std(close[-20:]) / max(close[-20:].mean(), 1e-9)
        if pre_move > 0 and post_move < 0 and vol < 0.02:
            patterns.append({
                'type': 'bull_flag',
                'signal': 'bullish_continuation',
                'strength': 'medium',
                'breakout_level': float(recent_highs.max())
            })
        if pre_move < 0 and post_move > 0 and vol < 0.02:
            patterns.append({
                'type': 'bear_flag',
                'signal': 'bearish_continuation',
                'strength': 'medium',
                'breakdown_level': float(recent_lows.min())
            })

    return patterns
def find_support_resistance(ohlcv: pd.DataFrame, sensitivity: float = 0.02) -> Dict[str, List[float]]:
    """Find key support and resistance levels using pivot points"""
    if len(ohlcv) < 50:
        return {'support': [], 'resistance': []}

    high = ohlcv['high'] if 'high' in ohlcv.columns else ohlcv['close']
    low = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']
    close = ohlcv['close']

    # Find local peaks and troughs
    resistance_levels = []
    support_levels = []

    for i in range(10, len(high) - 10):
        # Resistance: local maximum
        if high.iloc[i] == high.iloc[i-10:i+10].max():
            level = high.iloc[i]
            # Check if level was tested multiple times
            touches = ((high - level).abs() / level < sensitivity).sum()
            if touches >= 2:
                resistance_levels.append(float(level))

        # Support: local minimum
        if low.iloc[i] == low.iloc[i-10:i+10].min():
            level = low.iloc[i]
            touches = ((low - level).abs() / level < sensitivity).sum()
            if touches >= 2:
                support_levels.append(float(level))

    # Remove duplicates within sensitivity range
    def dedupe_levels(levels):
        if not levels:
            return []
        sorted_levels = sorted(set(levels))
        deduped = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            if (level - deduped[-1]) / deduped[-1] > sensitivity:
                deduped.append(level)
        return deduped[-3:]  # Keep top 3 most recent

    return {
        'support': dedupe_levels(support_levels),
        'resistance': dedupe_levels(resistance_levels)
    }

def analyze_volume_profile(ohlcv: pd.DataFrame, bins: int = 10) -> Dict[str, any]:
    """Analyze volume distribution across price levels"""
    if 'volume' not in ohlcv.columns or len(ohlcv) < 20:
        return {'poc': None, 'high_volume_nodes': [], 'volume_trend': 'neutral'}

    close = ohlcv['close']
    volume = ohlcv['volume']

    # Point of Control (POC) - price with highest volume
    price_bins = pd.cut(close, bins=bins)
    volume_profile = volume.groupby(price_bins).sum()

    if not volume_profile.empty:
        poc_range = volume_profile.idxmax()
        poc = float(poc_range.mid) if hasattr(poc_range, 'mid') else float(close.mean())
    else:
        poc = float(close.mean())

    # Volume trend
    recent_vol = volume.iloc[-5:].mean()
    older_vol = volume.iloc[-20:-5].mean()
    vol_ratio = recent_vol / older_vol if older_vol > 0 else 1

    if vol_ratio > 1.5:
        volume_trend = 'increasing'
    elif vol_ratio < 0.7:
        volume_trend = 'decreasing'
    else:
        volume_trend = 'stable'

    # High volume nodes (support/resistance areas)
    high_volume_nodes = []
    if not volume_profile.empty:
        threshold = volume_profile.quantile(0.7)
        for interval, vol in volume_profile.items():
            if vol > threshold and hasattr(interval, 'mid'):
                high_volume_nodes.append(float(interval.mid))

    return {
        'poc': poc,
        'high_volume_nodes': high_volume_nodes[:3],
        'volume_trend': volume_trend,
        'volume_ratio': float(vol_ratio)
    }


async def generate_probabilistic_signals(coin_id: str, ohlcv: pd.DataFrame) -> Optional[Dict]:
    """
    NEW: Generate probabilistic signals using AI synthesis engine

    This replaces the legacy generate_signals() with continuous confidence scores.

    Args:
        coin_id: Coin identifier
        ohlcv: OHLCV DataFrame

    Returns:
        Dict with 'signal_type', 'confidence' (0-1), 'analysis', or None if unavailable
    """
    try:
        from core.data_collectors.market_state_builder import get_market_state_builder
        from core.synthesis.ai_synthesis_engine import get_ai_synthesis_engine

        # Get builder and engine
        builder = get_market_state_builder()
        engine = get_ai_synthesis_engine()

        # Build market state
        market_state = await builder.build_market_state(coin_id)
        if market_state is None:
            return None

        # Run synthesis
        analysis = await engine.synthesize(market_state)

        return {
            'signal_type': analysis.signal_type,
            'confidence': analysis.confidence,
            'target_price': analysis.target_price,
            'stop_loss': analysis.stop_loss,
            'risk_level': analysis.risk_level,
            'analysis': analysis,  # Full SynthesizedAnalysis object
            '_probabilistic': True
        }

    except ImportError:
        # Synthesis engine not available
        return None
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error generating probabilistic signals: {e}")
        return None


