import logging
import hashlib
import time, json, os, config
import re, html
import fasteners

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from config import OWNER_ID, TOKEN
from locales.translator import t

from modules.news import get_significant_news

from telegram import Bot
from telegram.constants import ParseMode
from telegram import InlineKeyboardMarkup, InlineKeyboardButton

from core.api.binance_client import get_batch_ohlcv
from core.analysis.signal_generator import detect_patterns, detect_candlestick_patterns, detect_chart_formations
from core.analysis.pattern_tracker import pattern_tracker
from core.analysis.technical_analysis import rsi
from core.utils import COIN_ID_MAP
from services.coin_registry_service import get_coin_registry_service
from core.portfolio_db import get_portfolio_manager
from core.access.access_manager import get_access_manager, AccessTier

from pathlib import Path

# Set up logger before it's used
logger = logging.getLogger(__name__)

# Import AI synthesis components for feedback loop
try:
    from core.synthesis.feedback_loop import get_feedback_loop
    from core.synthesis.market_state_vector import create_empty_market_state, MarketContext
    _FEEDBACK_AVAILABLE = True
except ImportError:
    _FEEDBACK_AVAILABLE = False
    logger.warning("Feedback loop not available, using legacy pattern confidence")


ALERT_STATE_FILE = 'data/alerts_state.json'
LOCK = fasteners.InterProcessLock('data/alerts_state.lock')

def make_event_key(coin: str | None, alert_type: str, extra: str | None = None) -> str:
    """
    Строим ключ вида:
    BTC:RSI_EXTREME:4h
    или
    PORTFOLIO:portfolio_take_profit
    """
    parts = []
    if coin:
        parts.append(coin.upper())
    else:
        parts.append("GLOBAL")

    parts.append(alert_type)

    if extra:
        parts.append(extra)

    return ":".join(parts)

def _load_alert_state():
    if not os.path.exists(ALERT_STATE_FILE):
        return {}
    with LOCK:
        with open(ALERT_STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

def _save_alert_state(data):
    Path('data').mkdir(exist_ok=True)
    with LOCK:
        with open(ALERT_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)




def _alert_keyboard(alert_id: str, user_id: int = None) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t("alerts.button_done", user_id=user_id), callback_data=f"alert_done:{alert_id}"),
         InlineKeyboardButton(t("alerts.button_remind", user_id=user_id), callback_data=f"alert_remind:{alert_id}")],
        [InlineKeyboardButton(t("alerts.button_ignore", user_id=user_id), callback_data=f"alert_ignore:{alert_id}")]
    ])

class AlertConfig:
    """Конфигурация алертов для свинг + инвестиционной стратегии"""
    # Паттерны графиков
    PATTERN_BREAKOUT_THRESHOLD = 0.98  # 98% от уровня сопротивления
    VOLUME_SPIKE_MULTIPLIER = 3.0      # Объём x3 от среднего
    PATTERN_MIN_CONFIDENCE = 70        # Минимальная уверенность для паттерна
    # Пороги для портфеля (критические события)
    PORTFOLIO_STOP_LOSS = -12.0        # % убытка для стоп-лосс
    PORTFOLIO_TAKE_PROFIT = 20.0       # % прибыли для тейк-профит
    PORTFOLIO_RSI_OVERSOLD = 25        # RSI перепроданность
    PORTFOLIO_RSI_OVERBOUGHT = 75      # RSI перекупленность

    # Пороги для ТОП-20 (новые возможности)
    TOP20_PUMP_THRESHOLD = 12.0        # % роста за час
    TOP20_DUMP_THRESHOLD = -12.0       # % падения за час
    TOP20_VOLUME_MULTIPLIER = 2.0      # Минимальный рост объема
    TOP20_RSI_EXTREME_LOW = 20         # Экстремальная перепроданность
    TOP20_RSI_EXTREME_HIGH = 80        # Экстремальная перекупленность


    # Фильтры качества
    ALERT_COOLDOWN_HOURS = 6           # Часов между похожими алертами
    MAX_ALERTS_PER_CYCLE = 3           # Максимум алертов за проверку
    AI_CONFIDENCE_THRESHOLD = 70       # Минимальная оценка AI

    # ТОП-20 монет для мониторинга
    TOP20_COINS = [
        "bitcoin", "ethereum", "binancecoin", "solana", "ripple",
        "cardano", "avalanche-2", "dogecoin", "tron", "chainlink",
        "polkadot", "polygon", "internet-computer", "shiba-inu", "uniswap",
        "litecoin", "bitcoin-cash", "near", "leo-token", "stellar"
    ]

class AlertType:
    # Портфель алерты
    PORTFOLIO_STOP_LOSS = "portfolio_stop_loss"
    PORTFOLIO_TAKE_PROFIT = "portfolio_take_profit"
    PORTFOLIO_RSI_SIGNAL = "portfolio_rsi_signal"

    # ТОП-20 возможности
    MARKET_PUMP = "market_pump"
    MARKET_DUMP = "market_dump"
    MARKET_RSI_EXTREME = "market_rsi_extreme"

    # Новости
    CRITICAL_NEWS = "critical_news"
    # Графические паттерны
    PATTERN_BREAKOUT = "pattern_breakout"
    PATTERN_REVERSAL = "pattern_reversal"
    VOLUME_ANOMALY = "volume_anomaly"

class SmartAlertsSystem:
    def __init__(self):
        self.bot = Bot(token=TOKEN)
        self.subscribers = self._load_subscribers()
        self.last_prices = {}           # Кэш цен для сравнения
        self.last_alerts = {}           # Время последних алертов
        self.price_history = {}         # История цен для анализа трендов
        self.alerts_sent_today = 0      # Счетчик алертов за день
        self.registry = get_coin_registry_service()

    def _should_send_alert(self, coin: str, alert_type: str) -> bool:
        """Проверяет cooldown для алертов"""
        key = f"{coin}_{alert_type}"
        now = datetime.now()

        if key in self.last_alerts:
            hours_passed = (now - self.last_alerts[key]).total_seconds() / 3600
            if hours_passed < AlertConfig.ALERT_COOLDOWN_HOURS:
                return False

        self.last_alerts[key] = now
        return True

    def _load_subscribers(self):
        access_manager = get_access_manager()
        premium_users: list[int] = []

        for uid_str, data in access_manager.users.get("users", {}).items():
            try:
                uid = int(uid_str)
            except ValueError:
                continue
            tier = access_manager.get_user_tier(uid)
            if tier in (AccessTier.PREMIUM, AccessTier.BETA, AccessTier.OWNER):
                premium_users.append(uid)

        if OWNER_ID:
            premium_users.append(int(OWNER_ID))

        return {"free": [], "premium": sorted(set(premium_users))}

    def _calculate_ai_confidence(self, coin: str, price_change: float, volume_change: float, rsi: Optional[float]) -> int:
        """Рассчитывает уверенность AI в сигнале (0-100%)"""
        confidence = 50  # Базовая уверенность

        # Сила движения цены
        abs_change = abs(price_change)
        if abs_change > 15:
            confidence += 20
        elif abs_change > 10:
            confidence += 10

        # Подтверждение объемом
        if volume_change > 3.0:
            confidence += 15
        elif volume_change > 2.0:
            confidence += 10

        # RSI подтверждение
        if rsi:
            if (price_change > 0 and rsi < 30) or (price_change < 0 and rsi > 70):
                confidence += 15  # Дивергенция - сильный сигнал
            elif (price_change > 0 and rsi > 70) or (price_change < 0 and rsi < 30):
                confidence += 10  # Подтверждение тренда

        return min(100, max(0, confidence))

    def _format_recommendation(self, coin: str, signal_type: str, current_price: float, ai_confidence: int) -> str:
        """Генерирует конкретные рекомендации с уровнями"""

        if signal_type == "pump":
            target = current_price * 1.15  # +15% цель
            stop = current_price * 0.95    # -5% стоп
            return f"ЛОНГ от ${current_price:.2f}, цель ${target:.2f}, стоп ${stop:.2f}"

        elif signal_type == "dump":
            target = current_price * 0.85  # -15% цель для шорта
            stop = current_price * 1.05    # +5% стоп
            return f"ШОРТ от ${current_price:.2f}, цель ${target:.2f}, стоп ${stop:.2f}"

        elif signal_type == "oversold":
            target = current_price * 1.20  # +20% цель
            stop = current_price * 0.92    # -8% стоп
            return f"ПОКУПКА от ${current_price:.2f}, цель ${target:.2f}, стоп ${stop:.2f}"

        elif signal_type == "overbought":
            target = current_price * 0.80  # -20% цель
            stop = current_price * 1.08    # +8% стоп
            return f"ПРОДАЖА от ${current_price:.2f}, цель ${target:.2f}, стоп ${stop:.2f}"

        elif signal_type == "take_profit":
            partial_target = current_price * 0.95  # Зафиксировать по текущей с небольшим отступом
            return f"ФИКСИРОВАТЬ 50-70% позиции около ${partial_target:.2f}"

        elif signal_type == "stop_loss":
            return f"РАССМОТРЕТЬ выход ИЛИ усреднение если есть резерв"

        return "МОНИТОРИТЬ ситуацию"


    def _recent_alert_ok(self, alert_type: str, key: str) -> bool:
        """Return True if allowed to send; False if duplicate within dedup window."""
        st = _load_alert_state()
        now = time.time()
        window = int(os.getenv("ALERT_DEDUP_WINDOW_SECONDS", "7200"))
        # find any matching key within window
        for aid, rec in st.items():
            if rec.get("type")==alert_type and rec.get("key")==key and rec.get("status") in ("new","done","ignored","remind"):
                if now - rec.get("ts", 0) < window:
                    return False
        return True


    async def _send_alert(self, message: str, alert_type: str, priority: str = "normal", coin: Optional[str] = None):
            """Отправляет алерт с приоритизацией"""
            target_chat = int(os.getenv("ALERT_CHAT_ID", "0")) \
                          or getattr(config, "ALERT_CHAT_ID", None) \
                          or getattr(config, "OWNER_ID", None)
            try:
                # Проверяем лимит алертов
                if self.alerts_sent_today >= AlertConfig.MAX_ALERTS_PER_CYCLE * 8:  # 8 проверок в час × лимит
                    logger.warning("Достигнут дневной лимит алертов")
                    return
                # Эмодзи по типу алерта
                emoji_map = {
                    AlertType.PORTFOLIO_STOP_LOSS: "🚨",
                    AlertType.PORTFOLIO_TAKE_PROFIT: "🎯",
                    AlertType.PORTFOLIO_RSI_SIGNAL: "💎",
                    AlertType.MARKET_PUMP: "🚀",
                    AlertType.MARKET_DUMP: "💥",
                    AlertType.MARKET_RSI_EXTREME: "⚡",
                    AlertType.CRITICAL_NEWS: "📰"
                }

                priority_emoji = "🔥🔥🔥" if priority == "high" else "🔥" if priority == "medium" else ""
                emoji = emoji_map.get(alert_type, "🔔")

                alert_id = hashlib.md5(f"{alert_type}:{message}".encode()).hexdigest()[:10]
                key = make_event_key(coin, alert_type)
                # Сохраняем состояние алерта
                state = _load_alert_state()
                state[alert_id] = {
                    "type": alert_type,
                    "coin": coin,
                    "message": message,
                    "priority": priority,
                    "chat_id": target_chat,
                    "created_at": time.time(),
                    "status": "new",
                    "key": key,
                    "ts": time.time()
                }

                _save_alert_state(state)

                safe = html.escape(message)
                safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)  # поддержка **жирного** в тексте
                formatted_message = f"{emoji} <b>АЛЕРТ</b> {emoji} {priority_emoji}\n{safe}"

                self.subscribers = self._load_subscribers()
                for user_id in self.subscribers.get("premium", []):
                    try:
                        await self.bot.send_message(
                            chat_id=user_id,
                            text=formatted_message,
                            parse_mode=ParseMode.HTML,
                            reply_markup=_alert_keyboard(alert_id, user_id),
                        )
                    except Exception as e:
                        logger.error(f"Не удалось отправить алерт {user_id}: {e}")

                self.alerts_sent_today += 1
                logger.info(f"Алерт отправлен: {alert_type} (приоритет: {priority})")

            except Exception as e:
                logger.error(f"Ошибка отправки алерта: {e}")

    async def check_portfolio_alerts(self, context) -> List[str]:
        """Проверяет алерты по портфелю всех eligible пользователей, используя данные из снапшота."""
        pm = get_portfolio_manager()
        access_manager = get_access_manager()
        
        snap_service = context.application.bot_data.get("market_snapshot")
        markets_map = snap_service.get_latest().markets if snap_service and snap_service.get_latest() else {}
        if not markets_map:
            logger.warning("Не удалось получить markets_map из снапшота для алертов по портфелю.")
            return []

        alerts = []
        
        # Get all users with portfolios
        all_users = pm.get_all_users()
        
        # Filter by eligible users (PREMIUM/BETA/OWNER)
        eligible_tiers = {AccessTier.PREMIUM, AccessTier.BETA, AccessTier.OWNER}
        eligible_users = [uid for uid in all_users if access_manager.get_user_tier(uid) in eligible_tiers]
        
        if not eligible_users:
            logger.debug("No eligible users for portfolio alerts")
            return []
        
        # Check each eligible user's portfolio
        for user_id in eligible_users:
            portfolio = pm.load_portfolio(user_id)
            if not portfolio:
                continue
            
            for coin, data in portfolio.items():
                coin_id = COIN_ID_MAP.get(coin.upper(), coin.lower())
                market_data = markets_map.get(coin_id)
                if not market_data:
                    continue

                current_price_raw = market_data.get('current_price')
                try:
                    current_price = float(current_price_raw)
                except (TypeError, ValueError):
                    continue
                if current_price <= 0:
                    continue

                buy_price_raw = data.get('buy_price', data.get('price', 0))
                try:
                    buy_price = float(buy_price_raw)
                except (TypeError, ValueError):
                    continue
                if buy_price <= 0:
                    continue

                change_pct = ((current_price - buy_price) / buy_price) * 100
                # Пример правила
                if change_pct >= AlertConfig.PORTFOLIO_TAKE_PROFIT and self._should_send_alert(coin, AlertType.PORTFOLIO_TAKE_PROFIT):
                    msg = f"🎯 **{coin.upper()}: TAKE PROFIT**\nРост: **{change_pct:+.2f}%** (${buy_price:.2f} → ${current_price:.2f}).\n💡 **РЕКОМЕНДАЦИЯ:** {self._format_recommendation(coin, 'take_profit', current_price, 90)}"
                    await self._send_alert(msg, AlertType.PORTFOLIO_TAKE_PROFIT, "high", coin=coin)
                    alerts.append(f"Take Profit: {coin.upper()}")

                if change_pct <= AlertConfig.PORTFOLIO_STOP_LOSS and self._should_send_alert(coin, AlertType.PORTFOLIO_STOP_LOSS):
                    msg = f"🚨 **{coin.upper()}: STOP LOSS**\nПадение: **{change_pct:.2f}%** (${buy_price:.2f} → ${current_price:.2f}).\n💡 **РЕКОМЕНДАЦИЯ:** {self._format_recommendation(coin, 'stop_loss', current_price, 90)}"
                    await self._send_alert(msg, AlertType.PORTFOLIO_STOP_LOSS, "high", coin=coin)
                    alerts.append(f"Stop Loss: {coin.upper()}")
        return alerts

    async def check_top20_opportunities(self, context) -> List[str]:
        """Ищет торговые возможности в ТОП-20, используя данные из снапшотов."""
        snap_service = context.application.bot_data.get("market_snapshot")
        history_service = context.application.bot_data.get("history_snapshot")

        if not snap_service or not snap_service.get_latest() or not history_service or not history_service.get_latest():
            logger.warning("Снапшоты не готовы для анализа ТОП-20.")
            return []

        markets_map = snap_service.get_latest().markets
        history_set = history_service.get_latest()
        alerts = []
        opportunities_found = 0

        for coin_id in AlertConfig.TOP20_COINS:
            if opportunities_found >= AlertConfig.MAX_ALERTS_PER_CYCLE:
                break

            market_row = markets_map.get(coin_id, {})
            current_price = market_row.get('current_price')
            if not current_price:
                continue

            price_change_pct = market_row.get('price_change_percentage_1h_in_currency', 0) or 0
            volume_24h = market_row.get('total_volume', 0) or 0
            volume_ratio = min(volume_24h / 100000000, 5.0)

            real_rsi = None
            if coin_id in history_set.by_coin:
                df = history_set.by_coin[coin_id]
                if len(df) >= 15:
                    real_rsi = rsi(df['price'], period=14).iloc[-1]
            # Получаем OHLCV данные с Binance
            ohlcv_data = await get_batch_ohlcv([coin_id], interval="1h")
            if coin_id in ohlcv_data:
                df = ohlcv_data[coin_id]

                # Детектим паттерны
                patterns = detect_patterns(df)
                candles = detect_candlestick_patterns(df)
                formations = detect_chart_formations(df)

                # Проверяем каждый паттерн
                for pattern in patterns + candles + formations:
                    pattern_type = pattern.get('type')
                    pattern_stats = pattern_tracker.get_pattern_stats(pattern_type)

                    # NEW: Use feedback loop for confidence adjustment
                    confidence_boost = 0
                    adjusted_confidence = 70  # Base confidence
                    historical_context = None

                    if _FEEDBACK_AVAILABLE:
                        try:
                            # Create minimal market state for pattern
                            pattern_signal = 'bullish' if 'ascending' in pattern_type or 'bullish' in pattern.get('signal', '') else 'bearish'
                            signal_type = 'buy' if pattern_signal == 'bullish' else 'sell'

                            # Build minimal market state
                            market_state = create_empty_market_state(coin_id, current_price)
                            # Set market context based on available data
                            if real_rsi is not None:
                                if real_rsi < 35:
                                    market_state.context.rsi_zone = 'oversold'
                                elif real_rsi > 65:
                                    market_state.context.rsi_zone = 'overbought'

                            # Query feedback loop
                            feedback = get_feedback_loop()
                            historical_context = feedback.get_historical_context(
                                coin_id, market_state, signal_type
                            )

                            if historical_context:
                                # Convert continuous adjustment to confidence boost (0-100 scale)
                                # confidence_adjustment is -0.25 to +0.25
                                confidence_boost = int(historical_context.confidence_adjustment * 100)
                                adjusted_confidence = 70 + confidence_boost

                                logger.info(
                                    f"Feedback loop for {coin_id} {pattern_type}: "
                                    f"{historical_context.similar_signals_count} signals, "
                                    f"avg quality: {historical_context.avg_quality_score:.1%}, "
                                    f"boost: {confidence_boost:+d}"
                                )
                            else:
                                # Insufficient data, use pattern_stats as fallback
                                if pattern_stats['winrate'] >= 60 and pattern_stats['total'] >= 10:
                                    confidence_boost = 20
                                elif pattern_stats['total'] < 10:
                                    confidence_boost = 0
                                else:
                                    confidence_boost = -10
                                adjusted_confidence = 70 + confidence_boost
                        except Exception as e:
                            logger.error(f"Error querying feedback loop: {e}")
                            # Fallback to legacy logic
                            if pattern_stats['winrate'] >= 60 and pattern_stats['total'] >= 10:
                                confidence_boost = 20
                            elif pattern_stats['total'] < 10:
                                confidence_boost = 0
                            else:
                                confidence_boost = -10
                            adjusted_confidence = 70 + confidence_boost
                    else:
                        # LEGACY: Учитываем историческую эффективность
                        if pattern_stats['winrate'] >= 60 and pattern_stats['total'] >= 10:
                            confidence_boost = 20
                        elif pattern_stats['total'] < 10:
                            confidence_boost = 0  # Недостаточно данных
                        else:
                            confidence_boost = -10
                        adjusted_confidence = 70 + confidence_boost

                    # Если паттерн сильный и подтверждён объёмом
                    if pattern.get('strength') in ['strong', 'medium']:
                        if 'breakout' in pattern_type and self._should_send_alert(coin_id, AlertType.PATTERN_BREAKOUT):
                            # Сохраняем сигнал для трекинга
                            pattern_tracker.save_signal(
                                coin_id,
                                pattern_type,
                                'bullish' if 'ascending' in pattern_type or 'bullish' in pattern.get('signal', '') else 'bearish',
                                pattern.get('target', current_price * 1.05),
                                current_price
                            )

                            coin_symbol = next((k for k, v in COIN_ID_MAP.items() if v == coin_id), coin_id.upper())

                            # Build message with historical context
                            message_parts = [
                                f"**{coin_symbol}: ГРАФИЧЕСКИЙ ПАТТЕРН**",
                                f"📊 Паттерн: {pattern_type}",
                                f"💪 Сила: {pattern.get('strength')}",
                                f"🎯 Цель: ${pattern.get('target', current_price * 1.05):.4f}",
                            ]

                            # Add historical validation info
                            if historical_context:
                                message_parts.append(
                                    f"📈 AI Historical: {historical_context.similar_signals_count} signals, "
                                    f"{historical_context.avg_quality_score:.0%} avg quality (confidence: {confidence_boost:+d}%)"
                                )
                            else:
                                message_parts.append(
                                    f"📈 Pattern WR: {pattern_stats['winrate']:.1f}% ({pattern_stats['total']} signals)"
                                )

                            message_parts.append(
                                f"💡 **РЕКОМЕНДАЦИЯ:** {self._format_recommendation(coin_id, pattern_type, current_price, adjusted_confidence)}"
                            )

                            message = "\n".join(message_parts)
                            await self._send_alert(message, AlertType.PATTERN_BREAKOUT, "medium", coin = coin_id)
                            alerts.append(f"Pattern: {coin_symbol} - {pattern_type}")
                            opportunities_found += 1
            # Логика для PUMP сигнала
            if price_change_pct >= AlertConfig.TOP20_PUMP_THRESHOLD and volume_ratio >= AlertConfig.TOP20_VOLUME_MULTIPLIER:
                ai_confidence = self._calculate_ai_confidence(coin_id, price_change_pct, volume_ratio, real_rsi)
                if ai_confidence >= AlertConfig.AI_CONFIDENCE_THRESHOLD and self._should_send_alert(coin_id, AlertType.MARKET_PUMP):
                    recommendation = self._format_recommendation(coin_id, "pump", current_price, ai_confidence)
                    coin_symbol = next((k for k, v in COIN_ID_MAP.items() if v == coin_id), coin_id.upper())
                    message = (
                        f"**{coin_symbol}: МОЩНЫЙ ПРОБОЙ ВВЕРХ**\n"
                        f"🚀 Рост: +{price_change_pct:.1f}% за 1ч\n"
                        f"📊 Объем: x{volume_ratio:.1f} от среднего\n"
                        f"💰 Цена: ${current_price:.4f}\n"
                        f"📈 RSI: {f'{real_rsi:.1f}' if real_rsi else 'н/д'}\n"
                        f"🤖 AI уверенность: {ai_confidence}%\n"
                        f"💡 **РЕКОМЕНДАЦИЯ:** {recommendation}"
                    )
                    await self._send_alert(message, AlertType.MARKET_PUMP, "medium", coin = coin_id)
                    alerts.append(f"Pump: {coin_symbol}")
                    opportunities_found += 1

            # Логика для DUMP сигнала
            elif price_change_pct <= AlertConfig.TOP20_DUMP_THRESHOLD and volume_ratio >= AlertConfig.TOP20_VOLUME_MULTIPLIER:
                ai_confidence = self._calculate_ai_confidence(coin_id, price_change_pct, volume_ratio, real_rsi)
                if ai_confidence >= AlertConfig.AI_CONFIDENCE_THRESHOLD and self._should_send_alert(coin_id, AlertType.MARKET_DUMP):
                    recommendation = self._format_recommendation(coin_id, "dump", current_price, ai_confidence)
                    coin_symbol = next((k for k, v in COIN_ID_MAP.items() if v == coin_id), coin_id.upper())
                    message = (
                        f"**{coin_symbol}: ОБВАЛЬНОЕ ПАДЕНИЕ**\n"
                        f"💥 Падение: {price_change_pct:.1f}% за 1ч\n"
                        f"📊 Объем: x{volume_ratio:.1f} от среднего\n"
                        f"💰 Текущая цена: ${current_price:.4f}\n"
                        f"📉 RSI: {f'{real_rsi:.1f}' if real_rsi else 'н/д'}\n"
                        f"🤖 AI уверенность: {ai_confidence}%\n"
                        f"💡 **РЕКОМЕНДАЦИЯ:** {recommendation}"
                    )
                    await self._send_alert(message, AlertType.MARKET_DUMP, "medium", coin = coin_id)
                    alerts.append(f"Dump: {coin_symbol}")
                    opportunities_found += 1

            # Логика для EXTREME RSI
            if real_rsi:
                if real_rsi <= AlertConfig.TOP20_RSI_EXTREME_LOW:
                    if self._should_send_alert(coin_id, f"{AlertType.MARKET_RSI_EXTREME}_low"):
                        ai_confidence = self._calculate_ai_confidence(coin_id, -5, volume_ratio, real_rsi)
                        recommendation = self._format_recommendation(coin_id, "oversold", current_price, ai_confidence)
                        coin_symbol = next((k for k, v in COIN_ID_MAP.items() if v == coin_id), coin_id.upper())
                        message = (
                            f"**{coin_symbol}: ЭКСТРЕМАЛЬНАЯ ПЕРЕПРОДАННОСТЬ**\n"
                            f"⚡ RSI 14D: {real_rsi:.1f} (критически низкий)\n"
                            f"💰 Текущая цена: ${current_price:.4f}\n"
                            f"🔄 Потенциал отскока: высокий\n"
                            f"🤖 AI уверенность: {ai_confidence}%\n"
                            f"💡 **РЕКОМЕНДАЦИЯ:** {recommendation}"
                        )
                        await self._send_alert(message, AlertType.MARKET_RSI_EXTREME, "medium", coin = coin_id)
                        alerts.append(f"RSI extreme low: {coin_symbol}")
                        opportunities_found += 1

        return alerts

    async def check_critical_news(self) -> List[str]:
        """Проверяет критические новости"""
        alerts = []

        try:
            if not self._should_send_alert("market", AlertType.CRITICAL_NEWS):
                return alerts

            news_items = get_significant_news()

            for item in news_items[:1]:  # Только самая важная новость
                title = item['news'].get('title', 'Без заголовка')
                analysis = item['analysis']

                # Проверяем критичность новости
                critical_keywords = ['негативный', 'negative', 'hacked', 'ban', 'regulation', 'lawsuit']
                positive_keywords = ['позитивный', 'positive', 'etf', 'adoption', 'partnership']

                is_critical = any(keyword in analysis.lower() for keyword in critical_keywords)
                is_very_positive = any(keyword in analysis.lower() for keyword in positive_keywords)

                if is_critical or is_very_positive:
                    impact = "НЕГАТИВНОЕ" if is_critical else "ПОЗИТИВНОЕ"
                    recommendation = "Готовьтесь к волатильности, рассмотрите хеджирование" if is_critical else "Возможен рост рынка, следите за возможностями"

                    message = (
                        f"**КРИТИЧЕСКАЯ НОВОСТЬ**\n"
                        f"📰 {title}\n"
                        f"🧠 AI анализ: {analysis}\n"
                        f"⚡ Влияние на рынок: {impact}\n"
                        f"💡 **РЕКОМЕНДАЦИЯ:** {recommendation}"
                    )

                    await self._send_alert(message, AlertType.CRITICAL_NEWS, "high")
                    alerts.append("Critical news")
                    break

        except Exception as e:
            logger.error(f"Ошибка проверки новостей: {e}")

        return alerts

    async def run_full_scan(self, context):
        """Запускает полное сканирование всех алертов"""
        start_time = datetime.now()
        logger.info("🔍 Запуск полного сканирования алертов...")

        try:
            try:
                snap_service = context.application.bot_data.get("market_snapshot") if context and context.application else None
                snap = snap_service.get_latest() if snap_service else None
                if snap and snap.markets:
                    current_prices = {
                        cid: row.get("current_price")
                        for cid, row in snap.markets.items()
                        if row.get("current_price") is not None
                    }
                    if current_prices:
                        pattern_tracker.check_results(current_prices)
            except Exception as e:
                logger.warning("Pattern tracker update failed: %s", e)
            # Сброс счетчика если прошли сутки
            if hasattr(self, 'last_reset') and (datetime.now() - self.last_reset).days >= 1:
                self.alerts_sent_today = 0
                self.last_reset = datetime.now()
            elif not hasattr(self, 'last_reset'):
                self.last_reset = datetime.now()

            all_alerts = []

            # 1. Приоритет: проверяем портфель (критические события)
            portfolio_alerts = await self.check_portfolio_alerts(context)
            all_alerts.extend(portfolio_alerts)

            # 2. Проверяем возможности в ТОП-20 (если есть место)
            if len(all_alerts) < AlertConfig.MAX_ALERTS_PER_CYCLE:
                top20_alerts = await self.check_top20_opportunities(context)
                all_alerts.extend(top20_alerts)

            # 3. Проверяем критические новости (если есть место)
            if len(all_alerts) < AlertConfig.MAX_ALERTS_PER_CYCLE:
                news_alerts = await self.check_critical_news()
                all_alerts.extend(news_alerts)

            execution_time = (datetime.now() - start_time).total_seconds()

            if all_alerts:
                logger.info(f"✅ Сканирование завершено за {execution_time:.1f}с. Отправлено алертов: {len(all_alerts)}")
                logger.info(f"Алерты: {', '.join(all_alerts)}")
            else:
                logger.info(f"✅ Сканирование завершено за {execution_time:.1f}с. Критических событий не обнаружено.")

        except Exception as e:
            logger.error(f"Критическая ошибка в системе алертов: {e}")

# Глобальный экземпляр системы алертов
smart_alerts = SmartAlertsSystem()

# Функция для использования в job_queue
async def run_alerts_scan(context):
    """Функция для запуска из JobQueue"""
    await smart_alerts.run_full_scan(context)


