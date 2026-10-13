import logging
import time, json, os, config
import re, html
import secrets
import fasteners

from typing import List, Optional
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
from core.utils import COIN_ID_MAP, resolve_coin_id
from services.coin_registry_service import get_coin_registry_service
from core.portfolio_db import get_portfolio_manager
from core.access.access_manager import get_access_manager, AccessTier
from core.incidents import report_incident

from pathlib import Path

# Set up logger before it's used
logger = logging.getLogger(__name__)

# Historical pattern context: the previous implementation queried the
# feedback loop with a default-filled "empty market state", i.e. fabricated
# evidence. Until the prediction ledger (Package A3) provides real validated
# events, alerts use only the stored pattern outcome stats.


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


def _fresh_snapshot(service, max_age: float):
    if service and hasattr(service, "get_usable"):
        return service.get_usable(max_age)
    snapshot = service.get_latest() if service else None
    timestamp = getattr(snapshot, "ts", None)
    if not timestamp or time.time() - timestamp > max_age:
        return None
    return snapshot

def _load_alert_state():
    with LOCK:
        return _load_alert_state_unlocked()


def _load_alert_state_unlocked():
    if not os.path.exists(ALERT_STATE_FILE):
        return {}
    try:
        with open(ALERT_STATE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.exception("Alert state is unreadable; starting with an empty state")
        return {}


def _write_alert_state_unlocked(data):
    path = Path(ALERT_STATE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)

def _save_alert_state(data):
    with LOCK:
        _write_alert_state_unlocked(data)


def _mutate_alert_state(mutator):
    """Read-modify-write alert state while holding one inter-process lock."""
    with LOCK:
        state = _load_alert_state_unlocked()
        result = mutator(state)
        _write_alert_state_unlocked(state)
        return result


def update_alert_action(alert_id: str, user_id: int, action: str, now: Optional[float] = None) -> bool:
    """Apply a callback only to an alert owned by the requesting user."""
    timestamp = time.time() if now is None else now

    def mutate(state):
        record = state.get(alert_id)
        if not record or int(record.get("chat_id", 0)) != int(user_id):
            return False
        if action == "alert_done":
            record["status"] = "done"
        elif action == "alert_ignore":
            record["status"] = "ignored"
        elif action == "alert_remind":
            record["status"] = "remind"
            record["remind_at"] = timestamp + 3600
        else:
            return False
        record["updated_at"] = timestamp
        return True

    return bool(_mutate_alert_state(mutate))




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
    PATTERN_MIN_RULE_SCORE = 70        # Внутренний порог правила паттерна
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
    DAILY_ALERT_LIMIT = 24             # Максимум алертов в сутки
    SIGNAL_RULE_SCORE_THRESHOLD = 70   # Порог прозрачного набора правил

    # ТОП-20 монет для мониторинга
    TOP20_COINS = [
        "bitcoin", "ethereum", "binancecoin", "solana", "ripple",
        "cardano", "avalanche-2", "dogecoin", "tron", "chainlink",
        "polkadot", "polygon-ecosystem-token", "internet-computer", "shiba-inu", "uniswap",
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
        self.last_top20_skip_reason = None

    def _should_send_alert(self, coin: str, alert_type: str) -> bool:
        """Check cooldown without consuming it before delivery succeeds."""
        key = f"{coin}_{alert_type}"
        now = datetime.now()

        if key in self.last_alerts:
            hours_passed = (now - self.last_alerts[key]).total_seconds() / 3600
            if hours_passed < AlertConfig.ALERT_COOLDOWN_HOURS:
                return False

        return True

    def _mark_alert_sent(self, coin: str, alert_type: str) -> None:
        self.last_alerts[f"{coin}_{alert_type}"] = datetime.now()

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

    def _calculate_rule_score(self, price_change: float, volume_change: float, rsi: Optional[float]) -> int:
        """Calculate an internal trigger score; this is not a probability."""
        score = 50

        # Сила движения цены
        abs_change = abs(price_change)
        if abs_change > 15:
            score += 20
        elif abs_change > 10:
            score += 10

        # Подтверждение объемом
        if volume_change > 3.0:
            score += 15
        elif volume_change > 2.0:
            score += 10

        # RSI подтверждение
        if rsi:
            if (price_change > 0 and rsi < 30) or (price_change < 0 and rsi > 70):
                score += 15
            elif (price_change > 0 and rsi > 70) or (price_change < 0 and rsi < 30):
                score += 10

        return min(100, max(0, score))

    def _format_recommendation(self, signal_type: str, user_id: Optional[int] = None) -> str:
        """Return cautious next steps without fabricated targets or trade orders."""
        key = signal_type if signal_type in {
            "pump", "dump", "oversold", "overbought", "take_profit", "stop_loss"
        } else "pattern"
        return t(f"alerts.guidance_{key}", user_id=user_id)


    @staticmethod
    def _prune_alert_state(state: dict, max_age_days: int = 7) -> dict:
        """Drop stale alert records so alerts_state.json does not grow forever."""
        cutoff = time.time() - max_age_days * 86400
        return {aid: rec for aid, rec in state.items() if rec.get("ts", 0) >= cutoff}

    async def _send_alert(
        self,
        message: str,
        alert_type: str,
        priority: str = "normal",
        coin: Optional[str] = None,
        target_user_id: Optional[int] = None,
    ):
        """Send per-user alerts and persist state only for delivered messages."""
        if self.alerts_sent_today >= AlertConfig.DAILY_ALERT_LIMIT:
            logger.warning("Daily alert event limit reached")
            return 0

        emoji_map = {
            AlertType.PORTFOLIO_STOP_LOSS: "🚨",
            AlertType.PORTFOLIO_TAKE_PROFIT: "🎯",
            AlertType.PORTFOLIO_RSI_SIGNAL: "💎",
            AlertType.MARKET_PUMP: "🚀",
            AlertType.MARKET_DUMP: "💥",
            AlertType.MARKET_RSI_EXTREME: "⚡",
            AlertType.CRITICAL_NEWS: "📰",
        }
        priority_emoji = "🔥🔥🔥" if priority == "high" else "🔥" if priority == "medium" else ""
        emoji = emoji_map.get(alert_type, "🔔")
        if target_user_id is not None:
            recipients = [int(target_user_id)]
        else:
            self.subscribers = self._load_subscribers()
            recipients = self.subscribers.get("premium", [])
        # The allow-list (test bot) is enforced at delivery, so no alert path
        # can reach anyone outside it.
        recipients = [uid for uid in recipients if config.is_user_allowed(uid)]

        sent_count = 0
        for user_id in sorted(set(recipients)):
            alert_id = secrets.token_hex(5)
            try:
                localized_message = message(user_id) if callable(message) else message
                safe = html.escape(localized_message)
                safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
                header = html.escape(t("alerts.alert_header", user_id=user_id))
                formatted_message = f"{emoji} <b>{header}</b> {emoji} {priority_emoji}\n{safe}"
                await self.bot.send_message(
                    chat_id=user_id,
                    text=formatted_message,
                    parse_mode=ParseMode.HTML,
                    reply_markup=_alert_keyboard(alert_id, user_id),
                )
            except Exception:
                logger.exception("Failed to deliver alert type=%s user_id=%s", alert_type, user_id)
                continue

            created_at = time.time()

            def store_delivery(state, *, aid=alert_id, uid=user_id, ts=created_at):
                pruned = self._prune_alert_state(state)
                state.clear()
                state.update(pruned)
                state[aid] = {
                    "type": alert_type,
                    "coin": coin,
                    "message": localized_message,
                    "rendered_message": formatted_message,
                    "priority": priority,
                    "chat_id": uid,
                    "created_at": ts,
                    "status": "new",
                    "key": make_event_key(coin, alert_type),
                    "ts": ts,
                }

            _mutate_alert_state(store_delivery)
            sent_count += 1

        if sent_count:
            self.alerts_sent_today += 1
            logger.info(
                "Alert delivered type=%s priority=%s recipients=%s",
                alert_type,
                priority,
                sent_count,
            )
        return sent_count

    async def check_portfolio_alerts(self, context) -> List[str]:
        """Проверяет алерты по портфелю всех eligible пользователей, используя данные из снапшота."""
        pm = get_portfolio_manager()
        access_manager = get_access_manager()
        
        snap_service = context.application.bot_data.get("market_snapshot")
        snapshot = _fresh_snapshot(
            snap_service,
            max(120, config.SNAPSHOT_REFRESH_SECONDS * 3),
        )
        markets_map = snapshot.markets if snapshot else {}
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
                coin_id = resolve_coin_id(coin, getattr(self, "registry", None))
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
                user_coin_key = f"{user_id}:{coin}"

                if change_pct >= AlertConfig.PORTFOLIO_TAKE_PROFIT and self._should_send_alert(user_coin_key, AlertType.PORTFOLIO_TAKE_PROFIT):
                    msg = lambda uid, c=coin.upper(), ch=change_pct, buy=buy_price, cur=current_price: t(
                        "alerts.portfolio_profit",
                        user_id=uid,
                        coin=c,
                        change=ch,
                        buy=buy,
                        current=cur,
                        guidance=self._format_recommendation("take_profit", uid),
                    )
                    if await self._send_alert(msg, AlertType.PORTFOLIO_TAKE_PROFIT, "high", coin=coin, target_user_id=user_id):
                        self._mark_alert_sent(user_coin_key, AlertType.PORTFOLIO_TAKE_PROFIT)
                        alerts.append(f"Take Profit: {coin.upper()}")

                if change_pct <= AlertConfig.PORTFOLIO_STOP_LOSS and self._should_send_alert(user_coin_key, AlertType.PORTFOLIO_STOP_LOSS):
                    msg = lambda uid, c=coin.upper(), ch=change_pct, buy=buy_price, cur=current_price: t(
                        "alerts.portfolio_risk",
                        user_id=uid,
                        coin=c,
                        change=ch,
                        buy=buy,
                        current=cur,
                        guidance=self._format_recommendation("stop_loss", uid),
                    )
                    if await self._send_alert(msg, AlertType.PORTFOLIO_STOP_LOSS, "high", coin=coin, target_user_id=user_id):
                        self._mark_alert_sent(user_coin_key, AlertType.PORTFOLIO_STOP_LOSS)
                        alerts.append(f"Stop Loss: {coin.upper()}")
        return alerts

    async def check_top20_opportunities(self, context) -> List[str]:
        """Ищет торговые возможности в ТОП-20, используя данные из снапшотов."""
        snap_service = context.application.bot_data.get("market_snapshot")
        history_service = context.application.bot_data.get("history_snapshot")

        market_snapshot = _fresh_snapshot(
            snap_service,
            max(120, config.SNAPSHOT_REFRESH_SECONDS * 3),
        )
        history_snapshot = _fresh_snapshot(
            history_service,
            max(600, config.HISTORY_REFRESH_SECONDS * 3),
        )
        if not market_snapshot or not history_snapshot:
            self.last_top20_skip_reason = "snapshots_not_ready"
            logger.warning("Снапшоты не готовы для анализа ТОП-20.")
            return []

        self.last_top20_skip_reason = None

        markets_map = market_snapshot.markets
        history_set = history_snapshot
        alerts = []
        opportunities_found = 0

        # Один батч-запрос свечей для всех монет вместо 20 запросов в цикле
        try:
            all_ohlcv = await get_batch_ohlcv(AlertConfig.TOP20_COINS, interval="1h")
        except Exception as e:
            logger.warning(f"Binance OHLCV batch failed: {e}")
            all_ohlcv = {}

        for coin_id in AlertConfig.TOP20_COINS:
            if opportunities_found >= AlertConfig.MAX_ALERTS_PER_CYCLE:
                break

            market_row = markets_map.get(coin_id, {})
            current_price = market_row.get('current_price')
            if not current_price:
                continue

            price_change_pct = market_row.get('price_change_percentage_1h_in_currency', 0) or 0

            # Реальный всплеск объёма: текущая часовая свеча против среднего за 20 часов.
            # (Раньше здесь делили 24h-объём на константу $100M — метрика была фиктивной.)
            volume_ratio = None
            df_vol = all_ohlcv.get(coin_id)
            if df_vol is not None and 'volume' in df_vol.columns and len(df_vol) >= 21:
                avg_vol = float(df_vol['volume'].iloc[-21:-1].mean())
                cur_vol = float(df_vol['volume'].iloc[-1])
                if avg_vol > 0:
                    volume_ratio = cur_vol / avg_vol

            real_rsi = None
            if coin_id in history_set.by_coin:
                df = history_set.by_coin[coin_id]
                if len(df) >= 15:
                    real_rsi = rsi(df['price'], period=14).iloc[-1]

            if coin_id in all_ohlcv:
                df = all_ohlcv[coin_id]

                # Детектим паттерны
                patterns = detect_patterns(df)
                candles = detect_candlestick_patterns(df)
                formations = detect_chart_formations(df)

                # Проверяем каждый паттерн
                for pattern in patterns + candles + formations:
                    pattern_type = str(pattern.get('type') or pattern.get('pattern') or "unknown")
                    pattern_stats = pattern_tracker.get_pattern_stats(pattern_type)

                    # Real historical context requires the prediction ledger
                    # (Package A3); fabricating a market state to query the
                    # feedback loop produced meaningless lookups.
                    historical_context = None

                    # Если паттерн сильный и подтверждён объёмом
                    if pattern.get('strength') in ['strong', 'medium']:
                        if 'breakout' in pattern_type and self._should_send_alert(coin_id, AlertType.PATTERN_BREAKOUT):
                            pattern_target = pattern.get("target")
                            coin_symbol = next((k for k, v in COIN_ID_MAP.items() if v == coin_id), coin_id.upper())

                            def message(uid):
                                message_parts = [
                                    t("alerts.pattern_title", user_id=uid, coin=coin_symbol),
                                    t("alerts.pattern_name", user_id=uid, pattern=pattern_type),
                                    t(
                                        "alerts.pattern_strength",
                                        user_id=uid,
                                        strength=pattern.get("strength") or "n/a",
                                    ),
                                ]
                                if isinstance(pattern_target, (int, float)) and pattern_target > 0:
                                    message_parts.append(
                                        t(
                                            "alerts.pattern_level",
                                            user_id=uid,
                                            target=pattern_target,
                                        )
                                    )
                                if historical_context:
                                    message_parts.append(
                                        t(
                                            "alerts.pattern_sample",
                                            user_id=uid,
                                            count=historical_context.similar_signals_count,
                                        )
                                    )
                                elif pattern_stats.get("resolved", 0) and pattern_stats.get("winrate") is not None:
                                    message_parts.append(
                                        t(
                                            "alerts.pattern_outcomes",
                                            user_id=uid,
                                            rate=pattern_stats["winrate"],
                                            count=pattern_stats["resolved"],
                                        )
                                    )
                                else:
                                    message_parts.append(
                                        t("alerts.pattern_sample_insufficient", user_id=uid)
                                    )
                                message_parts.append(
                                    t(
                                        "alerts.what_to_check",
                                        user_id=uid,
                                        guidance=self._format_recommendation(pattern_type, uid),
                                    )
                                )
                                return "\n".join(message_parts)
                            if await self._send_alert(message, AlertType.PATTERN_BREAKOUT, "medium", coin=coin_id):
                                self._mark_alert_sent(coin_id, AlertType.PATTERN_BREAKOUT)
                                if isinstance(pattern_target, (int, float)) and pattern_target > 0:
                                    pattern_tracker.save_signal(
                                        coin_id,
                                        pattern_type,
                                        'bullish' if 'ascending' in pattern_type or 'bullish' in pattern.get('signal', '') else 'bearish',
                                        pattern_target,
                                        current_price,
                                    )
                                alerts.append(f"Pattern: {coin_symbol} - {pattern_type}")
                                opportunities_found += 1
            # PUMP/DUMP требуют подтверждения реальным объёмом; без данных объёма — не сигналим
            vol_confirmed = volume_ratio is not None and volume_ratio >= AlertConfig.TOP20_VOLUME_MULTIPLIER
            # Логика для PUMP сигнала
            if price_change_pct >= AlertConfig.TOP20_PUMP_THRESHOLD and vol_confirmed:
                rule_score = self._calculate_rule_score(price_change_pct, volume_ratio, real_rsi)
                if rule_score >= AlertConfig.SIGNAL_RULE_SCORE_THRESHOLD and self._should_send_alert(coin_id, AlertType.MARKET_PUMP):
                    coin_symbol = next((k for k, v in COIN_ID_MAP.items() if v == coin_id), coin_id.upper())
                    message = lambda uid: t(
                        "alerts.market_pump",
                        user_id=uid,
                        coin=coin_symbol,
                        change=price_change_pct,
                        volume=t(
                            "alerts.volume_ratio",
                            user_id=uid,
                            ratio=volume_ratio,
                        ),
                        price=current_price,
                        rsi=f"{real_rsi:.1f}" if real_rsi is not None else t("common.not_available", user_id=uid),
                        guidance=self._format_recommendation("pump", uid),
                    )
                    if await self._send_alert(message, AlertType.MARKET_PUMP, "medium", coin=coin_id):
                        self._mark_alert_sent(coin_id, AlertType.MARKET_PUMP)
                        alerts.append(f"Pump: {coin_symbol}")
                        opportunities_found += 1

            # Логика для DUMP сигнала
            elif price_change_pct <= AlertConfig.TOP20_DUMP_THRESHOLD and vol_confirmed:
                rule_score = self._calculate_rule_score(price_change_pct, volume_ratio, real_rsi)
                if rule_score >= AlertConfig.SIGNAL_RULE_SCORE_THRESHOLD and self._should_send_alert(coin_id, AlertType.MARKET_DUMP):
                    coin_symbol = next((k for k, v in COIN_ID_MAP.items() if v == coin_id), coin_id.upper())
                    message = lambda uid: t(
                        "alerts.market_dump",
                        user_id=uid,
                        coin=coin_symbol,
                        change=price_change_pct,
                        volume=t("alerts.volume_ratio", user_id=uid, ratio=volume_ratio),
                        price=current_price,
                        rsi=f"{real_rsi:.1f}" if real_rsi is not None else t("common.not_available", user_id=uid),
                        guidance=self._format_recommendation("dump", uid),
                    )
                    if await self._send_alert(message, AlertType.MARKET_DUMP, "medium", coin=coin_id):
                        self._mark_alert_sent(coin_id, AlertType.MARKET_DUMP)
                        alerts.append(f"Dump: {coin_symbol}")
                        opportunities_found += 1

            # Логика для EXTREME RSI (обе стороны)
            if real_rsi is not None:
                if real_rsi <= AlertConfig.TOP20_RSI_EXTREME_LOW:
                    if self._should_send_alert(coin_id, f"{AlertType.MARKET_RSI_EXTREME}_low"):
                        coin_symbol = next((k for k, v in COIN_ID_MAP.items() if v == coin_id), coin_id.upper())
                        message = lambda uid: t(
                            "alerts.rsi_low",
                            user_id=uid,
                            coin=coin_symbol,
                            rsi=real_rsi,
                            price=current_price,
                            guidance=self._format_recommendation("oversold", uid),
                        )
                        if await self._send_alert(message, AlertType.MARKET_RSI_EXTREME, "medium", coin=coin_id):
                            self._mark_alert_sent(coin_id, f"{AlertType.MARKET_RSI_EXTREME}_low")
                            alerts.append(f"RSI extreme low: {coin_symbol}")
                            opportunities_found += 1

                elif real_rsi >= AlertConfig.TOP20_RSI_EXTREME_HIGH:
                    if self._should_send_alert(coin_id, f"{AlertType.MARKET_RSI_EXTREME}_high"):
                        coin_symbol = next((k for k, v in COIN_ID_MAP.items() if v == coin_id), coin_id.upper())
                        message = lambda uid: t(
                            "alerts.rsi_high",
                            user_id=uid,
                            coin=coin_symbol,
                            rsi=real_rsi,
                            price=current_price,
                            guidance=self._format_recommendation("overbought", uid),
                        )
                        if await self._send_alert(message, AlertType.MARKET_RSI_EXTREME, "medium", coin=coin_id):
                            self._mark_alert_sent(coin_id, f"{AlertType.MARKET_RSI_EXTREME}_high")
                            alerts.append(f"RSI extreme high: {coin_symbol}")
                            opportunities_found += 1

        return alerts

    async def check_critical_news(self) -> List[str]:
        """Проверяет критические новости"""
        alerts = []

        try:
            import asyncio
            # Blocking RSS + sentiment — держим вне event loop
            news_items = await asyncio.to_thread(get_significant_news)

            for item in news_items[:1]:  # Только самая важная новость
                title = item['news'].get('title', 'Без заголовка')
                analysis = item['analysis']

                # Проверяем критичность новости
                critical_keywords = ['негативный', 'negative', 'hacked', 'ban', 'regulation', 'lawsuit']
                positive_keywords = ['позитивный', 'positive', 'etf', 'adoption', 'partnership']

                text_to_check = f"{title} {analysis}".lower()
                is_critical = any(keyword in text_to_check for keyword in critical_keywords)
                is_very_positive = any(keyword in text_to_check for keyword in positive_keywords)

                if is_critical or is_very_positive:
                    # Кулдаун проверяем ТОЛЬКО когда реально есть что отправить —
                    # иначе _should_send_alert "сжигает" окно впустую
                    if not self._should_send_alert("market", AlertType.CRITICAL_NEWS):
                        break
                    marker_key = "alerts.markers_negative" if is_critical else "alerts.markers_positive"
                    message = lambda uid: t(
                        "alerts.critical_news",
                        user_id=uid,
                        title=title,
                        analysis=analysis,
                        markers=t(marker_key, user_id=uid),
                        guidance=t("alerts.guidance_news", user_id=uid),
                    )

                    if await self._send_alert(message, AlertType.CRITICAL_NEWS, "high"):
                        self._mark_alert_sent("market", AlertType.CRITICAL_NEWS)
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
                snap = _fresh_snapshot(
                    snap_service,
                    max(120, config.SNAPSHOT_REFRESH_SECONDS * 3),
                )
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
            self.last_top20_skip_reason = None

            # 1. Приоритет: проверяем портфель (критические события)
            portfolio_alerts = await self.check_portfolio_alerts(context)
            all_alerts.extend(portfolio_alerts)

            # 2. Проверяем возможности в ТОП-20 (если есть место)
            if len(all_alerts) < AlertConfig.MAX_ALERTS_PER_CYCLE:
                top20_alerts = await self.check_top20_opportunities(context)
                all_alerts.extend(top20_alerts)
            skipped_sections = []
            if getattr(self, "last_top20_skip_reason", None):
                skipped_sections.append(
                    f"top20={self.last_top20_skip_reason}"
                )

            # 3. Проверяем критические новости (если есть место)
            if len(all_alerts) < AlertConfig.MAX_ALERTS_PER_CYCLE:
                news_alerts = await self.check_critical_news()
                all_alerts.extend(news_alerts)

            execution_time = (datetime.now() - start_time).total_seconds()

            if skipped_sections:
                logger.warning(
                    "Сканирование завершено частично за %.1fс; пропущено: %s; отправлено: %s",
                    execution_time,
                    ", ".join(skipped_sections),
                    len(all_alerts),
                )
            elif all_alerts:
                logger.info(f"✅ Сканирование завершено за {execution_time:.1f}с. Отправлено алертов: {len(all_alerts)}")
                logger.info(f"Алерты: {', '.join(all_alerts)}")
            else:
                logger.info(f"✅ Сканирование завершено за {execution_time:.1f}с. Критических событий не обнаружено.")

        except Exception as e:
            logger.exception("Критическая ошибка в системе алертов")
            await report_incident(
                context,
                e,
                subsystem="alerts_full_scan",
            )

# Глобальный экземпляр системы алертов
smart_alerts = SmartAlertsSystem()


async def run_alert_reminders(context):
    """Claim and deliver due reminders without duplicate concurrent sends."""
    now = time.time()

    def claim_due(state):
        due = []
        for alert_id, record in state.items():
            if record.get("status") != "remind":
                continue
            if float(record.get("remind_at", 0)) > now:
                continue
            record["status"] = "reminding"
            record["updated_at"] = now
            due.append((alert_id, dict(record)))
        return due

    due_alerts = _mutate_alert_state(claim_due)
    bot = getattr(context, "bot", None) or smart_alerts.bot

    for alert_id, record in due_alerts:
        user_id = int(record["chat_id"])
        rendered = record.get("rendered_message")
        if not rendered:
            safe = html.escape(str(record.get("message", "")))
            rendered = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        reminder = f"⏰ <b>{t('alerts.reminder_title', user_id=user_id)}</b>\n{rendered}"

        try:
            await bot.send_message(
                chat_id=user_id,
                text=reminder,
                parse_mode=ParseMode.HTML,
                reply_markup=_alert_keyboard(alert_id, user_id),
            )
        except Exception:
            logger.exception("Failed to deliver alert reminder id=%s user_id=%s", alert_id, user_id)

            def retry(state, *, aid=alert_id):
                current = state.get(aid)
                if current and current.get("status") == "reminding":
                    current["status"] = "remind"
                    current["remind_at"] = time.time() + 300

            _mutate_alert_state(retry)
            continue

        def delivered(state, *, aid=alert_id):
            current = state.get(aid)
            if current and current.get("status") == "reminding":
                current["status"] = "new"
                current.pop("remind_at", None)
                current["reminder_count"] = int(current.get("reminder_count", 0)) + 1
                current["updated_at"] = time.time()

        _mutate_alert_state(delivered)


# Функция для использования в job_queue
async def run_alerts_scan(context):
    """Функция для запуска из JobQueue"""
    await smart_alerts.run_full_scan(context)


