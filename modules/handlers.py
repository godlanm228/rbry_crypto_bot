import logging
import time
import asyncio, inspect, json, os
import html, re
import datetime
import pandas as pd

from core.batch_analyzer import BatchAnalyzer
from core.ai_analytics import ai_analyzer
from core.analysis.signal_generator import compute_rsi
from core.analysis.technical_analysis import realized_volatility
from core.analyzer_full import analyze_coin_full
from core.utils import COIN_ID_MAP, safe_run, _row_change_pct, _track_user_activity
from core.analytics.user_analytics import get_user_analytics
from core.api.coingecko_client import get_markets_data
from core.portfolio_db import get_portfolio_manager

from modules.news import get_significant_news, _format_news_block, render_news
from config import PORTFOLIO_FILE, OWNER_ID as USER_ID

from modules.alerts import _load_alert_state, _save_alert_state
from modules.user_prefs import (
    get_timezone, get_language, set_timezone, set_language,
    SUPPORTED_TIMEZONES, SUPPORTED_LANGS, get_profile,
    get_show_explanations, set_show_explanations
)
from locales.translator import t
from modules.admin_commands import _build_tier_text

from telegram.constants import ParseMode, ChatAction
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import ContextTypes
from zoneinfo import ZoneInfo

# Import access control and security
from core.access.access_manager import get_access_manager, AccessTier
from core.security.rate_limiter_decorator import get_rate_limiter

# Import payment manager
from modules.payments.payment_manager import get_payment_manager

batch = BatchAnalyzer(ttl_sec=60)
_global_app_ref = None  # will be set on first use
logger = logging.getLogger(__name__)

# Default timezone constant
DEFAULT_TIMEZONE = "Europe/Berlin"

def as_html(text: str) -> str:
    """Безопасный HTML для Telegram + конверт **bold** и _italic_."""
    txt = html.escape(text)
    # жирный
    txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
    # курсив
    txt = re.sub(r"_(.+?)_", r"<i>\1</i>", txt)
    return txt

def format_price(price: float) -> str:
    """Format price with appropriate decimal places based on magnitude."""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.2f}"
    elif price >= 0.01:
        return f"${price:.4f}"
    elif price >= 0.0001:
        return f"${price:.6f}"
    else:
        return f"${price:.8f}"

async def respond_message(
    message,
    text: str,
    reply_markup=None,
    parse_mode=ParseMode.HTML,
    disable_web_page_preview: bool = True,
    prefer_edit: bool = False
):
    """Edit if possible, otherwise send a new message."""
    if prefer_edit:
        try:
            await message.edit_text(
                text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
                reply_markup=reply_markup,
            )
            return
        except BadRequest as e:
            s = str(e)
            if "Message is not modified" in s or "Message text is empty" in s:
                return
            if "Message can't be edited" not in s:
                raise
    await message.reply_text(
        text,
        parse_mode=parse_mode,
        disable_web_page_preview=disable_web_page_preview,
        reply_markup=reply_markup,
    )

def build_help_text(user_id: int) -> str:
    parts = [
        t("help.title", user_id=user_id),
        "",
        t("help.commands_basic", user_id=user_id),
        t("help.cmd_start", user_id=user_id),
        t("help.cmd_help", user_id=user_id),
        t("help.cmd_portfolio", user_id=user_id),
        "",
        t("help.commands_portfolio", user_id=user_id),
        t("help.cmd_add_entry", user_id=user_id),
        t("help.cmd_sell", user_id=user_id),
        t("help.cmd_stats", user_id=user_id),
        t("help.cmd_movers", user_id=user_id),
        "",
        t("help.commands_analysis", user_id=user_id),
        t("help.cmd_analyze", user_id=user_id),
        t("help.cmd_news", user_id=user_id),
        "",
        t("help.commands_alerts", user_id=user_id),
        t("help.cmd_scan_now", user_id=user_id),
        "",
        t("help.example", user_id=user_id),
        "",
        t("help.alerts_active", user_id=user_id),
        t("help.alerts_desc", user_id=user_id),
    ]
    return "\n".join(parts)

def fmt_asof_for_user(user_id: int, ts: float | None) -> str:
    if not ts:
        return "n/a"
    tzname = get_timezone(user_id) or DEFAULT_TIMEZONE
    try:
        local = ZoneInfo(tzname)
    except Exception:
        local = ZoneInfo(DEFAULT_TIMEZONE)
    dt_utc = datetime.datetime.utcfromtimestamp(ts).replace(tzinfo=ZoneInfo("UTC"))
    dt_loc = dt_utc.astimezone(local)
    return f"{dt_loc.strftime('%Y-%m-%d %H:%M:%S')} ({tzname}) • {dt_utc.strftime('%H:%M:%S')} UTC"

def set_global_app_ref(app):
    """Устанавливает глобальную ссылку на приложение"""
    global _global_app_ref
    _global_app_ref = app

def get_global_app_ref():
    return _global_app_ref

def _row_price_usd(row):
    if not row: return None
    return row.get("current_price")
def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default
def _row_change_24h(row):
    if not row: return None
    return row.get("price_change_percentage_24h")

def _row_volume_24h(row):
    if not row: return None
    return row.get("total_volume")


def create_news_keyboard(user_id: int = None):
    """Создать клавиатуру для новостей"""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
    )

async def arun(func, *args, **kwargs):
    # Запускает синхронную функцию в отдельном потоке (не блокируя event loop)
    return await asyncio.to_thread(func, *args, **kwargs)

def create_action_buttons(user_id: int = None):
    """Кнопки для быстрых действий с портфелем"""
    keyboard = [
        [InlineKeyboardButton(t("menu.portfolio", user_id=user_id), callback_data="portfolio"),
         InlineKeyboardButton(t("menu.analysis", user_id=user_id), callback_data="analyze")],
        [InlineKeyboardButton(t("menu.stats", user_id=user_id), callback_data="stats"),
         InlineKeyboardButton(t("menu.movers", user_id=user_id), callback_data="movers")],
        [InlineKeyboardButton(t("menu.help", user_id=user_id), callback_data="help")]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_back_button(user_id: int = None):
    """Создать кнопку возврата в главное меню"""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")
    ]])

def create_navigation_buttons(user_id: int = None):
    """Создать расширенные кнопки навигации"""
    keyboard = [
        [InlineKeyboardButton(t("menu.stats", user_id=user_id), callback_data="stats"),
         InlineKeyboardButton(t("menu.movers", user_id=user_id), callback_data="movers")],
        [InlineKeyboardButton(t("menu.analysis", user_id=user_id), callback_data="analyze"),
         InlineKeyboardButton(t("menu.news", user_id=user_id), callback_data="news")],
        [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def generate_rule_based_signal(ch24, ch7d, user_id):
    """
    Generate rule-based trading signal based on price changes.
    
    Args:
        ch24: 24-hour price change percentage (float or None)
        ch7d: 7-day price change percentage (float or None)
        user_id: User ID for localization
    
    Returns:
        tuple: (signal_text, confidence_percentage)
    """
    # Handle missing data
    if not isinstance(ch24, (int, float)) or not isinstance(ch7d, (int, float)):
        return (t("analysis.signal_hold", user_id=user_id), 60)
    
    # Apply rules based on price changes
    if ch24 > 5 and ch7d > 10:
        # Strong bullish signal
        signal = t("analysis.signal_strong_buy", user_id=user_id)
        # Higher confidence for stronger moves (85-95%)
        confidence = min(95, 85 + abs(ch24 + ch7d) / 4)
    elif ch24 > 2 or ch7d > 5:
        # Mildly bullish signal
        signal = t("analysis.signal_buy", user_id=user_id)
        # Medium confidence (70-85%)
        confidence = min(85, 70 + abs(ch24 + ch7d) / 5)
    elif ch24 < -5 and ch7d < -10:
        # Strong bearish signal
        signal = t("analysis.signal_strong_sell", user_id=user_id)
        # Higher confidence for stronger moves (85-95%)
        confidence = min(95, 85 + abs(ch24 + ch7d) / 4)
    elif ch24 < -2 or ch7d < -5:
        # Mildly bearish signal
        signal = t("analysis.signal_sell", user_id=user_id)
        # Medium confidence (70-85%)
        confidence = min(85, 70 + abs(ch24 + ch7d) / 5)
    else:
        # Neutral signal
        signal = t("analysis.signal_hold", user_id=user_id)
        # Lower confidence for neutral (60-70%)
        confidence = 60 + abs(ch24 + ch7d) / 2
    
    return (signal, min(95, max(60, confidence)))

def create_main_menu(user_id: int = None):
    """
    Создать главное меню с учетом прав доступа

    Args:
        user_id: Telegram user ID для фильтрации owner-only buttons
    """
    keyboard = [
        [InlineKeyboardButton(t("menu.portfolio", user_id=user_id), callback_data="portfolio"),
         InlineKeyboardButton(t("menu.stats", user_id=user_id), callback_data="stats")],
        [InlineKeyboardButton(t("menu.analysis", user_id=user_id), callback_data="analyze"),
         InlineKeyboardButton(t("menu.movers", user_id=user_id), callback_data="movers")],
        [InlineKeyboardButton(t("menu.news", user_id=user_id), callback_data="news"),
         InlineKeyboardButton(t("menu.settings", user_id=user_id), callback_data="settings")],
    ]

    # Access tier button (visible to all users)
    keyboard.append([InlineKeyboardButton(t("menu.my_tier", user_id=user_id), callback_data="my_tier")])

    # Owner-only admin panel button
    if user_id and user_id == USER_ID:
        keyboard.append([InlineKeyboardButton(t("menu.admin_panel", user_id=user_id), callback_data="admin_panel")])

    keyboard.append([InlineKeyboardButton(t("menu.help", user_id=user_id), callback_data="help")])

    return InlineKeyboardMarkup(keyboard)

async def show_help(user_id: int, message, prefer_edit: bool):
    text = build_help_text(user_id)
    keyboard = [[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
    await respond_message(
        message,
        as_html(text),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
        prefer_edit=prefer_edit,
    )

async def show_settings(user_id: int, message, prefer_edit: bool):
    prof = get_profile(user_id)
    show_expl = get_show_explanations(user_id)
    expl_status = t("settings.explanations_on", user_id=user_id) if show_expl else t("settings.explanations_off", user_id=user_id)
    lang_code = prof.get("lang") or "ru"
    text = (
        f"{t('settings.title', user_id=user_id)}\n"
        f"{t('settings.user_id', user_id=user_id)}\n"
        f"{t('settings.timezone', user_id=user_id, tz=prof.get('tz') or t('settings.not_set', user_id=user_id))}\n"
        f"{t('settings.language', user_id=user_id, lang=lang_code, language=lang_code)}\n"
        f"{t('settings.explanations', user_id=user_id, status=expl_status)}\n"
    )
    keyboard = [
        [InlineKeyboardButton(t("settings.button_timezone", user_id=user_id), callback_data="settings_tz")],
        [InlineKeyboardButton(t("settings.button_language", user_id=user_id), callback_data="settings_lang")],
        [InlineKeyboardButton(t("settings.button_explanations", user_id=user_id), callback_data="settings_explanations")],
        [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
    ]
    await respond_message(
        message,
        as_html(text),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
        prefer_edit=prefer_edit,
    )

async def show_analyze_prompt(user_id: int, context: ContextTypes.DEFAULT_TYPE, message, prefer_edit: bool):
    keyboard = [
        [InlineKeyboardButton(t("analysis.my_portfolio", user_id=user_id), callback_data="analyze_portfolio")],
        [InlineKeyboardButton(t("analysis.top_10", user_id=user_id), callback_data="analyze_top10")],
        [InlineKeyboardButton(t("analysis.choose_coins", user_id=user_id), callback_data="analyze_custom")],
        [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]
    ]
    await respond_message(
        message,
        t("analysis.title", user_id=user_id),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
        prefer_edit=prefer_edit,
    )
async def show_portfolio(user_id: int, context: ContextTypes.DEFAULT_TYPE, message, prefer_edit: bool):
    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)

    if not portfolio:
        text = t("portfolio.empty_hint", user_id=user_id)
        keyboard = [
            [InlineKeyboardButton(t("portfolio.add_coin_button", user_id=user_id), callback_data="add_coin_prompt")],
            [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
        ]
        await respond_message(message, text, reply_markup=InlineKeyboardMarkup(keyboard), prefer_edit=prefer_edit)
        return

    snap_service = context.application.bot_data.get("market_snapshot")
    rows = {}
    as_of = "n/a"
    if snap_service:
        snap = snap_service.get_latest()
        if snap:
            rows = snap.markets
            if snap.ts:
                as_of = datetime.datetime.utcfromtimestamp(snap.ts).strftime("%H:%M:%S") + " UTC"

    lines = [f"<b>{t('portfolio.title', user_id=user_id)}</b>", "═" * 20, ""]
    total_value = 0.0
    total_cost = 0.0

    for coin, data in portfolio.items():
        amount = data.get("amount", 0)
        entry_price = data.get("buy_price", data.get("price", 0))
        coin_id = COIN_ID_MAP.get(coin.upper(), coin.lower())
        row = rows.get(coin_id) if rows else None
        current_price = row.get("current_price") if row else None

        if current_price is not None:
            current_value = amount * current_price
            cost_basis = amount * entry_price
            pnl_usd = current_value - cost_basis
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            pnl_emoji = "📈" if pnl_usd > 0 else "📉" if pnl_usd < 0 else "➡️"

            entry_str = format_price(entry_price)
            current_str = format_price(current_price)
            value_str = format_price(current_value)

            lines.append(
                f"<b>{coin.upper()}</b>\n"
                f"  {t('portfolio.amount_label', user_id=user_id)}: {amount:.6f}\n"
                f"  {t('portfolio.entry_price_label', user_id=user_id)}: {entry_str}\n"
                f"  {t('portfolio.current_price_label', user_id=user_id)}: {current_str}\n"
                f"  {t('portfolio.current_value_label', user_id=user_id)}: {value_str}\n"
                f"  {pnl_emoji} {t('portfolio.pnl_label', user_id=user_id)}: ${pnl_usd:+.2f} ({pnl_pct:+.1f}%)\n"
            )
            total_value += current_value
            total_cost += cost_basis
        else:
            entry_str = format_price(entry_price)
            lines.append(f"💰 <b>{coin.upper()}</b>: {amount:.6f} @ {entry_str}")
            total_cost += amount * entry_price

    if total_value > 0:
        total_pnl = total_value - total_cost
        total_pnl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
        pnl_emoji = "📈" if total_pnl > 0 else "📉" if total_pnl < 0 else "➡️"

        total_cost_str = format_price(total_cost)
        total_value_str = format_price(total_value)

        lines.append(
            f"\n{'═' * 20}\n"
            f"<b>{t('portfolio.total_header', user_id=user_id)}</b>\n"
            f"  {t('portfolio.invested_label', user_id=user_id)}: {total_cost_str}\n"
            f"  {t('portfolio.current_value_label', user_id=user_id)}: {total_value_str}\n"
            f"  {pnl_emoji} {t('portfolio.pnl_label', user_id=user_id)}: ${total_pnl:+.2f} ({total_pnl_pct:+.1f}%)"
        )

    lines.append(f"\n<i>{t('portfolio.as_of', user_id=user_id, time=as_of)}</i>")

    keyboard = [
        [InlineKeyboardButton(t("portfolio.add_coin_button", user_id=user_id), callback_data="add_coin_prompt"),
         InlineKeyboardButton(t("portfolio.sell_coin_button", user_id=user_id), callback_data="sell_coin_prompt")],
        [InlineKeyboardButton(t("portfolio.analyze_button", user_id=user_id), callback_data="analyze"),
         InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
    ]
    await respond_message(
        message,
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
        prefer_edit=prefer_edit,
    )
async def show_stats(user_id: int, context: ContextTypes.DEFAULT_TYPE, message, prefer_edit: bool):
    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)

    if not portfolio:
        text = t("stats.empty_message", user_id=user_id)
        keyboard = [
            [InlineKeyboardButton(t("portfolio.add_coin_button", user_id=user_id), callback_data="help")],
            [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
        ]
        await respond_message(message, text, reply_markup=InlineKeyboardMarkup(keyboard), prefer_edit=prefer_edit)
        return

    app = (context.application or get_global_app_ref())
    ms = app.bot_data.get("market_snapshot") if app else None
    snap = ms.get_latest() if ms else None

    if snap and snap.markets:
        rows = snap.markets
        as_of = fmt_asof_for_user(user_id, snap.ts)
    else:
        cids = [COIN_ID_MAP.get(c.upper(), c.lower()) for c in portfolio.keys()]
        rows = await get_markets_data(cids)
        as_of = "n/a"
        if not rows:
            await respond_message(
                message,
                t("stats.no_data", user_id=user_id),
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
                prefer_edit=prefer_edit,
            )
            return

    total_value = 0.0
    total_invested = 0.0
    msg = [t("stats.title", user_id=user_id), t("stats.divider", user_id=user_id), ""]

    def _f(x, default=None):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    for coin, data in portfolio.items():
        cid = COIN_ID_MAP.get(coin.upper(), coin.lower())
        amount = _f(data.get("amount", 0.0), 0.0)
        entry = _f(data.get("buy_price", data.get("price", 0.0)), 0.0)
        row = (rows or {}).get(cid, {}) or {}

        now = _f(row.get("current_price"), None)
        ch1 = row.get("price_change_percentage_1h_in_currency", row.get("price_change_percentage_1h"))
        ch24 = row.get("price_change_percentage_24h_in_currency", row.get("price_change_percentage_24h"))
        ch7 = row.get("price_change_percentage_7d_in_currency", row.get("price_change_percentage_7d"))

        if now is None:
            continue

        cur_val = (amount or 0.0) * now
        inv = (amount or 0.0) * (entry or 0.0)
        total_value += cur_val
        total_invested += inv
        change = ((now - (entry or 0.0)) / (entry or 1.0) * 100.0) if entry else 0.0

        def fmt(x): return f"{x:+.1f}%" if isinstance(x, (int, float)) else "н/д"

        msg += [
            "━━━━━━━━━━━━━━━━━",
            t("stats.coin_header", user_id=user_id, coin=coin.upper()),
            t("stats.price", user_id=user_id, price=now),
            t("stats.change_1h", user_id=user_id, change=fmt(ch1)),
            t("stats.change_24h", user_id=user_id, change=fmt(ch24)),
            t("stats.change_7d", user_id=user_id, change=fmt(ch7)),
            t("stats.profit_loss", user_id=user_id, pl=change),
        ]

    msg += [
        "",
        t("stats.total_invested", user_id=user_id, amount=total_invested),
        t("stats.current_value", user_id=user_id, amount=total_value),
        t("portfolio.as_of", user_id=user_id, time=as_of),
    ]

    await respond_message(
        message,
        as_html("\n".join(msg)),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
        prefer_edit=prefer_edit,
    )

async def show_movers(user_id: int, context: ContextTypes.DEFAULT_TYPE, message, prefer_edit: bool):
    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)
    if not portfolio:
        text = t("movers.empty_message", user_id=user_id)
        keyboard = [
            [InlineKeyboardButton(t("portfolio.add_coin_button", user_id=user_id), callback_data="help")],
            [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
        ]
        await respond_message(message, text, reply_markup=InlineKeyboardMarkup(keyboard), prefer_edit=prefer_edit)
        return

    app = (context.application or get_global_app_ref())
    ms = app.bot_data.get("market_snapshot") if app else None
    snap = ms.get_latest() if ms else None

    if snap and snap.markets:
        rows = snap.markets
        as_of = fmt_asof_for_user(user_id, snap.ts)
    else:
        cids = [COIN_ID_MAP.get(c.upper(), c.lower()) for c in portfolio.keys()]
        rows = await get_markets_data(cids)
        as_of = "n/a"
        if not rows:
            await respond_message(
                message,
                as_html(t("movers.no_data", user_id=user_id)),
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
                prefer_edit=prefer_edit,
            )
            return

    movers = []
    for coin, data in portfolio.items():
        cid = COIN_ID_MAP.get(coin.upper(), coin.lower())
        entry = float(data.get("buy_price", data.get("price", 0.0)) or 0.0)
        now = rows.get(cid, {}).get("current_price")
        if not isinstance(now, (int, float)) or entry <= 0:
            continue
        change_pct = (now - entry) / entry * 100.0
        movers.append((coin.upper(), change_pct, now, entry))

    movers.sort(key=lambda x: x[1], reverse=True)

    if not movers:
        await respond_message(
            message,
            as_html(t("movers.no_data", user_id=user_id)),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
            prefer_edit=prefer_edit,
        )
        return

    msg = [t("movers.title", user_id=user_id), t("movers.divider", user_id=user_id), ""]
    for coin, change, current, entry in movers:
        emoji = "🚀" if change > 10 else "📈" if change > 0 else "📉" if change > -10 else "💥"
        color = "🟢" if change > 0 else "🔴"
        change_str = f"{color}"
        msg += [
            t("movers.coin_entry", user_id=user_id, emoji=emoji, coin=coin, change_str=change_str),
            t("movers.change_line", user_id=user_id, change=change),
            t("movers.price_line", user_id=user_id, entry=entry, current=current),
            ""
        ]
    msg.append(t("portfolio.as_of", user_id=user_id, time=as_of))

    await respond_message(
        message,
        as_html("\n".join(msg)),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
        prefer_edit=prefer_edit,
    )

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Устанавливаем глобальную ссылку
    set_global_app_ref(context.application)

    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "start")
    profile = get_profile(user_id)

    if not profile.get("tz"):
        # первый запуск — спрашиваем TZ
        tz_kb = [[InlineKeyboardButton(tz, callback_data=f"set_tz:{tz}")] for tz in SUPPORTED_TIMEZONES[:6]]
        await update.message.reply_text(
            as_html(t("welcome.timezone_prompt", user_id=user_id)),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(tz_kb + [[InlineKeyboardButton(t("settings.skip_timezone", user_id=user_id), callback_data=f"set_tz:{DEFAULT_TIMEZONE}")]])
        )
        return

    features = t("welcome.features", user_id=user_id)
    welcome_message = (
        f"{t('welcome.title', user_id=user_id)}\n"
        f"{t('welcome.divider', user_id=user_id)}\n\n"
        f"{features}\n\n"
        f"{t('welcome.choose_action', user_id=user_id)}"
    )

    await update.message.reply_text(
        as_html(welcome_message),
        reply_markup=create_main_menu(user_id),
        parse_mode=ParseMode.HTML,
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "help")
    await show_help(user_id, update.message, prefer_edit=False)

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "settings")
    await show_settings(user_id, update.message, prefer_edit=False)

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "analyze")
    await show_analyze_prompt(user_id, context, update.message, prefer_edit=False)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "stats")
    await show_stats(user_id, context, update.message, prefer_edit=False)

async def movers_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "movers")
    await show_movers(user_id, context, update.message, prefer_edit=False)

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip() if update.message and update.message.text else ""
    cmd = text.split()[0] if text else "Команда"
    _track_user_activity(update.effective_user, "unknown_command", {"text": text})
    await update.message.reply_text(f"? {cmd} не найдена. Введите /help для списка команд.")

async def handle_coin_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle custom coin input when user is in analyze_custom mode"""
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "custom_coin_input", {"text": update.message.text if update.message else ""})
    
    # Check if we're waiting for coin input
    if not context.user_data.get("awaiting_coin_input", False):
        return  # Not waiting for input, ignore
    
    # Clear the flag
    context.user_data["awaiting_coin_input"] = False
    
    # Get the coin symbol from message
    coin_symbol = update.message.text.strip().upper()
    
    # Check access control
    access_manager = get_access_manager()
    rate_limiter = get_rate_limiter()
    tier = access_manager.get_user_tier(user_id)
    
    # Check rate limit
    allowed, reason = rate_limiter.check_rate_limit(user_id, tier.value)
    if not allowed:
        await update.message.reply_text(reason)
        return
    
    # Check basic analysis access
    allowed, reason = access_manager.check_access(user_id, 'basic_analysis')
    if not allowed:
        await update.message.reply_text(f"? {reason}")
        return
    
    # Try to analyze the coin
    coin_id = COIN_ID_MAP.get(coin_symbol, coin_symbol.lower())
    
    # Show analyzing message
    await update.message.reply_text(t("analysis.analyzing_coin", user_id=user_id, coin=coin_symbol))
    
    # Get market data
    try:
        snap_service = context.application.bot_data.get("market_snapshot")
        row = None
        
        if snap_service:
            snap = snap_service.get_latest()
            if snap and snap.markets:
                row = snap.markets.get(coin_id)
        
        # Fallback to API if not in snapshot
        if not row:
            from core.api.coingecko_client import get_markets_data
            rows = await get_markets_data([coin_id])
            row = rows.get(coin_id) if rows else None
        
        if not row:
            await update.message.reply_text(t("analysis.coin_not_found", user_id=user_id))
            return
        
        current_price = row.get("current_price")
        
        pm = get_portfolio_manager()
        portfolio = pm.load_portfolio(user_id)
        entry = portfolio.get(coin_symbol, {}) or portfolio.get(coin_id, {})
        
        show_explanations = get_show_explanations(user_id)
        news_context = None
        if show_explanations:
            try:
                items = get_significant_news(limit=3)
                sources = []
                combined_text = ""
                for it in items:
                    n = it.get("news", {})
                    if n.get("source"):
                        sources.append(n.get("source"))
                    combined_text += " " + (n.get("summary") or "") + " " + (n.get("title") or "")
                sentiment = "neutral"
                if combined_text and ai_analyzer.finbert and ai_analyzer.finbert.status.get("enabled"):
                    try:
                        label, _score, _backend = await ai_analyzer.finbert.analyze_async(combined_text[:512])
                        sentiment = "positive" if label.startswith("pos") else "negative" if label.startswith("neg") else "neutral"
                    except Exception:
                        sentiment = "neutral"
                news_context = {
                    "count": len(items),
                    "sentiment": sentiment,
                    "sources": sources or ["cryptonews", "cointelegraph", "decrypt"],
                }
            except Exception:
                news_context = {"count": 0, "sentiment": "neutral", "sources": ["cryptonews", "cointelegraph", "decrypt"]}

        text = await analyze_coin_full(
            coin_id=coin_id,
            current_price=float(current_price or 0.0),
            portfolio_entry=entry,
            indicators=row,
            context=context,
            user_id=user_id,
            show_explanations=show_explanations,
            news_context=news_context
        )
        
        # Send with back button
        keyboard = [[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="analyze")]]
        await update.message.reply_text(
            as_html(text), 
            parse_mode=ParseMode.HTML, 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    except Exception as e:
        logger.error(f"Error analyzing coin {coin_symbol}: {e}", exc_info=True)
        await update.message.reply_text(t("analysis.analysis_error", user_id=user_id))

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "portfolio")
    await show_portfolio(user_id, context, update.message, prefer_edit=False)

async def stats_command_inline(query, context):
    """Статистика — сначала пытаемся взять из снапшота, если его нет — делаем разовый запрос рынка."""
    await query.answer()
    await show_stats(query.from_user.id, context, query.message, prefer_edit=True)

async def top_movers_command_inline(query, context):
    await query.answer()
    await show_movers(query.from_user.id, context, query.message, prefer_edit=True)

async def back_to_main_menu(query):
    """Возврат в главное меню"""
    user_id = query.from_user.id
    welcome_message = (
        f"{t('welcome.title', user_id=user_id)}\n"
        f"{t('welcome.divider', user_id=user_id)}\n\n"
        f"{t('welcome.choose_action', user_id=user_id)}"
    )

    await query.edit_message_text(
        as_html(welcome_message),
        reply_markup=create_main_menu(user_id),
        parse_mode=ParseMode.HTML,
    )

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query and query.from_user and query.data:
        action = query.data.split(":", 1)[0]
        _track_user_activity(query.from_user, action, {"data": query.data})

    # Устанавливаем глобальную ссылку
    set_global_app_ref(context.application)

    if query.data == "portfolio":
        user_id = query.from_user.id
        await show_portfolio(user_id, context, query.message, prefer_edit=True)
        return
    elif query.data.startswith("alert_"):
        user_id = query.from_user.id
        action, alert_id = query.data.split(":", 1)
        state = _load_alert_state()
        if alert_id in state:
            if action == "alert_done":
                state[alert_id]["status"] = "done"
                await query.answer(t("alerts.status_done", user_id=user_id))
            elif action == "alert_remind":
                state[alert_id]["remind_at"] = time.time() + 3600
                await query.answer(t("alerts.status_remind", user_id=user_id))
            elif action == "alert_ignore":
                state[alert_id]["status"] = "ignored"
                await query.answer(t("alerts.status_ignored", user_id=user_id))
            _save_alert_state(state)
            await query.edit_message_reply_markup(reply_markup=None)
        return

    elif query.data == "settings":
        user_id = query.from_user.id
        await show_settings(user_id, query.message, prefer_edit=True)
        return
    elif query.data == "settings_tz":
        user_id = query.from_user.id
        tz_kb = [[InlineKeyboardButton(tz, callback_data=f"set_tz:{tz}")] for tz in SUPPORTED_TIMEZONES[:6]]
        tz_kb.append([InlineKeyboardButton(t("common.back_to_settings", user_id=user_id), callback_data="settings")])
        await query.edit_message_text(
            as_html(t("settings.choose_timezone", user_id=user_id)),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(tz_kb)
        )

    elif query.data == "settings_lang":
        user_id = query.from_user.id
        lang_kb = [
            [InlineKeyboardButton("🇷🇺 Русский", callback_data="set_lang:ru")],
            [InlineKeyboardButton("🇺🇦 Українська", callback_data="set_lang:uk")],
            [InlineKeyboardButton("🇬🇧 English", callback_data="set_lang:en")],
            [InlineKeyboardButton("🇩🇪 Deutsch", callback_data="set_lang:de")],
            [InlineKeyboardButton(t("common.back_to_settings", user_id=user_id), callback_data="settings")]
        ]
        await query.edit_message_text(
            t("settings.choose_language", user_id=user_id),
            reply_markup=InlineKeyboardMarkup(lang_kb)
        )
    elif query.data == "settings_explanations":
        user_id = query.from_user.id
        current = get_show_explanations(user_id)
        set_show_explanations(user_id, not current)
        await query.answer(t("settings.explanations_saved", user_id=user_id))
        await show_settings(user_id, query.message, prefer_edit=True)

    elif query.data.startswith("set_lang:"):
        user_id = query.from_user.id
        lang = query.data.split("set_lang:", 1)[1]
        set_language(user_id, lang)
        await query.answer(t("settings.language_saved", user_id=user_id))
        await show_settings(user_id, query.message, prefer_edit=True)

    elif query.data.startswith("set_tz:"):
        user_id = query.from_user.id
        tz = query.data.split("set_tz:", 1)[1]
        set_timezone(user_id, tz)
        await query.answer(t("settings.timezone_saved", user_id=user_id))
        await show_settings(user_id, query.message, prefer_edit=True)

    elif query.data == "analyze":
        user_id = query.from_user.id
        await show_analyze_prompt(user_id, context, query.message, prefer_edit=True)
    elif query.data == "analyze_custom":
        user_id = query.from_user.id
        # Set flag that we're waiting for coin input
        context.user_data["awaiting_coin_input"] = True
        
        # Показываем популярные монеты для выбора
        popular_coins = ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX"]
        keyboard = []
        for i in range(0, len(popular_coins), 2):
            row = []
            for j in range(2):
                if i+j < len(popular_coins):
                    coin = popular_coins[i+j]
                    row.append(InlineKeyboardButton(coin, callback_data=f"analyze_coin:{coin}"))
            keyboard.append(row)
        keyboard.append([InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="analyze")])

        # Updated message with hint for custom coin input
        message_text = f"{t('analysis.choose_coin', user_id=user_id)}\n\n{t('analysis.custom_hint', user_id=user_id)}"
        
        await query.edit_message_text(
            message_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    elif query.data == "analyze_portfolio":
        # ACCESS CONTROL CHECK
        user_id = query.from_user.id
        access_manager = get_access_manager()

        # Check rate limit
        rate_limiter = get_rate_limiter()
        tier = access_manager.get_user_tier(user_id)
        allowed, reason = rate_limiter.check_rate_limit(user_id, tier.value)
        if not allowed:
            await query.answer(reason, show_alert=True)
            return

        # Check basic analysis access (FREE tier can use this)
        allowed, reason = access_manager.check_access(user_id, 'basic_analysis')
        if not allowed:
            keyboard = [[InlineKeyboardButton(t("tier.button_premium", user_id=user_id), callback_data="premium")],
                       [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
            await query.edit_message_text(
                as_html(t("errors.access_denied_feature", user_id=user_id, reason=reason)),
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        # Record request
        rate_limiter.record_request(user_id)
        access_manager.record_request(user_id)

        t_start = time.time()
        timestamp = time.strftime("%H:%M:%S")
        pm = get_portfolio_manager()
        portfolio = pm.load_portfolio(user_id)
        if not portfolio:
            await query.edit_message_text(as_html(t("portfolio.empty", user_id=user_id)), parse_mode=ParseMode.HTML, reply_markup=create_back_button(user_id))
            return

        await query.edit_message_text(as_html(t("analysis.analyzing_portfolio", user_id=user_id, time=timestamp)), parse_mode=ParseMode.HTML)

        coin_ids = [COIN_ID_MAP.get(c.upper(), c.lower()) for c in portfolio.keys()]

        show_explanations = get_show_explanations(user_id)
        news_context = None
        if show_explanations:
            try:
                items = get_significant_news(limit=3)
                sources = []
                combined_text = ""
                for it in items:
                    n = it.get("news", {})
                    if n.get("source"):
                        sources.append(n.get("source"))
                    combined_text += " " + (n.get("summary") or "") + " " + (n.get("title") or "")
                sentiment = "neutral"
                if combined_text and ai_analyzer.finbert and ai_analyzer.finbert.status.get("enabled"):
                    try:
                        label, _score, _backend = await ai_analyzer.finbert.analyze_async(combined_text[:512])
                        sentiment = "positive" if label.startswith("pos") else "negative" if label.startswith("neg") else "neutral"
                    except Exception:
                        sentiment = "neutral"
                news_context = {
                    "count": len(items),
                    "sentiment": sentiment,
                    "sources": sources or ["cryptonews", "cointelegraph", "decrypt"],
                }
            except Exception:
                news_context = {"count": 0, "sentiment": "neutral", "sources": ["cryptonews", "cointelegraph", "decrypt"]}

        # Пробуем снапшот
        t_snap = time.time()
        snap_service = context.application.bot_data.get("market_snapshot")
        data_map = {}

        if snap_service and snap_service.get_latest():
            snap = snap_service.get_latest()
            # Проверяем свежесть (120 сек)
            if snap.ts and (time.time() - snap.ts) < 120:
                data_map = {cid: {"market": snap.markets.get(cid, {})} for cid in coin_ids}
                logger.info(f"✅ Snapshot used, age: {time.time() - snap.ts:.1f}s, fetch time: {time.time() - t_snap:.3f}s")

        # Если снапшот старый или пустой - используем batch
        if not data_map or not any(d["market"] for d in data_map.values()):
            t_batch = time.time()
            try:
                data_map = await batch.analyze(coin_ids, need_history=False)
                logger.info(f"✅ Batch.analyze used, time: {time.time() - t_batch:.3f}s")
            except Exception as e:
                await query.edit_message_text(as_html(t("errors.market_error", user_id=user_id, error=str(e))), parse_mode=ParseMode.HTML, reply_markup=create_back_button(user_id))
                return

        t_analysis = time.time()
        parts = []
        for coin_sym, entry in portfolio.items():
            t_coin = time.time()
            cid = COIN_ID_MAP.get(coin_sym.upper(), coin_sym.lower())
            pack = data_map.get(cid, {}) or {}
            m = pack.get("market", {}) or {}
            h = pack.get("history", {}) or {}

            price = m.get("current_price")
            if not isinstance(price, (int, float)):
                parts.append(t("analysis.no_market_data", user_id=user_id, coin=coin_sym.upper()))
                continue

            # RSI / волатильность (если есть история)
            rsi_val = None
            vol = None
            prices = h.get("prices") or []
            if prices:
                df = pd.DataFrame(prices, columns=["ts","close"])
                rsi_ser = compute_rsi(df["close"], 14)
                rsi_val = float(rsi_ser.dropna().iloc[-1]) if not rsi_ser.dropna().empty else None
                vol = realized_volatility(df["close"], 14)

            # базовый анализ
            analysis_txt = await analyze_coin_full(
                cid, price, entry, m, context,
                user_id=user_id, show_explanations=show_explanations, news_context=news_context
            )

            parts.append(analysis_txt)
            logger.info(f"  - {coin_sym} total: {time.time() - t_coin:.3f}s")

        logger.info(f"✅ Total analysis time: {time.time() - t_analysis:.3f}s")
        logger.info(f"✅ TOTAL REQUEST TIME: {time.time() - t_start:.3f}s")

        blocks = parts if parts else ["Ошибка анализа"]
        max_len = 4096
        current = ""
        first = True
        for block in blocks:
            if not current:
                current = block
            elif len(current) + 2 + len(block) > max_len:
                if first:
                    await query.edit_message_text(as_html(current), parse_mode=ParseMode.HTML, reply_markup=create_back_button(user_id))
                    first = False
                else:
                    await query.message.reply_text(as_html(current), parse_mode=ParseMode.HTML)
                current = block
            else:
                current += "\n\n" + block
        if current:
            if first:
                await query.edit_message_text(as_html(current), parse_mode=ParseMode.HTML, reply_markup=create_back_button(user_id))
            else:
                await query.message.reply_text(as_html(current), parse_mode=ParseMode.HTML)

    elif query.data == "analyze_top10":
        # ACCESS CONTROL CHECK
        user_id = query.from_user.id
        access_manager = get_access_manager()

        # Check rate limit
        rate_limiter = get_rate_limiter()
        tier = access_manager.get_user_tier(user_id)
        allowed, reason = rate_limiter.check_rate_limit(user_id, tier.value)
        if not allowed:
            await query.answer(reason, show_alert=True)
            return

        # Check basic analysis access (FREE tier can use this)
        allowed, reason = access_manager.check_access(user_id, 'basic_analysis')
        if not allowed:
            keyboard = [[InlineKeyboardButton(t("tier.button_premium", user_id=user_id), callback_data="premium")],
                       [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
            await query.edit_message_text(
                as_html(t("errors.access_denied_feature", user_id=user_id, reason=reason)),
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        # Record request
        rate_limiter.record_request(user_id)
        access_manager.record_request(user_id)

        top_coins = ["bitcoin","ethereum","binancecoin","solana","ripple",
                     "cardano","avalanche-2","dogecoin","polkadot","chainlink"]

        await query.edit_message_text(as_html(t("analysis.analyzing_top10", user_id=user_id)), parse_mode=ParseMode.HTML)

        try:
            snap_service = context.application.bot_data.get("market_snapshot")
            if snap_service and snap_service.get_latest():
                markets = snap_service.get_latest().markets
                data_map = {cid: {"market": markets.get(cid, {})} for cid in top_coins}
            else:
                # fallback
                data_map = await batch.analyze(top_coins, need_history=False)
        except Exception as e:
            error_text = t("analysis.error_with_nav", user_id=user_id, error=str(e))
            keyboard = [
                [InlineKeyboardButton(t("analysis.retry", user_id=user_id), callback_data="analyze_top10")],
                [InlineKeyboardButton(t("analysis.back_to_analysis", user_id=user_id), callback_data="analyze")],
                [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
            ]
            await query.edit_message_text(as_html(error_text), parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))
            return

        parts = [t("analyze.top10_title", user_id=user_id), ""]

        for cid in top_coins:
            m = (data_map.get(cid) or {}).get("market", {}) or {}
            price = m.get("current_price")
            ch24 = m.get("price_change_percentage_24h_in_currency", m.get("price_change_percentage_24h"))
            ch7d = m.get("price_change_percentage_7d_in_currency", m.get("price_change_percentage_7d"))
            name = (m.get("name") or cid).upper()
            symbol = (m.get("symbol") or "").upper()
            label = f"{name} ({symbol})" if symbol else name
            
            # Market cap and volume
            market_cap = m.get("market_cap")
            volume_24h = m.get("total_volume")
            
            # Format market cap
            if market_cap:
                if market_cap >= 1_000_000_000:
                    market_cap_str = f"${market_cap / 1_000_000_000:.2f}B"
                else:
                    market_cap_str = f"${market_cap / 1_000_000:.2f}M"
            else:
                market_cap_str = t("common.not_available", user_id=user_id)
            
            # Format volume
            if volume_24h:
                if volume_24h >= 1_000_000_000:
                    volume_str = f"${volume_24h / 1_000_000_000:.2f}B"
                else:
                    volume_str = f"${volume_24h / 1_000_000:.2f}M"
            else:
                volume_str = t("common.not_available", user_id=user_id)

            price_str = f"${price:.2f}" if isinstance(price, (int, float)) else t("common.not_available", user_id=user_id)
            ch24_str = f"{ch24:+.1f}%" if isinstance(ch24, (int, float)) else t("common.not_available", user_id=user_id)
            ch7d_str = f"{ch7d:+.1f}%" if isinstance(ch7d, (int, float)) else t("common.not_available", user_id=user_id)
            
            # Emoji signal based on 24h change
            if isinstance(ch24, (int, float)):
                if ch24 > 3:
                    signal_emoji = "📈"
                elif ch24 < -3:
                    signal_emoji = "📉"
                else:
                    signal_emoji = "➡️"
            else:
                signal_emoji = "➡️"

            # Generate rule-based signal with confidence
            signal_text, confidence = generate_rule_based_signal(ch24, ch7d, user_id)

            # Enhanced formatting
            parts += [
                f"{signal_emoji} **{label}**",
                f"+ {t('analysis.price_label', user_id=user_id)}: {price_str} | 24h: {ch24_str} | 7d: {ch7d_str}",
                f"+ {t('analysis.cap_label', user_id=user_id)}: {market_cap_str} | {t('analysis.vol_label', user_id=user_id)}: {volume_str}",
                f"L **{signal_text}** ({t('analysis.confidence', user_id=user_id)}: {confidence:.0f}%)",
                ""
            ]

        await query.edit_message_text(as_html("\n".join(parts)), parse_mode=ParseMode.HTML, reply_markup=create_back_button(user_id))

    elif query.data.startswith("analyze_coin:"):
        # ACCESS CONTROL CHECK
        user_id = query.from_user.id
        access_manager = get_access_manager()
        rate_limiter = get_rate_limiter()
        tier = access_manager.get_user_tier(user_id)

        # Check rate limit
        allowed, reason = rate_limiter.check_rate_limit(user_id, tier.value)
        if not allowed:
            await query.answer(reason, show_alert=True)
            return

        # 1) монета
        symbol = query.data.split(":", 1)[1].upper()
        coin_id = COIN_ID_MAP.get(symbol, symbol.lower())  # маппинг уже есть в utils.py

        # Check basic analysis access (FREE tier can use this)
        allowed, reason = access_manager.check_access(user_id, 'basic_analysis')
        if not allowed:
            keyboard = [
                [InlineKeyboardButton(t("premium.button", user_id=user_id), callback_data="premium")],
                [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]
            ]
            await query.edit_message_text(
                f"? {reason}",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        # Check coin limit (FREE users can only analyze BTC/ETH)
        allowed, reason = access_manager.check_coin_limit(user_id, coin_id)
        if not allowed:
            keyboard = [
                [InlineKeyboardButton(t("tier.button_premium", user_id=user_id), callback_data="premium")],
                [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]
            ]
            await query.edit_message_text(
                t("errors.access_denied_feature", user_id=user_id, reason=reason),
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        # Record request
        rate_limiter.record_request(user_id)
        access_manager.record_request(user_id)

        # 2) свежий маркет из снапшота (без сети)
        app = get_global_app_ref()
        row = {}
        snap_ts = None
        if app:
            snap_svc = app.bot_data.get("market_snapshot")
            snap = snap_svc.get_latest() if snap_svc else None
            if snap and snap.markets:
                row = snap.markets.get(coin_id) or {}
                snap_ts = snap.ts

        # 3) если снапшота нет или монеты нет в нем — как фолбэк, дергаем ТОЛЬКО ОДНУ монету
        if not row:
            try:
                mkts = await get_markets_data([coin_id])  # вернётся dict id->row
                row = mkts.get(coin_id) or {}
            except Exception as e:
                error_text = t("analysis.error_with_nav", user_id=user_id, error=str(e))
                keyboard = [
                    [InlineKeyboardButton(t("analysis.retry", user_id=user_id), callback_data=f"analyze_coin:{symbol}")],
                    [InlineKeyboardButton(t("analysis.back_to_analysis", user_id=user_id), callback_data="analyze")],
                    [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
                ]
                await query.edit_message_text(as_html(error_text), parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))
                return

        current_price = row.get("current_price")  # аккуратно: None-guard уже есть утилитами в файле

        pm = get_portfolio_manager()
        portfolio = pm.load_portfolio(user_id)
        entry = portfolio.get(symbol, {}) or portfolio.get(coin_id, {})  # поддержим оба ключа
        
        try:
            show_explanations = get_show_explanations(user_id)
            news_context = None
            if show_explanations:
                try:
                    items = get_significant_news(limit=3)
                    sources = []
                    combined_text = ""
                    for it in items:
                        n = it.get("news", {})
                        if n.get("source"):
                            sources.append(n.get("source"))
                        combined_text += " " + (n.get("summary") or "") + " " + (n.get("title") or "")
                    sentiment = "neutral"
                    if combined_text and ai_analyzer.finbert and ai_analyzer.finbert.status.get("enabled"):
                        try:
                            label, _score, _backend = await ai_analyzer.finbert.analyze_async(combined_text[:512])
                            sentiment = "positive" if label.startswith("pos") else "negative" if label.startswith("neg") else "neutral"
                        except Exception:
                            sentiment = "neutral"
                    news_context = {
                        "count": len(items),
                        "sentiment": sentiment,
                        "sources": sources or ["cryptonews", "cointelegraph", "decrypt"],
                    }
                except Exception:
                    news_context = {"count": 0, "sentiment": "neutral", "sources": ["cryptonews", "cointelegraph", "decrypt"]}

            text = await analyze_coin_full(
                coin_id=coin_id,
                current_price=float(current_price or 0.0),
                portfolio_entry=entry,
                indicators=row,
                context=context,
                user_id=user_id,
                show_explanations=show_explanations,
                news_context=news_context
            )
            await query.edit_message_text(as_html(text), parse_mode=ParseMode.HTML, reply_markup=create_action_buttons(user_id))
        except Exception as e:
            error_text = t("analysis.error_with_nav", user_id=user_id, error=str(e))
            keyboard = [
                [InlineKeyboardButton(t("analysis.retry", user_id=user_id), callback_data=f"analyze_coin:{symbol}")],
                [InlineKeyboardButton(t("analysis.back_to_analysis", user_id=user_id), callback_data="analyze")],
                [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
            ]
            await query.edit_message_text(as_html(error_text), parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))


    elif query.data == "stats":
        try:
            await show_stats(query.from_user.id, context, query.message, prefer_edit=True)
        except Exception as e:
            user_id = query.from_user.id
            await query.edit_message_text(t("errors.stats_error", user_id=user_id, error=str(e)))
    elif query.data == "movers":
        try:
            await show_movers(query.from_user.id, context, query.message, prefer_edit=True)
        except Exception as e:
            user_id = query.from_user.id
            await query.edit_message_text(t("errors.movers_error", user_id=user_id, error=str(e)))
    elif query.data == "add_coin_prompt":
        # Handler for "Add Coin" button
        user_id = query.from_user.id
        help_text = (
            f"<b>{t('portfolio.add_coin_help_title', user_id=user_id)}</b>\n\n"
            f"<code>{t('portfolio.add_coin_help_text', user_id=user_id)}</code>\n\n"
            f"<b>{t('portfolio.add_coin_help_example', user_id=user_id)}</b>"
        )
        keyboard = [[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
        await query.edit_message_text(help_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "sell_coin_prompt":
        # Handler for "Sell" button - list user's coins
        user_id = query.from_user.id
        pm = get_portfolio_manager()
        portfolio = pm.load_portfolio(user_id)
        
        coin_list = ", ".join([coin.upper() for coin in portfolio.keys()]) if portfolio else t("common.not_available", user_id=user_id)
        
        help_text = (
            f"<b>{t('portfolio.sell_coin_help_title', user_id=user_id)}</b>\n\n"
            f"<code>{t('portfolio.sell_coin_help_text', user_id=user_id)}</code>\n\n"
            f"<b>{t('portfolio.sell_coin_help_coins', user_id=user_id, coins=coin_list)}</b>\n\n"
            f"<b>{t('portfolio.sell_coin_help_example', user_id=user_id)}</b>"
        )
        keyboard = [[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
        await query.edit_message_text(help_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "back_to_menu":
        await back_to_main_menu(query)

    elif query.data == "news":
        user_id = query.from_user.id
        await context.bot.send_chat_action(chat_id=query.message.chat.id, action=ChatAction.TYPING)
        items = get_significant_news()
        await render_news(status_msg=query.message, items=items, reply_markup=create_news_keyboard(user_id), user_id=user_id, fallback_bot=context.bot, fallback_chat_id=query.message.chat.id)
    elif query.data == "help":
        user_id = query.from_user.id
        await show_help(user_id, query.message, prefer_edit=True)
    elif query.data == "my_tier":
        # Show user's current tier and limits
        user_id = query.from_user.id
        output = _build_tier_text(user_id)

        access_manager = get_access_manager()
        tier = access_manager.get_user_tier(user_id).value

        keyboard = []
        if tier == 'free':
            keyboard.append([InlineKeyboardButton(t("tier.button_premium", user_id=user_id), callback_data="premium")])

        keyboard.append([InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")])

        await query.edit_message_text(
            as_html(output),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif query.data == "premium":
        # Show premium pricing tiers with payment options
        payment_manager = get_payment_manager()
        await payment_manager.show_premium_options(update, context)

    elif query.data.startswith("buy_plan:"):
        # Handle plan purchase
        plan_id = query.data.split(":", 1)[1]
        payment_manager = get_payment_manager()
        await payment_manager.create_invoice(update, context, plan_id)

    elif query.data == "admin_panel":
        # Owner-only admin panel
        user_id = query.from_user.id

        if user_id != USER_ID:
            await query.answer(t("admin.access_denied", user_id=user_id), show_alert=True)
            return

        output = t("admin.title", user_id=user_id) + "\n\n"
        output += t("admin.subtitle", user_id=user_id) + "\n\n"

        access_manager = get_access_manager()
        users = access_manager.users.get("users", {})

        # Count by tier
        tier_counts = {}
        for user_data in users.values():
            tier = user_data.get('tier', 'free')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Check if OWNER is already in users dict to avoid double-counting
        # OWNER is determined by config.OWNER_ID, not in access_control.json
        owner_in_users = any(uid == USER_ID for uid in users.keys())
        if not owner_in_users:
            tier_counts['owner'] = tier_counts.get('owner', 0) + 1

        # Total users count
        total_users = len(users) + (0 if owner_in_users else 1)
        output += t("admin.total_users", user_id=user_id, count=total_users) + "\n"

        output += "\n" + t("admin.by_tiers", user_id=user_id) + "\n"
        for tier, count in tier_counts.items():
            output += t("admin.tier_count", user_id=user_id, tier=tier.upper(), count=count) + "\n"

        # Show total portfolio users from database
        try:
            pm = get_portfolio_manager()
            all_portfolio_users = pm.get_all_users()
            output += "\n" + t("admin.portfolio_users", user_id=user_id, count=len(all_portfolio_users)) + "\n"
        except Exception as e:
            logger.error(f"Error getting portfolio users: {e}")

        keyboard = [
            [InlineKeyboardButton(t("admin.button_stats", user_id=user_id), callback_data="admin_stats"),
             InlineKeyboardButton(t("admin.button_grant", user_id=user_id), callback_data="admin_grant_help")],
            [InlineKeyboardButton(t("admin.button_users_list", user_id=user_id), callback_data="admin_users_list")],
            [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
        ]

        await query.edit_message_text(
            as_html(output),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif query.data == "admin_stats":
        # Owner-only: detailed statistics
        user_id = query.from_user.id
        if user_id != USER_ID:
            await query.answer(t("admin.access_denied", user_id=user_id), show_alert=True)
            return

        access_manager = get_access_manager()
        users = access_manager.users.get("users", {})

        output = t("admin.stats_title", user_id=user_id) + "\n\n"
        output += t("admin.total_users", user_id=user_id, count=len(users)) + "\n\n"

        # Active premium users
        now = datetime.datetime.now().timestamp()

        active_premium = 0
        expired = 0

        for user_data in users.values():
            tier = user_data.get('tier', 'free')
            expires = user_data.get('expires_date')

            if expires and expires < now:
                expired += 1
            elif tier in ['premium', 'beta']:
                active_premium += 1

        output += t("admin.active_premium", user_id=user_id, count=active_premium) + "\n"
        output += t("admin.expired", user_id=user_id, count=expired) + "\n\n"

        # By tier
        tier_counts = {}
        for user_data in users.values():
            tier = user_data.get('tier', 'free')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        output += t("admin.by_tiers", user_id=user_id) + "\n"
        for tier, count in tier_counts.items():
            output += t("admin.tier_count", user_id=user_id, tier=tier.upper(), count=count) + "\n"

        keyboard = [[InlineKeyboardButton(t("common.back_to_menu", user_id=user_id), callback_data="admin_panel")]]

        await query.edit_message_text(
            as_html(output),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif query.data == "admin_grant_help":
        # Owner-only: show grant command help
        user_id = query.from_user.id
        if user_id != USER_ID:
            await query.answer(t("admin.access_denied", user_id=user_id), show_alert=True)
            return

        output = t("admin.grant_title", user_id=user_id) + "\n\n"
        output += t("admin.grant_command", user_id=user_id) + "\n"
        output += t("admin.grant_usage", user_id=user_id) + "\n\n"

        output += t("admin.grant_tiers", user_id=user_id) + "\n"
        output += t("admin.grant_tiers_list", user_id=user_id) + "\n\n"

        output += t("admin.grant_days", user_id=user_id) + "\n"
        output += t("admin.grant_days_info", user_id=user_id) + "\n\n"

        output += t("admin.grant_examples", user_id=user_id) + "\n"
        output += t("admin.grant_example_1", user_id=user_id) + "\n"
        output += t("admin.grant_example_2", user_id=user_id) + "\n\n"

        output += t("admin.grant_other", user_id=user_id) + "\n"
        output += t("admin.grant_revoke", user_id=user_id) + "\n"
        output += t("admin.grant_user_stats", user_id=user_id) + "\n"

        keyboard = [[InlineKeyboardButton(t("common.back_to_menu", user_id=user_id), callback_data="admin_panel")]]

        await query.edit_message_text(
            output,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif query.data == "admin_users_list":
        # Owner-only: show list of users
        user_id = query.from_user.id
        if user_id != USER_ID:
            await query.answer(t("admin.access_denied", user_id=user_id), show_alert=True)
            return

        access_manager = get_access_manager()
        users = access_manager.users.get("users", {})

        if not users:
            output = t("admin.users_list_title", user_id=user_id) + "\n\n"
            output += t("admin.users_list_empty", user_id=user_id)
        else:
            output = t("admin.users_list_title", user_id=user_id) + "\n\n"

            # Sort by tier
            sorted_users = sorted(
                users.items(),
                key=lambda x: (x[1].get('tier', 'free'), x[0])
            )

            for uid, data in sorted_users[:15]:  # Show first 15
                tier = data.get('tier', 'free')
                expires = data.get('expires_date')

                user_line = f"• <code>{uid}</code> - {tier.upper()}"

                if expires:
                    expires_dt = datetime.datetime.fromtimestamp(expires)
                    days_left = (expires_dt - datetime.datetime.now()).days

                    if days_left > 0:
                        user_line += f" ({days_left}д)"
                    else:
                        user_line += f" (истек)"

                output += user_line + "\n"

            if len(users) > 15:
                output += "\n" + t("admin.users_list_more", user_id=user_id, count=len(users) - 15)

        keyboard = [[InlineKeyboardButton(t("common.back_to_menu", user_id=user_id), callback_data="admin_panel")]]

        await query.edit_message_text(
            output,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    else:
        user_id = query.from_user.id
        await query.edit_message_text(t("errors.unknown_command", user_id=user_id))















