import logging
import time
import asyncio
import html, re
import datetime
import math
import config

from core.ai_analytics import ai_analyzer
from core.analyzer_full import analyze_coin_full
from core.utils import COIN_ID_MAP, _track_user_activity, resolve_coin_id
from core.analytics.user_analytics import get_user_analytics
from core.api.coingecko_client import get_markets_data
from core.portfolio_db import get_portfolio_manager
from core.portfolio_valuation import value_portfolio
from core.formatting import as_html, format_price, plain
from core.market_signal import classify_price_momentum
from core.incidents import public_temporary_message, report_incident
from core.access.presentation import format_access_decision

from modules.news import get_significant_news, _format_news_block, render_news
from config import PORTFOLIO_FILE, OWNER_ID as USER_ID

from modules.alerts import update_alert_action
from modules.user_prefs import (
    get_timezone, get_language, set_timezone, set_language,
    SUPPORTED_TIMEZONES, SUPPORTED_LANGS, get_profile,
    get_show_explanations, set_show_explanations, init_profile_language
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
from modules.callback_utils import answer_once

_global_app_ref = None  # will be set on first use
logger = logging.getLogger(__name__)

# Default timezone constant
DEFAULT_TIMEZONE = "Europe/Berlin"


async def _public_incident(context, error: BaseException, user_id: int, subsystem: str) -> str:
    await report_incident(context, error, subsystem=subsystem, user_id=user_id)
    return public_temporary_message(user_id)


def _fresh_market_snapshot(application):
    if not application:
        return None
    service = application.bot_data.get("market_snapshot")
    max_age = max(120, config.SNAPSHOT_REFRESH_SECONDS * 3)
    if service and hasattr(service, "get_usable"):
        return service.get_usable(max_age)
    snapshot = service.get_latest() if service else None
    if not snapshot or not getattr(snapshot, "ts", None):
        return None
    return snapshot if time.time() - snapshot.ts <= max_age else None


async def _get_market_rows(application, coin_ids):
    """Merge a fresh snapshot with a one-shot fetch for missing IDs."""
    snapshot = _fresh_market_snapshot(application)
    rows = dict(snapshot.markets) if snapshot and snapshot.markets else {}
    missing = [coin_id for coin_id in dict.fromkeys(coin_ids) if coin_id not in rows]
    if missing:
        try:
            fetched = await get_markets_data(missing)
            rows.update(fetched)
            if fetched and snapshot is None:
                return rows, time.time()
        except Exception:
            logger.warning("Market row fallback failed for %s coins", len(missing), exc_info=True)
    return rows, getattr(snapshot, "ts", None)

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

async def build_news_context(user_id: int = None):
    """Собирает лёгкий news-context для AI-объяснений.

    Единственное место этой логики (раньше блок был скопирован в 3 хендлерах).
    RSS + sentiment — блокирующий I/O, поэтому уводим в отдельный поток.
    """
    # "No news analysis" is None, never "neutral" (missing evidence is not neutral evidence).
    fallback = {"count": 0, "sentiment": None, "sources": ["cryptonews", "cointelegraph", "decrypt"]}
    try:
        items = await asyncio.to_thread(get_significant_news, 3)
        sources = []
        combined_text = ""
        for it in items:
            n = it.get("news", {})
            if n.get("source"):
                sources.append(n.get("source"))
            combined_text += " " + (n.get("summary") or "") + " " + (n.get("title") or "")

        sentiment = None
        if combined_text and ai_analyzer.finbert and ai_analyzer.finbert.status.get("available"):
            try:
                label, _score, _backend = await ai_analyzer.finbert.analyze_async(combined_text[:512])
                sentiment = "positive" if label.startswith("pos") else "negative" if label.startswith("neg") else "neutral"
            except Exception:
                sentiment = None

        return {
            "count": len(items),
            "sentiment": sentiment,
            "sources": sources or fallback["sources"],
        }
    except Exception:
        return fallback

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
    """Return a localized rule signal and a qualitative evidence strength."""
    result = classify_price_momentum(ch24, ch7d)
    return (
        t(result.translation_key, user_id=user_id),
        t(f"analysis.strength_{result.strength}", user_id=user_id),
    )

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

    coin_ids = [resolve_coin_id(coin) for coin in portfolio]
    rows, snapshot_ts = await _get_market_rows(context.application, coin_ids)
    as_of = (
        datetime.datetime.fromtimestamp(snapshot_ts, datetime.timezone.utc).strftime("%H:%M:%S") + " UTC"
        if snapshot_ts
        else "n/a"
    )

    valuation = value_portfolio(
        portfolio,
        rows,
        coin_id_resolver=resolve_coin_id,
    )
    lines = [f"<b>{plain(t('portfolio.title', user_id=user_id))}</b>", "═" * 20, ""]

    for position in valuation.positions:
        coin = position.symbol
        amount = position.amount
        entry_price = position.entry_price
        current_price = position.current_price

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
                f"<b>{html.escape(coin.upper())}</b>\n"
                f"  {t('portfolio.amount_label', user_id=user_id)}: {amount:.6f}\n"
                f"  {t('portfolio.entry_price_label', user_id=user_id)}: {entry_str}\n"
                f"  {t('portfolio.current_price_label', user_id=user_id)}: {current_str}\n"
                f"  {t('portfolio.current_value_label', user_id=user_id)}: {value_str}\n"
                f"  {pnl_emoji} {t('portfolio.pnl_label', user_id=user_id)}: ${pnl_usd:+.2f} ({pnl_pct:+.1f}%)\n"
            )
        else:
            entry_str = format_price(entry_price)
            lines.append(
                f"<b>{html.escape(coin.upper())}</b>\n"
                f"  {t('portfolio.amount_label', user_id=user_id)}: {amount:.6f}\n"
                f"  {t('portfolio.entry_price_label', user_id=user_id)}: {entry_str}\n"
                f"  {t('portfolio.price_temporarily_unavailable', user_id=user_id)}\n"
            )

    total_cost_str = format_price(valuation.total_invested)
    if valuation.is_complete:
        total_value = valuation.total_current_value or 0.0
        total_pnl = valuation.total_pnl or 0.0
        total_pnl_pct = valuation.total_pnl_pct
        pnl_emoji = "📈" if total_pnl > 0 else "📉" if total_pnl < 0 else "➡️"
        total_value_str = format_price(total_value)
        pnl_pct_text = f" ({total_pnl_pct:+.1f}%)" if total_pnl_pct is not None else ""

        lines.append(
            f"\n{'═' * 20}\n"
            f"<b>{plain(t('portfolio.total_header', user_id=user_id))}</b>\n"
            f"  {t('portfolio.invested_label', user_id=user_id)}: {total_cost_str}\n"
            f"  {t('portfolio.current_value_label', user_id=user_id)}: {total_value_str}\n"
            f"  {pnl_emoji} {t('portfolio.pnl_label', user_id=user_id)}: ${total_pnl:+.2f}{pnl_pct_text}"
        )
    else:
        known_value_str = format_price(valuation.known_current_value)
        missing = ", ".join(valuation.missing_symbols)
        lines.append(
            f"\n{'═' * 20}\n"
            f"<b>{plain(t('portfolio.total_header', user_id=user_id))}</b>\n"
            f"  {t('portfolio.invested_label', user_id=user_id)}: {total_cost_str}\n"
            f"  {t('portfolio.known_value_label', user_id=user_id)}: {known_value_str}\n"
            f"  {t('portfolio.incomplete_totals', user_id=user_id, coins=missing)}"
        )

    lines.append(f"\n<i>{plain(t('portfolio.as_of', user_id=user_id, time=as_of))}</i>")

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
    cids = [resolve_coin_id(c) for c in portfolio.keys()]
    rows, snapshot_ts = await _get_market_rows(app, cids)
    as_of = fmt_asof_for_user(user_id, snapshot_ts) if snapshot_ts else "n/a"
    if not rows:
        await respond_message(
            message,
            t("stats.no_data", user_id=user_id),
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
            prefer_edit=prefer_edit,
        )
        return

    valuation = value_portfolio(
        portfolio,
        rows or {},
        coin_id_resolver=resolve_coin_id,
    )
    msg = [t("stats.title", user_id=user_id), t("stats.divider", user_id=user_id), ""]

    def _f(x, default=None):
        try:
            value = float(x)
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    for coin, data in portfolio.items():
        cid = resolve_coin_id(coin)
        entry = _f(data.get("buy_price", data.get("price", 0.0)), 0.0)
        row = (rows or {}).get(cid, {}) or {}

        now = _f(row.get("current_price"), None)
        ch1 = row.get("price_change_percentage_1h_in_currency", row.get("price_change_percentage_1h"))
        ch24 = row.get("price_change_percentage_24h_in_currency", row.get("price_change_percentage_24h"))
        ch7 = row.get("price_change_percentage_7d_in_currency", row.get("price_change_percentage_7d"))

        if now is None:
            continue

        change = ((now - (entry or 0.0)) / (entry or 1.0) * 100.0) if entry else 0.0

        def fmt(x):
            return (
                f"{x:+.1f}%"
                if isinstance(x, (int, float))
                else t("common.not_available", user_id=user_id)
            )

        msg += [
            "━━━━━━━━━━━━━━━━━",
            t("stats.coin_header", user_id=user_id, coin=coin.upper()),
            t("stats.price", user_id=user_id, price=format_price(now)),
            t("stats.change_1h", user_id=user_id, change=fmt(ch1)),
            t("stats.change_24h", user_id=user_id, change=fmt(ch24)),
            t("stats.change_7d", user_id=user_id, change=fmt(ch7)),
            t("stats.profit_loss", user_id=user_id, pl=change),
        ]

    msg += ["", t("stats.total_invested", user_id=user_id, amount=valuation.total_invested)]
    if valuation.is_complete:
        msg.append(t("stats.current_value", user_id=user_id, amount=valuation.total_current_value or 0.0))
    else:
        msg += [
            t("portfolio.known_value", user_id=user_id, amount=valuation.known_current_value),
            t("portfolio.incomplete_totals", user_id=user_id, coins=", ".join(valuation.missing_symbols)),
        ]
    msg.append(t("portfolio.as_of", user_id=user_id, time=as_of))

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
    cids = [resolve_coin_id(c) for c in portfolio.keys()]
    rows, snapshot_ts = await _get_market_rows(app, cids)
    as_of = fmt_asof_for_user(user_id, snapshot_ts) if snapshot_ts else "n/a"
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
        cid = resolve_coin_id(coin)
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
            t("movers.price_line", user_id=user_id, entry=format_price(entry), current=format_price(current)),
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
    init_profile_language(user_id, update.effective_user.language_code)
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
    _track_user_activity(update.effective_user, "unknown_command", {"text": text})
    await update.message.reply_text(
        t("errors.unknown_command", user_id=update.effective_user.id)
    )

async def handle_coin_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle custom coin input when user is in analyze_custom mode"""
    user_id = update.effective_user.id

    # Check if we're waiting for coin input FIRST — otherwise every random text
    # message in the chat lands in analytics and hits the disk
    if not context.user_data.get("awaiting_coin_input", False):
        return  # Not waiting for input, ignore

    _track_user_activity(update.effective_user, "custom_coin_input", {"text": update.message.text if update.message else ""})

    # Clear the flag
    context.user_data["awaiting_coin_input"] = False

    # Get the coin symbol from message (basic input hygiene)
    coin_symbol = update.message.text.strip().upper()[:20]
    if not re.fullmatch(r"[A-Z0-9\-]{2,20}", coin_symbol):
        await update.message.reply_text(t("analysis.coin_not_found", user_id=user_id))
        return

    # Check access control
    access_manager = get_access_manager()
    rate_limiter = get_rate_limiter()
    tier = access_manager.get_user_tier(user_id)

    # Check rate limit
    allowed, reason = rate_limiter.check_rate_limit(user_id, tier.value)
    if not allowed:
        await update.message.reply_text(t("errors.system_busy", user_id=user_id))
        return

    # Check basic analysis access
    decision = access_manager.evaluate_access(user_id, 'basic_analysis')
    if not decision.allowed:
        await update.message.reply_text(format_access_decision(decision, user_id))
        return

    registry = context.application.bot_data.get("coin_registry")
    coin_id = resolve_coin_id(coin_symbol, registry)

    # Coin limit must apply to typed input too, not only to buttons —
    # otherwise FREE users bypass the BTC/ETH restriction by typing
    decision = access_manager.evaluate_coin_limit(user_id, coin_id)
    if not decision.allowed:
        await update.message.reply_text(format_access_decision(decision, user_id))
        return

    # Count this request against the user's cooldown (was missing: the FREE
    # 2h cooldown never applied to the typed-input path)
    rate_limiter.record_request(user_id)
    access_manager.record_request(user_id)

    # Show analyzing message
    await update.message.reply_text(t("analysis.analyzing_coin", user_id=user_id, coin=coin_symbol))
    
    # Get market data
    try:
        rows, _snapshot_ts = await _get_market_rows(context.application, [coin_id])
        row = rows.get(coin_id)
        
        if not row:
            registry = context.application.bot_data.get("coin_registry")
            is_known = coin_symbol in COIN_ID_MAP or (
                registry is not None and registry.resolve(coin_symbol) is not None
            )
            key = "analysis.market_data_updating" if is_known else "analysis.coin_not_found"
            await update.message.reply_text(
                t(key, user_id=user_id, coin=coin_symbol)
            )
            return
        
        current_price = row.get("current_price")
        
        pm = get_portfolio_manager()
        portfolio = pm.load_portfolio(user_id)
        entry = portfolio.get(coin_symbol, {}) or portfolio.get(coin_id, {})

        show_explanations = get_show_explanations(user_id)
        news_context = await build_news_context(user_id) if show_explanations else None

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
        await report_incident(
            context,
            e,
            subsystem="typed_coin_analysis",
            user_id=user_id,
        )
        await update.message.reply_text(public_temporary_message(user_id))

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "portfolio")
    await show_portfolio(user_id, context, update.message, prefer_edit=False)

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
    # Branches answer with their own toast where they have one; whatever is
    # left unanswered gets the plain answer here, exactly once.
    try:
        await _menu_callback(update, context)
    finally:
        await answer_once(update.callback_query)


async def _menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
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
        if update_alert_action(alert_id, user_id, action):
            if action == "alert_done":
                await answer_once(query, t("alerts.status_done", user_id=user_id))
            elif action == "alert_remind":
                await answer_once(query, t("alerts.status_remind", user_id=user_id))
            elif action == "alert_ignore":
                await answer_once(query, t("alerts.status_ignored", user_id=user_id))
            await query.edit_message_reply_markup(reply_markup=None)
        else:
            await answer_once(query, t("alerts.status_unavailable", user_id=user_id), show_alert=True)
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
        await answer_once(query, t("settings.explanations_saved", user_id=user_id))
        await show_settings(user_id, query.message, prefer_edit=True)

    elif query.data.startswith("set_lang:"):
        user_id = query.from_user.id
        lang = query.data.split("set_lang:", 1)[1]
        set_language(user_id, lang)
        await answer_once(query, t("settings.language_saved", user_id=user_id))
        await show_settings(user_id, query.message, prefer_edit=True)

    elif query.data.startswith("set_tz:"):
        user_id = query.from_user.id
        tz = query.data.split("set_tz:", 1)[1]
        first_run = get_profile(user_id).get("tz") is None
        set_timezone(user_id, tz)
        await answer_once(query, t("settings.timezone_saved", user_id=user_id))
        if first_run:
            # Onboarding: continue to the main menu, not into Settings.
            await back_to_main_menu(query)
        else:
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
            await answer_once(query, t("errors.system_busy", user_id=user_id), show_alert=True)
            return

        decision = access_manager.evaluate_access(user_id, 'basic_analysis')
        if not decision.allowed:
            keyboard = [[InlineKeyboardButton(t("tier.button_premium", user_id=user_id), callback_data="premium")],
                       [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
            await query.edit_message_text(
                as_html(format_access_decision(decision, user_id)),
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        t_start = time.time()
        timestamp = time.strftime("%H:%M:%S")
        pm = get_portfolio_manager()
        portfolio = pm.load_portfolio(user_id)
        if not portfolio:
            await query.edit_message_text(as_html(t("portfolio.empty", user_id=user_id)), parse_mode=ParseMode.HTML, reply_markup=create_back_button(user_id))
            return

        await answer_once(query)
        await query.edit_message_text(as_html(t("analysis.analyzing_portfolio", user_id=user_id, time=timestamp)), parse_mode=ParseMode.HTML)

        coin_ids = [resolve_coin_id(c) for c in portfolio.keys()]
        unavailable = [
            coin_id
            for coin_id in coin_ids
            if not access_manager.check_coin_limit(user_id, coin_id)[0]
        ]
        if unavailable:
            limits = access_manager.get_limits(user_id)
            reason = t(
                "errors.portfolio_coins_not_in_tier",
                user_id=user_id,
                coins=", ".join(unavailable),
                allowed=", ".join(limits.get('allowed_coins') or ["all"]),
            )
            # `decision` here is the *allowed* basic-analysis decision, whose
            # text is empty; the refusal reason is the coin-tier message.
            await query.edit_message_text(
                as_html(reason),
                parse_mode=ParseMode.HTML,
                reply_markup=create_back_button(user_id),
            )
            return

        rate_limiter.record_request(user_id)
        access_manager.record_request(user_id)

        show_explanations = get_show_explanations(user_id)
        news_context = await build_news_context(user_id) if show_explanations else None

        market_rows, market_ts = await _get_market_rows(context.application, coin_ids)
        data_map = {cid: {"market": market_rows.get(cid, {})} for cid in coin_ids}
        available_count = sum(1 for coin_id in coin_ids if market_rows.get(coin_id))
        logger.info(
            "Portfolio market coverage rows=%s requested=%s snapshot_ts=%s",
            available_count,
            len(coin_ids),
            market_ts,
        )

        t_analysis = time.time()
        parts = []
        for coin_sym, entry in portfolio.items():
            t_coin = time.time()
            cid = resolve_coin_id(coin_sym)
            pack = data_map.get(cid, {}) or {}
            m = pack.get("market", {}) or {}

            price = m.get("current_price")
            if not isinstance(price, (int, float)):
                parts.append(t("analysis.no_market_data", user_id=user_id, coin=coin_sym.upper()))
                continue

            # базовый анализ
            analysis_txt = await analyze_coin_full(
                cid, price, entry, m, context,
                user_id=user_id, show_explanations=show_explanations, news_context=news_context
            )

            parts.append(analysis_txt)
            logger.info(f"  - {coin_sym} total: {time.time() - t_coin:.3f}s")

        logger.info(f"✅ Total analysis time: {time.time() - t_analysis:.3f}s")
        logger.info(f"✅ TOTAL REQUEST TIME: {time.time() - t_start:.3f}s")

        blocks = parts if parts else [t("errors.analysis_error", user_id=user_id)]
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
            await answer_once(query, t("errors.system_busy", user_id=user_id), show_alert=True)
            return

        # TOP-10 is a paid batch feature in the central tier policy.
        decision = access_manager.evaluate_access(user_id, 'top10_analysis')
        if not decision.allowed:
            keyboard = [[InlineKeyboardButton(t("tier.button_premium", user_id=user_id), callback_data="premium")],
                       [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
            await query.edit_message_text(
                as_html(format_access_decision(decision, user_id)),
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        # Record request
        rate_limiter.record_request(user_id)
        access_manager.record_request(user_id)

        top_coins = ["bitcoin","ethereum","binancecoin","solana","ripple",
                     "cardano","avalanche-2","dogecoin","polkadot","chainlink"]

        await answer_once(query)
        await query.edit_message_text(as_html(t("analysis.analyzing_top10", user_id=user_id)), parse_mode=ParseMode.HTML)

        try:
            markets, _market_ts = await _get_market_rows(context.application, top_coins)
            if not any(markets.get(coin_id) for coin_id in top_coins):
                raise RuntimeError("top10 market coverage is empty")
            data_map = {cid: {"market": markets.get(cid, {})} for cid in top_coins}
        except Exception as e:
            error_text = await _public_incident(context, e, user_id, "top10_analysis")
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

            price_str = format_price(price) if isinstance(price, (int, float)) else t("common.not_available", user_id=user_id)
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

            signal_text, evidence_strength = generate_rule_based_signal(ch24, ch7d, user_id)

            # Enhanced formatting
            parts += [
                f"{signal_emoji} **{label}**",
                f"+ {t('analysis.price_label', user_id=user_id)}: {price_str} | 24h: {ch24_str} | 7d: {ch7d_str}",
                f"+ {t('analysis.cap_label', user_id=user_id)}: {market_cap_str} | {t('analysis.vol_label', user_id=user_id)}: {volume_str}",
                f"L **{signal_text}** ({t('analysis.evidence_strength', user_id=user_id)}: {evidence_strength})",
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
            await answer_once(query, t("errors.system_busy", user_id=user_id), show_alert=True)
            return

        # 1) монета
        symbol = query.data.split(":", 1)[1].upper()
        coin_id = resolve_coin_id(symbol)

        # Check basic analysis access (FREE tier can use this)
        decision = access_manager.evaluate_access(user_id, 'basic_analysis')
        if not decision.allowed:
            keyboard = [
                [InlineKeyboardButton(t("premium.button", user_id=user_id), callback_data="premium")],
                [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]
            ]
            await query.edit_message_text(
                format_access_decision(decision, user_id),
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        # Check coin limit (FREE users can only analyze BTC/ETH)
        decision = access_manager.evaluate_coin_limit(user_id, coin_id)
        if not decision.allowed:
            keyboard = [
                [InlineKeyboardButton(t("tier.button_premium", user_id=user_id), callback_data="premium")],
                [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]
            ]
            await query.edit_message_text(
                format_access_decision(decision, user_id),
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        # Record request
        rate_limiter.record_request(user_id)
        access_manager.record_request(user_id)
        await answer_once(query)

        # 2) свежий маркет из снапшота (без сети)
        app = get_global_app_ref()
        markets, _snapshot_ts = await _get_market_rows(app, [coin_id])
        row = markets.get(coin_id) or {}

        current_price = row.get("current_price")  # аккуратно: None-guard уже есть утилитами в файле

        pm = get_portfolio_manager()
        portfolio = pm.load_portfolio(user_id)
        entry = portfolio.get(symbol, {}) or portfolio.get(coin_id, {})  # поддержим оба ключа
        
        try:
            show_explanations = get_show_explanations(user_id)
            news_context = await build_news_context(user_id) if show_explanations else None

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
            error_text = await _public_incident(context, e, user_id, "coin_analysis")
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
            await query.edit_message_text(await _public_incident(context, e, user_id, "portfolio_stats"))
    elif query.data == "movers":
        try:
            await show_movers(query.from_user.id, context, query.message, prefer_edit=True)
        except Exception as e:
            user_id = query.from_user.id
            await query.edit_message_text(await _public_incident(context, e, user_id, "portfolio_movers"))
    elif query.data == "add_coin_prompt":
        # Handler for "Add Coin" button
        user_id = query.from_user.id
        help_text = (
            f"<b>{plain(t('portfolio.add_coin_help_title', user_id=user_id))}</b>\n\n"
            f"<code>{t('portfolio.add_coin_help_text', user_id=user_id)}</code>\n\n"
            f"<b>{plain(t('portfolio.add_coin_help_example', user_id=user_id))}</b>"
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
            f"<b>{plain(t('portfolio.sell_coin_help_title', user_id=user_id))}</b>\n\n"
            f"<code>{t('portfolio.sell_coin_help_text', user_id=user_id)}</code>\n\n"
            f"<b>{plain(t('portfolio.sell_coin_help_coins', user_id=user_id, coins=coin_list))}</b>\n\n"
            f"<b>{plain(t('portfolio.sell_coin_help_example', user_id=user_id))}</b>"
        )
        keyboard = [[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
        await query.edit_message_text(help_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "back_to_menu":
        await back_to_main_menu(query)

    elif query.data == "news":
        user_id = query.from_user.id
        await context.bot.send_chat_action(chat_id=query.message.chat.id, action=ChatAction.TYPING)
        items = await asyncio.to_thread(get_significant_news)
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
            await answer_once(query, t("admin.access_denied", user_id=user_id), show_alert=True)
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
            await answer_once(query, t("admin.access_denied", user_id=user_id), show_alert=True)
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
            await answer_once(query, t("admin.access_denied", user_id=user_id), show_alert=True)
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
            await answer_once(query, t("admin.access_denied", user_id=user_id), show_alert=True)
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















