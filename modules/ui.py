"""What the bot's screens share: market rows, long messages, the keyboards, the way back to the main menu."""

import datetime
import logging
import time
from zoneinfo import ZoneInfo

import config
from config import OWNER_ID as USER_ID
from core.api.coingecko_client import get_markets_data
from core.formatting import as_html
from core.incidents import public_temporary_message, report_incident
from locales.translator import t
from modules.user_prefs import get_timezone
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest

logger = logging.getLogger(__name__)
_global_app_ref = None  # set by the first handler that runs
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


TELEGRAM_TEXT_LIMIT = 4096


def split_message(text: str, limit: int = 4000) -> list:
    """Split at blank lines (then lines) so no chunk exceeds Telegram's limit.

    HTML tags in this bot never span lines, so line boundaries are safe cuts.
    """
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for block in text.split("\n\n"):
        pieces = [block] if len(block) <= limit else block.split("\n")
        for piece in pieces:
            sep = "\n\n" if piece is block else "\n"
            candidate = f"{current}{sep}{piece}" if current else piece
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece[:limit]
    if current:
        chunks.append(current)
    return chunks


async def respond_message(
    message,
    text: str,
    reply_markup=None,
    parse_mode=ParseMode.HTML,
    disable_web_page_preview: bool = True,
    prefer_edit: bool = False
):
    """Edit if possible, otherwise send a new message.

    Texts over the Telegram limit (large portfolios) go out in several
    messages; the keyboard is attached to the last one.
    """
    chunks = split_message(text)
    first, rest = chunks[0], chunks[1:]
    first_markup = None if rest else reply_markup
    sent = False
    if prefer_edit:
        try:
            await message.edit_text(
                first,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
                reply_markup=first_markup,
            )
            sent = True
        except BadRequest as e:
            s = str(e)
            if "Message is not modified" in s or "Message text is empty" in s:
                sent = True
            elif "Message can't be edited" not in s:
                raise
    if not sent:
        await message.reply_text(
            first,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
            reply_markup=first_markup,
        )
    for index, chunk in enumerate(rest):
        await message.reply_text(
            chunk,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
            reply_markup=reply_markup if index == len(rest) - 1 else None,
        )


def fmt_asof_for_user(user_id: int, ts: float | None) -> str:
    if not ts:
        return "n/a"
    tzname = get_timezone(user_id) or DEFAULT_TIMEZONE
    try:
        local = ZoneInfo(tzname)
    except Exception:
        local = ZoneInfo(DEFAULT_TIMEZONE)
    dt_utc = datetime.datetime.fromtimestamp(ts, ZoneInfo("UTC"))
    dt_loc = dt_utc.astimezone(local)
    return f"{dt_loc.strftime('%Y-%m-%d %H:%M:%S')} ({tzname}) • {dt_utc.strftime('%H:%M:%S')} UTC"


def set_global_app_ref(app):
    """Устанавливает глобальную ссылку на приложение"""
    global _global_app_ref
    _global_app_ref = app


def get_global_app_ref():
    return _global_app_ref


def create_news_keyboard(user_id: int = None):
    """Создать клавиатуру для новостей"""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
    )


def create_action_buttons(user_id: int = None, plan_symbol: str = None):
    """Кнопки для быстрых действий с портфелем"""
    keyboard = [
        [InlineKeyboardButton(t("plan.button_for", user_id=user_id, symbol=plan_symbol),
                              callback_data=f"tp:{plan_symbol}")],
    ] if plan_symbol else []
    keyboard += [
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


def create_main_menu(user_id: int = None):
    """
    Создать главное меню с учетом прав доступа

    Args:
        user_id: Telegram user ID для фильтрации owner-only buttons
    """
    keyboard = [
        [InlineKeyboardButton(t("menu.risk", user_id=user_id), callback_data="risk")],
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
