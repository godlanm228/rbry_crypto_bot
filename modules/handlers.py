"""The bot's commands, its menu buttons and the text typed into it: who handles what.

The screens themselves live next door: modules/screens.py (help, settings, portfolio, statistics, movers),
modules/menu_settings.py, menu_analysis.py, menu_portfolio.py, menu_account.py, menu_admin.py, and what they share
in modules/ui.py.
"""

import asyncio
import logging
import re

from config import OWNER_ID as USER_ID
from core.formatting import as_html, plain
from core.utils import COIN_ID_MAP, _track_user_activity
from locales.translator import t
from modules import menu_account, menu_admin, menu_analysis, menu_portfolio, menu_settings, paper_view
from modules.alerts import update_alert_action
from modules.callback_utils import answer_once
from modules.news import get_significant_news, render_news
from modules.screens import (
    show_analyze_prompt, show_help, show_movers, show_paper, show_portfolio, show_settings, show_stats,
)
from modules.ui import (
    DEFAULT_TIMEZONE, _public_incident, back_to_main_menu, create_main_menu, create_news_keyboard,
    set_global_app_ref,
)
from modules.user_prefs import SUPPORTED_TIMEZONES, get_profile, init_profile_language
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Устанавливаем глобальную ссылку
    set_global_app_ref(context.application)

    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "start")
    init_profile_language(user_id, update.effective_user.language_code)
    profile = get_profile(user_id)

    if not profile.get("tz"):
        # первый запуск - спрашиваем TZ
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
    # Only the command word is logged, never the rest of the message.
    _track_user_activity(update.effective_user, "unknown_command", {"command": (text or "").split(" ", 1)[0][:32]})
    await update.message.reply_text(
        t("errors.unknown_command", user_id=update.effective_user.id)
    )


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "portfolio")
    await show_portfolio(user_id, context, update.message, prefer_edit=False)


def _is_known_ticker(symbol: str, registry) -> bool:
    if symbol in COIN_ID_MAP:
        return True
    try:
        if registry is None:
            from services.coin_registry_service import get_coin_registry_service

            registry = get_coin_registry_service()
        resolved = registry.resolve(symbol) if registry else None
    except Exception:
        logger.exception("Coin registry lookup failed symbol=%s", symbol)
        return False
    return bool(resolved and resolved.get("coingecko_id"))


async def _offer_ticker_actions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A bare ticker typed outside any prompt gets a choice instead of silence.

    Other text is still ignored, so chatter never reaches the coin lookup twice.
    """
    message = update.message
    symbol = (message.text or "").strip().upper() if message else ""
    if not re.fullmatch(r"[A-Z0-9]{2,10}", symbol):
        return
    if not _is_known_ticker(symbol, context.application.bot_data.get("coin_registry")):
        return
    user_id = update.effective_user.id
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(t("menu.analysis", user_id=user_id), callback_data=f"analyze_coin:{symbol}"),
        InlineKeyboardButton(t("plan.button_for", user_id=user_id, symbol=symbol), callback_data=f"tp:{symbol}"),
    ]])
    text = f"🪙 <b>{plain(symbol)}</b>\n{plain(t('plan.ticker_choice', user_id=user_id))}"
    await message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


async def handle_coin_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Text typed into the bot: the reply a screen is waiting for (a position size, a plan's coin, an amount to
    add, a coin to analyse), or a bare ticker, which gets a choice of what to do with it."""
    if context.user_data.get("awaiting_risk_size"):
        from modules.risk_handlers import handle_risk_text

        await handle_risk_text(update, context)
        return

    if context.user_data.get("awaiting_plan_coin"):
        from modules.plan_handlers import handle_plan_text

        await handle_plan_text(update, context)
        return

    pending_add = context.user_data.get("awaiting_add")
    if pending_add is not None:
        await menu_portfolio._handle_add_reply(update, context, pending_add)
        return

    # Check if we're waiting for coin input FIRST - otherwise every random text
    # message in the chat lands in analytics and hits the disk
    if not context.user_data.get("awaiting_coin_input", False):
        await _offer_ticker_actions(update, context)
        return

    await menu_analysis.typed_coin(update, context)


async def _cb_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The portfolio (in the owner's test bot: the paper account)."""
    query = update.callback_query
    user_id = query.from_user.id
    await show_portfolio(user_id, context, query.message, prefer_edit=True)
    return


async def _cb_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Done, remind or ignore under an alert."""
    query = update.callback_query
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


async def _cb_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The portfolio's statistics."""
    query = update.callback_query
    try:
        await show_stats(query.from_user.id, context, query.message, prefer_edit=True)
    except Exception as e:
        user_id = query.from_user.id
        await query.edit_message_text(await _public_incident(context, e, user_id, "portfolio_stats"))


async def _cb_movers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The portfolio's movers."""
    query = update.callback_query
    try:
        await show_movers(query.from_user.id, context, query.message, prefer_edit=True)
    except Exception as e:
        user_id = query.from_user.id
        await query.edit_message_text(await _public_incident(context, e, user_id, "portfolio_movers"))


async def _cb_back_to_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Back to the main menu."""
    query = update.callback_query
    await back_to_main_menu(query)


async def _cb_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The news digest."""
    query = update.callback_query
    user_id = query.from_user.id
    await context.bot.send_chat_action(chat_id=query.message.chat.id, action=ChatAction.TYPING)
    items = await asyncio.to_thread(get_significant_news)
    await render_news(status_msg=query.message, items=items, reply_markup=create_news_keyboard(user_id), user_id=user_id, fallback_bot=context.bot, fallback_chat_id=query.message.chat.id)


async def _cb_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The help screen."""
    query = update.callback_query
    user_id = query.from_user.id
    await show_help(user_id, query.message, prefer_edit=True)


ROUTES = {
    "portfolio": _cb_portfolio, "stats": _cb_stats, "movers": _cb_movers, "back_to_menu": _cb_back_to_menu,
    "news": _cb_news, "help": _cb_help,
    "settings": menu_settings.settings, "settings_tz": menu_settings.settings_tz,
    "settings_lang": menu_settings.settings_lang, "settings_explanations": menu_settings.settings_explanations,
    "analyze": menu_analysis.analyze, "analyze_custom": menu_analysis.analyze_custom,
    "analyze_portfolio": menu_analysis.analyze_portfolio, "analyze_top10": menu_analysis.analyze_top10,
    "add_coin_prompt": menu_portfolio.add_coin_prompt, "add_other": menu_portfolio.add_pick,
    "sell_coin_prompt": menu_portfolio.sell_coin_prompt,
    "my_tier": menu_account.my_tier, "premium": menu_account.premium,
    "admin_panel": menu_admin.admin_panel, "admin_stats": menu_admin.admin_stats,
    "admin_grant_help": menu_admin.admin_grant_help, "admin_users_list": menu_admin.admin_users_list,
}
# buttons that carry a value after their name; no name above starts with one of these
PREFIX_ROUTES = (
    ("alert_", _cb_alert), ("set_lang:", menu_settings.set_lang), ("set_tz:", menu_settings.set_tz),
    ("analyze_coin:", menu_analysis.analyze_coin), ("add_pick:", menu_portfolio.add_pick),
    ("buy_plan:", menu_account.buy_plan),
)


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Branches answer with their own toast where they have one; whatever is
    # left unanswered gets the plain answer here, exactly once.
    try:
        await _menu_callback(update, context)
    finally:
        await answer_once(update.callback_query)


async def _menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Every menu button (the risk and trade plan buttons have handlers of their own, see main.py): the state a
    button leaves behind is cleared, then the button's screen runs (modules/menu_*.py, modules/screens.py)."""

    query = update.callback_query
    if query and query.data:
        if not (query.data.startswith("add_pick:") or query.data == "add_other"):
            context.user_data.pop("awaiting_add", None)
        if query.data != "analyze_custom":
            context.user_data.pop("awaiting_coin_input", None)
        context.user_data.pop("awaiting_risk_size", None)
        context.user_data.pop("awaiting_plan_coin", None)
    if query and query.from_user and query.data:
        action = query.data.split(":", 1)[0]
        _track_user_activity(query.from_user, action, {"data": query.data})

    # Устанавливаем глобальную ссылку
    set_global_app_ref(context.application)

    data = query.data
    if data.startswith("paper:") and paper_view.enabled(query.from_user.id, USER_ID):
        await show_paper(query.message, data.split(":", 1)[1], prefer_edit=True)
        return
    handler = ROUTES.get(data) or next((h for prefix, h in PREFIX_ROUTES if data.startswith(prefix)), None)
    if handler is None:
        await query.edit_message_text(t("errors.unknown_command", user_id=query.from_user.id))
        return
    await handler(update, context)
