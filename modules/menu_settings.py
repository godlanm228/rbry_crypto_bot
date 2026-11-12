"""Settings buttons: the time zone, the language, the explanations switch."""

from core.formatting import as_html
from locales.translator import t
from modules.callback_utils import answer_once
from modules.screens import show_settings
from modules.ui import back_to_main_menu
from modules.user_prefs import (
    SUPPORTED_TIMEZONES, get_profile, get_show_explanations, set_language, set_show_explanations, set_timezone,
)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The settings screen."""
    query = update.callback_query
    user_id = query.from_user.id
    await show_settings(user_id, query.message, prefer_edit=True)
    return


async def settings_tz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The list of time zones."""
    query = update.callback_query
    user_id = query.from_user.id
    tz_kb = [[InlineKeyboardButton(tz, callback_data=f"set_tz:{tz}")] for tz in SUPPORTED_TIMEZONES[:6]]
    tz_kb.append([InlineKeyboardButton(t("common.back_to_settings", user_id=user_id), callback_data="settings")])
    await query.edit_message_text(
        as_html(t("settings.choose_timezone", user_id=user_id)),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(tz_kb)
    )


async def settings_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The list of languages."""
    query = update.callback_query
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


async def settings_explanations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch the explanations on or off."""
    query = update.callback_query
    user_id = query.from_user.id
    current = get_show_explanations(user_id)
    set_show_explanations(user_id, not current)
    await answer_once(query, t("settings.explanations_saved", user_id=user_id))
    await show_settings(user_id, query.message, prefer_edit=True)


async def set_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """A language was picked."""
    query = update.callback_query
    user_id = query.from_user.id
    lang = query.data.split("set_lang:", 1)[1]
    set_language(user_id, lang)
    await answer_once(query, t("settings.language_saved", user_id=user_id))
    await show_settings(user_id, query.message, prefer_edit=True)


async def set_tz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """A time zone was picked (the first run goes on to the main menu)."""
    query = update.callback_query
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
