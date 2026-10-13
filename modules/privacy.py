"""Privacy notice, user data erasure, payment support and owner refunds.

Telegram requires bots that sell digital goods for Stars to answer
/paysupport and to handle refunds; GDPR requires a way to see what is
stored and to erase it.
"""

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

import config
from core.access.access_manager import get_access_manager
from core.analytics.user_analytics import get_user_analytics
from core.formatting import as_html
from core.portfolio_db import get_portfolio_manager
from locales.translator import t
from modules.callback_utils import answer_once
from modules.user_prefs import delete_profile

logger = logging.getLogger(__name__)


def delete_user_data(user_id: int) -> None:
    """Erase portfolio, preferences, analytics and alert history of a user.

    Kept on purpose: the payment ledger (accounting obligation) and the
    access grant itself (the user paid for it).
    """
    uid = int(user_id)
    conn = get_portfolio_manager()._get_conn()
    with conn:
        for table in ("sales", "entries", "portfolio", "users"):
            conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (uid,))

    analytics_conn = get_user_analytics().db.conn()
    with analytics_conn:
        analytics_conn.execute("DELETE FROM analytics_activity WHERE user_id = ?", (uid,))
        analytics_conn.execute("DELETE FROM analytics_users WHERE user_id = ?", (uid,))

    delete_profile(uid)

    from modules.alerts import _mutate_alert_state

    def drop_user_alerts(state):
        for alert_id in [aid for aid, rec in state.items() if rec.get("chat_id") == uid]:
            del state[alert_id]

    _mutate_alert_state(drop_user_alerts)
    logger.info("User data erased user_id=%s", uid)


async def privacy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text(
        as_html(t("privacy.notice", user_id=user_id)), parse_mode=ParseMode.HTML
    )


async def delete_me_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(t("privacy.delete_confirm_button", user_id=user_id), callback_data="delete_me:confirm")],
        [InlineKeyboardButton(t("privacy.delete_cancel_button", user_id=user_id), callback_data="delete_me:cancel")],
    ])
    await update.message.reply_text(
        as_html(t("privacy.delete_prompt", user_id=user_id)),
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


async def delete_me_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    try:
        if query.data == "delete_me:confirm":
            # Read the language before the profile is gone.
            done_text = t("privacy.deleted", user_id=user_id)
            delete_user_data(user_id)
            await query.edit_message_text(as_html(done_text), parse_mode=ParseMode.HTML)
        else:
            await query.edit_message_text(
                as_html(t("privacy.delete_cancelled", user_id=user_id)), parse_mode=ParseMode.HTML
            )
    finally:
        await answer_once(query)


async def paysupport_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    keyboard = None
    if config.SUPPORT_URL:
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(t("premium.button_contact", user_id=user_id), url=config.SUPPORT_URL)
        ]])
    await update.message.reply_text(
        as_html(t("privacy.paysupport", user_id=user_id)),
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


async def refund_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner only: /refund USER_ID TELEGRAM_CHARGE_ID — refund Stars and revoke access."""
    if update.effective_user.id != config.OWNER_ID:
        return
    if len(context.args) != 2 or not context.args[0].isdigit():
        await update.message.reply_text("Usage: /refund USER_ID TELEGRAM_CHARGE_ID")
        return
    target, charge_id = int(context.args[0]), context.args[1]
    try:
        await context.bot.refund_star_payment(user_id=target, telegram_payment_charge_id=charge_id)
    except Exception as error:
        await update.message.reply_text(f"Refund failed: {error}")
        return
    result = get_access_manager().revoke_access(config.OWNER_ID, target)
    await update.message.reply_text(f"Refunded {charge_id} to {target}. {result}")
