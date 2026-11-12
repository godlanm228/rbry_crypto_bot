"""Account buttons: the user's plan and limits, the premium plans, a purchase."""

from core.access.access_manager import get_access_manager
from core.formatting import as_html
from locales.translator import t
from modules.admin_commands import _build_tier_text
from modules.payments.payment_manager import get_payment_manager
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes


async def my_tier(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The user's plan and limits."""
    query = update.callback_query
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


async def premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The premium plans with their prices."""
    payment_manager = get_payment_manager()
    await payment_manager.show_premium_options(update, context)


async def buy_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """An invoice for the plan that was picked."""
    query = update.callback_query
    # Handle plan purchase
    plan_id = query.data.split(":", 1)[1]
    payment_manager = get_payment_manager()
    await payment_manager.create_invoice(update, context, plan_id)
