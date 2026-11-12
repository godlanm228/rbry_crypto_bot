"""The owner's panel: users by plan, statistics, the grant command's help, the list of users."""

import datetime
import logging

from config import OWNER_ID as USER_ID
from core.access.access_manager import get_access_manager
from core.formatting import as_html
from core.portfolio_db import get_portfolio_manager
from locales.translator import t
from modules.callback_utils import answer_once
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Users by plan and the panel's buttons."""
    query = update.callback_query
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
    owner_in_users = any(str(uid) == str(USER_ID) for uid in users)     # the ids are stored as text
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
        logger.error(f"Error getting portfolio users: {e!r}")

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


async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Active and expired plans."""
    query = update.callback_query
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


async def admin_grant_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """How to grant and revoke access."""
    query = update.callback_query
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


async def admin_users_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The first users with their plans."""
    query = update.callback_query
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
                    user_line += " (истек)"

            output += user_line + "\n"

        if len(users) > 15:
            output += "\n" + t("admin.users_list_more", user_id=user_id, count=len(users) - 15)

    keyboard = [[InlineKeyboardButton(t("common.back_to_menu", user_id=user_id), callback_data="admin_panel")]]

    await query.edit_message_text(
        output,
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
