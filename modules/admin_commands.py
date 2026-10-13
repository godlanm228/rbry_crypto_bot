"""
Admin Commands - Owner-only user management commands

Commands:
- /grant <user_id> <tier> <days> - Grant access to user
- /revoke <user_id> - Revoke user access
- /user_stats - Show active users statistics
- /check_user <user_id> - Check user's current tier and limits

All commands require OWNER tier.
"""

import logging
from core.formatting import as_html
from telegram import Update
from telegram.ext import ContextTypes

from core.access.access_manager import get_access_manager, AccessTier
from core.analytics.user_analytics import get_user_analytics
from locales.translator import t

logger = logging.getLogger(__name__)

def _build_tier_text(user_id: int) -> str:
    access_manager = get_access_manager()
    user_info = access_manager.get_user_info(user_id)

    tier = user_info['tier']
    limits = user_info['limits']

    output = t("tier.title", user_id=user_id, tier=tier.upper()) + "\n\n"
    output += t("tier.limits_title", user_id=user_id) + "\n"

    max_coins = limits.get('max_coins')
    if max_coins:
        output += t("tier.max_coins", user_id=user_id, count=max_coins) + "\n"
    else:
        output += t("tier.max_coins_unlimited", user_id=user_id) + "\n"

    allowed_coins = limits.get('allowed_coins')
    if allowed_coins:
        output += t("tier.allowed_coins", user_id=user_id, coins=', '.join(allowed_coins)) + "\n"
    else:
        output += t("tier.allowed_coins_all", user_id=user_id) + "\n"

    cooldown = limits.get('request_cooldown', 0)
    if cooldown > 0:
        cooldown_min = cooldown // 60
        cooldown_hours = cooldown_min // 60
        if cooldown_hours > 0:
            output += t("tier.request_cooldown_hours", user_id=user_id, hours=cooldown_hours) + "\n"
        else:
            output += t("tier.request_cooldown_minutes", user_id=user_id, minutes=cooldown_min) + "\n"
    else:
        output += t("tier.request_cooldown_none", user_id=user_id) + "\n"

    ai_status = t("common.enabled", user_id=user_id) if limits.get('ai_synthesis') else t("common.disabled", user_id=user_id)
    patterns_status = t("common.enabled", user_id=user_id) if limits.get('patterns') else t("common.disabled", user_id=user_id)
    output += t("tier.ai_synthesis", user_id=user_id, status=ai_status) + "\n"
    output += t("tier.patterns", user_id=user_id, status=patterns_status) + "\n"

    expires_date = user_info.get('expires_date')
    if tier != 'free' and expires_date:
        from datetime import datetime
        expires_dt = datetime.fromtimestamp(expires_date)
        days_left = (expires_dt - datetime.now()).days

        if days_left > 0:
            output += "\n" + t("tier.expires_in", user_id=user_id, days=days_left, date=expires_dt.strftime('%Y-%m-%d')) + "\n"
        else:
            output += "\n" + t("tier.expired", user_id=user_id) + "\n"

    if tier == 'free':
        output += "\n" + t("tier.want_unlimited", user_id=user_id)

    return output


async def cmd_grant(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /grant <user_id> <tier> <days>

    Grant access to a user.

    Examples:
        /grant 123456789 premium 30
        /grant 987654321 beta 0
    """
    admin_id = update.effective_user.id
    try:
        analytics = get_user_analytics()
        analytics.register_user(
            user_id=update.effective_user.id,
            username=update.effective_user.username,
            first_name=update.effective_user.first_name,
            language_code=update.effective_user.language_code,
        )
        analytics.log_activity(admin_id, "grant", {"args": context.args})
    except Exception as e:
        logger.debug(f"User analytics error: {e}")
    access_manager = get_access_manager()

    try:
        # Parse arguments
        args = context.args

        if len(args) < 3:
            await update.message.reply_text(
                "❌ Usage: /grant <user_id> <tier> <days>\n\n"
                "Tiers: free, premium, beta, owner\n"
                "Days: Number of days (0 = permanent)\n\n"
                "Example: /grant 123456789 premium 30"
            )
            return

        target_user_id = int(args[0])
        tier = args[1].lower()
        days = int(args[2])

        # Validate
        if days < 0:
            await update.message.reply_text("❌ Days must be >= 0")
            return

        # Grant access
        result = access_manager.grant_access(admin_id, target_user_id, tier, days)
        await update.message.reply_text(result)

    except ValueError as e:
        await update.message.reply_text(f"❌ Invalid arguments: {e}")
    except Exception as e:
        logger.error(f"Error in /grant command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_revoke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /revoke <user_id>

    Revoke user's special access (reverts to FREE tier).

    Example:
        /revoke 123456789
    """
    admin_id = update.effective_user.id
    try:
        analytics = get_user_analytics()
        analytics.register_user(
            user_id=update.effective_user.id,
            username=update.effective_user.username,
            first_name=update.effective_user.first_name,
            language_code=update.effective_user.language_code,
        )
        analytics.log_activity(admin_id, "revoke", {"args": context.args})
    except Exception as e:
        logger.debug(f"User analytics error: {e}")
    access_manager = get_access_manager()

    try:
        if not context.args:
            await update.message.reply_text(
                "❌ Usage: /revoke <user_id>\n\n"
                "Example: /revoke 123456789"
            )
            return

        target_user_id = int(context.args[0])

        # Revoke access
        result = access_manager.revoke_access(admin_id, target_user_id)
        await update.message.reply_text(result)

    except ValueError as e:
        await update.message.reply_text(f"❌ Invalid user_id: {e}")
    except Exception as e:
        logger.error(f"Error in /revoke command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_user_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /user_stats

    Show statistics about active users and tiers.
    Owner-only command.
    """
    admin_id = update.effective_user.id
    try:
        analytics = get_user_analytics()
        analytics.register_user(
            user_id=update.effective_user.id,
            username=update.effective_user.username,
            first_name=update.effective_user.first_name,
            language_code=update.effective_user.language_code,
        )
        analytics.log_activity(admin_id, "user_stats")
    except Exception as e:
        logger.debug(f"User analytics error: {e}")
    access_manager = get_access_manager()

    try:
        result = access_manager.get_user_stats(admin_id)
        await update.message.reply_text(as_html(result), parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error in /user_stats command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_check_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /check_user <user_id>

    Check a user's current tier and access limits.
    Owner-only command.

    Example:
        /check_user 123456789
    """
    admin_id = update.effective_user.id
    try:
        analytics = get_user_analytics()
        analytics.register_user(
            user_id=update.effective_user.id,
            username=update.effective_user.username,
            first_name=update.effective_user.first_name,
            language_code=update.effective_user.language_code,
        )
        analytics.log_activity(admin_id, "check_user", {"args": context.args})
    except Exception as e:
        logger.debug(f"User analytics error: {e}")
    access_manager = get_access_manager()

    # Verify admin is OWNER
    if access_manager.get_user_tier(admin_id) != AccessTier.OWNER:
        await update.message.reply_text("❌ This command is only available to OWNER tier")
        return

    try:
        # Check if user wants to see their own info
        if not context.args:
            target_user_id = admin_id
        else:
            target_user_id = int(context.args[0])

        # Get user info
        user_info = access_manager.get_user_info(target_user_id)
        tier = user_info['tier']
        limits = user_info['limits']

        output = f"**User {target_user_id}**\n\n"
        output += f"**Tier:** {tier.upper()}\n\n"

        output += "**Limits:**\n"
        output += f"  Max coins: {limits.get('max_coins', 'unlimited')}\n"

        allowed_coins = limits.get('allowed_coins')
        if allowed_coins:
            output += f"  Allowed coins: {', '.join(allowed_coins)}\n"
        else:
            output += f"  Allowed coins: all\n"

        cooldown = limits.get('request_cooldown', 0)
        if cooldown > 0:
            cooldown_min = cooldown // 60
            output += f"  Request cooldown: {cooldown_min}m\n"
        else:
            output += f"  Request cooldown: none\n"

        output += f"  AI synthesis: {limits.get('ai_synthesis', False)}\n"
        output += f"  Patterns: {limits.get('patterns', False)}\n"

        # Expiration info
        granted_date = user_info.get('granted_date')
        expires_date = user_info.get('expires_date')

        if granted_date:
            from datetime import datetime
            granted_dt = datetime.fromtimestamp(granted_date)
            output += f"\n**Granted:** {granted_dt.strftime('%Y-%m-%d %H:%M')}\n"

            if expires_date:
                expires_dt = datetime.fromtimestamp(expires_date)
                output += f"**Expires:** {expires_dt.strftime('%Y-%m-%d %H:%M')}\n"
            else:
                output += f"**Expires:** Never (permanent)\n"

        await update.message.reply_text(as_html(output), parse_mode='HTML')

    except ValueError as e:
        await update.message.reply_text(f"❌ Invalid user_id: {e}")
    except Exception as e:
        logger.error(f"Error in /check_user command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_my_tier(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /my_tier

    Show your current tier and limits.
    Available to all users.
    """
    user_id = update.effective_user.id
    try:
        analytics = get_user_analytics()
        analytics.register_user(
            user_id=update.effective_user.id,
            username=update.effective_user.username,
            first_name=update.effective_user.first_name,
            language_code=update.effective_user.language_code,
        )
        analytics.log_activity(user_id, "my_tier")
    except Exception as e:
        logger.debug(f"User analytics error: {e}")

    try:
        output = _build_tier_text(user_id)
        await update.message.reply_text(as_html(output), parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error in /my_tier command: {e}", exc_info=True)
        await update.message.reply_text(t("common.error", user_id=user_id))
