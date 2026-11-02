"""
Rate Limiter - Global and per-user rate limiting

Prevents API abuse with:
- Global limit: 100 requests per minute (system protection)
- User limits: Tier-specific (handled by access_manager)

Usage:
    @rate_limit
    async def analyze_coin_full(update, context, coin_id):
        ...
"""

import time
import logging
from functools import wraps
from typing import Dict, Callable, Optional
from telegram import Update
from telegram.ext import ContextTypes
from locales.translator import t

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Global and per-user rate limiting to prevent API abuse
    """

    def __init__(self):
        self.user_requests: Dict[int, list] = {}  # user_id -> [timestamps]
        self.global_requests: list = []  # [timestamps]

        # Limits
        self.GLOBAL_LIMIT = 100  # requests per minute
        self.CLEANUP_INTERVAL = 3600  # Clean old data every hour
        self._last_cleanup = time.time()

    def check_rate_limit(self, user_id: int, tier: str) -> tuple[bool, str]:
        """
        Check if user can make a request

        Args:
            user_id: Telegram user ID
            tier: User's access tier

        Returns:
            (allowed, reason)
        """
        now = time.time()

        # Cleanup old data periodically
        if now - self._last_cleanup > self.CLEANUP_INTERVAL:
            self._cleanup_old_requests(now)

        # Check global limit (protect system)
        self.global_requests = [ts for ts in self.global_requests if now - ts < 60]
        if len(self.global_requests) >= self.GLOBAL_LIMIT:
            logger.warning(f"Global rate limit exceeded: {len(self.global_requests)} requests/min")
            return False, "⚠️ System overload. Try again in a minute."

        # User-specific limits already handled by access_manager
        # This is an additional safety layer

        return True, "OK"

    def record_request(self, user_id: int):
        """Record request for rate limiting"""
        now = time.time()

        if user_id not in self.user_requests:
            self.user_requests[user_id] = []

        self.user_requests[user_id].append(now)
        self.global_requests.append(now)

    def _cleanup_old_requests(self, now: float):
        """Cleanup old request data (keep last hour)"""
        # Cleanup user requests
        for user_id in list(self.user_requests.keys()):
            self.user_requests[user_id] = [
                ts for ts in self.user_requests[user_id] if now - ts < 3600
            ]
            # Remove users with no recent requests
            if not self.user_requests[user_id]:
                del self.user_requests[user_id]

        # Cleanup global requests
        self.global_requests = [ts for ts in self.global_requests if now - ts < 3600]

        self._last_cleanup = now
        logger.debug("Rate limiter cleanup completed")


# Global singleton
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def rate_limit(func: Callable):
    """
    Decorator for rate limiting handlers

    Usage:
        @rate_limit
        async def analyze_coin_full(update, context, coin_id):
            ...
    """
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        try:
            from core.access.access_manager import get_access_manager

            user_id = update.effective_user.id
            access_manager = get_access_manager()
            tier = access_manager.get_user_tier(user_id)

            # Check global rate limit
            limiter = get_rate_limiter()
            allowed, reason = limiter.check_rate_limit(user_id, tier.value)

            if not allowed:
                await update.message.reply_text(t("errors.system_busy", user_id=user_id))
                return

            # Record request
            limiter.record_request(user_id)

        except Exception as e:
            logger.error(f"Rate limit check error: {e!r}")
            # Continue execution on error (fail open)

        # Execute handler
        return await func(update, context, *args, **kwargs)

    return wrapper
