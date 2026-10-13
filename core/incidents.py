"""Internal incident reporting and a stable public error boundary."""

from __future__ import annotations

import logging
import secrets
import time
import traceback
from typing import Dict, Optional, Tuple

import config
from core.logging_config import redact_text
from locales.translator import t


logger = logging.getLogger(__name__)

# One owner notification per (subsystem, exception type) per window; the log
# keeps every incident. Without this any user could flood the owner's chat.
OWNER_NOTIFY_WINDOW_SECONDS = 600
_last_notified: Dict[Tuple[str, str], float] = {}
_suppressed: Dict[Tuple[str, str], int] = {}


def _should_notify_owner(subsystem: str, error: BaseException) -> Tuple[bool, int]:
    key = (subsystem, type(error).__name__)
    now = time.monotonic()
    last = _last_notified.get(key)
    if last is not None and now - last < OWNER_NOTIFY_WINDOW_SECONDS:
        _suppressed[key] = _suppressed.get(key, 0) + 1
        return False, 0
    _last_notified[key] = now
    return True, _suppressed.pop(key, 0)


def public_temporary_message(user_id: Optional[int] = None) -> str:
    """A calm public message that never exposes implementation details."""
    return t("errors.temporary_unavailable", user_id=user_id)


async def report_incident(
    context,
    error: BaseException,
    *,
    subsystem: str,
    user_id: Optional[int] = None,
) -> str:
    """Log full diagnostics and notify the owner without leaking them publicly."""
    incident_id = secrets.token_hex(4).upper()
    logger.error(
        "INCIDENT id=%s subsystem=%s user_id=%s",
        incident_id,
        subsystem,
        user_id,
        exc_info=(type(error), error, error.__traceback__),
    )

    owner_id = getattr(config, "OWNER_ID", None)
    bot = getattr(context, "bot", None)
    if bot is None:
        application = getattr(context, "application", None)
        bot = getattr(application, "bot", None)

    notify, suppressed = _should_notify_owner(subsystem, error)
    if owner_id and bot and notify:
        trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        detail = redact_text(trace)[-2800:]
        repeats = f"Similar incidents suppressed since last notice: {suppressed}\n" if suppressed else ""
        message = (
            f"⚠️ Incident {incident_id}\n"
            f"Subsystem: {subsystem}\n"
            f"User: {user_id if user_id is not None else 'n/a'}\n"
            f"{repeats}"
            f"{detail}"
        )
        try:
            await bot.send_message(chat_id=owner_id, text=message)
        except Exception:
            logger.exception("Failed to deliver incident %s to owner", incident_id)

    return incident_id
