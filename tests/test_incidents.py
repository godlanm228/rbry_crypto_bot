import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

from core.incidents import public_temporary_message, report_incident


@pytest.mark.asyncio
async def test_public_message_never_contains_exception_details():
    bot = SimpleNamespace(send_message=AsyncMock())
    context = SimpleNamespace(bot=bot)
    error = RuntimeError("postgres password=super-secret")

    incident_id = await report_incident(context, error, subsystem="portfolio", user_id=77)
    public = public_temporary_message(77)

    assert incident_id not in public
    assert "super-secret" not in public
    assert "RuntimeError" not in public
    owner_text = bot.send_message.await_args.kwargs["text"]
    assert incident_id in owner_text
    assert "portfolio" in owner_text


@pytest.mark.asyncio
async def test_incident_notification_redacts_tokens():
    bot = SimpleNamespace(send_message=AsyncMock())
    context = SimpleNamespace(bot=bot)
    token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ_123456"

    await report_incident(context, RuntimeError(token), subsystem="telegram")

    owner_text = bot.send_message.await_args.kwargs["text"]
    assert token not in owner_text
    assert "redacted-telegram-token" in owner_text
