import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

import main


def _update(user_id):
    message = SimpleNamespace(reply_text=AsyncMock())
    return SimpleNamespace(
        effective_user=SimpleNamespace(
            id=user_id,
            username=None,
            first_name="Test",
            language_code="en",
        ),
        message=message,
        effective_message=message,
    )


@pytest.mark.asyncio
async def test_health_is_not_exposed_to_regular_users():
    update = _update(999)
    context = SimpleNamespace(application=None)

    await main.health_command(update, context)

    update.message.reply_text.assert_awaited_once()
    assert "Health" not in update.message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_owner_health_reads_snapshot_contract():
    update = _update(main.config.OWNER_ID)
    market = SimpleNamespace(get_latest=lambda: SimpleNamespace(ts=1_700_000_000))
    history = SimpleNamespace(get_latest=lambda: SimpleNamespace(ts=1_700_000_100))
    context = SimpleNamespace(
        application=SimpleNamespace(
            bot_data={"market_snapshot": market, "history_snapshot": history}
        )
    )

    await main.health_command(update, context)

    text = update.message.reply_text.await_args.args[0]
    assert "Health" in text
    assert "n/a" not in text


@pytest.mark.asyncio
async def test_unhandled_error_reports_owner_and_gives_user_safe_response(monkeypatch):
    update = _update(999)
    context = SimpleNamespace(error=RuntimeError("secret backend detail"))
    report = AsyncMock(return_value="ABC123")
    monkeypatch.setattr(main, "report_incident", report)

    await main.on_error(update, context)

    report.assert_awaited_once()
    public = update.message.reply_text.await_args.args[0]
    assert "secret backend detail" not in public
    assert "RuntimeError" not in public
    assert "ABC123" not in public
