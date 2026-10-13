"""User input validation and owner-incident throttling."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

import core.incidents as incidents
from core.utils import parse_positive_number


def test_parse_positive_number_accepts_decimal_comma():
    assert parse_positive_number("0,5") == 0.5
    assert parse_positive_number(" 60000 ") == 60000.0


def test_parse_positive_number_rejects_invalid_values():
    for raw in ("-1", "0", "nan", "inf", "-inf", "abc", ""):
        assert parse_positive_number(raw) is None, raw


def test_owner_gets_one_incident_per_window(monkeypatch):
    monkeypatch.setattr(incidents, "_last_notified", {})
    monkeypatch.setattr(incidents, "_suppressed", {})
    bot = SimpleNamespace(send_message=AsyncMock())
    context = SimpleNamespace(bot=bot)

    for _ in range(5):
        asyncio.run(incidents.report_incident(context, ValueError("bad"), subsystem="portfolio_add"))
    assert bot.send_message.await_count == 1

    # A different exception type in the same subsystem is still reported.
    asyncio.run(incidents.report_incident(context, KeyError("x"), subsystem="portfolio_add"))
    assert bot.send_message.await_count == 2

    # After the window the next notice carries the suppressed count.
    monkeypatch.setattr(incidents, "OWNER_NOTIFY_WINDOW_SECONDS", 0)
    asyncio.run(incidents.report_incident(context, ValueError("bad"), subsystem="portfolio_add"))
    assert "suppressed since last notice: 4" in bot.send_message.await_args.kwargs["text"]
