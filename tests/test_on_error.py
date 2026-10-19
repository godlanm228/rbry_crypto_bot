"""Transient network errors are not owner incidents; real API errors are."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from telegram.error import BadRequest, NetworkError, TimedOut

import main


def _run(error, monkeypatch):
    report = AsyncMock()
    monkeypatch.setattr(main, "report_incident", report)
    asyncio.run(main.on_error(None, SimpleNamespace(error=error)))
    return report


def test_network_errors_are_not_incidents(monkeypatch):
    assert _run(NetworkError("getaddrinfo failed"), monkeypatch).await_count == 0
    assert _run(TimedOut(), monkeypatch).await_count == 0


def test_bad_request_is_still_an_incident(monkeypatch):
    assert _run(BadRequest("Message is too long"), monkeypatch).await_count == 1
