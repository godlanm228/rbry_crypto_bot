"""Callback queries must be answered exactly once (Telegram rejects a second answer)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from telegram.error import BadRequest

import modules.handlers as handlers
from modules.callback_utils import answer_once


def _query(data, qid):
    return SimpleNamespace(
        id=qid,
        data=data,
        from_user=SimpleNamespace(id=42, username="u", first_name="U", language_code="en"),
        message=SimpleNamespace(),
        answer=AsyncMock(),
    )


def test_answer_once_answers_a_single_time():
    query = _query("x", "q-once")
    assert asyncio.run(answer_once(query, "first")) is True
    assert asyncio.run(answer_once(query, "second")) is False
    query.answer.assert_awaited_once_with(text="first", show_alert=False)


def test_answer_once_swallows_telegram_rejection():
    query = _query("x", "q-rejected")
    query.answer.side_effect = BadRequest("Query is too old")
    assert asyncio.run(answer_once(query)) is False


def test_menu_branch_toast_is_the_only_answer(monkeypatch):
    monkeypatch.setattr(handlers, "_track_user_activity", lambda *a, **k: None)
    monkeypatch.setattr(handlers, "set_global_app_ref", lambda *a, **k: None)
    monkeypatch.setattr(handlers, "set_language", lambda *a, **k: None)
    show_settings = AsyncMock()
    monkeypatch.setattr(handlers, "show_settings", show_settings)

    query = _query("set_lang:en", "q-lang")
    update = SimpleNamespace(callback_query=query)
    context = SimpleNamespace(application=None, user_data={})

    asyncio.run(handlers.menu_callback(update, context))

    assert query.answer.await_count == 1
    assert query.answer.await_args.kwargs["text"]  # the "language saved" toast
    show_settings.assert_awaited_once()


def test_menu_branch_without_toast_still_gets_answered(monkeypatch):
    monkeypatch.setattr(handlers, "_track_user_activity", lambda *a, **k: None)
    monkeypatch.setattr(handlers, "set_global_app_ref", lambda *a, **k: None)
    show_settings = AsyncMock()
    monkeypatch.setattr(handlers, "show_settings", show_settings)

    query = _query("settings", "q-settings")
    update = SimpleNamespace(callback_query=query)
    context = SimpleNamespace(application=None, user_data={})

    asyncio.run(handlers.menu_callback(update, context))

    query.answer.assert_awaited_once_with(text=None, show_alert=False)
