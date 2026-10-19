"""Add-coin menu: pick a coin, then send the amount (price optional)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import modules.handlers as handlers
import core.utils as utils


def _setup(monkeypatch, price=82000.0):
    monkeypatch.setattr(handlers, "_track_user_activity", lambda *a, **k: None)
    monkeypatch.setattr(handlers, "set_global_app_ref", lambda *a, **k: None)

    async def rows(app, ids):
        return {ids[0]: {"current_price": price}} if price else {}, None

    monkeypatch.setattr(handlers, "_get_market_rows", rows)
    added = []

    async def fake_add(update, context, coin, raw_amount, raw_price, **kwargs):
        added.append((coin, raw_amount, raw_price))
        return True

    monkeypatch.setattr(utils, "add_portfolio_entry", fake_add)
    return added


def _press(data, context, qid):
    query = SimpleNamespace(id=qid, data=data, from_user=SimpleNamespace(id=7),
                            message=SimpleNamespace(), answer=AsyncMock(), edit_message_text=AsyncMock())
    asyncio.run(handlers.menu_callback(SimpleNamespace(callback_query=query), context))
    return query


def _say(text, context):
    message = SimpleNamespace(text=text, reply_text=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=7), message=message)
    asyncio.run(handlers.handle_coin_input(update, context))
    return message


def test_pick_coin_then_amount_uses_current_price(monkeypatch):
    added = _setup(monkeypatch)
    context = SimpleNamespace(application=None, user_data={})
    _press("add_pick:BTC", context, "q-pick")
    assert context.user_data["awaiting_add"] == {"coin": "BTC"}
    _say("0.5", context)
    assert added == [("BTC", "0.5", "82000.0")]
    assert "awaiting_add" not in context.user_data


def test_other_coin_with_ticker_amount_and_price(monkeypatch):
    added = _setup(monkeypatch)
    context = SimpleNamespace(application=None, user_data={})
    _press("add_other", context, "q-other")
    _say("ton 10 1,4", context)
    assert added == [("TON", "10", "1,4")]


def test_bad_reply_keeps_waiting_and_other_buttons_cancel(monkeypatch):
    added = _setup(monkeypatch)
    context = SimpleNamespace(application=None, user_data={})
    _press("add_pick:ETH", context, "q-eth")
    reply = _say("one two three", context)
    assert added == [] and context.user_data.get("awaiting_add")
    reply.reply_text.assert_awaited()
    monkeypatch.setattr(handlers, "show_settings", AsyncMock())
    _press("settings", context, "q-settings")
    assert "awaiting_add" not in context.user_data
