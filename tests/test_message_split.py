"""Long replies are split under Telegram's 4096-character limit."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from modules.ui import respond_message, split_message


def _portfolio_text(coins):
    rows = [f"<b>COIN{i}</b>\nAmount: 1.000000\nEntry: $1.00 → Now: $1.10\n📈 P&L: +$0.10 (+10.0%)" for i in range(coins)]
    return "<b>PORTFOLIO</b>\n\n" + "\n\n".join(rows)


def test_split_keeps_every_chunk_under_the_limit_and_loses_nothing():
    text = _portfolio_text(120)
    chunks = split_message(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 4000 for chunk in chunks)
    assert "".join(chunk.replace("\n", "") for chunk in chunks) == text.replace("\n", "")


def test_short_text_is_one_chunk():
    assert split_message("hello") == ["hello"]


def test_keyboard_goes_on_the_last_message():
    message = MagicMock()
    message.edit_text = AsyncMock()
    message.reply_text = AsyncMock()
    keyboard = object()
    asyncio.run(respond_message(message, _portfolio_text(120), reply_markup=keyboard, prefer_edit=True))
    assert message.edit_text.await_args.kwargs["reply_markup"] is None
    assert message.reply_text.await_args_list[-1].kwargs["reply_markup"] is keyboard
