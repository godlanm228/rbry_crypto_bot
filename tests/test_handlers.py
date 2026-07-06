import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_start_command_first_run_asks_timezone(tmp_path, monkeypatch):
    """First /start (no timezone in profile) must ask for a timezone."""
    # Isolate user prefs in a temp file so the test never touches real data/
    import modules.user_prefs as prefs
    monkeypatch.setattr(prefs, "PREFS_FILE", tmp_path / "prefs.json")
    monkeypatch.setattr(prefs, "_cache", None)
    monkeypatch.setattr(prefs, "_cache_mtime", None)

    from modules.handlers import start_command

    update = MagicMock()
    update.effective_user.id = 424242
    update.effective_user.username = "test_user"
    update.effective_user.first_name = "Test"
    update.effective_user.language_code = "ru"
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    context = MagicMock()

    await start_command(update, context)

    update.message.reply_text.assert_called_once()
    # Reply must contain the timezone keyboard
    kwargs = update.message.reply_text.call_args.kwargs
    assert kwargs.get("reply_markup") is not None
