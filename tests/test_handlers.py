import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
pytestmark = pytest.mark.asyncio

from modules.handlers import start_command
from unittest.mock import AsyncMock, MagicMock
import pytest

@pytest.mark.asyncio
async def test_start_command():
    update = MagicMock()
    update.message = MagicMock()
    context = MagicMock()
    await start_command(update, context)
    update.message.reply_text.assert_called_once()
