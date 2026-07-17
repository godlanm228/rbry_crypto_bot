import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fasteners
import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

import modules.alerts as alerts


@pytest.fixture
def isolated_alert_state(tmp_path, monkeypatch):
    state_path = tmp_path / "alerts.json"
    monkeypatch.setattr(alerts, "ALERT_STATE_FILE", str(state_path))
    monkeypatch.setattr(
        alerts,
        "LOCK",
        fasteners.InterProcessLock(str(tmp_path / "alerts.lock")),
    )
    return state_path


@pytest.mark.asyncio
async def test_broadcast_creates_one_owned_alert_per_recipient(isolated_alert_state):
    system = object.__new__(alerts.SmartAlertsSystem)
    system.bot = SimpleNamespace(send_message=AsyncMock())
    system.alerts_sent_today = 0
    system.subscribers = {"premium": [111, 222]}
    system._load_subscribers = lambda: {"premium": [111, 222]}

    sent = await system._send_alert("**BTC** moved", alerts.AlertType.MARKET_PUMP)

    assert sent == 2
    state = alerts._load_alert_state()
    assert len(state) == 2
    assert {record["chat_id"] for record in state.values()} == {111, 222}
    callback_ids = {
        call.kwargs["reply_markup"].inline_keyboard[0][0].callback_data.split(":", 1)[1]
        for call in system.bot.send_message.await_args_list
    }
    assert callback_ids == set(state)


@pytest.mark.asyncio
async def test_broadcast_renders_message_separately_for_each_user(isolated_alert_state):
    system = object.__new__(alerts.SmartAlertsSystem)
    system.bot = SimpleNamespace(send_message=AsyncMock())
    system.alerts_sent_today = 0
    system.subscribers = {"premium": [111, 222]}
    system._load_subscribers = lambda: {"premium": [111, 222]}

    await system._send_alert(
        lambda user_id: f"localized-for-{user_id}",
        alerts.AlertType.MARKET_PUMP,
    )

    texts = {
        call.kwargs["chat_id"]: call.kwargs["text"]
        for call in system.bot.send_message.await_args_list
    }
    assert "localized-for-111" in texts[111]
    assert "localized-for-222" in texts[222]
    state = alerts._load_alert_state()
    assert {record["message"] for record in state.values()} == {
        "localized-for-111",
        "localized-for-222",
    }


def test_alert_action_cannot_modify_another_users_record(isolated_alert_state):
    alerts._save_alert_state(
        {"abc": {"chat_id": 111, "status": "new", "ts": 100.0}}
    )

    assert alerts.update_alert_action("abc", 222, "alert_done", now=200.0) is False
    assert alerts._load_alert_state()["abc"]["status"] == "new"
    assert alerts.update_alert_action("abc", 111, "alert_remind", now=200.0) is True
    record = alerts._load_alert_state()["abc"]
    assert record["status"] == "remind"
    assert record["remind_at"] == 3800.0


@pytest.mark.asyncio
async def test_due_reminder_is_delivered_and_rearmed(isolated_alert_state):
    alerts._save_alert_state(
        {
            "abc": {
                "chat_id": 111,
                "status": "remind",
                "remind_at": 1.0,
                "message": "BTC moved",
                "ts": 1.0,
            }
        }
    )
    bot = SimpleNamespace(send_message=AsyncMock())
    context = SimpleNamespace(bot=bot)

    await alerts.run_alert_reminders(context)

    bot.send_message.assert_awaited_once()
    record = alerts._load_alert_state()["abc"]
    assert record["status"] == "new"
    assert record["reminder_count"] == 1
    assert "remind_at" not in record
