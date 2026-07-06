import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")


@pytest.mark.asyncio
async def test_portfolio_alerts_are_sent_to_portfolio_owner_only(monkeypatch):
    import modules.alerts as alerts_mod

    system = object.__new__(alerts_mod.SmartAlertsSystem)
    system.alerts_sent_today = 0
    system.last_alerts = {}

    sent_targets = []

    async def fake_send_alert(message, alert_type, priority="normal", coin=None, target_user_id=None):
        sent_targets.append(target_user_id)

    monkeypatch.setattr(system, "_send_alert", fake_send_alert)
    monkeypatch.setattr(system, "_should_send_alert", lambda coin, alert_type: True)

    class FakePortfolioManager:
        def get_all_users(self):
            return [111, 222]

        def load_portfolio(self, user_id):
            return {"BTC": {"amount": 1.0, "buy_price": 100.0}}

    class FakeAccessManager:
        def get_user_tier(self, user_id):
            return alerts_mod.AccessTier.PREMIUM

    class FakeSnapshotService:
        def get_latest(self):
            return SimpleNamespace(markets={"bitcoin": {"current_price": 130.0}})

    monkeypatch.setattr(alerts_mod, "get_portfolio_manager", lambda: FakePortfolioManager())
    monkeypatch.setattr(alerts_mod, "get_access_manager", lambda: FakeAccessManager())

    context = SimpleNamespace(
        application=SimpleNamespace(bot_data={"market_snapshot": FakeSnapshotService()})
    )

    await system.check_portfolio_alerts(context)

    assert sent_targets == [111, 222]
