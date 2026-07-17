import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

from services.history_snapshot import HistorySnapshotService
from services.market_snapshot import MarketSnapshotService
from services import healthchecks


@pytest.mark.asyncio
async def test_market_snapshot_tracks_failure_and_recovery(monkeypatch):
    service = MarketSnapshotService(["bitcoin"], ttl=60)

    async def fail():
        raise RuntimeError("upstream timeout")

    monkeypatch.setattr(service, "_fetch_markets", fail)
    assert await service.refresh_once() is False
    assert service.consecutive_failures == 1
    assert "upstream timeout" in service.last_error

    async def succeed():
        return {"bitcoin": {"current_price": 100.0}}

    monkeypatch.setattr(service, "_fetch_markets", succeed)
    assert await service.refresh_once() is True
    assert service.consecutive_failures == 0
    assert service.last_error is None
    assert service.get_latest().markets["bitcoin"]["current_price"] == 100.0


@pytest.mark.asyncio
async def test_partial_snapshot_is_diagnostic_only(monkeypatch):
    service = MarketSnapshotService(["bitcoin", "ethereum"], ttl=60)

    async def partial():
        return {"bitcoin": {"current_price": 100.0}}

    monkeypatch.setattr(service, "_fetch_markets", partial)

    assert await service.refresh_once() is False
    assert service.get_latest() is None
    assert service.last_partial_snapshot.markets["bitcoin"]["current_price"] == 100.0
    assert service.coverage_ratio == 0.5
    assert "incomplete market coverage" in service.last_error


@pytest.mark.asyncio
async def test_partial_refresh_does_not_replace_last_healthy_snapshot(monkeypatch):
    service = MarketSnapshotService(["bitcoin", "ethereum"], ttl=60)

    async def complete():
        return {
            "bitcoin": {"current_price": 100.0},
            "ethereum": {"current_price": 200.0},
        }

    monkeypatch.setattr(service, "_fetch_markets", complete)
    assert await service.refresh_once() is True
    healthy = service.get_latest()

    async def partial():
        return {"bitcoin": {"current_price": 110.0}}

    monkeypatch.setattr(service, "_fetch_markets", partial)
    assert await service.refresh_once() is False
    assert service.get_latest() is healthy
    assert service.get_latest().markets["bitcoin"]["current_price"] == 100.0


@pytest.mark.asyncio
async def test_history_snapshot_tracks_failure_and_recovery(monkeypatch):
    service = HistorySnapshotService(["bitcoin"])

    async def fail():
        raise RuntimeError("history unavailable")

    monkeypatch.setattr(service, "_fetch_histories", fail)
    assert await service.refresh_once() is False
    assert service.consecutive_failures == 1

    async def succeed():
        return {"bitcoin": object()}

    monkeypatch.setattr(service, "_fetch_histories", succeed)
    assert await service.refresh_once() is True
    assert "bitcoin" in service.get_latest().by_coin
    assert service.consecutive_failures == 0


@pytest.mark.asyncio
async def test_history_ready_event_fires_only_after_validated_snapshot(monkeypatch):
    service = HistorySnapshotService(["bitcoin"])

    async def succeed():
        return {"bitcoin": object()}

    monkeypatch.setattr(service, "_fetch_histories", succeed)
    waiter = asyncio.create_task(service.wait_until_ready(timeout=1))

    assert await service.refresh_once() is True
    assert await waiter is True


@pytest.mark.asyncio
async def test_snapshot_watchdog_notifies_on_degradation_and_recovery():
    healthchecks._notification_state.clear()
    market = SimpleNamespace(
        get_latest=lambda: SimpleNamespace(ts=1.0),
        consecutive_failures=3,
        last_error="timeout",
    )
    history = SimpleNamespace(
        get_latest=lambda: SimpleNamespace(ts=time.time()),
        consecutive_failures=0,
        last_error=None,
    )
    bot = SimpleNamespace(send_message=AsyncMock())
    context = SimpleNamespace(
        bot=bot,
        application=SimpleNamespace(
            bot=bot,
            bot_data={"market_snapshot": market, "history_snapshot": history},
        ),
    )

    await healthchecks.run_snapshot_watchdog(context)
    await healthchecks.run_snapshot_watchdog(context)
    assert bot.send_message.await_count == 1

    market.get_latest = lambda: SimpleNamespace(ts=time.time())
    market.consecutive_failures = 0
    market.last_error = None
    await healthchecks.run_snapshot_watchdog(context)

    assert bot.send_message.await_count == 2
    assert "recovered" in bot.send_message.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_scheduled_healthcheck_does_not_spend_coingecko_budget(monkeypatch):
    checked_urls = []

    async def http_check(name, url, timeout=5.0):
        checked_urls.append(url)
        return healthchecks.HealthCheckResult(name, True, "ok")

    async def ok_check():
        return healthchecks.HealthCheckResult("local", True, "ok")

    monkeypatch.setattr(healthchecks, "_http_check", http_check)
    monkeypatch.setattr(healthchecks, "_finbert_status_check", ok_check)
    monkeypatch.setattr(healthchecks, "_locales_check", ok_check)
    monkeypatch.setattr(healthchecks, "_portfolio_db_check", ok_check)

    healthy = SimpleNamespace(
        get_latest=lambda: SimpleNamespace(ts=time.time()),
        coverage_ratio=1.0,
        expected_count=20,
        available_count=20,
        consecutive_failures=0,
        last_error=None,
    )
    context = SimpleNamespace(
        bot=None,
        application=SimpleNamespace(
            bot=None,
            bot_data={"market_snapshot": healthy, "history_snapshot": healthy},
        ),
    )

    await healthchecks.run_healthchecks(context)

    assert "https://api.coingecko.com/api/v3/ping" not in checked_urls
    assert "https://api.telegram.org" in checked_urls
