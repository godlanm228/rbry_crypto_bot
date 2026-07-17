import asyncio

import pytest

from core.api import coingecko_client
from core.rate_limiter import ApiRateLimiter


class FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status = status
        self.payload = payload
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def read(self):
        return b""

    async def text(self):
        return ""

    async def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)

    def get(self, _url, params=None):
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_rate_limiter_serializes_concurrent_callers(monkeypatch):
    clock = [100.0]
    real_sleep = asyncio.sleep

    def monotonic():
        return clock[0]

    async def sleep(delay):
        clock[0] += delay
        await real_sleep(0)

    monkeypatch.setattr("core.rate_limiter.time.monotonic", monotonic)
    monkeypatch.setattr("core.rate_limiter.asyncio.sleep", sleep)
    limiter = ApiRateLimiter(cps=0.5)
    grants = []

    async def caller():
        await limiter.wait_if_needed()
        grants.append(clock[0])

    await asyncio.gather(caller(), caller(), caller())

    assert grants == [100.0, 102.0, 104.0]


@pytest.mark.asyncio
async def test_rate_limiter_applies_shared_cooldown(monkeypatch):
    clock = [100.0]

    monkeypatch.setattr("core.rate_limiter.time.monotonic", lambda: clock[0])

    async def sleep(delay):
        clock[0] += delay

    monkeypatch.setattr("core.rate_limiter.asyncio.sleep", sleep)
    limiter = ApiRateLimiter(cps=10)

    await limiter.wait_if_needed()
    await limiter.cooldown(5)
    await limiter.wait_if_needed()

    assert clock[0] == 105.0


@pytest.mark.asyncio
async def test_history_batch_opens_circuit_after_repeated_failures(monkeypatch):
    calls = []

    async def fake_session():
        return object()

    async def fail(_session, url, _params=None):
        calls.append(url)
        raise coingecko_client.CoinGeckoError("offline")

    monkeypatch.setattr(coingecko_client, "get_session", fake_session)
    monkeypatch.setattr(coingecko_client, "_fetch_json", fail)
    monkeypatch.setattr(coingecko_client, "_BATCH_FAILURE_BUDGET", 3)

    result = await coingecko_client.get_batch_historical_data(
        ["bitcoin", "ethereum", "solana", "cardano", "ripple"]
    )

    assert len(calls) == 3
    assert result["cardano"]["error"] == "batch_circuit_open"
    assert result["ripple"]["error"] == "batch_circuit_open"


@pytest.mark.asyncio
async def test_retry_after_is_shared_and_request_recovers(monkeypatch):
    class Limiter:
        def __init__(self):
            self.waits = 0
            self.cooldowns = []

        async def wait_if_needed(self):
            self.waits += 1

        async def cooldown(self, seconds):
            self.cooldowns.append(seconds)

    limiter = Limiter()
    session = FakeSession([
        FakeResponse(429, headers={"Retry-After": "7"}),
        FakeResponse(200, payload={"ok": True}),
    ])
    monkeypatch.setattr(coingecko_client, "api_limiter", limiter)
    monkeypatch.setattr(coingecko_client, "_MAX_ATTEMPTS", 2)

    result = await coingecko_client._fetch_json(
        session,
        "https://api.coingecko.com/api/v3/ping",
    )

    assert result == {"ok": True}
    assert limiter.waits == 2
    assert limiter.cooldowns == [7.0]
