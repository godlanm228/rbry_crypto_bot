import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

from core.utils import resolve_coin_id


class Registry:
    def resolve(self, symbol):
        if symbol == "ARB":
            return {"coingecko_id": "arbitrum"}
        return None


def test_curated_alias_wins_before_registry():
    assert resolve_coin_id("btc", Registry()) == "bitcoin"


def test_registry_id_is_used_for_non_curated_ticker():
    assert resolve_coin_id("arb", Registry()) == "arbitrum"


def test_unknown_ticker_keeps_legacy_lowercase_fallback():
    assert resolve_coin_id("UNKNOWN", Registry()) == "unknown"
