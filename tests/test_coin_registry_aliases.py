"""Ticker collisions resolve to the coin with the best market-cap rank."""

import asyncio
import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

import services.coin_registry_service as registry_module
from services.coin_registry_service import CoinRegistryService, choose_aliases

# Real collisions seen in the CoinGecko list (2026-10-14).
ITEMS = [
    {"id": "the-open-network", "symbol": "ton", "name": "Toncoin"},
    {"id": "tontoken", "symbol": "ton", "name": "TON Community"},
    {"id": "optimism", "symbol": "op", "name": "Optimism"},
    {"id": "one-path", "symbol": "op", "name": "One Path"},
    {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
    {"id": "harrypotterobamasonic10in", "symbol": "bitcoin", "name": "HarryPotterObamaSonic10Inu (ETH)"},
    {"id": "tia", "symbol": "tia", "name": "Tiamonds [OLD]"},
    {"id": "celestia", "symbol": "tia", "name": "Celestia"},
    {"id": "deadcoin-a", "symbol": "zzz", "name": "Dead A"},
    {"id": "zzz", "symbol": "zzzz", "name": "Exact Id"},
]
RANKS = {"bitcoin": 1, "the-open-network": 30, "celestia": 109, "optimism": 130,
         "harrypotterobamasonic10in": 600}


def test_ranked_coin_wins_ticker_collision():
    aliases = choose_aliases(ITEMS, RANKS)
    assert aliases["ton"][0] == "the-open-network"
    assert aliases["op"][0] == "optimism"
    assert aliases["tia"][0] == "celestia"


def test_coin_name_or_id_beats_a_meme_ticker():
    aliases = choose_aliases(ITEMS, RANKS)
    assert aliases["bitcoin"][0] == "bitcoin"
    assert aliases["toncoin"][0] == "the-open-network"


def test_unranked_exact_id_beats_unranked_ticker():
    aliases = choose_aliases(ITEMS, RANKS)
    assert aliases["zzz"][0] == "zzz"


def test_service_resolves_through_alias_table(tmp_path, monkeypatch):
    async def fake_list():
        return ITEMS

    async def fake_ranks():
        return RANKS

    monkeypatch.setattr(registry_module, "get_coin_list", fake_list)
    monkeypatch.setattr(registry_module, "get_market_cap_ranks", fake_ranks)
    db = tmp_path / "registry.db"
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE coin_registry (symbol TEXT PRIMARY KEY, coingecko_id TEXT NOT NULL,"
            " binance_symbol TEXT, name TEXT, updated_at TIMESTAMP)"
        )
    service = CoinRegistryService(db_path=db)
    assert asyncio.run(service.refresh_if_needed(force=True)) is True
    assert service.resolve("TON")["coingecko_id"] == "the-open-network"
    assert service.resolve("Bitcoin")["coingecko_id"] == "bitcoin"
    assert service.resolve("ton")["market_cap_rank"] == 30
    assert service.resolve("nope") is None
    # Fresh registry with aliases does not refresh again.
    assert asyncio.run(service.refresh_if_needed()) is False
