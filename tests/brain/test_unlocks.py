"""Token unlocks: the team's, investors' and airdrops' cliffs of 1%+ of the supply, three days ahead and on the day."""

import pandas as pd

from core.brain import unlocks as ul
from core.sandbox.context_sources import parse_unlocks
from core.sandbox.market_store import MarketStore

HOUR = 3_600_000
T = int(pd.Timestamp("2024-03-16 12:20").value // 10**6)


def test_a_schedule_keeps_the_cliffs_to_people_not_the_treasury_and_sums_a_moment():
    payload = {"gecko_id": "aptos", "defillamaIds": ["2725"], "metadata": {"total": 1000, "token": "coingecko:aptos", "events": [
        {"timestamp": T // 1000, "noOfTokens": [10, 5], "category": "insiders", "unlockType": "cliff"},
        {"timestamp": T // 1000, "noOfTokens": [4], "category": "privateSale", "unlockType": "cliff"},
        {"timestamp": T // 1000, "noOfTokens": [500], "category": "noncirculating", "unlockType": "cliff"},
        {"timestamp": T // 1000 + 86400, "noOfTokens": [9], "category": "insiders", "unlockType": "linear"}]}}
    events, total, ids = parse_unlocks(payload)
    assert events == [(T, 19.0)] and total == 1000 and ids == ["aptos", "aptos", "llama:2725"]


def test_a_big_unlock_is_signalled_three_days_ahead_and_at_its_hours_close(tmp_path):
    close = int(pd.Timestamp("2024-03-16 13:00").value // 10**6)
    assert ul.unlock_events([(T, 0.019), (T + 86_400_000, 0.005)]) == [("UNLOCK-3D", close - 72 * HOUR),
                                                                        ("UNLOCK-DAY", close)]
    on_hour = int(pd.Timestamp("2024-04-16 00:00").value // 10**6)
    assert ul.unlock_events([(on_hour, 0.02)])[1] == ("UNLOCK-DAY", on_hour)   # at 00:00: the hour that ends then
    store = MarketStore(tmp_path / "m.db")
    store.save_hourly("unlock:APT", [(T, 0.019)])
    store.save_hourly("unlock:XYZ", [(T, 0.5)])

    class _Market:
        base_of = {"APT": "APT"}
    assert ul.all_events(store, _Market()) == {"APT": [("UNLOCK-3D", close - 72 * HOUR), ("UNLOCK-DAY", close)]}
