"""Token unlocks: supply forced onto the market on a day known in advance.

The schedules are DefiLlama's (their free unlock pages, ~370 tokens; market_data.py unlocks, weekly):
the cliff unlocks to a token's team, investors, airdrop and the like, not a treasury's. A moment counts
when it frees 1%+ of the token's whole supply at once (set before any test). Two events on that coin:
  UNLOCK-3D   three days before it: the hour's close 72 hours before the unlock's hour closes
  UNLOCK-DAY  the close of the hour the unlock falls in
A schedule is public long before (vesting contracts, the token's papers), so the dates are no look-ahead;
the list of tokens is, though: DefiLlama documents the ones people follow now, survivors among them.
"""

from __future__ import annotations

from core.sandbox.market_store import HOUR_MS

MIN_SHARE = 0.01
AHEAD_MS = 72 * HOUR_MS


def unlock_events(moments) -> list[tuple[str, int]]:
    """[(event, signal ms)] from [(unlock ms, share of the supply)]."""
    out = []
    for ts, share in moments:
        if share < MIN_SHARE:
            continue
        close = -(-int(ts) // HOUR_MS) * HOUR_MS          # the close of the hour it falls in (or the hour it ends)
        out += [("UNLOCK-3D", close - AHEAD_MS), ("UNLOCK-DAY", close)]
    return out


def all_events(store, market) -> dict[str, list[tuple[str, int]]]:
    """{base: [(event, signal ms)]} for the library's coins with a documented schedule."""
    with store.connect() as conn:
        names = [r[0] for r in conn.execute("SELECT DISTINCT series FROM hourly_series WHERE series LIKE 'unlock:%'")]
    out = {}
    for name in names:
        base = name.split(":", 1)[1]
        if base in market.base_of:
            events = unlock_events(store.hourly(name).itertuples(index=False, name=None))
            if events:
                out[base] = events
    return out
