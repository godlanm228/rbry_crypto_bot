#!/usr/bin/env python3
"""Market snapshots for the trades decided before 07.11.2026, when decisions started recording what they saw.

Vlad, 08.11: a trade without the market it was decided in cannot be judged. The hour of each such trade is
replayed by the watcher's own code, as `panic_watch.py hour --at` replays one (watch.see over market.db, the
same formulas, the same universe and filters): BTC's numbers, the regime, the breadth, every coin with an event,
fear and greed and the dominances of that day; the traded coin's own numbers then. Next to them, what the hourly
log wrote at the time (nervous, the coins in panic), so a replay that disagrees with the record shows it.
Not replayed: the urgent watcher's stories (only today's state is kept) and the selector's reasons (its model of
the day is not kept).

Written once to runtime/learning/watch/rebuilt.json (read only by the site, marked «восстановлено»).

    python scripts/rebuild_snapshots.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pandas as pd  # noqa: E402

import panic_watch as pw  # noqa: E402
from core.brain import exits as ex  # noqa: E402
from core.brain import factory as fa  # noqa: E402
from core.brain import watch  # noqa: E402
from core.console import utf8_output  # noqa: E402
from core.sandbox.market_store import MarketStore  # noqa: E402

OUT = watch.WATCH_DIR / "rebuilt.json"
HOUR_MS = 3_600_000
logger = logging.getLogger("rebuild_snapshots")


def wanting() -> set:
    """The events the live watcher looks for: the panic's and every one the selector sees."""
    return set(ex.PANIC_EVENTS) | set(fa.LIVE_EVENTS)


def logged_hours() -> dict[int, dict]:
    return {h["hour"]: h for h in watch.tail(pw.HOURS, 100_000) if isinstance(h, dict) and "hour" in h}


def coin_then(store: MarketStore, seen, symbol: str, base: str, hour_ms: int, wanted: set) -> dict:
    """The traded coin as the hour saw it: from the replay when it had an event, else its own numbers."""
    for c in seen.coins:
        if c["base"] == base:
            return watch.coin_snapshot(c)
    btc_ctx = watch.btc_now(store, hour_ms, None)[0]
    frame = watch.hours(store, symbol, hour_ms, watch.HISTORY_DAYS, None)
    found, nums = watch.coin_events(frame, symbol, base, hour_ms, btc_ctx, None, wanted, None)
    return watch.coin_snapshot({"events": found, **nums}) if nums else {}


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    journal = json.loads((watch.WATCH_DIR / "journal.json").read_text(encoding="utf-8"))
    missing: dict[int, set] = {}                      # hour -> {(base, symbol)}
    for book in (journal.get("books") or {}).values():
        for t in book.get("trades") or []:
            if not (t.get("decision") or {}).get("market") and t.get("moment"):
                missing.setdefault(int(t["moment"]) - HOUR_MS, set()).add((t["base"], t.get("symbol") or f"{t['base']}USDT"))
    if not missing:
        print("every trade has its own snapshot")
        return 0
    store = MarketStore()
    pairs, perp_of, stocks = pw.universe(store)
    wanted, log = wanting(), logged_hours()
    out = {"made": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
           "how": "watch.see over market.db (as panic_watch.py hour --at), background without urgent stories",
           "trades": {}}
    for hour_ms in sorted(missing):
        started = time.time()
        seen = watch.see(store, hour_ms, pairs, None, pw.db_oi(store), perp_of, stocks, wanted, pw.fund_fn(store, False))
        context = pw.background(store, hour_ms)
        context.pop("urgent", None)                   # the urgent state keeps today's stories only
        context.pop("alarms", None)
        seen.context = context
        market = seen.market()
        rec = log.get(hour_ms) or {}
        logged = ({"nervous": rec.get("nervous"), "panic": rec.get("panic"),
                   "stocks": sorted(set(rec.get("panic") or []) & set(stocks))} if rec else None)
        for base, symbol in sorted(missing[hour_ms]):
            out["trades"][f"{hour_ms + HOUR_MS}:{base}"] = {
                "hour": hour_ms, "market": market, "coin": coin_then(store, seen, symbol, base, hour_ms, wanted),
                "logged": logged}
        logger.info("%s: %d coins checked, regime %s, nervous %s (log: %s), panic %s (log: %s), %.0f s",
                    pd.Timestamp(hour_ms, unit="ms"), seen.checked, seen.regime, seen.nervous,
                    None if rec is None else rec.get("nervous"), [c["base"] for c in seen.panic],
                    None if rec is None else rec.get("panic"), time.time() - started)
    if args.dry_run:
        print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])
        return 0
    watch.WATCH_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUT)
    print(f"{len(out['trades'])} snapshots for {len(missing)} hours -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
