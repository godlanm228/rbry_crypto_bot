#!/usr/bin/env python3
"""Re-download every futures metrics file already ingested and store it again with the fixed parser.

08.11.2026: archive.parse_metrics took the hour's "last" 5-minute snapshot in the file's order, and Binance's
files are not in time order, so futures_hourly (open interest, the crowd's and the top traders' ratios) held a
snapshot from anywhere in the hour since 2020, and zeros where Binance wrote holes. The raw files were never
kept, so they are fetched again. Newest first (the live replay and recent research need them most); resumable
(the keys done go to a progress file); stops by itself at --stop-at (local time) so it never runs into the
night job's use of market.db.

    python scripts/rebuild_metrics.py [--workers 12] [--stop-at 23:50] [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import httpx  # noqa: E402

from core.sandbox import archive  # noqa: E402
from core.sandbox.market_store import MarketStore  # noqa: E402

DONE = ROOT / "runtime" / "learning" / "metrics-rebuild-done.txt"
logger = logging.getLogger("rebuild_metrics")


def keys_to_do(store: MarketStore) -> list[str]:
    with store.connect() as conn:
        keys = [r[0] for r in conn.execute("SELECT key FROM fetched WHERE key >= ? AND key < ? AND status = 'ok'",
                                           (archive.METRICS_DAILY, archive.METRICS_DAILY + "￿"))]
    done = set(DONE.read_text(encoding="utf-8").split()) if DONE.exists() else set()
    return sorted((k for k in keys if k not in done), key=lambda k: (archive.key_period(k), k), reverse=True)


def pair_of(key: str) -> str:
    return key[len(archive.METRICS_DAILY):].split("/", 1)[0]


def fetch(client: httpx.Client, key: str):
    raw = archive.download(client, key)
    return None if raw is None else archive.parse_metrics(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--stop-at", default="", help="HH:MM local: submit nothing after it")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-at", default="", help="HH:MM local: wait until then (after the night job)")
    args = parser.parse_args()
    if args.start_at:
        h, m = map(int, args.start_at.split(":"))
        when = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
        if when <= datetime.now():
            when = when.replace(day=when.day) + (datetime(2000, 1, 2) - datetime(2000, 1, 1))
        print(f"waiting until {when:%d.%m %H:%M}", flush=True)
        time.sleep(max(0.0, (when - datetime.now()).total_seconds()))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    stop = None
    if args.stop_at:
        h, m = map(int, args.stop_at.split(":"))
        stop = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    store = MarketStore()
    todo = keys_to_do(store)
    if args.limit:
        todo = todo[:args.limit]
    logger.info("%d files to fetch again", len(todo))
    stats = {"files": 0, "rows": 0, "missing": 0, "failed": 0}
    started, queue, pending = time.time(), iter(todo), {}
    DONE.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True,
                      limits=httpx.Limits(max_connections=args.workers * 2,
                                          max_keepalive_connections=args.workers * 2)) as client, \
            ThreadPoolExecutor(args.workers) as pool, DONE.open("a", encoding="utf-8") as done_file:
        while True:
            late = stop is not None and datetime.now() >= stop
            while not late and len(pending) < args.workers * 3 and (key := next(queue, None)) is not None:
                pending[pool.submit(fetch, client, key)] = key
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                key = pending.pop(fut)
                try:
                    frame = fut.result()
                except Exception as exc:  # noqa: BLE001 - one file never stops the rest; it stays to do
                    stats["failed"] += 1
                    logger.warning("%s: %r", key, exc)
                    continue
                if frame is None:
                    stats["missing"] += 1
                else:
                    stats["rows"] += store.save_futures_hourly(pair_of(key), frame)
                    stats["files"] += 1
                done_file.write(key + "\n")
                n = stats["files"] + stats["missing"]
                if n % 2000 == 0:
                    done_file.flush()
                    rate = n / (time.time() - started)
                    logger.info("%d/%d files, %.0f/s, ~%.0f min left", n, len(todo), rate, (len(todo) - n) / rate / 60)
    left = len(keys_to_do(store))
    logger.info("%s in %.0f s; %d files left%s", stats, time.time() - started, left,
                " (stopped at --stop-at; run again to go on)" if left else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
