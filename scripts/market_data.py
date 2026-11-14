"""Market data for the brain (step 1): fill and keep runtime/learning/market.db.

    python scripts/market_data.py bootstrap              # every USDT pair since 2017, then 5m, futures, context
    python scripts/market_data.py bootstrap --pairs BTCUSDT,LUNAUSDT --no-futures
    python scripts/market_data.py update                 # daily: yesterday's files, new listings, context
    python scripts/market_data.py hourly                 # hourly: the last closed 1h candles of trading pairs
    python scripts/market_data.py news [--backfill]      # headlines, (de)listing announcements, BTC dominance
    python scripts/market_data.py cex                    # Binance's reserves by token a day (laptop only: 45 MB reply)
    python scripts/market_data.py sentiment              # who is buying: Coinbase, Upbit, Deribit's volatility, the won
    python scripts/market_data.py unlocks                # token unlock schedules (weekly)
    python scripts/market_data.py btc-pairs              # the pairs quoted in BTC before 2021
    python scripts/market_data.py fine                   # 5-minute candles of every pair, in a folder of their own
    python scripts/market_data.py report

Everything comes from public, keyless sources (see core/sandbox/archive.py
and context_sources.py). Safe to stop and re-run: finished files are skipped.
A second bootstrap or update waits until the first one finishes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.console import utf8_output  # noqa: E402

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from core.sandbox import archive, context_sources  # noqa: E402
from core.sandbox.data import EXCHANGE_INFO_URLS, _get_json  # noqa: E402
from core.sandbox.market_store import DEFAULT_DB_PATH, FINE_DB_PATH, HOUR_MS, MarketStore  # noqa: E402

logger = logging.getLogger("market_data")

FIVE_MINUTE_PAIRS = 100        # top pairs by 30-day USDT volume
FIVE_MINUTE_MONTHS = 24
METRICS_PAIRS = 100
SNAPSHOT_HOURS = 3             # dominance after this many hours past midnight is no longer "the close"
FUTURES_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
FUNDING_API = "https://fapi.binance.com/fapi/v1/fundingRate"


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _client(workers: int) -> httpx.Client:
    return httpx.Client(timeout=60, follow_redirects=True,
                        limits=httpx.Limits(max_connections=workers * 2, max_keepalive_connections=workers * 2))


# tasks: (kind, pair, tf, key) -> download in threads, write in the main thread

def _work(client: httpx.Client, task: tuple):
    kind, _, _, key = task
    raw = archive.download(client, key)
    if raw is None:
        return None
    if kind == "kline":
        return archive.parse_klines(raw)
    if kind == "funding":
        return archive.parse_funding(raw)
    return archive.parse_metrics(raw)


def _ingest(store: MarketStore, task: tuple, fut, stats: dict, today: date) -> None:
    kind, pair, tf, key = task
    try:
        result = fut.result()
    except Exception as exc:
        stats["failed"] += 1
        logger.warning("%s: %r", key, exc)
        return
    if result is None:
        stats["missing"] += 1
        if archive.is_settled_missing(key, today):
            store.mark_fetched(key, "missing")
        return
    if kind == "kline":
        rows = store.save_klines(pair, tf, result)
    elif kind == "funding":
        rows = store.save_funding(pair, result)
    else:
        rows = store.save_futures_hourly(pair, result)
    store.mark_fetched(key, "ok", rows)
    stats["files"] += 1
    stats["rows"] += rows


def run_tasks(store: MarketStore, client: httpx.Client, tasks: list[tuple], workers: int, label: str) -> dict:
    """Download in threads, write in this one; at most a few files per worker wait in memory."""
    stats = {"files": 0, "rows": 0, "missing": 0, "failed": 0}
    if not tasks:
        return stats
    today, started, done = _today(), time.time(), 0
    logger.info("%s: %d files to fetch", label, len(tasks))
    queue, pending = iter(tasks), {}
    with ThreadPoolExecutor(workers) as pool:
        while True:
            while len(pending) < workers * 3 and (task := next(queue, None)) is not None:
                pending[pool.submit(_work, client, task)] = task
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                _ingest(store, pending.pop(fut), fut, stats, today)
                done += 1
                if done % 1000 == 0:
                    rate = done / (time.time() - started)
                    logger.info("%s: %d/%d files, %.0f/s, ~%.0f min left", label, done, len(tasks), rate,
                                (len(tasks) - done) / rate / 60)
    logger.info("%s: %s in %.0f s", label, stats, time.time() - started)
    return stats


# universe

def trading_pairs() -> set[str]:
    info = _get_json(EXCHANGE_INFO_URLS, {}, timeout=60)
    return {s["symbol"] for s in info["symbols"] if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"}


def refresh_universe(store: MarketStore, client: httpx.Client) -> dict[str, str]:
    """All archive USDT pairs plus today's listings, with trading / delisted status."""
    listed = archive.pair_names(archive.list_dir(client, archive.SPOT_MONTHLY))
    trading = trading_pairs()
    pairs = archive.usdt_bases(set(listed) | trading)
    store.upsert_symbols((p, b, "trading" if p in trading else "delisted") for p, b in pairs.items())
    logger.info("universe: %d pairs, %d trading", len(pairs), len(set(pairs) & trading))
    try:
        logger.info("products: %d bases, %d tokenized shares", store.save_products(context_sources.fetch_products(client)),
                    len(store.tokenized_stocks()))
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("products: %r", exc)
    return pairs


def top_by_volume(store: MarketStore, pairs: list[str], n: int, days: int = 30) -> list[str]:
    since = int((time.time() - days * 86400) * 1000)
    volume = {p: float(store.klines(p, "1h", since)["quote_volume"].sum()) for p in pairs}
    return [p for p, v in sorted(volume.items(), key=lambda kv: -kv[1]) if v > 0][:n]


def _listings(client: httpx.Client, prefixes: list[str], workers: int) -> dict[str, list[str]]:
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(prefixes, pool.map(lambda p: archive.list_dir(client, p, dirs=False), prefixes)))


# steps

def klines_step(store, client, pairs: list[str], tf: str, workers: int, bootstrap: bool, since_month: str = ""):
    """Monthly archive files (bootstrap), then the API from the last stored candle to now."""
    today = _today()
    if bootstrap:
        fetched = store.fetched("data/spot/")
        prefixes = {p: f"{archive.SPOT_MONTHLY}{p}/{tf}/" for p in pairs}
        listed = _listings(client, list(prefixes.values()), workers)
        tasks = [("kline", p, tf, k) for p in pairs
                 for k in archive.plan_klines(listed[prefixes[p]], fetched, today, since_month)]
        run_tasks(store, client, tasks, workers, f"klines {tf}")
    trading = {s for s, _, _ in store.symbols("trading")}
    tail_step(store, [p for p in pairs if p in trading], tf)


def tail_step(store: MarketStore, pairs: list[str], tf: str, workers: int = 4) -> None:
    now = int(time.time() * 1000)
    month_start = int(datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                      .timestamp() * 1000)

    def one(pair):
        last = store.last_ts(pair, tf)
        return archive.fetch_recent(pair, tf, month_start if last is None else last + archive.TF_MS[tf], now)

    started, rows, failed = time.time(), 0, []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, p): p for p in pairs}
        for fut in as_completed(futures):
            pair = futures.pop(fut)
            try:
                frame = fut.result()
            except RuntimeError as exc:
                failed.append(pair)
                logger.debug("tail %s %s: %s", pair, tf, exc)
                continue
            rows += store.save_klines(pair, tf, frame) if not frame.empty else 0
    logger.info("klines %s (api): %d candles for %d pairs in %.0f s%s", tf, rows, len(pairs),
                time.time() - started, f", failed: {' '.join(sorted(failed))}" if failed else "")


def futures_step(store, client, spot: dict[str, str], workers: int, bootstrap: bool):
    """Funding for every perpetual of our coins; metrics (open interest, long/short): the bootstrap
    fetches today's top 100 back to 2020, the daily update every live perpetual."""
    today = _today()
    archive_futures = archive.pair_names(archive.list_dir(client, archive.FUNDING_MONTHLY))
    try:
        info = _get_json(FUTURES_INFO, {}, timeout=60)
        live = {s["symbol"] for s in info["symbols"]
                if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL"}
    except RuntimeError as exc:
        logger.warning("futures exchangeInfo unavailable (%r): archive only", exc)
        live = set()
    mapping = archive.futures_pairs(set(archive_futures) | live, set(spot.values()))
    fetched = store.fetched("data/futures/")
    tasks = []
    prev_month = f"{archive.daily_window(today)[0]:%Y-%m}"
    if bootstrap:
        listed = _listings(client, [f"{archive.FUNDING_MONTHLY}{f}/" for f in mapping], workers)
        for f in mapping:
            tasks += [("funding", f, "", k) for k in listed[f"{archive.FUNDING_MONTHLY}{f}/"]
                      if k.endswith(".zip") and k not in fetched]
    else:
        tasks += [("funding", f, "", archive.funding_monthly_key(f, prev_month)) for f in mapping
                  if f in live and archive.funding_monthly_key(f, prev_month) not in fetched]
    run_tasks(store, client, tasks, workers, "funding (archive)")
    _funding_tail(store, sorted(f for f in mapping if f in live))

    by_base = {b: f for f, b in mapping.items() if f in live}
    spot_top = top_by_volume(store, [p for p, b in spot.items() if b in by_base], METRICS_PAIRS)
    chosen = [by_base[spot[p]] for p in spot_top]
    tasks = []
    if bootstrap:
        listed = _listings(client, [f"{archive.METRICS_DAILY}{f}/" for f in chosen], workers)
        for f in chosen:
            tasks += [("metrics", f, "", k) for k in listed[f"{archive.METRICS_DAILY}{f}/"]
                      if k.endswith(".zip") and k not in fetched]
    else:
        # every live perpetual from 28.10.2026 on, not only today's top 100: in a few months the
        # metrics no longer say which coins survived (the archive before that stays top-100 only)
        for f in sorted(f for f in mapping if f in live):
            tasks += [("metrics", f, "", k) for d in archive.daily_window(today)
                      for k in [archive.metrics_daily_key(f, d)] if k not in fetched]
    run_tasks(store, client, tasks, workers, "futures metrics")


def _funding_tail(store: MarketStore, pairs: list[str]) -> None:
    """This month's funding from the API: monthly files appear only after the month ends."""
    month_start = int(datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0).timestamp() * 1000)
    total = 0
    for f in pairs:
        last = store.funding(f)["ts"].max()
        since = max(month_start, int(last) + 1) if pd.notna(last) else month_start
        try:
            rows = _get_json(FUNDING_API, {"symbol": f, "startTime": since, "limit": 1000}, timeout=30)
        except RuntimeError as exc:
            logger.warning("funding API %s: %r", f, exc)
            return
        total += store.save_funding(f, [(int(r["fundingTime"]), float(r["fundingRate"])) for r in rows])
        time.sleep(0.7)   # the endpoint allows 500 calls per 5 minutes per IP
    logger.info("funding (api): %d rows for %d perpetuals", total, len(pairs))


def context_step(store: MarketStore, client: httpx.Client) -> None:
    for name, fetch in (("stablecoins_usd", context_sources.fetch_stablecoins),
                        ("fear_greed", context_sources.fetch_fear_greed)):
        try:
            logger.info("%s: %d days", name, store.save_series(name, fetch(client)))
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("%s: %r", name, exc)
    try:
        for name, rows in context_sources.fetch_macro(client).items():
            logger.info("macro %s: %d days", name, store.save_series(name, rows))
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("macro: %r", exc)
    try:
        for name, rows in context_sources.fetch_pageviews(client).items():
            logger.info("attention %s: %d days", name, store.save_series(name, rows))
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("attention: %r", exc)
    for name, fetch in (("defi_tvl", context_sources.fetch_tvl), ("fomc_day", context_sources.fetch_fomc)):
        try:                                    # plan item 8: the background
            logger.info("%s: %d days", name, store.save_series(name, fetch(client)))
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("%s: %r", name, exc)


def snapshot_step(store: MarketStore, client: httpx.Client, now: Optional[datetime] = None) -> None:
    """BTC/ETH/stablecoin shares as of the last UTC close: the first run after midnight saves them.

    The value belongs to the day that just closed, like that day's candle; a
    night with no answer before SNAPSHOT_HOURS stays a gap rather than a
    reading taken hours later.
    """
    now = now or datetime.now(timezone.utc)
    if now.hour >= SNAPSHOT_HOURS:
        return
    day = f"{(now - timedelta(days=1)).date():%Y-%m-%d}"
    saved = store.series("btc_dominance")
    if not saved.empty and saved["day"].max() >= pd.Timestamp(day):
        return
    try:
        values = context_sources.fetch_global(client)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("dominance: %r", exc)
        return
    for name, value in values.items():
        store.save_series(name, [(day, value)])
    logger.info("dominance for %s: %s", day, {k: round(v, 2) for k, v in values.items() if k.endswith("dominance")})


def collect_news(store: MarketStore, backfill: bool = False) -> None:
    items, failed = context_sources.fetch_feeds()
    new = store.save_news(items)
    with httpx.Client(timeout=30) as client:
        for catalog in context_sources.ANNOUNCEMENTS:
            try:
                if backfill:
                    batch = context_sources.all_announcements(client, catalog)
                else:
                    batch, _ = context_sources.fetch_announcements(client, catalog)
                new += store.save_news(batch)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                failed.append(f"binance catalog {catalog}: {exc}")
        snapshot_step(store, client)
    logger.info("news: %d new%s", new, f", failed: {failed}" if failed else "")


# commands

def bootstrap(args) -> None:
    store = MarketStore(args.db)
    with store.fill_lock(), _client(args.workers) as client:
        spot = refresh_universe(store, client)
        pairs = [p for p in spot if not args.pairs or p in args.pairs]
        klines_step(store, client, pairs, "1h", args.workers, bootstrap=True)
        if not args.no_5m:
            trading = {s for s, _, st in store.symbols("trading")}
            top = top_by_volume(store, [p for p in pairs if p in trading], FIVE_MINUTE_PAIRS)
            cutoff = (pd.Period(_today(), "M") - FIVE_MINUTE_MONTHS).strftime("%Y-%m")
            klines_step(store, client, top, "5m", args.workers, bootstrap=True, since_month=cutoff)
        if not args.no_futures:
            futures_step(store, client, {p: spot[p] for p in pairs}, args.workers, bootstrap=True)
        context_step(store, client)


def update(args) -> None:
    store = MarketStore(args.db)
    with store.fill_lock(), _client(args.workers) as client:
        known = {s for s, _, _ in store.symbols()}
        spot = refresh_universe(store, client)
        trading = sorted(s for s, _, _ in store.symbols("trading"))
        fresh = [p for p in trading if p not in known or store.last_ts(p, "1h") is None]
        if fresh:
            logger.info("new pairs: %s", " ".join(fresh))
            klines_step(store, client, fresh, "1h", args.workers, bootstrap=True)
        klines_step(store, client, [p for p in trading if p not in fresh], "1h", args.workers, bootstrap=False)
        klines_step(store, client, sorted(set(store.pairs_with("5m")) & set(trading)), "5m", args.workers,
                    bootstrap=False)
        futures_step(store, client, spot, args.workers, bootstrap=False)
        context_step(store, client)


def fine(args) -> None:
    """Every USDT pair's 5-minute candles since its listing, delisted pairs included.

    About 30 GB, so they live in a store of their own (``FINE_DB_PATH``), on the
    laptop only: market.db keeps the hundred most traded pairs for two years.
    The first run takes hours; a stopped one carries on from the files it has,
    and every later run adds the finished months and the days since.
    """
    store = MarketStore(args.fine_db)
    with store.fill_lock(), _client(args.workers) as client:
        spot = refresh_universe(store, client)
        klines_step(store, client, sorted(spot), "5m", args.workers, bootstrap=True)
    files, rows = store.ingested(archive.SPOT_MONTHLY)
    logger.info("fine store: %d monthly files, %s candles, %.1f GB", files, f"{rows:,}",
                store.db_path.stat().st_size / 1e9)


BTC_PAIRS_UNTIL = "2021-12"          # by then every coin worth trading had a deep USDT market


def btc_pairs(args) -> None:
    """Hourly candles of every BTC-quoted pair up to 2021 (plan item 5).

    Until 2020 most altcoins traded against BTC on Binance, not USDT: 19 coins had a $5M USDT day in
    2018. Stored under their own pairs (ADABTC, ...), never in ``symbols``, so nothing that reads the
    USDT universe sees them; MarketStore.btc_histories turns them into dollars for the years before
    a coin's USDT pair.
    """
    store = MarketStore(args.db)
    with store.fill_lock(), _client(args.workers) as client:
        pairs = [p for p in archive.pair_names(archive.list_dir(client, archive.SPOT_MONTHLY))
                 if p.endswith("BTC") and len(p) > 3]
        fetched = store.fetched("data/spot/")
        prefixes = {p: f"{archive.SPOT_MONTHLY}{p}/1h/" for p in pairs}
        listed = _listings(client, list(prefixes.values()), args.workers)
        tasks = [("kline", p, "1h", k) for p in pairs for k in listed[prefixes[p]]
                 if k.endswith(".zip") and archive.key_period(k) <= BTC_PAIRS_UNTIL and k not in fetched]
        run_tasks(store, client, tasks, args.workers, "klines 1h (BTC pairs)")


def cex(args) -> None:
    """Binance's reserves by token a day (DefiLlama, free): the on-chain flows' history (core/brain/onchain.py).
    One reply of ~45 MB: run on the laptop, not on the 1 GB VM."""
    store = MarketStore(args.db)
    with httpx.Client(timeout=180, follow_redirects=True, headers=context_sources.BROWSER_HEADERS) as client:
        reserves = context_sources.fetch_cex_reserves(client, "binance-cex")
    rows = sum(store.save_series(f"cex:binance:{token}", days) for token, days in reserves.items())
    logger.info("binance reserves: %d tokens, %d days", len(reserves), rows)


def sentiment(args) -> None:
    """Who is buying, by the hour (core/brain/sentiment.py): Coinbase and Upbit closes, Deribit's volatility
    index, the won's rate. Light (a few hundred small requests the first time, a few a night later)."""
    from core.brain import sentiment as se
    store = MarketStore(args.db)
    now = int(time.time() * 1000) // HOUR_MS * HOUR_MS - HOUR_MS          # the last closed hour
    with httpx.Client(timeout=60, follow_redirects=True, headers=context_sources.BROWSER_HEADERS) as client:
        store.save_series(se.FX, context_sources.fetch_ecb_krw(client))
        for name, (fetch, ticker, first) in se.SOURCES.items():
            last = store.last_hourly(name)
            start = first if last is None else last - 24 * HOUR_MS
            rows = fetch(client, ticker, start, now)
            store.save_hourly(name, rows)
            logger.info("%s: %d hours from %s", name, len(rows), _when(start))


def unlocks(args) -> None:
    """Token unlock schedules (core/brain/unlocks.py), from DefiLlama's public unlock pages' data; weekly."""
    store = MarketStore(args.db)
    with httpx.Client(timeout=120, follow_redirects=True, headers=context_sources.BROWSER_HEADERS) as client:
        found = context_sources.fetch_unlocks(client)
    rows = sum(store.save_hourly(f"unlock:{symbol}", events) for symbol, events in found.items())
    logger.info("unlocks: %d tokens, %d unlock moments", len(found), rows)


def hourly(args) -> None:
    """The last closed hour of every trading pair, so market breadth is never more than an hour old.

    Skipped while a bootstrap or the daily update runs: they fetch the same tail.
    """
    store = MarketStore(args.db)
    with store.fill_lock(wait=False) as free:
        if not free:
            logger.info("hourly: a bootstrap or update is running, skipped")
            return
        tail_step(store, sorted(s for s, _, _ in store.symbols("trading")), "1h")


def _when(ms) -> str:
    return "—" if ms is None else datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def report(args) -> None:
    c = MarketStore(args.db).coverage()
    print(f"market.db: {c['size_mb']:.0f} MB; pairs: {c['symbols']}")
    for tf, symbols, rows, first, last in c["klines"]:
        print(f"  klines {tf}: {symbols} pairs, {rows:,} candles, {_when(first)} … {_when(last)}")
    f = c["funding"]
    print(f"  funding: {f[0]} perpetuals, {f[1]:,} payments, {_when(f[2])} … {_when(f[3])}")
    m = c["futures"]
    print(f"  futures metrics: {m[0]} perpetuals, {m[1]:,} hours, {_when(m[2])} … {_when(m[3])}")
    for name, n, first, last in c["series"]:
        print(f"  {name}: {n} days, {first} … {last}")
    for kind, source, n, first, last in c["news"]:
        print(f"  news {kind}/{source}: {n}, {_when(first)} … {_when(last)}")


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--fine-db", type=Path, default=FINE_DB_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--pairs", type=lambda s: set(s.split(",")), default=None)
    b.add_argument("--no-5m", action="store_true")
    b.add_argument("--no-futures", action="store_true")
    u = sub.add_parser("update")
    for p in (b, u):
        p.add_argument("--workers", type=int, default=8)
    sub.add_parser("hourly")
    sub.add_parser("cex")
    sub.add_parser("sentiment")
    sub.add_parser("unlocks")
    bp = sub.add_parser("btc-pairs")
    bp.add_argument("--workers", type=int, default=8)
    fn = sub.add_parser("fine")
    fn.add_argument("--workers", type=int, default=8)
    n = sub.add_parser("news")
    n.add_argument("--backfill", action="store_true")
    sub.add_parser("report")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command == "bootstrap":
        bootstrap(args)
    elif args.command == "update":
        update(args)
    elif args.command == "hourly":
        hourly(args)
    elif args.command == "btc-pairs":
        btc_pairs(args)
    elif args.command == "fine":
        fine(args)
    elif args.command == "cex":
        cex(args)
    elif args.command == "sentiment":
        sentiment(args)
    elif args.command == "unlocks":
        unlocks(args)
    elif args.command == "news":
        collect_news(MarketStore(args.db), args.backfill)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
