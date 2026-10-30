"""The mass-panic watcher on paper (core/brain/watch.py): every hour, a minute after it closes.

    python scripts/panic_watch.py hour [--notify] [--dry-run]
    python scripts/panic_watch.py hour --at "2026-03-05 13:00"     # one past hour (UTC start), db only
    python scripts/panic_watch.py report [--weekly] [--notify]

Runs on the test VM (deploy/rbry-panic.timer; Sunday's report deploy/rbry-panic-report.*).
--notify writes to the owner only: a card for every mass moment, a line when a
paper half is sold or a paper trade closes (no sound from 00:00 to 08:00 Berlin),
and anything that stops the watcher. --at replays a past hour on market.db alone
(open interest from the archive) and writes and sends nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RBRY_ENV_FILE", str(ROOT / ".env.test"))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import watch  # noqa: E402
from core.notify import notify_owner  # noqa: E402
from core.sandbox import archive  # noqa: E402
from core.sandbox.market_store import HOUR_MS, MarketStore  # noqa: E402

logger = logging.getLogger("panic_watch")
DECISIONS = watch.WATCH_DIR / "decisions.jsonl"
SEEN = watch.WATCH_DIR / "seen.jsonl"
FILLS = watch.WATCH_DIR / "fills.json"
SENT = watch.WATCH_DIR / "sent.json"
STATE = watch.WATCH_DIR / "state.json"
OI_API = "https://fapi.binance.com/futures/data/openInterestHist"
WEEK_MS = 7 * 86_400_000


def _send(text: str, notify: bool, silent: bool = False) -> None:
    print(text, flush=True)
    if notify and not notify_owner(text, silent=silent):
        logger.warning("the message to the owner did not go out")


def _json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def api_tail(symbol: str, since_ms: int) -> pd.DataFrame:
    try:
        return archive.fetch_recent(symbol, "1h", since_ms)
    except RuntimeError as exc:
        logger.info("tail %s: %s", symbol, exc)
        return pd.DataFrame()


def api_oi(perp: str, hour_ms: int):
    """Open interest by hour from the futures API in the archive's terms: ts the hour's start, value at its end."""
    try:
        resp = httpx.get(OI_API, params={"symbol": perp, "period": "1h", "limit": 30}, timeout=15)
        rows = resp.json() if resp.status_code == 200 else []
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("open interest %s: %s", perp, exc)
        return None
    if not rows:
        return None
    frame = pd.DataFrame({"ts": [int(r["timestamp"]) - HOUR_MS for r in rows],
                          "open_interest_value": [float(r["sumOpenInterestValue"]) for r in rows]})
    frame["ls_accounts"] = np.nan
    return frame[frame["ts"] <= hour_ms]


def db_oi(store: MarketStore):
    def one(perp: str, hour_ms: int):
        frame = store.futures_hourly(perp)
        return frame[(frame["ts"] <= hour_ms) & (frame["ts"] > hour_ms - 30 * HOUR_MS)]
    return one


def universe(store: MarketStore) -> tuple[list, dict, set]:
    pairs = [(s, b) for s, b, _ in store.symbols("trading")]
    spot = {b for _, b, _ in store.symbols()}
    with store.connect() as conn:
        perpetuals = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM funding")]
    perp_of: dict[str, str] = {}
    for perp, base in sorted(archive.futures_pairs(perpetuals, spot).items(), key=lambda kv: kv[0] != kv[1] + "USDT"):
        perp_of.setdefault(base, perp)
    return pairs, perp_of, store.tokenized_stocks()


def prices_fn(store: MarketStore, newest_hour: int, live: bool):
    def one(symbol: str, from_ms: int) -> np.ndarray:
        frame = store.klines(symbol, "1h", start_ts=from_ms)
        frame = watch.with_tail(frame[frame["ts"] <= newest_hour], symbol, newest_hour, 4,
                                api_tail if live else None)
        return frame[["ts", "open", "high", "low", "close"]].to_numpy(dtype=float)
    return one


def update_fills(decisions: list[dict], fills: dict, now_ms: int) -> bool:
    """The paper entry of every pick: the 5-minute candle 10 minutes after the hour opened, once it exists."""
    changed = False
    for d in decisions:
        at = d["moment"] + watch.REACTION_MS
        for p in d["picks"]:
            key = f"{d['moment']}:{p['base']}"
            if key in fills or now_ms < at + 5 * 60_000:
                continue
            try:
                five = archive.fetch_recent(p["symbol"], "5m", at, at + 5 * 60_000 + 1)
            except RuntimeError as exc:
                logger.info("fill %s: %s", key, exc)
                continue
            row = five[five["ts"] == at]
            if len(row):
                fills[key] = float(row["open"].iloc[0])
                changed = True
    return changed


def books(store: MarketStore, decisions: list[dict], fills: dict, newest_hour: int, live: bool) -> dict:
    prices = prices_fn(store, newest_hour, live)
    return {v: watch.book(decisions, prices, fills, v, newest_hour) for v in watch.VARIANTS}


def hour(args) -> None:
    now = int(time.time() * 1000)
    replay = bool(args.at)
    hour_ms = (int(pd.Timestamp(args.at, tz="UTC").value // 10**6) // HOUR_MS * HOUR_MS if replay
               else watch.newest_closed_hour(now))
    store = MarketStore()
    pairs, perp_of, stocks = universe(store)
    started = time.time()
    seen = watch.see(store, hour_ms, pairs, None if replay else api_tail, db_oi(store) if replay else api_oi,
                     perp_of, stocks)
    logger.info("hour %s: %d pairs checked, %d in panic (%s), nervous %s, %.0f s",
                pd.Timestamp(hour_ms, unit="ms"), seen.checked, len(seen.panic),
                " ".join(p["base"] for p in seen.panic), seen.nervous, time.time() - started)
    decision = watch.decide(seen, now if not replay else hour_ms + HOUR_MS + 60_000)
    if replay or args.dry_run:
        print(json.dumps({"hour": str(pd.Timestamp(hour_ms, unit="ms")), "checked": seen.checked,
                          "nervous": seen.nervous, "panic": seen.panic}, ensure_ascii=False, indent=1, default=str))
        if decision:
            print(watch.card(decision))
        return
    if seen.panic:
        watch.append(SEEN, {"hour": hour_ms, "nervous": seen.nervous, "btc_r1": seen.btc_r1, "checked": seen.checked,
                            "panic": seen.panic})
    decisions = watch.load(DECISIONS)
    silent = watch.quiet(now)
    if decision and not any(d["moment"] == decision["moment"] for d in decisions):
        watch.append(DECISIONS, decision)
        decisions.append(decision)
        _send(watch.card(decision), args.notify, silent)
    if not decisions:
        _write(STATE, {"hour": hour_ms, "checked": seen.checked, "panic": len(seen.panic), "nervous": seen.nervous})
        return
    fills = _json(FILLS, {})
    if update_fills(decisions, fills, now):
        _write(FILLS, fills)
    result = books(store, decisions, fills, hour_ms, live=True)
    sent = set(_json(SENT, []))
    lines, keys = watch.news(result, sent)
    if lines:
        _send("\n".join(lines), args.notify, silent)
        _write(SENT, sorted(sent | keys))
    _write(STATE, {"hour": hour_ms, "checked": seen.checked, "panic": len(seen.panic), "nervous": seen.nervous,
                   "books": {v: watch.summary(b) for v, b in result.items()}})


def report(args) -> None:
    now = int(time.time() * 1000)
    store = MarketStore()
    decisions = watch.load(DECISIONS)
    newest = watch.newest_closed_hour(now)
    result = books(store, decisions, _json(FILLS, {}), newest, live=True) if decisions else {}
    _send(watch.report_text(result, decisions, now - WEEK_MS, now, "неделя" if args.weekly else "сейчас"),
          args.notify)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hour")
    h.add_argument("--notify", action="store_true")
    h.add_argument("--dry-run", action="store_true")
    h.add_argument("--at", default="")
    r = sub.add_parser("report")
    r.add_argument("--weekly", action="store_true")
    r.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        {"hour": hour, "report": report}[args.cmd](args)
    except Exception as exc:                      # the owner hears when the watcher stops
        if getattr(args, "notify", False):
            notify_owner(f"🌊 Дозорный паники: сбой ({type(exc).__name__}: {str(exc)[:200]}). Смотрю.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
