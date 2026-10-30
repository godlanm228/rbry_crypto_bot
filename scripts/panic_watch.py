"""The paper watcher (core/brain/watch.py): the mass panic and the factory's candidates, every hour.

    python scripts/panic_watch.py hour [--notify] [--dry-run]
    python scripts/panic_watch.py hour --at "2026-03-05 13:00"     # one past hour (UTC start), db only
    python scripts/panic_watch.py report [--weekly | --daily] [--notify]
    python scripts/panic_watch.py account                          # open the one $100 account from now

Runs on the test VM (deploy/rbry-panic.timer; Sunday's report deploy/rbry-panic-report.*).
--notify writes to the owner only: a card for every decision, a line when a
paper half is taken or a paper trade closes (no sound from 00:00 to 08:00 Berlin),
and anything that stops the watcher. --at replays a past hour on market.db alone
(open interest from the archive) and writes and sends nothing. The factory's
candidates come from runtime/learning/watch/promoted.json, which the laptop's
night copies here (scripts/night.py).
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

from core.brain import exits as ex  # noqa: E402
from core.brain import factory as fa  # noqa: E402
from core.brain import selector as se  # noqa: E402
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
PROMOTED = watch.WATCH_DIR / "promoted.json"
SELECTOR_DIR = watch.WATCH_DIR / "selector"            # the live model, copied from the laptop every week
SELECTOR_DECISIONS = watch.WATCH_DIR / "decisions-selector.jsonl"
ACCOUNT = watch.WATCH_DIR / "account.json"             # the one $100 account (Vlad, 30.10.2026), opened once
EQUITY = watch.WATCH_DIR / "equity.jsonl"               # every book's value at every hour
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


def runnable(entries: list[dict]) -> list[dict]:
    """The factory's candidates the watcher can run, but not the panic it runs anyway."""
    out = []
    for s in entries:
        if s["id"] in watch.BUILT_IN:
            continue
        if not fa.live_ready(fa.Genome.from_dict(s["genome"])):
            logger.warning("%s needs data the watcher has no live feed for; skipped", s["id"])
            continue
        out.append(s)
    return out


def decisions_path(strategy: dict) -> Path:
    return watch.WATCH_DIR / f"decisions-{strategy['id']}.jsonl"


def funding_fn(store: MarketStore, perp_of: dict):
    def one(base: str):
        perp = perp_of.get(base)
        return store.funding(perp)[["ts", "rate"]].to_numpy(dtype=float) if perp else None
    return one


def books(store: MarketStore, decisions: list[dict], fills: dict, newest_hour: int, live: bool,
          plans: list[dict] = (), plan_decisions: dict = None, perp_of: dict = None,
          selector_decisions: list = ()) -> dict:
    prices = prices_fn(store, newest_hour, live)
    out = {v: watch.book(decisions, prices, fills, v, newest_hour) for v in watch.VARIANTS} if decisions else {}
    funding = funding_fn(store, perp_of or {})
    if selector_decisions:
        out["selector"] = watch.selector_book(list(selector_decisions), prices, fills, newest_hour, funding)
    for s in plans:
        ds = (plan_decisions or {}).get(s["id"], [])
        if ds:
            out[s["id"]] = watch.strategy_book(s, ds, prices, fills, newest_hour, funding)
    return out


def hour(args) -> None:
    now = int(time.time() * 1000)
    replay = bool(args.at)
    hour_ms = (int(pd.Timestamp(args.at, tz="UTC").value // 10**6) // HOUR_MS * HOUR_MS if replay
               else watch.newest_closed_hour(now))
    store = MarketStore()
    pairs, perp_of, stocks = universe(store)
    plans = runnable(_json(Path(args.promoted) if args.promoted else PROMOTED, []))
    selector = se.load(Path(args.selector) if args.selector else SELECTOR_DIR)
    wanted = set(ex.PANIC_EVENTS) | {e for s in plans for e in s["genome"]["events"]}
    if selector:
        wanted |= set(fa.LIVE_EVENTS)                  # the selector looks at every event the watcher can see
    started = time.time()
    seen = watch.see(store, hour_ms, pairs, None if replay else api_tail, db_oi(store) if replay else api_oi,
                     perp_of, stocks, wanted)
    logger.info("hour %s: %d pairs checked, %d in panic (%s), %d with any event, nervous %s, %d strategies, %.0f s",
                pd.Timestamp(hour_ms, unit="ms"), seen.checked, len(seen.panic),
                " ".join(p["base"] for p in seen.panic), len(seen.coins), seen.nervous, len(plans),
                time.time() - started)
    decided = now if not replay else hour_ms + HOUR_MS + 60_000
    decision = watch.decide(seen, decided)
    fresh = {s["id"]: watch.strategy_decide(seen, s, decided) for s in plans}
    chosen = watch.selector_decide(seen, *selector, decided) if selector else None
    if replay or args.dry_run:
        print(json.dumps({"hour": str(pd.Timestamp(hour_ms, unit="ms")), "checked": seen.checked,
                          "nervous": seen.nervous, "coins": seen.coins}, ensure_ascii=False, indent=1, default=str))
        if decision:
            print(watch.card(decision))
        for s in plans:
            if fresh[s["id"]]:
                print(watch.strategy_card(fresh[s["id"]], s))
        if chosen:
            print(watch.selector_card(chosen))
        return
    if seen.coins:
        watch.append(SEEN, {"hour": hour_ms, "nervous": seen.nervous, "btc_r1": seen.btc_r1, "btc_up": seen.btc_up,
                            "regime": seen.regime, "checked": seen.checked, "coins": seen.coins})
    account = _json(ACCOUNT, None)
    silent = watch.quiet(now)
    talk = account is None            # with the account open, only the account speaks
    decisions = watch.load(DECISIONS)
    if decision and not any(d["moment"] == decision["moment"] for d in decisions):
        watch.append(DECISIONS, decision)
        decisions.append(decision)
        if talk:
            _send(watch.card(decision), args.notify, silent)
    plan_decisions = {}
    for s in plans:
        ds = watch.load(decisions_path(s))
        d = fresh[s["id"]]
        if d and not any(x["moment"] == d["moment"] for x in ds):
            watch.append(decisions_path(s), d)
            ds.append(d)
            if talk:
                _send(watch.strategy_card(d, s), args.notify, silent)
        plan_decisions[s["id"]] = ds
    picked = watch.load(SELECTOR_DECISIONS)
    if chosen and not any(x["moment"] == chosen["moment"] for x in picked):
        watch.append(SELECTOR_DECISIONS, chosen)
        picked.append(chosen)
        if talk:
            _send(watch.selector_card(chosen), args.notify, silent)
    every = decisions + [d for ds in plan_decisions.values() for d in ds] + picked
    fills = _json(FILLS, {})
    if every and update_fills(every, fills, now):
        _write(FILLS, fills)
    result = books(store, decisions, fills, hour_ms, True, plans, plan_decisions, perp_of, picked)
    if account:
        result = {watch.ACCOUNT: account_of(store, account, decisions, plans, plan_decisions, picked, fills, hour_ms,
                                            perp_of), **result}
    sent = set(_json(SENT, []))
    shown = {watch.ACCOUNT: result[watch.ACCOUNT]} if account else result
    lines, keys = watch.news(shown, sent)
    if account:
        opened, open_keys = watch.entry_lines(result[watch.ACCOUNT], sent)
        if opened:
            lines = [f"📒 {watch.ACCOUNT_LABEL}: вход (бумага, деньги не двигаются)"] + opened + lines
        keys |= open_keys
        if lines:
            lines.append(f"Баланс ${result[watch.ACCOUNT].value:.2f} из $100.")
    if lines:
        _send("\n".join(lines), args.notify, silent)
        _write(SENT, sorted(sent | keys))
    watch.append(EQUITY, {"hour": hour_ms, "books": {v: round(b.value, 4) for v, b in result.items()}})
    _write(STATE, {"hour": hour_ms, "updated": f"{watch.berlin(now):%d.%m %H:%M}", "checked": seen.checked,
                   "panic": len(seen.panic), "nervous": seen.nervous, "regime": seen.regime,
                   "strategies": len(plans), "selector": bool(selector), "account": account,
                   "books": {v: watch.summary(b) for v, b in result.items()}})


def account_of(store: MarketStore, account: dict, decisions: list, plans: list, plan_decisions: dict, picked: list,
               fills: dict, newest_hour: int, perp_of: dict) -> watch.Book:
    """The one $100 account: the panic first, then the factory's candidates, then the selector."""
    sources = [("паника", decisions, lambda p: (watch.VARIANTS["nostop"], 1))]
    for s in plans:
        g = fa.Genome.from_dict(s["genome"])
        sources.append((watch.label(s), plan_decisions.get(s["id"], []), lambda p, g=g: (g.rule(), g.side)))
    sources.append((watch.SELECTOR_LABEL, picked, lambda p: se.RULES[p["action"]]))
    return watch.account_book(sources, prices_fn(store, newest_hour, True), fills, newest_hour, account["start_ms"],
                              funding_fn(store, perp_of))


def report(args) -> None:
    now = int(time.time() * 1000)
    store = MarketStore()
    _, perp_of, _ = universe(store)
    decisions = watch.load(DECISIONS)
    plans = runnable(_json(PROMOTED, []))
    plan_decisions = {s["id"]: watch.load(decisions_path(s)) for s in plans}
    picked = watch.load(SELECTOR_DECISIONS)
    newest = watch.newest_closed_hour(now)
    fills = _json(FILLS, {})
    result = books(store, decisions, fills, newest, True, plans, plan_decisions, perp_of, picked)
    account = _json(ACCOUNT, None)
    if account:
        result = {watch.ACCOUNT: account_of(store, account, decisions, plans, plan_decisions, picked, fills, newest,
                                            perp_of), **result}
    if args.daily and account:
        _send(watch.day_text(result[watch.ACCOUNT], result, watch.load(EQUITY), now, account["start_ms"]),
              args.notify)
        return
    _send(watch.report_text(result, decisions, now - WEEK_MS, now, "неделя" if args.weekly else "сейчас",
                            {s["id"]: s for s in plans}), args.notify)


def open_account(args) -> None:
    """Open the one $100 account from now (once; --force starts it again)."""
    if ACCOUNT.exists() and not args.force:
        print(f"already open: {ACCOUNT.read_text(encoding='utf-8')}")
        return
    now = int(time.time() * 1000)
    _write(ACCOUNT, {"start_ms": now, "start_usd": 100.0, "opened": f"{watch.berlin(now):%d.%m.%Y %H:%M}"})
    print(ACCOUNT.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hour")
    h.add_argument("--notify", action="store_true")
    h.add_argument("--dry-run", action="store_true")
    h.add_argument("--at", default="")
    h.add_argument("--promoted", default="", help="the candidates to run (default: the watcher's promoted.json)")
    h.add_argument("--selector", default="", help="the live model's folder (default: the watcher's selector/)")
    r = sub.add_parser("report")
    r.add_argument("--weekly", action="store_true")
    r.add_argument("--daily", action="store_true")
    r.add_argument("--notify", action="store_true")
    a = sub.add_parser("account")
    a.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        {"hour": hour, "report": report, "account": open_account}[args.cmd](args)
    except Exception as exc:                      # the owner hears when the watcher stops
        if getattr(args, "notify", False):
            notify_owner(f"🌊 Дозорный паники: сбой ({type(exc).__name__}: {str(exc)[:200]}). Смотрю.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
