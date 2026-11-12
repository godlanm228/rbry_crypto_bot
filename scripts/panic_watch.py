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
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RBRY_ENV_FILE", str(ROOT / ".env.test"))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import allocator as al  # noqa: E402
from core.brain import exits as ex  # noqa: E402
from core.brain import factory as fa  # noqa: E402
from core.brain import radar as rd  # noqa: E402
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
RADAR = watch.WATCH_DIR / "radar-model.json"             # the listing radar's base rates, copied from the laptop
SELECTOR_DIR = watch.WATCH_DIR / "selector"            # the live model, copied from the laptop every week
SELECTOR_PREV = watch.WATCH_DIR / "selector-prev"      # the one before it (the night keeps it on a change)
SELECTOR_TOLD = watch.WATCH_DIR / "selector-told.json"  # the rollbacks and stops the owner was told of
SELECTOR_DECISIONS = watch.WATCH_DIR / "decisions-selector.jsonl"
ACCOUNT = watch.WATCH_DIR / "account.json"             # the one $100 account, opened once
EQUITY = watch.WATCH_DIR / "equity.jsonl"               # every book's value at every hour
HOURS = watch.WATCH_DIR / "hours.jsonl"                 # every hour: the market as the watcher saw it, what it took
PLACES = watch.WATCH_DIR / "places.jsonl"               # the account's places, each from the moment it took force
# the places and their order from the account's opening (30.10.2026 16:29) until places.jsonl began: the allocator's
FIRST_PLACES = {"паника": 4, watch.SELECTOR_LABEL: 1}
FIRST_ORDER = ["паника", watch.SELECTOR_LABEL]
PLANS_SEEN = watch.WATCH_DIR / "plans-seen.json"        # every factory candidate ever on paper: retired ones too
JOURNAL = watch.WATCH_DIR / "journal.json"              # every paper trade with its decision and its outcome
URGENT_STATE = ROOT / "runtime" / "learning" / "urgent_state.json"   # the urgent watcher's (scripts/urgent_watch.py)
NEAR_HOURLY = 5                                         # the selector's best coins of every hour, taken or not
OI_API = "https://fapi.binance.com/futures/data/openInterestHist"
CROWD_API = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
FUNDING_API = "https://fapi.binance.com/fapi/v1/fundingRate"
CROWD_HOURS = 800                                       # the crowd's month (720 hours) and some; the API keeps 30 days
CROWD_PAGE = 500                                        # the API's most rows a call
WEEK_MS = 7 * 86_400_000


def _send(text: str, notify: bool, silent: bool = False) -> None:
    print(text, flush=True)
    if notify and not notify_owner(text, silent=silent):
        logger.warning("the message to the owner did not go out")


def _json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write(path: Path, value, indent: Optional[int] = 1) -> None:
    """Whole or not at all: the test bot reads state.json at any moment, and a half-written sent.json or
    fills.json would stop every hour after it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=indent), encoding="utf-8")
    tmp.replace(path)


def api_tail(symbol: str, since_ms: int) -> pd.DataFrame:
    try:
        return archive.fetch_recent(symbol, "1h", since_ms)
    except RuntimeError as exc:
        logger.info("tail %s: %r", symbol, exc)
        return pd.DataFrame()


def _api_rows(url: str, params: dict, what: str) -> list:
    try:
        resp = httpx.get(url, params=params, timeout=15)
        return resp.json() if resp.status_code == 200 else []
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("%s: %r", what, exc)
        return []


def api_oi(perp: str, hour_ms: int):
    """Open interest and the accounts' long/short ratio by hour from the futures API, in the archive's terms:
    ts the hour's start, the value at its end (the API stamps a snapshot with its time)."""
    oi_rows = _api_rows(OI_API, {"symbol": perp, "period": "1h", "limit": 30}, f"open interest {perp}")
    ls_rows = _api_rows(CROWD_API, {"symbol": perp, "period": "1h", "limit": CROWD_PAGE}, f"crowd {perp}")
    if ls_rows:                                         # the month before: a second page
        older = _api_rows(CROWD_API, {"symbol": perp, "period": "1h", "limit": CROWD_PAGE,
                                      "endTime": min(int(r["timestamp"]) for r in ls_rows) - 1}, f"crowd {perp}")
        ls_rows = older + ls_rows
    if not oi_rows and not ls_rows:
        return None
    oi_frame = pd.DataFrame({"ts": [int(r["timestamp"]) - HOUR_MS for r in oi_rows],
                             "open_interest_value": [float(r["sumOpenInterestValue"]) for r in oi_rows]})
    ls_frame = pd.DataFrame({"ts": [int(r["timestamp"]) - HOUR_MS for r in ls_rows],
                             "ls_accounts": [float(r["longShortRatio"]) for r in ls_rows]})
    frame = oi_frame.merge(ls_frame, on="ts", how="outer").sort_values("ts")
    for col in ("open_interest_value", "ls_accounts"):
        if col not in frame:
            frame[col] = np.nan
    return frame[frame["ts"] <= hour_ms].reset_index(drop=True)


def fund_fn(store: MarketStore, live: bool):
    """Every funding payment of a perpetual up to the hour's close: market.db, and the newest from the API."""
    def one(perp: str, hour_ms: int):
        frame = store.funding(perp)
        if live:
            last = int(frame["ts"].max()) if len(frame) else hour_ms - 30 * 86_400_000
            rows = _api_rows(FUNDING_API, {"symbol": perp, "startTime": last + 1, "limit": 100}, f"funding {perp}")
            if rows:
                fresh = pd.DataFrame({"ts": [int(r["fundingTime"]) for r in rows],
                                      "rate": [float(r["fundingRate"]) for r in rows]})
                frame = pd.concat([frame, fresh[fresh["ts"] > last]], ignore_index=True)
        return frame[frame["ts"] <= hour_ms + HOUR_MS].reset_index(drop=True)
    return one


def db_oi(store: MarketStore):
    def one(perp: str, hour_ms: int):
        frame = store.futures_hourly(perp)
        return frame[(frame["ts"] <= hour_ms) & (frame["ts"] > hour_ms - CROWD_HOURS * HOUR_MS)]
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
    memo: dict = {}                                   # symbol -> (from ms, its hours from then): one path a symbol

    def one(symbol: str, from_ms: int) -> np.ndarray:
        have = memo.get(symbol)
        if have is None or from_ms < have[0]:
            frame = store.klines(symbol, "1h", start_ts=from_ms)
            frame = watch.with_tail(frame[frame["ts"] <= newest_hour], symbol, newest_hour, 4,
                                    api_tail if live else None)
            have = memo[symbol] = (from_ms, frame[["ts", "open", "high", "low", "close"]].to_numpy(dtype=float))
        rows = have[1]
        return rows[rows[:, 0] >= from_ms] if from_ms > have[0] else rows
    return one


def background(store: MarketStore, hour_ms: int) -> dict:
    """What surrounded the hour, as known at its close: fear and greed and BTC's share of the market (their newest
    days), and the urgent watcher's stories and market alarms of the last day."""
    out, close = {}, pd.Timestamp(hour_ms + HOUR_MS, unit="ms")
    for name, known_after in (("fear_greed", 0), ("btc_dominance", 1), ("usdt_dominance", 1)):
        s = store.series(name)                        # fear and greed is out at its day's start, a share after it
        s = s[s["day"] + pd.Timedelta(days=known_after) <= close]
        if len(s):
            out[name] = {"day": str(s["day"].iloc[-1].date()), "value": float(s["value"].iloc[-1])}
    try:
        urgent = json.loads(URGENT_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        urgent = {}
    since, until = hour_ms + HOUR_MS - 86_400_000, hour_ms + HOUR_MS
    out["urgent"] = [{"story": r.get("entity"), "kinds": r.get("kinds", []), "told_at": r.get("sent_at")}
                     for r in urgent.get("stories", {}).values()
                     if r.get("last_ts", 0) >= since and r.get("sent_at", 0) <= until]
    out["alarms"] = sorted(k for k, ts in urgent.get("alerts", {}).items() if since <= ts <= until)
    return out


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
                logger.info("fill %s: %r", key, exc)
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
          selector_decisions: list = (), prices=None) -> dict:
    prices = prices or prices_fn(store, newest_hour, live)
    stocks = store.tokenized_stocks()
    decisions, selector_decisions = watch.tradable(decisions, stocks), watch.tradable(list(selector_decisions), stocks)
    plan_decisions = {k: watch.tradable(v, stocks) for k, v in (plan_decisions or {}).items()}
    out = {v: watch.book(decisions, prices, fills, v, newest_hour) for v in watch.VARIANTS}   # from their first $100
    funding = funding_fn(store, perp_of or {})
    for s in plans:
        out[s["id"]] = watch.strategy_book(s, (plan_decisions or {}).get(s["id"], []), prices, fills, newest_hour,
                                           funding)
    out["selector"] = watch.selector_book(list(selector_decisions), prices, fills, newest_hour, funding)
    return out


def hour(args) -> None:
    now = int(time.time() * 1000)
    replay = bool(args.at)
    hour_ms = (int(pd.Timestamp(args.at, tz="UTC").value // 10**6) // HOUR_MS * HOUR_MS if replay
               else watch.newest_closed_hour(now))
    store = MarketStore()
    pairs, perp_of, stocks = universe(store)
    plans = runnable(_json(Path(args.promoted) if args.promoted else PROMOTED, []))
    if not (args.at or args.dry_run):
        remember_plans(plans)
    selector = se.load(Path(args.selector) if args.selector else SELECTOR_DIR)
    if not args.selector:                             # a version falling short on paper: back to the one before
        selector = pick_selector(selector, args)
    wanted = set(ex.PANIC_EVENTS) | {e for s in plans for e in s["genome"]["events"]}
    if selector:
        wanted |= set(fa.LIVE_EVENTS)                  # the selector looks at every event the watcher can see
    started = time.time()
    seen = watch.see(store, hour_ms, pairs, None if replay else api_tail, db_oi(store) if replay else api_oi,
                     perp_of, stocks, wanted, fund_fn(store, not replay))
    try:
        seen.context = background(store, hour_ms)
    except Exception:                                  # what surrounded the hour never costs the hour itself
        logger.exception("background")
    logger.info("hour %s: %d pairs checked, %d in panic (%s), %d with any event, nervous %s, %d strategies, %.0f s",
                pd.Timestamp(hour_ms, unit="ms"), seen.checked, len(seen.panic),
                " ".join(p["base"] for p in seen.panic), len(seen.coins), seen.nervous, len(plans),
                time.time() - started)
    decided = now if not replay else hour_ms + HOUR_MS + 60_000
    decision = watch.decide(seen, decided)
    fresh = {s["id"]: watch.strategy_decide(seen, s, decided) for s in plans}
    view = watch.selector_view(seen, *selector) if selector else pd.DataFrame()
    chosen = watch.selector_decide(seen, *selector, decided, view=view) if selector else None
    bar = selector[1].get("bar", se.LIVE_BAR) if selector else se.LIVE_BAR
    if replay or args.dry_run:
        print(json.dumps({"hour": str(pd.Timestamp(hour_ms, unit="ms")), "checked": seen.checked,
                          "nervous": seen.nervous, "coins": seen.coins, "market": seen.market(),
                          "selector_best": watch.selector_near(view, bar, NEAR_HOURLY)},
                         ensure_ascii=False, indent=1, default=str))
        for d in (decision, chosen, *fresh.values()):
            if d:
                print(json.dumps(d, ensure_ascii=False, default=str)[:4000])
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
    try:                                               # the hour as seen, whatever was taken: the panic's pulse
        if any(h.get("hour") == hour_ms for h in watch.tail(HOURS, 24 * 14)):   # a re-run of any hour of 2 weeks
            raise FileExistsError(f"hour {hour_ms} is already in {HOURS.name}")
        watch.append(HOURS, {**seen.market(), "at": f"{pd.Timestamp(hour_ms, unit='ms'):%Y-%m-%d %H:%M} UTC",
                             "taken": {"panic": [p["base"] for p in decision["picks"]] if decision else [],
                                       "selector": [p["base"] for p in chosen["picks"]] if chosen else [],
                                       **{s["id"]: [p["base"] for p in fresh[s["id"]]["picks"]]
                                          for s in plans if fresh[s["id"]]}},
                             "selector_best": watch.selector_near(view, bar, NEAR_HOURLY)})
    except FileExistsError as exc:                    # a rerun within the hour
        logger.info("hours: %r", exc)
    except Exception:
        logger.exception("hours")
    every = watch.tradable(decisions + [d for ds in plan_decisions.values() for d in ds] + picked, stocks)
    fills = _json(FILLS, {})
    if every and update_fills(every, fills, now):
        _write(FILLS, fills)
    prices = prices_fn(store, hour_ms, True)
    result = books(store, decisions, fills, hour_ms, True, plans, plan_decisions, perp_of, picked, prices)
    places = allocation(account, plans, result, selector[1] if selector else {}) if account else []
    if account:
        history = places_history(account, places, hour_ms + HOUR_MS)   # today's places from this hour's decisions
        result = {watch.ACCOUNT: account_of(store, account, decisions, plans, plan_decisions, picked, fills, hour_ms,
                                            perp_of, places, prices, history), **result}
    sources = {"паника": decisions, watch.VARIANT_RU["nostop"]: decisions, watch.SELECTOR_LABEL: picked,
               **{watch.label(s): plan_decisions.get(s["id"], []) for s in plans}}
    try:
        _write(JOURNAL, watch.ledger(result, sources, prices, hour_ms), indent=None)
    except Exception:                                  # the journal never costs the hour's messages
        logger.exception("journal")
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
                   "panic": len(seen.panic), "nervous": seen.nervous, "regime": seen.regime, "market": seen.states,
                   "strategies": len(plans), "selector": bool(selector), "account": account,
                   "allocation": [{"label": l, "places": n, "worth_pct": round(100 * w, 2)} for l, n, w in places],
                   "selector_model": selector[1].get("trained_at") if selector else None,
                   "selector_record": watch.selector_record(picked, result["selector"]) if "selector" in result else {},
                   "books": {v: watch.summary(b) for v, b in result.items()}})


def allocation(account: dict, plans: list, books_: dict, selector_meta: dict) -> list:
    """Each source's places (core/brain/allocator.py): history's expectation blended with its own book's paper
    trades since the account opened."""
    def live(book_id: str) -> list:
        b = books_.get(book_id)
        return [p.net for p in b.positions if p.status != "open" and p.moment >= account["start_ms"]] if b else []
    sources = [al.Source("паника", sum(watch.EXPECT["nostop"]["usd_per_100"]) / 200, live("nostop"))]
    for s in plans:
        ys = [s.get("years", {}).get(y, {}).get("usd_per_100") for y in ("2025", "2026")]
        ys = [y for y in ys if y is not None and np.isfinite(y)]
        sources.append(al.Source(watch.label(s), (sum(ys) / len(ys) / 100) if ys else 0.0, live(s["id"])))
    sources.append(al.Source(watch.SELECTOR_LABEL, float(selector_meta.get("prior", 0.0)), live("selector"),
                             proven=False))
    return al.allocate(sources)


def places_history(account: dict, places: list, moment: Optional[int] = None) -> list[dict]:
    """The account's places through time; with ``moment``, today's places take force from it when they changed
    (written to places.jsonl), without it the file is only read."""
    loaded = watch.load(PLACES)
    opening = {"from": account["start_ms"], "places": FIRST_PLACES, "order": FIRST_ORDER}
    history = loaded or [opening]
    now, order = {label: n for label, n, _ in places}, [label for label, _, _ in places]
    last = history[-1]
    if moment is not None and now and (now != last["places"] or order != list(last.get("order") or last["places"])):
        entry = {"from": moment, "places": now, "order": order}
        if not loaded:                                # no readable line yet: the opening first, even after a torn one
            watch.append(PLACES, opening)
        watch.append(PLACES, entry)
        history.append(entry)
    return history


def pick_selector(live, args):
    """The selector this hour (watch.choose_selector on the paper record of every version, kept in state.json);
    the owner is told once of a rollback or a stop."""
    records = _json(STATE, {}).get("selector_record", {})
    chosen, what = watch.choose_selector(live, se.load(SELECTOR_PREV) if SELECTOR_PREV.exists() else None, records)
    if what and not (args.at or args.dry_run):
        version = live[1].get("trained_at", "?")
        told = _json(SELECTOR_TOLD, {})
        if told.get(version) != what:
            r = records.get(version, {})
            numbers = (f"{r.get('n', 0)} закрытых сделок, в среднем {100 * r.get('mean_net', 0):+.2f}% при прогнозе "
                       f"{100 * r.get('mean_forecast', 0):+.2f}%")
            text = (f"🧠 Модель выбора {version} не сбывается на бумаге ({numbers}). "
                    + (f"Вернул прежнюю версию {chosen[1].get('trained_at', '?')}." if what == "rollback" else
                       "Прежней версии нет или она тоже не сбывалась — модель выбора не берёт новых сделок до "
                       "следующей недельной версии."))
            _send(text, args.notify, watch.quiet(int(time.time() * 1000)))
            told[version] = what
            _write(SELECTOR_TOLD, told)
    return chosen


def remember_plans(plans: list) -> dict:
    """Every factory candidate that was ever on paper, by id (plans-seen.json): one that left promoted.json keeps
    its past trades in the account."""
    seen = _json(PLANS_SEEN, {})
    fresh = {s["id"]: s for s in plans if s["id"] not in seen}
    if fresh:
        seen.update(fresh)
        _write(PLANS_SEEN, seen)
    return seen


def account_of(store: MarketStore, account: dict, decisions: list, plans: list, plan_decisions: dict, picked: list,
               fills: dict, newest_hour: int, perp_of: dict, places: list, prices=None,
               history: Optional[list] = None) -> watch.Book:
    """The one $100 account: the sources in the allocator's order, each with the places it had at every moment
    (``history``: places_history's; without it today's places at every moment)."""
    rules = {"паника": (decisions, lambda p: (watch.VARIANTS["nostop"], 1)),
             watch.SELECTOR_LABEL: (picked, lambda p: se.RULES[p["action"]])}
    for s in {**_json(PLANS_SEEN, {}), **{s["id"]: s for s in plans}}.values():   # retired candidates too
        g = fa.Genome.from_dict(s["genome"])
        ds = plan_decisions.get(s["id"]) if s["id"] in plan_decisions else watch.load(decisions_path(s))
        rules[watch.label(s)] = (ds, lambda p, g=g: (g.rule(), g.side))
    stocks = store.tokenized_stocks()
    rules = {label: (watch.tradable(ds, stocks), rule) for label, (ds, rule) in rules.items()}
    ever = {label for h in (history or []) for label, n in h["places"].items() if n > 0}
    now = [label for label, n, _ in places if n > 0]
    order = now + sorted(ever - set(now))           # the default priority; with a history each moment has its own
    sources = [(label, *rules[label]) for label in order if label in rules]
    caps = watch.caps_at(history) if history else {label: n for label, n, _ in places}
    return watch.account_book(sources, prices or prices_fn(store, newest_hour, True), fills, newest_hour,
                              account["start_ms"], funding_fn(store, perp_of), caps,
                              watch.order_at(history) if history else None)


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
    selector = se.load(SELECTOR_DIR)
    places = allocation(account, plans, result, selector[1] if selector else {}) if account else []
    if account:
        result = {watch.ACCOUNT: account_of(store, account, decisions, plans, plan_decisions, picked, fills, newest,
                                            perp_of, places, history=places_history(account, places)), **result}
    if args.daily and account:
        titles = store.news(since_ts=now - 86_400_000, kind="listing")["title"].tolist()
        radar = rd.day_line(titles, _json(RADAR, None))       # plan item 9: a warning, not a pick
        day0 = int(watch.berlin(now).normalize().tz_convert("UTC").value // 10**6)
        stocks = store.tokenized_stocks()
        taken = sum(day0 <= d["moment"] <= now for d in watch.tradable(decisions, stocks) if d["picks"])
        pulse = watch.pulse(watch.tail(HOURS, 48), day0, now, stocks, taken)
        night = watch.night_line(PROMOTED.stat().st_mtime if PROMOTED.exists() else None, now / 1000)
        _send("\n".join(x for x in (watch.day_text(result[watch.ACCOUNT], result, watch.load(EQUITY), now,
                                                   account["start_ms"], places, pulse), radar, night) if x),
              args.notify)
        return
    _send(watch.report_text(result, decisions, now - WEEK_MS, now, "неделя" if args.weekly else "сейчас",
                            {s["id"]: s for s in plans}), args.notify)


def digest(args) -> None:
    """What the night researcher and the weekly analyst read (over ssh): the paper trades open, opened or closed in
    the last ``--hours``, the account's skips, the hours, and C1's newest decision; compact and bounded."""
    now = int(time.time() * 1000)
    since = now - int(args.hours * HOUR_MS)
    journal = _json(JOURNAL, {"books": {}})
    c1 = watch.tail(ROOT / "runtime" / "learning" / "paper" / "decisions.jsonl", 1)
    print(watch.digest(journal, watch.tail(HOURS, int(args.hours) + 2), c1[-1] if c1 else None, since,
                       args.max), flush=True)


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
    g = sub.add_parser("digest")
    g.add_argument("--hours", type=float, default=36)
    g.add_argument("--max", type=int, default=25_000, help="the most characters printed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        {"hour": hour, "report": report, "account": open_account, "digest": digest}[args.cmd](args)
    except Exception as exc:                      # the owner hears when the watcher stops
        if getattr(args, "notify", False):
            notify_owner(f"🌊 Дозорный паники: сбой ({type(exc).__name__}: {str(exc)[:200]}). Смотрю.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
