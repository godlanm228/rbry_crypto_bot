"""The paper watcher: the mass panic and the factory's candidates on live prices, cards to Vlad.

Every hour, a minute after it closes (scripts/panic_watch.py hour, deploy/rbry-panic.timer):
1. see: every trading pair is read from market.db (45 days of hours, kept fresh by
   market-hourly) and, when it turns over $5M a day, its newest closed hours from
   the klines API; BTC for 400 days (its trend, the nervous market); open interest
   from the futures API only for coins that moved 3% in 4 hours when an open
   interest event is wanted. Events are found with the library's own code and
   cooldowns (core/brain/events.py), so the watcher sees what history saw;
2. decide: the mass panic (4+ coins in panic in a nervous market) takes the 3 most
   liquid, and every factory candidate on paper (runtime/learning/watch/promoted.json,
   copied from the laptop's factory, core/brain/factory.py) takes what its genome
   takes, and the selector (core/brain/selector.py: a model the laptop retrains
   every week on every closed trade, copied to runtime/learning/watch/selector/)
   takes for each coin the action with the best forecast above +1%; each
   decision goes into its own decisions file before any outcome exists, never
   changed;
3. book: the panic has two $100 books (half out at +5%, the rest at the entry or
   after 3 days; without a stop and with a disaster stop at -30%), each candidate
   one with its own exit, long on spot or 1x short; all replay their decisions on
   hourly prices (core/brain/exits.py). Entry: the open of the 5-minute candle 10
   minutes after the hour opened (Vlad's reaction), until it exists the hour's open;
4. account (Vlad, 30.10.2026): one $100 paper account, «Разгон», from the moment it
   was opened: every strategy's picks go into it, the panic first, then the
   factory's candidates, then the selector, 20% of the account a trade, at most 5
   open; the strategies' own books keep running unseen, to show which one pulls;
5. tell: the account's entries, halves taken and closes, from 00:00 to 08:00 Berlin
   without sound, and its day at 21:00.
No money moves; messages go to the owner only.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.brain import events as ev
from core.brain import exits as ex
from core.brain import factory as fa
from core.brain import regime as rg
from core.brain import selector as se
from core.brain.dataset import DAY_MS, slippage_pct
from core.sandbox.market_store import HOUR_MS, History, MarketStore, PEG_MEDIAN_MOVE

WATCH_DIR = Path(__file__).resolve().parents[2] / "runtime" / "learning" / "watch"
MASS = ex.MASS["mass4"]
PICKS = ex.PICKS
HISTORY_DAYS = 45
BTC_DAYS = 400
OI_MOVE = 0.03                      # open interest is fetched only for coins that moved this much in 4 hours
OI_EVENTS = ("OI-UP-UP", "OI-UP-DOWN", "OI-FLUSH")
TAIL_WORKERS = 8
REACTION_MS = ev.REACTION_MS
VARIANTS = {"nostop": ex.BY_RULE["H72-HALF5"], "stop30": ex.BY_RULE["H72-HALF5-S30"]}
VARIANT_RU = {"nostop": "паника без стопа", "stop30": "паника со стопом −30%"}
BUILT_IN = {g.id for g in fa.seeds()[:2]}          # the panic's two books are the factory's first two seeds
QUIET_HOURS = (0, 8)                # Berlin: no sound
WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
# history (scripts/brain_panic.py, panic-20261030-1420): mass4, 3 most liquid; 2025 and 2026
EXPECT = {"nostop": {"up_of10": (7.7, 7.8), "usd_per_100": (1.90, 2.21), "yearly_pct": (10, 19)},
          "stop30": {"up_of10": (7.6, 7.6), "usd_per_100": (1.22, 1.76), "yearly_pct": (5, 19)}}
TailFn = Callable[[str, int], pd.DataFrame]          # (symbol, from ms) -> closed hourly candles
OiFn = Callable[[str, int], Optional[pd.DataFrame]]  # (perpetual, newest hour) -> ts, open_interest_value


# -- seeing ----------------------------------------------------------------------------------

def newest_closed_hour(now_ms: int) -> int:
    """The start of the newest hour that has closed."""
    return now_ms // HOUR_MS * HOUR_MS - HOUR_MS


def hours(store: MarketStore, symbol: str, hour_ms: int, days: int, tail: Optional[TailFn]) -> pd.DataFrame:
    """A pair's hourly candles over ``days`` up to the hour starting at ``hour_ms``, the newest from the API."""
    frame = store.klines(symbol, "1h", start_ts=hour_ms - days * DAY_MS)
    return with_tail(frame[frame["ts"] <= hour_ms], symbol, hour_ms, days, tail)


def with_tail(frame: pd.DataFrame, symbol: str, hour_ms: int, days: int, tail: Optional[TailFn]) -> pd.DataFrame:
    """The candles market.db does not have yet, up to the hour starting at ``hour_ms``, from the API."""
    last = int(frame["ts"].iloc[-1]) if len(frame) else hour_ms - days * DAY_MS - HOUR_MS
    if tail is not None and last < hour_ms:
        fresh = tail(symbol, last + HOUR_MS)
        if fresh is not None and len(fresh):
            fresh = fresh[(fresh["ts"] > last) & (fresh["ts"] <= hour_ms)]
            frame = pd.concat([frame, fresh[frame.columns]], ignore_index=True)
    return frame.reset_index(drop=True)


def turnover(frame: pd.DataFrame) -> float:
    """The 30-day median USDT turnover a day of finished days, as the library's features count it."""
    day = frame["ts"] // DAY_MS
    daily = frame["quote_volume"].groupby(day).sum()
    full = frame["quote_volume"].groupby(day).count() >= 20
    daily = daily[full].iloc[-30:]
    return float(daily.median()) if len(daily) >= 20 else float("nan")


def _history(symbol: str, base: str, frame: pd.DataFrame) -> History:
    return History(base, symbol, base, frame, frame.iloc[:0], False)


def btc_now(store: MarketStore, hour_ms: int, tail: Optional[TailFn]) -> tuple[pd.DataFrame, str]:
    """BTC's hours as the library's context, and the regime of the hour starting at ``hour_ms``."""
    frame = hours(store, "BTCUSDT", hour_ms, BTC_DAYS, tail)
    ctx = ev.btc_context(ev.grid(_history("BTCUSDT", "BTC", frame)))
    regimes = rg.btc_regimes(frame[["ts", "open", "high", "low", "close"]].to_numpy(dtype=float))
    return ctx, str(rg.at(regimes, np.array([hour_ms]))[0])


def coin_events(frame: pd.DataFrame, symbol: str, base: str, hour_ms: int, btc: pd.DataFrame,
                oi: Optional[pd.DataFrame] = None, wanted=ex.PANIC_EVENTS) -> tuple[list[str], dict]:
    """(the ``wanted`` events of the hour starting at ``hour_ms``, the coin's numbers then), as on history."""
    g = ev.grid(_history(symbol, base, frame))
    if not len(g) or int(g["ts"].iloc[-1]) != hour_ms or len(g) < ev.MIN_HOURS + 2:
        return [], {}
    x = ev.features(g, None, oi, btc)
    usual = (x["turnover"].to_numpy() >= ev.MIN_TURNOVER) & x["sigma"].notna().to_numpy()
    last = len(x) - 1
    found = []
    for e in sorted(wanted):
        if e in ev.PAIR_EVENTS and base == "BTC":
            continue
        event = ev.BY_ID[e]
        mask = event.detect(x).fillna(False).to_numpy(dtype=bool) & usual
        fired = ev.fire(mask, event.edge)
        if len(fired) and fired[-1] == last:
            found.append(e)
    row = x.iloc[last]
    closes = g["close"].to_numpy(dtype=float)
    return found, {"turnover": float(row["turnover"]), "r1": float(row["r1"]), "r4": float(row["r4"]),
                   "r24": float(row["r24"]), "close": float(row["close"]),
                   "sigma_day": ex.daily_sigma(closes, len(closes)), "usual": bool(usual[last]),
                   "big": bool(row["turnover"] >= ev.BIG_TURNOVER)}


@dataclass
class Seen:
    hour_ms: int
    nervous: bool
    btc_r1: float
    checked: int
    coins: list = field(default_factory=list)       # {base, symbol, events, turnover, r1, r24, close, sigma_day, ...}
    btc_up: Optional[bool] = None
    regime: str = ""                                # regime.REGIMES, '' while unknown

    @property
    def panic(self) -> list:
        """The coins in panic, with their panic events only."""
        out = []
        for c in self.coins:
            events = [e for e in c["events"] if e in ex.PANIC_EVENTS]
            if events:
                out.append({**c, "events": events})
        return out

    def when(self) -> pd.Timestamp:
        return berlin(self.hour_ms + HOUR_MS)


def see(store: MarketStore, hour_ms: int, pairs: list[tuple[str, str]], tail: Optional[TailFn], oi: OiFn,
        perp_of: dict[str, str], stocks: set[str] = frozenset(), wanted=ex.PANIC_EVENTS) -> Seen:
    """Every coin with a ``wanted`` event (the panic's always included) in the hour starting at ``hour_ms``."""
    wanted = set(wanted) | set(ex.PANIC_EVENTS)
    btc, regime = btc_now(store, hour_ms, tail)
    at = btc[btc["ts"] == hour_ms]
    nervous = bool(at["nervous"].iloc[0]) if len(at) else False
    seen = Seen(hour_ms, nervous, float(at["btc_r1"].iloc[0]) if len(at) else float("nan"), 0,
                btc_up=bool(at["btc_up"].iloc[0]) if len(at) else None, regime=regime)
    liquid = []
    for symbol, base in pairs:
        if base in stocks:
            continue
        frame = hours(store, symbol, hour_ms, HISTORY_DAYS, None)
        known = frame[frame["ts"] < hour_ms // DAY_MS * DAY_MS]
        if not len(known) or not turnover(known) >= ev.MIN_TURNOVER * 0.8:
            continue                                   # far below $5M a day: not worth an API call
        if known["close"].groupby(known["ts"] // DAY_MS).last().pct_change().abs().median() < PEG_MEDIAN_MOVE:
            continue                                   # a stablecoin or a peg
        liquid.append((symbol, base, frame))
    with ThreadPoolExecutor(TAIL_WORKERS) as pool:     # the newest hours of every liquid pair at once
        frames = list(pool.map(lambda s: with_tail(s[2], s[0], hour_ms, HISTORY_DAYS, tail), liquid))
    wants_oi = bool(wanted & set(OI_EVENTS))
    for (symbol, base, _), frame in zip(liquid, frames):
        seen.checked += 1
        found, nums = coin_events(frame, symbol, base, hour_ms, btc, wanted=wanted)
        perp = perp_of.get(base)
        if (wants_oi and nums and nums["usual"] and perp and math.isfinite(nums["r4"])
                and abs(nums["r4"]) >= OI_MOVE):
            interest = oi(perp, hour_ms)
            if interest is not None and len(interest):
                found, nums = coin_events(frame, symbol, base, hour_ms, btc, interest, wanted)
        if found:
            seen.coins.append({"base": base, "symbol": symbol, "events": found, "perp": bool(perp), **nums})
    return seen


# -- deciding ---------------------------------------------------------------------------------

def _decision(seen: Seen, decided_ms: int, size: int, picks: list[dict]) -> dict:
    return {"moment": seen.hour_ms + HOUR_MS, "hour": seen.hour_ms, "decided_ms": decided_ms,
            "decided_at": pd.Timestamp(decided_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "nervous": seen.nervous, "btc_r1": seen.btc_r1, "size": size,
            "picks": [{k: p[k] for k in ("base", "symbol", "turnover", "close", "sigma_day")} for p in picks]}


def decide(seen: Seen, decided_ms: int) -> Optional[dict]:
    """A mass moment in a nervous market: the 3 most liquid coins in panic; else None."""
    panic = seen.panic
    if not seen.nervous or len(panic) < MASS:
        return None
    d = _decision(seen, decided_ms, len(panic), sorted(panic, key=lambda p: -p["turnover"])[:PICKS])
    d["panic"] = [{k: p[k] for k in ("base", "symbol", "events", "turnover", "r1", "r24")} for p in panic]
    return d


def seen_frame(seen: Seen) -> pd.DataFrame:
    """The hour's coins as the factory's trade table: one row per coin and event, the hour's conditions."""
    when = seen.when()
    rows = [{"event": e, "history": c["base"], "base": c["base"], "signal_ts": seen.hour_ms + HOUR_MS,
             "turnover": c["turnover"], "fell24": c["r24"], "nervous": seen.nervous, "btc": seen.btc_up,
             "weekend": when.dayofweek >= 5, "big": c["big"], "daypart": str(ev._daypart(np.array([when.hour]))[0]),
             "perp": c["perp"], "squeeze_chance": np.nan, "sigma_day": c["sigma_day"]}
            for c in seen.coins for e in c["events"]]
    cols = ["event", "history", "base", "signal_ts", "turnover", "fell24", "nervous", "btc", "weekend", "big",
            "daypart", "perp", "squeeze_chance", "sigma_day"]
    frame = pd.DataFrame(rows, columns=cols)
    for c in ("nervous", "btc", "weekend", "big", "perp"):
        frame[c] = frame[c].astype(object)
    return frame


def strategy_decide(seen: Seen, strategy: dict, decided_ms: int) -> Optional[dict]:
    """What a factory candidate takes this hour, exactly as on history (factory.chosen); else None."""
    g = fa.Genome.from_dict(strategy["genome"])
    frame = seen_frame(seen)
    if not len(frame):
        return None
    picks = fa.chosen(g, frame)
    if not len(picks):
        return None
    coins = {c["base"]: c for c in seen.coins}
    matched = fa.chosen(replace(g, pick="all"), frame)["base"].nunique()
    order = picks.sort_values("turnover", ascending=False, kind="stable")["base"]   # as the backtest's book
    d = _decision(seen, decided_ms, int(matched), [coins[b] for b in order])
    d["strategy"] = strategy["id"]
    return d


def selector_decide(seen: Seen, model, meta: dict, decided_ms: int) -> Optional[dict]:
    """What the selector takes this hour: per coin the action with the best forecast above its bar."""
    rows = seen_frame(seen)
    rows = rows[rows["event"].isin(fa.LIVE_EVENTS)]
    if not len(rows) or not seen.regime:
        return None
    trend, mood = seen.regime.split("|")
    btc = pd.DataFrame({"ts": [seen.hour_ms], "trend": [trend], "mood": [mood]})
    best = se.best_actions(se.candidates(rows, None, btc), model, tuple(meta["features"]), meta.get("bar", se.LIVE_BAR))
    if not len(best):
        return None
    coins = {c["base"]: c for c in seen.coins}
    d = _decision(seen, decided_ms, int(rows["base"].nunique()), [coins[b] for b in best["base"]])
    for p, r in zip(d["picks"], best.itertuples(index=False)):
        p.update(action=r.action, side=int(r.side), forecast=float(r.forecast))
    d.update(strategy="selector", regime=seen.regime, model=meta.get("trained_at"))
    return d


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# -- booking -----------------------------------------------------------------------------------

PricesFn = Callable[[str, int], np.ndarray]          # (symbol, from ms) -> rows of ts, open, high, low, close
FundingFn = Callable[[str], Optional[np.ndarray]]    # (base) -> rows of ts, rate


@dataclass
class Position:
    moment: int
    base: str
    symbol: str
    size: float
    entry: float
    status: str                  # open | stop | breakeven | time | target | keep | liquidated
    net: float                   # closed: the result; open: if closed at the newest close
    exit_ts: Optional[int]
    half_ts: Optional[int]       # when half was taken
    due_ts: int                  # when the rest is closed at the latest
    side: int = 1
    source: str = ""             # the strategy that picked it (the account)
    rule: Optional[ex.Rule] = None
    sigma_day: float = float("nan")


@dataclass
class Book:
    variant: str                 # nostop | stop30 | a factory genome's id
    cash: float
    value: float
    positions: list = field(default_factory=list)
    skipped: int = 0
    label: str = ""
    side: int = 1
    half_pct: Optional[float] = None
    hours: int = 72


def follow(prices: np.ndarray, moment: int, entry: Optional[float], sigma_day: float, rule: ex.Rule,
           newest_hour: int, slip: float, side: int = 1, funding: Optional[np.ndarray] = None) -> Optional[dict]:
    """One pick under one rule up to the newest closed hour: closed, or open and marked at the newest close.
    A short walks the mirrored path 2*entry - price, like the factory."""
    if not len(prices):
        return None
    ts = prices[:, 0]
    a = int(np.searchsorted(ts, moment))
    if a >= len(ts) or ts[a] != moment:
        return None
    b = int(np.searchsorted(ts, min(moment + rule.hours * HOUR_MS, newest_hour + HOUR_MS)))
    o, h, l, c = (np.asarray(prices[a:b, k], dtype=float) for k in (1, 2, 3, 4))
    price = float(entry) if entry else float(o[0])
    path = (o, h, l, c) if side > 0 else (2 * price - o, 2 * price - l, 2 * price - h, 2 * price - c)
    exits, held, reason, _ = ex.walk_long(*path, price, ex._stop_dist(rule, sigma_day), rule)
    half_at = price * (1 + rule.half_pct) if rule.half_pct else np.inf
    hits = path[1] >= half_at
    half_ts = int(ts[a + int(np.argmax(hits))]) + HOUR_MS if len(exits) == 2 and hits.any() else None
    over = moment + rule.hours * HOUR_MS <= newest_hour + HOUR_MS
    closed = reason != "time" or over
    if not closed:
        exits = exits[:-1] + [(exits[-1][0], float(path[3][-1]))]
    end_ts = int(ts[a + held - 1]) + HOUR_MS
    if side > 0:
        net = ex.net_of(exits, price, slip)
    elif float(np.max(h[:held])) >= 2 * price:
        net, reason, closed = -1.0, "liquidated", True
    else:
        net = fa.net_short(exits, price, slip, funding, int(ts[a]), end_ts)
    return {"entry": price, "status": reason if closed else "open", "net": net,
            "exit_ts": end_ts if closed else None, "half_ts": half_ts}


def book_rule(decisions: list[dict], prices: PricesFn, fills: dict, rule: ex.Rule, newest_hour: int,
              name: str, label: str, side: int = 1, funding: Optional[FundingFn] = None,
              slips: Optional[dict] = None, caps: Optional[dict] = None) -> Book:
    """A $100 paper book of decisions: 20% of the equity a pick, at most 5 open, a coin once at a time;
    ``caps``: a source's most open positions at once (the account's exploration)."""
    cash, held = ex.START_USD, []
    out = Book(name, cash, cash, label=label, side=side, half_pct=rule.half_pct, hours=rule.hours)
    for d in sorted(decisions, key=lambda d: d["moment"]):
        for p in [p for p in held if p.exit_ts is not None and p.exit_ts <= d["moment"]]:
            cash += p.size * (1 + p.net)
            held.remove(p)
        for pick in d["picks"]:
            key = f"{d['moment']}:{pick['base']}"
            if len(held) >= ex.MAX_OPEN or any(p.base == pick["base"] for p in held):
                out.skipped += 1
                continue
            cap = (caps or {}).get(pick.get("_source"))
            if cap is not None and sum(p.source == pick["_source"] for p in held) >= cap:
                out.skipped += 1
                continue
            size = ex.SHARE * (cash + sum(p.size for p in held))
            if size > cash:
                out.skipped += 1
                continue
            slip = (slips or {}).get(key, float(slippage_pct(np.array([pick["turnover"]]))[0]) / 100.0)
            if "_rule" in pick:
                rule_p, side_p = pick["_rule"], pick["_side"]
            else:
                rule_p, side_p = se.RULES[pick["action"]] if "action" in pick else (rule, side)
            f = follow(prices(pick["symbol"], d["moment"]), d["moment"], fills.get(key), pick["sigma_day"], rule_p,
                       newest_hour, slip, side_p, funding(pick["base"]) if funding and side_p < 0 else None)
            if f is None:
                out.skipped += 1
                continue
            cash -= size
            pos = Position(d["moment"], pick["base"], pick["symbol"], size, f["entry"], f["status"], f["net"],
                           f["exit_ts"], f["half_ts"], d["moment"] + rule_p.hours * HOUR_MS, side_p,
                           pick.get("_source", label), rule_p, float(pick["sigma_day"]))
            out.positions.append(pos)
            held.append(pos)
    for p in [p for p in held if p.exit_ts is not None]:
        cash += p.size * (1 + p.net)
        held.remove(p)
    out.cash = cash
    out.value = cash + sum(p.size * (1 + p.net) for p in held)
    return out


def book(decisions: list[dict], prices: PricesFn, fills: dict, variant: str, newest_hour: int,
         slips: Optional[dict] = None) -> Book:
    """One of the panic's two books."""
    return book_rule(decisions, prices, fills, VARIANTS[variant], newest_hour, variant, VARIANT_RU[variant],
                     slips=slips)


SELECTOR_LABEL = "модель выбора"
ACCOUNT = "account"
ACCOUNT_LABEL = "Счёт Разгон"
# the selector trades thousands of times a year and its forecasts do not hold yet (2018-2026 walked forward):
# in the account it is exploration, one position at a time, so it never crowds the panic out
ACCOUNT_CAPS = {SELECTOR_LABEL: 1}


def account_book(sources: list, prices: PricesFn, fills: dict, newest_hour: int, start_ms: int,
                 funding: Optional[FundingFn] = None) -> Book:
    """The one $100 account: ``sources`` = [(label, decisions, rule_of)] in priority order, rule_of(pick) ->
    (rule, side). Only decisions from ``start_ms`` count; within a moment the first source goes first."""
    merged: dict[int, list] = {}
    for prio, (label, decisions, rule_of) in enumerate(sources):
        for d in decisions:
            if d["moment"] < start_ms:
                continue
            for pick in d["picks"]:
                r, s = rule_of(pick)
                key = (prio, -float(pick.get("forecast", pick["turnover"])))
                merged.setdefault(d["moment"], []).append((key, {**pick, "_rule": r, "_side": s, "_source": label}))
    decisions = [{"moment": m, "picks": [p for _, p in sorted(ps, key=lambda x: x[0])]}
                 for m, ps in sorted(merged.items())]
    return book_rule(decisions, prices, fills, VARIANTS["nostop"], newest_hour, ACCOUNT, ACCOUNT_LABEL, 1, funding,
                     caps=ACCOUNT_CAPS)


def selector_book(decisions: list[dict], prices: PricesFn, fills: dict, newest_hour: int,
                  funding: Optional[FundingFn] = None) -> Book:
    """The selector's $100: every pick with its own action (side and exit)."""
    b = book_rule(decisions, prices, fills, se.RULES["L24"][0], newest_hour, "selector", SELECTOR_LABEL, 1, funding)
    b.half_pct = 0.05                         # its only half is the 3-day action's +5%
    return b


def strategy_book(strategy: dict, decisions: list[dict], prices: PricesFn, fills: dict, newest_hour: int,
                  funding: Optional[FundingFn] = None) -> Book:
    g = fa.Genome.from_dict(strategy["genome"])
    return book_rule(decisions, prices, fills, g.rule(), newest_hour, strategy["id"], label(strategy), g.side,
                     funding)


# -- telling -----------------------------------------------------------------------------------

def berlin(ms: int) -> pd.Timestamp:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").tz_convert("Europe/Berlin")


def quiet(now_ms: int) -> bool:
    return QUIET_HOURS[0] <= berlin(now_ms).hour < QUIET_HOURS[1]


def _price(x: float) -> str:
    if x >= 100:
        return f"${x:,.2f}"
    if x >= 1:
        return f"${x:.3f}"
    return f"${x:.{max(4, 3 - math.floor(math.log10(x)))}f}" if x > 0 else "$0"


def _due(d: dict, hours: int) -> str:
    due = berlin(d["moment"] + hours * HOUR_MS)
    return f"{WEEKDAYS[due.dayofweek]} {due:%d.%m %H:%M}"


def label(strategy: dict) -> str:
    return f"фабрика · {fa.GOALS[strategy['goal']]['title']} · {strategy['id'][-4:]}"


def card(d: dict) -> str:
    """The message for a new mass moment."""
    start = berlin(d["hour"])
    lines = [f"🌊 Массовая паника — учебная сделка (бумага, деньги не двигаются)",
             f"Час {start:%H:%M}–{berlin(d['moment']):%H:%M} (Берлин): в панике сразу {d['size']} монет, BTC "
             f"{100 * d['btc_r1']:+.1f}% за час, рынок нервный.",
             f"Бумажный счёт берёт {len(d['picks'])} самые ликвидные, по 20% счёта (≈ $20 из $100):"]
    for p in d["picks"]:
        px = p["close"]
        lines.append(f"• {p['base']} ≈ {_price(px)} → лимитка на продажу половины {_price(px * 1.05)} (+5%); "
                     f"аварийный стоп {_price(px * 0.7)} (−30%)")
    lines += ["Когда лимитка сработала — стоп на остаток = цена входа.",
              f"Остаток закрыть: {_due(d, VARIANTS['nostop'].hours)} (через 3 дня).",
              f"История 2025–2026: в плюс ~{EXPECT['nostop']['up_of10'][0]:.1f} из 10, в среднем "
              f"+${EXPECT['nostop']['usd_per_100'][0] * 0.2:.2f}…{EXPECT['nostop']['usd_per_100'][1] * 0.2:.2f} на сделку "
              f"в $20 (без стопа). Зайти через час — примерно вдвое меньше, через 6 ч — ещё меньше, но в плюсе.",
              "Это проверка на бумаге: решает только то, как пройдут такие сделки дальше."]
    return "\n".join(lines)


def strategy_card(d: dict, strategy: dict) -> str:
    """The message for a factory candidate's new decision: what to do in prices."""
    g = fa.Genome.from_dict(strategy["genome"])
    start = berlin(d["hour"])
    side = "купить (спот)" if g.side > 0 else "шорт (фьючерс, маржа 1x)"
    room = d["picks"][:ex.MAX_OPEN]
    lines = [f"🏭 {label(strategy)} — учебная сделка (бумага, деньги не двигаются)",
             f"Стратегия: {strategy['title']}",
             f"Час {start:%H:%M}–{berlin(d['moment']):%H:%M} (Берлин): подходят {d['size']} монет; бумажный счёт "
             f"этой стратегии ($100) берёт до {len(room)}, по 20% счёта, самые ликвидные первыми"
             + (f" (ещё {len(d['picks']) - len(room)} — нет места: больше {ex.MAX_OPEN} сделок сразу счёт не держит)"
                if len(d["picks"]) > len(room) else "") + ":"]
    for p in room:
        px, steps = p["close"], []
        dist = ex._stop_dist(g.rule(), p["sigma_day"])
        if g.half_pct:
            steps.append(f"половину закрыть на {_price(px * (1 + g.side * g.half_pct))} "
                         f"({'+' if g.side > 0 else '−'}{100 * g.half_pct:.0f}%), остаток — стоп на цену входа")
        if dist:
            steps.append(f"стоп {_price(px * (1 - g.side * dist))}")
        if g.take and dist:
            steps.append(f"всё забрать на {_price(px * (1 + g.side * g.take * dist))}")
        if g.keep:
            steps.append(f"от +5% держать {100 * g.keep:.0f}% лучшего")
        lines.append(f"• {p['base']} ≈ {_price(px)} → {side}" + ("; " + "; ".join(steps) if steps else ""))
    ys = strategy.get("years", {})
    lines += [f"Закрыть остаток: {_due(d, g.hours)} (через {g.hours} ч).",
              "История фабрики: " + "; ".join(
                  f"{y}: рост {ys[y]['growth_pct']:+.0f}%, просадка {ys[y]['drawdown_pct']:.0f}%"
                  for y in ("2025", "2026") if y in ys) + ".",
              "Это проверка на бумаге: фабрика нашла это на истории, решает живой рынок."]
    return "\n".join(lines)


def selector_card(d: dict) -> str:
    """The message for the selector's new decision: each coin's action in prices and its forecast."""
    start = berlin(d["hour"])
    room = d["picks"][:ex.MAX_OPEN]
    lines = [f"🧠 Модель выбора — учебная сделка (бумага, деньги не двигаются)",
             f"Час {start:%H:%M}–{berlin(d['moment']):%H:%M} (Берлин), режим «{rg.ru(d['regime'])}»: события у "
             f"{d['size']} монет; модель (обучена {d.get('model', '?')}) берёт до {len(room)}, по 20% счёта, лучшие "
             f"прогнозы первыми:"]
    for p in room:
        rule, side = se.RULES[p["action"]]
        px, steps = p["close"], []
        dist = ex._stop_dist(rule, p["sigma_day"])
        if rule.half_pct:
            steps.append(f"половину закрыть на {_price(px * (1 + side * rule.half_pct))}, остаток — стоп на цену входа")
        if dist:
            steps.append(f"стоп {_price(px * (1 - side * dist))}")
        if rule.take and dist:
            steps.append(f"цель {_price(px * (1 + side * rule.take * dist))}")
        lines.append(f"• {p['base']} ≈ {_price(px)} → {se.ACTION_RU[p['action']]} (прогноз {100 * p['forecast']:+.1f}%)"
                     + ("; " + "; ".join(steps) if steps else "") + f"; закрыть до {_due(d, rule.hours)}")
    lines += ["История модели: в 2022–2024 не умела (минус), в 2025–2026 её прогнозы начали сбываться. "
              "Бумага покажет, учится ли она вживую."]
    return "\n".join(lines)


def news(books: dict[str, Book], sent: set[str]) -> tuple[list[str], set[str]]:
    """Lines for halves taken and trades closed since the last message, and their keys."""
    lines, keys = [], set()
    for b in books.values():
        own = "" if b.variant in VARIANTS else f"{b.variant}:"     # the panic's two books share their halves
        for p in b.positions:
            k_half = f"{own}{p.moment}:{p.base}:half"
            if p.half_ts and k_half not in sent and k_half not in keys:
                keys.add(k_half)
                half = (p.rule.half_pct if p.rule is not None and p.rule.half_pct else b.half_pct) or 0.05
                gain = 100 * half
                at = p.entry * (1 + p.side * half)
                lines.append(f"✂️ {p.base} ({b.label}): {gain:+.0f}% в нашу сторону — половина закрыта на бумаге по "
                             f"{_price(at)}, стоп остатка → {_price(p.entry)} ({berlin(p.half_ts):%d.%m %H:%M})")
            k_done = f"{p.moment}:{p.base}:{b.variant}:closed"
            if p.status != "open" and k_done not in sent:
                keys.add(k_done)
                word = {"stop": "стоп", "breakeven": "остаток по цене входа", "time": "срок вышел",
                        "target": "цель", "keep": "трейлинг", "liquidated": "ликвидация (маржа 1x)"}.get(p.status,
                                                                                                  p.status)
                lines.append(f"🏁 {p.base} ({b.label}) закрыта: {word}, итог "
                             f"{100 * p.net:+.1f}% = {p.size * p.net:+.2f} $ на {p.size:.0f} $")
    return lines, keys


def summary(b: Book) -> dict:
    closed = [p for p in b.positions if p.status != "open"]
    last = sorted(closed, key=lambda p: p.exit_ts or 0, reverse=True)[:20]
    return {"label": b.label, "value": b.value, "cash": b.cash, "trades": len(b.positions), "closed": len(closed),
            "open": [asdict_position(p) for p in b.positions if p.status == "open"], "skipped": b.skipped,
            "last": [asdict_position(p) for p in last],
            "up_of10": 10 * float(np.mean([p.net > 0 for p in closed])) if closed else float("nan"),
            "usd_per_100": 100 * float(np.mean([p.net for p in closed])) if closed else float("nan")}


def levels(p: Position) -> dict:
    """A position's stop, half and target in prices."""
    out = {}
    if p.rule is None:
        return out
    dist = ex._stop_dist(p.rule, p.sigma_day) if math.isfinite(p.sigma_day) or p.rule.stop is None else None
    if dist:
        out["stop"] = p.entry * (1 - p.side * dist)
    if p.rule.half_pct:
        out["half"] = p.entry * (1 + p.side * p.rule.half_pct)
    if p.rule.take and dist:
        out["target"] = p.entry * (1 + p.side * p.rule.take * dist)
    return out


def asdict_position(p: Position) -> dict:
    return {"moment": p.moment, "base": p.base, "side": p.side, "source": p.source, "size": round(p.size, 2),
            "entry": p.entry, "net_pct": round(100 * p.net, 2), "pnl_usd": round(p.size * p.net, 2),
            "half_sold": p.half_ts is not None, "status": p.status, "due": f"{berlin(p.due_ts):%d.%m %H:%M}",
            "exit": f"{berlin(p.exit_ts):%d.%m %H:%M}" if p.exit_ts else None,
            **{k: round(v, 8) for k, v in levels(p).items()}}


def entry_lines(b: Book, sent: set[str]) -> tuple[list[str], set[str]]:
    """The account's new positions, with what to do in prices."""
    lines, keys = [], set()
    for p in b.positions:
        key = f"{p.moment}:{p.base}:{b.variant}:open"
        if key in sent:
            continue
        keys.add(key)
        lv = levels(p)
        steps = []
        if "half" in lv:
            steps.append(f"половину закрыть на {_price(lv['half'])}, остаток — стоп на цену входа")
        if "stop" in lv:
            steps.append(f"стоп {_price(lv['stop'])}")
        if "target" in lv:
            steps.append(f"цель {_price(lv['target'])}")
        side = "лонг (спот)" if p.side > 0 else "шорт (фьючерс 1x)"
        due = berlin(p.due_ts)
        lines.append(f"🟢 {p.base}: {side} на ${p.size:.2f} по ≈ {_price(p.entry)} · {p.source}"
                     + ("; " + "; ".join(steps) if steps else "")
                     + f"; закрыть до {WEEKDAYS[due.dayofweek]} {due:%d.%m %H:%M}")
    return lines, keys


SPARK = "▁▂▃▄▅▆▇█"


def spark(values: list[float]) -> str:
    """A line of the account's equity, low to high."""
    v = [x for x in values if x is not None and math.isfinite(x)]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    if hi - lo < 1e-9:
        return SPARK[3] * len(v)
    return "".join(SPARK[int((x - lo) / (hi - lo) * (len(SPARK) - 1))] for x in v)


def day_text(account: Book, books: dict[str, Book], equity: list[dict], now_ms: int, start_ms: int) -> str:
    """The account's day at 21:00: the balance, today's trades, what is open, and the unseen books."""
    s = summary(account)
    day0 = int(berlin(now_ms).normalize().tz_convert("UTC").value // 10**6)
    closed_today = [p for p in account.positions if p.exit_ts and p.exit_ts >= day0]
    opened_today = [p for p in account.positions if p.moment >= day0]
    values = [e["books"].get(ACCOUNT) for e in equity]
    lines = [f"📒 {ACCOUNT_LABEL} — итог дня {berlin(now_ms):%d.%m} (бумага)",
             f"Баланс ${s['value']:.2f} из $100 ({s['value'] - 100:+.2f} $ с {berlin(start_ms):%d.%m %H:%M}); "
             f"свободно ${s['cash']:.2f}. {spark(values[-48:])}",
             f"Сегодня: открыто {len(opened_today)}, закрыто {len(closed_today)}"
             + (f" на {sum(p.size * p.net for p in closed_today):+.2f} $" if closed_today else "") + "."]
    for p in s["open"]:
        lines.append(f"• {p['base']} {'лонг' if p['side'] > 0 else 'шорт'} ${p['size']:.0f}: {p['net_pct']:+.1f}% "
                     f"({p['pnl_usd']:+.2f} $)" + (", половина закрыта" if p["half_sold"] else "")
                     + (f", стоп {_price(p['stop'])}" if "stop" in p else "") + f", до {p['due']} · {p['source']}")
    if not s["open"] and not s["trades"]:
        lines.append("Сделок пока не было: рынок спокойный, ни одна стратегия не нашла входа. Так и задумано.")
    others = [f"{b.label} ${b.value:.2f}" for v, b in books.items() if v != ACCOUNT]
    if others:
        lines.append("Отдельные счета стратегий (по $100): " + "; ".join(others) + ".")
    return "\n".join(lines)


def report_text(books: dict[str, Book], decisions: list[dict], since_ms: int, now_ms: int,
                title: str = "неделя", strategies: Optional[dict] = None) -> str:
    """The week: the panic's books against history, then every factory candidate on paper."""
    fresh = [d for d in decisions if d["moment"] >= since_ms]
    lines = [f"🌊 Дозорный — {title} (бумага)",
             f"Массовых моментов паники: {len(fresh)} за {title}, {len(decisions)} всего."]
    if not decisions:
        lines.append("Паники не было — бот молчал, так и задумано. Её счета по $100 не тронуты.")
    for variant, b in books.items():
        s = summary(b)
        if variant in VARIANTS:
            if not decisions:
                continue
            e = EXPECT[variant]
            expect = (f" История 2025–2026: {e['up_of10'][0]:.1f}–{e['up_of10'][1]:.1f} из 10, "
                      f"{e['usd_per_100'][0] * 0.2:+.2f}…{e['usd_per_100'][1] * 0.2:+.2f} $ на $20.")
        else:
            st = (strategies or {}).get(variant, {})
            ys = st.get("years", {})
            expect = (" История фабрики: " + "; ".join(f"{y} {ys[y]['growth_pct']:+.0f}%" for y in ("2025", "2026")
                                                         if y in ys) + ".") if ys else ""
        lines.append(f"• {b.label}: счёт ${s['value']:.2f} из $100; закрыто {s['closed']} сделок"
                     + (f", в плюс {s['up_of10']:.1f} из 10, в среднем {s['usd_per_100'] * 0.2:+.2f} $ на $20"
                        if s["closed"] else "")
                     + f"; открыто {len(s['open'])}.{expect}")
        for p in s["open"]:
            lines.append(f"   открыта {p['base']}: {p['net_pct']:+.1f}%"
                         + (", половина закрыта" if p["half_sold"] else "") + f", закрыть до {p['due']}")
    if sum(summary(b)["closed"] for b in books.values()) < 20:
        lines.append("Сделок пока мало: выводы — после 20 закрытых, до того это наблюдение, не оценка.")
    return "\n".join(lines)
