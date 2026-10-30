"""Stage 3 of the watcher: the mass panic on live prices, on paper, cards to Vlad.

Every hour, a minute after it closes (scripts/panic_watch.py hour, deploy/rbry-panic.timer):
1. see: every trading pair is read from market.db (45 days of hours, kept fresh by
   market-hourly) and, when it turns over $5M a day, its newest closed hours from
   the klines API; BTC for 400 days (the nervous market); open interest from the
   futures API only for coins already 5% down in 4 hours. The four panic events
   are found with the library's own code and cooldowns (core/brain/events.py), so
   the watcher sees what history saw;
2. decide: a mass moment (4+ coins in panic in a nervous market) takes the 3
   most liquid; the moment, every coin in panic and the picks go into
   decisions.jsonl before any outcome exists, never changed;
3. book: two $100 books replay every decision on hourly prices with the exits of
   core/brain/exits.py: half out at +5%, the rest at the entry or after 3 days;
   one without a stop, one with a disaster stop at -30%. Entry: the open of the
   5-minute candle 10 minutes after the hour opened (Vlad's reaction), until it
   exists the hour's open;
4. tell: a card for a new moment, a line when a half is sold or a trade closes;
   from 00:00 to 08:00 Berlin without sound.
No money moves; messages go to the owner only.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.brain import events as ev
from core.brain import exits as ex
from core.brain.dataset import DAY_MS, slippage_pct
from core.sandbox.market_store import HOUR_MS, History, MarketStore, PEG_MEDIAN_MOVE

WATCH_DIR = Path(__file__).resolve().parents[2] / "runtime" / "learning" / "watch"
MASS = ex.MASS["mass4"]
PICKS = ex.PICKS
HISTORY_DAYS = 45
BTC_DAYS = 400
OI_FIRST = -0.05                    # open interest is fetched only for coins this far down in 4 hours
TAIL_WORKERS = 8
REACTION_MS = ev.REACTION_MS
VARIANTS = {"nostop": ex.BY_RULE["H72-HALF5"], "stop30": ex.BY_RULE["H72-HALF5-S30"]}
VARIANT_RU = {"nostop": "без стопа", "stop30": "со стопом −30%"}
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


def btc_now(store: MarketStore, hour_ms: int, tail: Optional[TailFn]) -> pd.DataFrame:
    return ev.btc_context(ev.grid(_history("BTCUSDT", "BTC", hours(store, "BTCUSDT", hour_ms, BTC_DAYS, tail))))


def coin_panic(frame: pd.DataFrame, symbol: str, base: str, hour_ms: int, btc: pd.DataFrame,
               oi: Optional[pd.DataFrame] = None) -> tuple[list[str], dict]:
    """(the panic events of the hour starting at ``hour_ms``, the coin's numbers then), as on history."""
    g = ev.grid(_history(symbol, base, frame))
    if not len(g) or int(g["ts"].iloc[-1]) != hour_ms or len(g) < ev.MIN_HOURS + 2:
        return [], {}
    x = ev.features(g, None, oi, btc)
    usual = (x["turnover"].to_numpy() >= ev.MIN_TURNOVER) & x["sigma"].notna().to_numpy()
    last = len(x) - 1
    found = []
    for e in ex.PANIC_EVENTS:
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
                   "sigma_day": ex.daily_sigma(closes, len(closes)), "usual": bool(usual[last])}


@dataclass
class Seen:
    hour_ms: int
    nervous: bool
    btc_r1: float
    checked: int
    panic: list = field(default_factory=list)       # {base, symbol, events, turnover, r1, r24, close, sigma_day}


def see(store: MarketStore, hour_ms: int, pairs: list[tuple[str, str]], tail: Optional[TailFn], oi: OiFn,
        perp_of: dict[str, str], stocks: set[str] = frozenset()) -> Seen:
    """Every coin in panic in the hour starting at ``hour_ms``."""
    btc = btc_now(store, hour_ms, tail)
    at = btc[btc["ts"] == hour_ms]
    nervous = bool(at["nervous"].iloc[0]) if len(at) else False
    seen = Seen(hour_ms, nervous, float(at["btc_r1"].iloc[0]) if len(at) else float("nan"), 0)
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
    for (symbol, base, _), frame in zip(liquid, frames):
        seen.checked += 1
        found, nums = coin_panic(frame, symbol, base, hour_ms, btc)
        perp = perp_of.get(base)
        if nums and nums["usual"] and perp and math.isfinite(nums["r4"]) and nums["r4"] <= OI_FIRST:
            interest = oi(perp, hour_ms)
            if interest is not None and len(interest):
                found, nums = coin_panic(frame, symbol, base, hour_ms, btc, interest)
        if found:
            seen.panic.append({"base": base, "symbol": symbol, "events": found, **nums})
    return seen


# -- deciding ---------------------------------------------------------------------------------

def decide(seen: Seen, decided_ms: int) -> Optional[dict]:
    """A mass moment in a nervous market: the 3 most liquid coins in panic; else None."""
    if not seen.nervous or len(seen.panic) < MASS:
        return None
    picks = sorted(seen.panic, key=lambda p: -p["turnover"])[:PICKS]
    return {"moment": seen.hour_ms + HOUR_MS, "hour": seen.hour_ms, "decided_ms": decided_ms,
            "decided_at": pd.Timestamp(decided_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "nervous": seen.nervous, "btc_r1": seen.btc_r1, "size": len(seen.panic),
            "panic": [{k: p[k] for k in ("base", "symbol", "events", "turnover", "r1", "r24")} for p in seen.panic],
            "picks": [{k: p[k] for k in ("base", "symbol", "turnover", "close", "sigma_day")} for p in picks]}


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


@dataclass
class Position:
    moment: int
    base: str
    symbol: str
    size: float
    entry: float
    status: str                  # open | stop | breakeven | time | ...
    net: float                   # closed: the result; open: if closed at the newest close
    exit_ts: Optional[int]
    half_ts: Optional[int]       # when half was sold at +5%
    due_ts: int                  # when the rest is closed at the latest


@dataclass
class Book:
    variant: str
    cash: float
    value: float
    positions: list = field(default_factory=list)
    skipped: int = 0


def follow(prices: np.ndarray, moment: int, entry: Optional[float], sigma_day: float, rule: ex.Rule,
           newest_hour: int, slip: float) -> Optional[dict]:
    """One pick under one rule up to the newest closed hour: closed, or open and marked at the newest close."""
    if not len(prices):
        return None
    ts = prices[:, 0]
    a = int(np.searchsorted(ts, moment))
    if a >= len(ts) or ts[a] != moment:
        return None
    b = int(np.searchsorted(ts, min(moment + rule.hours * HOUR_MS, newest_hour + HOUR_MS)))
    o, h, l, c = (np.asarray(prices[a:b, k], dtype=float) for k in (1, 2, 3, 4))
    price = float(entry) if entry else float(o[0])
    dist = rule.stop_pct if rule.stop is None else min(rule.stop * sigma_day, 0.9)
    exits, held, reason, _ = ex.walk_long(o, h, l, c, price, dist, rule)
    half_at = price * (1 + rule.half_pct) if rule.half_pct else np.inf
    k_half = int(np.argmax(h >= half_at)) if (h >= half_at).any() else None
    half_ts = int(ts[a + k_half]) + HOUR_MS if len(exits) == 2 and k_half is not None else None
    over = moment + rule.hours * HOUR_MS <= newest_hour + HOUR_MS
    if reason != "time" or over:
        return {"entry": price, "status": reason, "net": ex.net_of(exits, price, slip),
                "exit_ts": int(ts[a + held - 1]) + HOUR_MS, "half_ts": half_ts}
    marked = exits[:-1] + [(exits[-1][0], float(c[-1]))]
    return {"entry": price, "status": "open", "net": ex.net_of(marked, price, slip), "exit_ts": None,
            "half_ts": half_ts}


def book(decisions: list[dict], prices: PricesFn, fills: dict, variant: str, newest_hour: int,
         slips: Optional[dict] = None) -> Book:
    """A $100 paper book of every decision: 20% of the equity a pick, at most 5 open, a coin once at a time."""
    rule = VARIANTS[variant]
    cash, held = ex.START_USD, []
    out = Book(variant, cash, cash)
    for d in sorted(decisions, key=lambda d: d["moment"]):
        for p in [p for p in held if p.exit_ts is not None and p.exit_ts <= d["moment"]]:
            cash += p.size * (1 + p.net)
            held.remove(p)
        for pick in d["picks"]:
            key = f"{d['moment']}:{pick['base']}"
            if len(held) >= ex.MAX_OPEN or any(p.base == pick["base"] for p in held):
                out.skipped += 1
                continue
            size = ex.SHARE * (cash + sum(p.size for p in held))
            if size > cash:
                out.skipped += 1
                continue
            slip = (slips or {}).get(key, float(slippage_pct(np.array([pick["turnover"]]))[0]) / 100.0)
            f = follow(prices(pick["symbol"], d["moment"]), d["moment"], fills.get(key), pick["sigma_day"], rule,
                       newest_hour, slip)
            if f is None:
                out.skipped += 1
                continue
            cash -= size
            pos = Position(d["moment"], pick["base"], pick["symbol"], size, f["entry"], f["status"], f["net"],
                           f["exit_ts"], f["half_ts"], d["moment"] + rule.hours * HOUR_MS)
            out.positions.append(pos)
            held.append(pos)
    for p in [p for p in held if p.exit_ts is not None]:
        cash += p.size * (1 + p.net)
        held.remove(p)
    out.cash = cash
    out.value = cash + sum(p.size * (1 + p.net) for p in held)
    return out


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


def card(d: dict) -> str:
    """The message for a new mass moment."""
    start, due = berlin(d["hour"]), berlin(d["moment"] + VARIANTS["nostop"].hours * HOUR_MS)
    lines = [f"🌊 Массовая паника — учебная сделка (бумага, деньги не двигаются)",
             f"Час {start:%H:%M}–{berlin(d['moment']):%H:%M} (Берлин): в панике сразу {d['size']} монет, BTC "
             f"{100 * d['btc_r1']:+.1f}% за час, рынок нервный.",
             f"Бумажный счёт берёт {len(d['picks'])} самые ликвидные, по 20% счёта (≈ $20 из $100):"]
    for p in d["picks"]:
        px = p["close"]
        lines.append(f"• {p['base']} ≈ {_price(px)} → лимитка на продажу половины {_price(px * 1.05)} (+5%); "
                     f"аварийный стоп {_price(px * 0.7)} (−30%)")
    lines += ["Когда лимитка сработала — стоп на остаток = цена входа.",
              f"Остаток закрыть: {WEEKDAYS[due.dayofweek]} {due:%d.%m %H:%M} (через 3 дня).",
              f"История 2025–2026: в плюс ~{EXPECT['nostop']['up_of10'][0]:.1f} из 10, в среднем "
              f"+${EXPECT['nostop']['usd_per_100'][0] * 0.2:.2f}…{EXPECT['nostop']['usd_per_100'][1] * 0.2:.2f} на сделку "
              f"в $20 (без стопа). Зайти через час — примерно вдвое меньше, через 6 ч — ещё меньше, но в плюсе.",
              "Это проверка на бумаге: решает только то, как пройдут такие сделки дальше."]
    return "\n".join(lines)


def news(books: dict[str, Book], sent: set[str]) -> tuple[list[str], set[str]]:
    """Lines for halves sold and trades closed since the last message, and their keys."""
    lines, keys = [], set()
    for variant, b in books.items():
        for p in b.positions:
            k_half = f"{p.moment}:{p.base}:half"
            if p.half_ts and k_half not in sent and k_half not in keys:
                keys.add(k_half)
                lines.append(f"✂️ {p.base}: +5% — половина продана на бумаге по {_price(p.entry * 1.05)}, стоп "
                             f"остатка → {_price(p.entry)} ({berlin(p.half_ts):%d.%m %H:%M})")
            k_done = f"{p.moment}:{p.base}:{variant}:closed"
            if p.status != "open" and k_done not in sent:
                keys.add(k_done)
                word = {"stop": "аварийный стоп", "breakeven": "остаток по цене входа", "time": "3 дня прошли",
                        "target": "цель", "keep": "трейлинг"}.get(p.status, p.status)
                lines.append(f"🏁 {p.base} ({VARIANT_RU[variant]}) закрыта: {word}, итог "
                             f"{100 * p.net:+.1f}% = {p.size * p.net:+.2f} $ на {p.size:.0f} $")
    return lines, keys


def summary(b: Book) -> dict:
    closed = [p for p in b.positions if p.status != "open"]
    return {"value": b.value, "cash": b.cash, "trades": len(b.positions), "closed": len(closed),
            "open": [asdict_position(p) for p in b.positions if p.status == "open"], "skipped": b.skipped,
            "up_of10": 10 * float(np.mean([p.net > 0 for p in closed])) if closed else float("nan"),
            "usd_per_100": 100 * float(np.mean([p.net for p in closed])) if closed else float("nan")}


def asdict_position(p: Position) -> dict:
    return {"moment": p.moment, "base": p.base, "size": round(p.size, 2), "entry": p.entry,
            "net_pct": round(100 * p.net, 2), "half_sold": p.half_ts is not None,
            "due": f"{berlin(p.due_ts):%d.%m %H:%M}"}


def report_text(books: dict[str, Book], decisions: list[dict], since_ms: int, now_ms: int,
                title: str = "неделя") -> str:
    fresh = [d for d in decisions if d["moment"] >= since_ms]
    lines = [f"🌊 Дозорный паники — {title} (бумага)",
             f"Массовых моментов: {len(fresh)} за {title}, {len(decisions)} всего."]
    if not decisions:
        lines.append("Паники не было — бот молчал, так и задумано. Счета по $100 не тронуты.")
        return "\n".join(lines)
    for variant, b in books.items():
        s = summary(b)
        e = EXPECT[variant]
        lines.append(f"• {VARIANT_RU[variant]}: счёт ${s['value']:.2f} из $100; закрыто {s['closed']} сделок"
                     + (f", в плюс {s['up_of10']:.1f} из 10, в среднем {s['usd_per_100'] * 0.2:+.2f} $ на $20"
                        if s["closed"] else "")
                     + f"; открыто {len(s['open'])}. История 2025–2026: {e['up_of10'][0]:.1f}–{e['up_of10'][1]:.1f} "
                     f"из 10, {e['usd_per_100'][0] * 0.2:+.2f}…{e['usd_per_100'][1] * 0.2:+.2f} $ на $20.")
        for p in s["open"]:
            lines.append(f"   открыта {p['base']}: {p['net_pct']:+.1f}%"
                         + (", половина продана" if p["half_sold"] else "") + f", закрыть до {p['due']}")
    if sum(summary(b)["closed"] for b in books.values()) < 20:
        lines.append("Сделок пока мало: выводы — после 20 закрытых, до того это наблюдение, не оценка.")
    return "\n".join(lines)
