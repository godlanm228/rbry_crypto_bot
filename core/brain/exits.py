"""Stage 2 of the watcher: how to get out of a panic buy, and which coins of a panic moment to buy.

PANIC is one signal from four events of the library (core/brain/events.py) that
all said the same in 2018-2026: buy what forced selling pushed down, in a nervous
market only. It was put together after their 2025 and 2026 results were seen, so
history no longer judges PANIC itself cleanly; the paper journal will. Here
history chooses, on 2018-2024 only, how to exit and which coins of one moment to
take, and 2025 and 2026 show whether that choice holds.

A trade is a long on spot, walked hour by hour on the simulator's hourly cache:
entry at the open of the hour after the signal, moved by the event's average
10-minute reaction; within an hour the low comes first; a level the hour opens
through exits at that open; a trailing floor uses the peak of earlier hours.
Stops are in the coin's usual daily moves (its hourly sigma over the week before
the signal times sqrt(24)), so a calm coin and a wild one get stops of the same
meaning. A trade's twin is the same exit rule from random hours of the same coin
and month: what the rule makes without the panic.

A panic moment is one signal hour; its size is how many coins panicked in it,
known at that hour's close. The first run showed the edge in mass
moments (11+ coins: forced selling of the whole market) and none in a lone
coin's panic, while a trade-weighted mean hides that: a mass moment is 30
trades, a lone coin one. So results are also given per moment, and the picks
include mass moments only. Found while reading the results, mass moments are a
hypothesis for the paper journal, not a finding of history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.brain.dataset import FEE_PCT
from core.brain.events import HOUR_MS, LEARN_END_MS, CHECK_END_MS, PERIODS, period
from core.sandbox.evaluate import t_cluster

PANIC_EVENTS = ("FELL-WITH-BTC", "CAPITULATION-TURN", "OI-FLUSH", "DUMP")   # the order breaks ties of one coin-hour
SIGMA_HOURS = 168
TWINS = 2
SHARE = 0.2                 # of the equity a trade: $20 of $100
MAX_OPEN = 5
MAX_DRAWDOWN = 0.5          # losing 50 of 100 is acceptable, not more
START_USD = 100.0


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    hours: int
    stop: Optional[float] = None         # usual daily moves below the entry
    stop_pct: Optional[float] = None     # or this share below it
    take: Optional[float] = None         # everything out this many stop distances above
    half: Optional[float] = None         # half out this many stop distances above, the rest's stop to the entry
    half_pct: Optional[float] = None     # ... or this share above
    keep: Optional[float] = None         # trailing floor: keep this share of the best gain so far
    keep_from: float = 0.05              # ... once the best gain reached this


RULES = (
    Rule("H24", "держать сутки", 24),
    Rule("H72", "держать 3 дня", 72),
    Rule("H24-S2", "сутки, стоп 2 обычных хода", 24, stop=2),
    Rule("H72-S2", "3 дня, стоп 2 обычных хода", 72, stop=2),
    Rule("H72-S3", "3 дня, стоп 3 обычных хода", 72, stop=3),
    Rule("H72-S10", "3 дня, стоп −10%", 72, stop_pct=0.10),
    Rule("H72-S2-T2", "3 дня, стоп 2 хода, всё забрать на +2 стопа", 72, stop=2, take=2),
    Rule("H72-S2-K", "3 дня, стоп 2 хода, от +4% держать половину лучшего", 72, stop=2, keep=0.5, keep_from=0.04),
    Rule("H72-S3-HALF", "3 дня, стоп 3 хода, половину на +1 стоп, остаток без убытка", 72, stop=3, half=1),
    Rule("H72-HALF5", "3 дня без стопа, половину на +5%, остаток без убытка", 72, half_pct=0.05),
    # added after the first runs showed LUNA (05.2022) going to zero under H72-HALF5: a disaster stop only
    Rule("H72-HALF5-S30", "3 дня, половину на +5%, остаток без убытка, аварийный стоп −30%", 72, stop_pct=0.30,
         half_pct=0.05),
    Rule("H168-S3-K", "неделя, стоп 3 хода, от +5% держать 60% лучшего", 168, stop=3, keep=0.6),
    Rule("H168-S2-T3", "неделя, стоп 2 хода, всё забрать на +3 стопа", 168, stop=2, take=3),
)
BY_RULE = {r.id: r for r in RULES}
SELECTIONS = {"all": "все монеты момента", "fell": "3 самых упавших за сутки", "liquid": "3 самых ликвидных",
              "mass4": "моменты от 4 монет, 3 самых ликвидных", "mass11": "моменты от 11 монет, 3 самых ликвидных"}
PICKS = 3
MASS = {"mass4": 4, "mass11": 11}


def _first(mask: np.ndarray) -> int:
    return int(np.argmax(mask)) if mask.any() else len(mask)


def walk_long(o, h, l, c, entry: float, dist: Optional[float], rule: Rule) -> tuple[list, int, str, float]:
    """(exits as [(share, price)], hours held, reason, worst low on the way vs the entry) of one long."""
    n = len(o)
    stop = entry * (1 - dist) if dist is not None else -np.inf
    target = entry * (1 + rule.take * dist) if rule.take and dist else np.inf
    half_at = (entry * (1 + rule.half * dist) if rule.half and dist
               else entry * (1 + rule.half_pct) if rule.half_pct else np.inf)
    trail = np.full(n, -np.inf)
    if rule.keep is not None:
        gain = np.maximum.accumulate(np.r_[entry, h[:-1]]) / entry - 1       # the best gain of the earlier hours
        trail = np.where(gain >= rule.keep_from, entry * (1 + rule.keep * gain), -np.inf)
    floor = np.maximum(trail, stop)
    k_floor, k_half = _first(l <= floor), _first(h >= half_at)
    exits, share, after = [], 1.0, -1
    if k_half < k_floor:                                                       # the low of an hour comes first
        exits.append((0.5, max(float(o[k_half]), half_at)))
        share, after = 0.5, k_half
        floor = np.maximum(floor, entry)
        k_floor = _first((l <= floor) & (np.arange(n) > k_half))
    k_target = _first((h >= target) & (np.arange(n) > after))
    if k_floor < n and k_floor <= k_target:
        k = k_floor
        price = min(float(o[k]), float(floor[k]))
        reason = "breakeven" if after >= 0 and floor[k] == entry else ("keep" if trail[k] > stop else "stop")
    elif k_target < n:
        k, price, reason = k_target, max(float(o[k_target]), target), "target"
    else:
        k, price, reason = n - 1, float(c[-1]), "time"
    exits.append((share, price))
    return exits, k + 1, reason, float(np.min(l[:k + 1]) / entry - 1)


def net_of(exits: list, entry: float, slip: float) -> float:
    fee = FEE_PCT / 100.0
    return sum(s * p * (1 - slip) * (1 - fee) for s, p in exits) / (entry * (1 + slip) * (1 + fee)) - 1


def _window(prices: np.ndarray, entry_ts: int, hours: int, newest: int) -> Optional[tuple]:
    """Hours a..b of a position entered at ``entry_ts``; None when not entered or not over by the newest data.
    A coin that stopped trading inside the window ends at its last close."""
    ts = prices[:, 0]
    a = int(np.searchsorted(ts, entry_ts))
    if a >= len(ts) or ts[a] != entry_ts or entry_ts + hours * HOUR_MS > newest + HOUR_MS:
        return None
    b = int(np.searchsorted(ts, entry_ts + hours * HOUR_MS))
    return (a, b) if b > a else None


def daily_sigma(closes: np.ndarray, a: int) -> float:
    """The coin's usual daily move from the week of hours before index ``a``."""
    lo = max(a - SIGMA_HOURS, 0)
    lr = np.diff(np.log(np.asarray(closes[lo:a], dtype=float)))
    lr = lr[np.isfinite(lr)]
    return float(lr.std() * math.sqrt(24)) if len(lr) >= SIGMA_HOURS // 2 else float("nan")


def panic_trades(rows: pd.DataFrame, market) -> pd.DataFrame:
    """One row per coin and hour of a panic in a nervous market, with what an exit and a pick need."""
    react = {}
    for e in PANIC_EVENTS:
        r = rows[rows["event"] == e]
        g, x = r["g4_0"].to_numpy(dtype=float), r["react4"].to_numpy(dtype=float)
        delta = (1 + g) / (1 + g + x) - 1                                      # the entry 10 minutes later
        react[e] = float(np.nanmean(delta)) if np.isfinite(delta).any() else 0.0
    p = rows[rows["event"].isin(PANIC_EVENTS) & (rows["nervous"] == True)    # noqa: E712 (object column)
             & rows["history"].isin(set(market.base_of))].copy()           # the hourly cache: 60+ days of trading
    p["order"] = p["event"].map({e: i for i, e in enumerate(PANIC_EVENTS)})
    p = p.sort_values(["signal_ts", "order"]).drop_duplicates(["history", "signal_ts"]).reset_index(drop=True)
    p["delta"] = p["event"].map(react)
    fell, sigma = np.full(len(p), np.nan), np.full(len(p), np.nan)
    for i, (hist, ts) in enumerate(zip(p["history"], p["signal_ts"])):
        prices = market.prices(hist)
        a = int(np.searchsorted(prices[:, 0], ts))
        if 25 <= a <= len(prices):
            fell[i] = prices[a - 1, 4] / prices[a - 25, 4] - 1
            sigma[i] = daily_sigma(prices[:, 4], a)
    p["fell24"], p["sigma_day"] = fell, sigma
    p["moment_size"] = p.groupby("signal_ts")["history"].transform("size")
    return p[np.isfinite(p["sigma_day"])].reset_index(drop=True)


def picked(trades: pd.DataFrame, how: str) -> np.ndarray:
    """Which trades of each panic moment (one signal hour) are taken."""
    if how == "all":
        return np.ones(len(trades), dtype=bool)
    key = trades["fell24"] if how == "fell" else -trades["turnover"]
    top = (key.groupby(trades["signal_ts"]).rank(method="first") <= PICKS).to_numpy()
    return top & (trades["moment_size"].to_numpy() >= MASS[how]) if how in MASS else top


def _stop_dist(rule: Rule, sigma_day: float) -> Optional[float]:
    if rule.stop is not None:
        return min(rule.stop * sigma_day, 0.9)
    return rule.stop_pct


def run_rule(trades: pd.DataFrame, market, rule: Rule, newest: int, seed: int = 7, delay: int = 0,
             twins: bool = True) -> pd.DataFrame:
    """Every trade under one exit rule: net, hours held, reason, worst on the way, and its twins' mean net.
    ``delay``: entered that many hours after the signal (at that hour's open, no 10-minute reaction)."""
    rng = np.random.default_rng(seed)
    out = []
    for t in trades.itertuples(index=False):
        prices = market.prices(t.history)
        w = _window(prices, int(t.signal_ts) + delay * HOUR_MS, rule.hours, newest)
        if w is None:
            continue
        a, b = w
        o, h, l, c = (np.asarray(prices[a:b, k], dtype=float) for k in (1, 2, 3, 4))
        entry = float(o[0]) * (1 + (t.delta if not delay else 0.0))
        exits, held, reason, worst = walk_long(o, h, l, c, entry, _stop_dist(rule, t.sigma_day), rule)
        twin = _twin(prices, int(t.signal_ts), rule, newest, t.slip, rng) if twins else float("nan")
        out.append({"history": t.history, "signal_ts": int(t.signal_ts), "net": net_of(exits, entry, t.slip),
                    "held": held + delay, "reason": reason, "worst": worst, "twin": twin})
    return pd.DataFrame(out)


def _twin(prices: np.ndarray, signal_ts: int, rule: Rule, newest: int, slip: float, rng) -> float:
    """The rule's mean net from random hours of the same coin and month."""
    ts = prices[:, 0]
    first = pd.Timestamp(int(signal_ts), unit="ms").to_period("M").start_time
    lo = max(int(np.searchsorted(ts, first.value // 10**6)), SIGMA_HOURS)
    hi = int(np.searchsorted(ts, (first + pd.offsets.MonthBegin(1)).value // 10**6))
    if hi <= lo:
        return float("nan")
    nets = []
    for a in rng.integers(lo, hi, size=TWINS):
        w = _window(prices, int(ts[a]), rule.hours, newest)
        sigma = daily_sigma(prices[:, 4], int(a))
        if w is None or not math.isfinite(sigma):
            continue
        x, y = w
        o, h, l, c = (np.asarray(prices[x:y, k], dtype=float) for k in (1, 2, 3, 4))
        exits, _, _, _ = walk_long(o, h, l, c, float(o[0]), _stop_dist(rule, sigma), rule)
        nets.append(net_of(exits, float(o[0]), slip))
    return float(np.mean(nets)) if nets else float("nan")


def describe(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"n": 0, "up_of10": float("nan"), "usd_per_100": float("nan"), "worst_p10_pct": float("nan"),
                "lost_p10_pct": float("nan"), "excess_pct": float("nan"), "t": float("nan"), "held_h": float("nan"),
                "moments": 0, "moment_usd_per_100": float("nan")}
    ex = (t["net"] - t["twin"]).to_numpy(dtype=float)
    ok = np.isfinite(ex)
    weeks = pd.to_datetime(t["signal_ts"].to_numpy(), unit="ms").strftime("%G-W%V").to_numpy()
    return {"n": len(t), "up_of10": 10 * float((t["net"] > 0).mean()), "usd_per_100": 100 * float(t["net"].mean()),
            "worst_p10_pct": 100 * float(np.quantile(t["worst"], 0.1)),
            "lost_p10_pct": 100 * float(np.quantile(t["net"], 0.1)),
            "excess_pct": 100 * float(ex[ok].mean()) if ok.any() else float("nan"),
            "t": t_cluster(ex[ok], weeks[ok]) if ok.sum() > 1 else float("nan"), "held_h": float(t["held"].mean()),
            "moments": int(t["signal_ts"].nunique()),
            "moment_usd_per_100": 100 * float(t.groupby("signal_ts")["net"].mean().mean())}


def portfolio(t: pd.DataFrame, start_ms: int, end_ms: int, share: float = SHARE, max_open: int = MAX_OPEN,
              start_usd: float = START_USD, by: str = "turnover") -> dict:
    """$100 from ``start_ms``: every trade entered in [start, end) gets ``share`` of the equity while fewer than
    ``max_open`` are open and the coin is not held; equity (open trades at cost) after every close. Within a
    moment the trade with the highest ``by`` goes first (the most liquid, or the best forecast)."""
    order = ["signal_ts", by] if by in t else ["signal_ts"]
    t = t[(t["signal_ts"] >= start_ms) & (t["signal_ts"] < end_ms)].sort_values(
        order, ascending=[True, False][:len(order)], kind="stable")     # within a moment the most liquid first
    cash, open_, curve, taken = start_usd, [], [start_usd], 0
    for r in t.itertuples(index=False):
        for p in sorted([p for p in open_ if p[0] <= r.signal_ts]):
            cash += p[1] * (1 + p[2])
            open_.remove(p)
            curve.append(cash + sum(q[1] for q in open_))
        if len(open_) >= max_open or any(q[3] == r.history for q in open_):
            continue
        size = share * (cash + sum(q[1] for q in open_))
        if size > cash or size <= 0:
            continue
        cash -= size
        open_.append((int(r.signal_ts) + int(r.held) * HOUR_MS, size, float(r.net), r.history))
        taken += 1
    for p in sorted(open_):
        cash += p[1] * (1 + p[2])
        open_.remove(p)
        curve.append(cash + sum(q[1] for q in open_))
    c = np.asarray(curve)
    peak = np.maximum.accumulate(c)
    return {"end_usd": float(c[-1]), "growth_pct": 100 * (float(c[-1]) / start_usd - 1),
            "drawdown_pct": 100 * float(((c - peak) / peak).min()), "trades": taken,
            "years": (end_ms - start_ms) / (365.25 * 86_400_000)}


def yearly(growth_pct: float, years: float) -> float:
    g = 1 + growth_pct / 100
    return 100 * (g ** (1 / years) - 1) if g > 0 and years > 0 else -100.0


def tag(t: pd.DataFrame, hours: int) -> pd.DataFrame:
    return t.assign(period=period(t["signal_ts"].to_numpy(), hours))


SPANS = {PERIODS[0]: (int(pd.Timestamp("2018-01-01").value // 10**6), LEARN_END_MS),
         PERIODS[1]: (LEARN_END_MS, CHECK_END_MS)}
