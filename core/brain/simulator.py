"""Simulator v2: every position follows its real hourly prices.

The first portfolio booked each position as a precomputed outcome and counted
it at cost while open, so drawdowns inside trades were invisible and exits
could not depend on the path. Here a position is walked hour by hour from the
open of the hour after its signal:

- exits: a stop (ATRs from the entry), a target (a multiple of the stop), a
  trailing floor that keeps a share of the best gain seen so far (``keep``,
  e.g. 0.7 = never give back more than 30% of the peak profit, active once the
  gain reached ``keep_from``), a time limit, and for shorts the liquidation of
  1x isolated margin (the margin is gone, no more);
- within one hour the adverse extreme is assumed to come first, an hour that
  opens through a level exits at that open, and the trailing floor only uses
  the peak of earlier hours (the high of an hour may come after its low);
- costs: spot fees for longs, futures fees and funding for shorts, slippage by
  the coin's turnover on the signal day; a short needs a live perpetual and is
  settled at its last price if the perpetual stops;
- the portfolio is valued at every day's close with open positions at market;
- early exits on events, each switched on by its own ``Exit`` field: Binance
  announced the coin's delisting (out at the next hour's open), BTC fell
  ``btc_crash`` in 24 hours (longs out), the shorted coin rose ``squeeze`` in
  24 hours, the held coin fell ``crash`` in 24 hours. A 24-hour move is read
  at an hour's close and acted on at the next hour's open.

Hourly prices come from a cache of one .npy per history (``build_cache``) that
is read memory-mapped, so several worker processes can share it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.brain.dataset import (DAY_MS, ENTRY_WITHIN_HOURS, FEE_PCT, FUTURES_FEE_PCT, PERP_SETTLE_MS,
                                slippage_pct)
from core.brain.portfolio import DRAWDOWN_BRAKE, MAX_POSITIONS, MAX_SHORT_SHARE, POSITION_SHARE, START_USD
from core.sandbox import archive
from core.sandbox.market_store import HOUR_MS, MarketStore

CACHE_DIR = Path(__file__).resolve().parents[2] / "runtime" / "learning" / "brain" / "hourly"


# -- the hourly cache ---------------------------------------------------------------------

def _file(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9@_-]", "_", name)


def build_cache(store: MarketStore, folder: Path = CACHE_DIR) -> int:
    """Hourly ts/open/high/low/close of every history, and every perpetual's funding, one .npy each."""
    folder = Path(folder)
    (folder / "funding").mkdir(parents=True, exist_ok=True)
    names = {}
    for h in store.histories(min_days=60):
        frame = h.hourly
        np.save(folder / f"{_file(h.name)}.npy",
                frame[["ts", "open", "high", "low", "close"]].to_numpy(dtype=np.float64))
        names[h.name] = h.base
    spot = {b for _, b, _ in store.symbols()}
    with store.connect() as conn:
        perpetuals = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM funding")]
    for pair, base in archive.futures_pairs(perpetuals, spot).items():
        rates = store.funding(pair)
        np.save(folder / "funding" / f"{_file(base)}.npy", rates[["ts", "rate"]].to_numpy(dtype=np.float64))
    (folder / "meta.json").write_text(json.dumps({"histories": names}), encoding="utf-8")
    save_notices(store, folder)
    return len(names)


def save_notices(store: MarketStore, folder: Path = CACHE_DIR) -> None:
    """Binance's spot delisting notices, {base: [publication ms]}, next to the prices."""
    (Path(folder) / "notices.json").write_text(json.dumps(store.delisting_notices()), encoding="utf-8")


class Market:
    """Memory-mapped hourly prices and funding, loaded on first use."""

    def __init__(self, folder: Path = CACHE_DIR):
        self.folder = Path(folder)
        self.base_of = json.loads((self.folder / "meta.json").read_text(encoding="utf-8"))["histories"]
        self._prices: dict[str, np.ndarray] = {}
        self._funding: dict[str, Optional[np.ndarray]] = {}
        path = self.folder / "notices.json"
        self._notices = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def prices(self, history: str) -> np.ndarray:
        if history not in self._prices:
            self._prices[history] = np.load(self.folder / f"{_file(history)}.npy", mmap_mode="r")
        return self._prices[history]

    def notices(self, history: str) -> np.ndarray:
        base = self.base_of.get(history, history.split("@")[0])
        return np.sort(np.asarray(self._notices.get(base, []), dtype=np.float64))

    def btc(self) -> np.ndarray:
        return self.prices("BTC")

    def funding(self, history: str) -> Optional[np.ndarray]:
        base = self.base_of.get(history, history.split("@")[0])
        if base not in self._funding:
            path = self.folder / "funding" / f"{_file(base)}.npy"
            self._funding[base] = np.load(path, mmap_mode="r") if path.exists() else None
        return self._funding[base]


# -- one position --------------------------------------------------------------------------

@dataclass(frozen=True)
class Exit:
    days: int = 14
    stop_atr: Optional[float] = None       # stop this many ATRs against the entry
    stop_pct: Optional[float] = None       # or this share of the entry against it
    target_r: Optional[float] = None       # target this many stop distances in favour (needs stop_atr)
    keep: Optional[float] = None           # trailing floor: keep this share of the best gain so far
    keep_from: float = 0.05                # ... once the best gain reached this
    leverage: float = 1.0                  # shorts on isolated margin: liquidated at 1/leverage against
    on_delist: bool = False                # out after a Binance delisting notice for the coin
    btc_crash: Optional[float] = None      # longs out when BTC fell this much in 24 hours
    squeeze: Optional[float] = None        # shorts out when the coin rose this much in 24 hours
    crash: Optional[float] = None          # longs out when the coin fell this much in 24 hours


@dataclass
class Walk:
    entry_ts: int
    exit_ts: int
    ret: float                             # net result as a share of the position
    reason: str                            # stop | target | keep | liquidated | time | settled | ended
    marks: dict = field(default_factory=dict)   # day -> net result if closed at that day's close
    worst: float = 0.0                     # worst unrealised result on the way (closes)


def _costs(side: int, slip_pct: float) -> tuple[float, float]:
    """(entry multiplier, exit multiplier) on the price, fees and slippage included."""
    fee = (FEE_PCT if side > 0 else FUTURES_FEE_PCT) / 100.0
    s = slip_pct / 100.0
    if side > 0:
        return (1 + s) * (1 + fee), (1 - s) * (1 - fee)
    return (1 - s) * (1 - fee), (1 + s) * (1 + fee)


def _net(side: int, entry: float, exit_: float, slip_pct: float) -> float:
    e, x = _costs(side, slip_pct)
    if side > 0:
        return exit_ * x / (entry * e) - 1.0
    return (entry * e - exit_ * x) / entry


def _first_event(prices: np.ndarray, a: int, b: int, side: int, rule: Exit,
                 notices: Optional[np.ndarray], btc: Optional[np.ndarray]) -> tuple[int, str]:
    """(window index of the hour whose open is the first event exit, its name), or (b - a, "")."""
    ts, c = prices[:, 0], prices[:, 4]
    span, best, name = b - a, b - a, ""
    if rule.on_delist and notices is not None and len(notices):
        i = int(np.searchsorted(notices, ts[a]))
        if i < len(notices) and notices[i] <= ts[b - 1]:
            k = int(np.searchsorted(ts[a:b], notices[i]))              # the first hour starting after it
            if k < best:
                best, name = k, "delist"
    lo = max(a - 24, 0)
    # 24-hour moves of the coin, read at each hour's close (needs the 24 hours before the entry)
    closes = np.asarray(c[lo:b])
    back = np.full(span, np.nan)
    offset = a - lo
    for_hours = np.arange(span) + offset
    has = for_hours - 24 >= 0
    back[has] = closes[for_hours[has] - 24]
    change = np.asarray(c[a:b]) / back - 1.0
    triggers = []
    if side > 0 and rule.crash is not None:
        triggers.append(("crash", change <= -rule.crash))
    if side < 0 and rule.squeeze is not None:
        triggers.append(("squeeze", change >= rule.squeeze))
    if side > 0 and rule.btc_crash is not None and btc is not None and len(btc):
        b_ts, b_c = btc[:, 0], btc[:, 4]
        i = np.clip(np.searchsorted(b_ts, ts[a:b], side="right") - 1, 0, len(b_ts) - 1)
        j = np.clip(np.searchsorted(b_ts, ts[a:b] - DAY_MS, side="right") - 1, 0, len(b_ts) - 1)
        triggers.append(("btc_crash", b_c[i] / b_c[j] - 1.0 <= -rule.btc_crash))
    for label, hit in triggers:
        if hit.any():
            k = int(np.argmax(hit)) + 1                                  # acted on at the next hour's open
            if k < best:
                best, name = k, label
    return best, name


def walk(prices: np.ndarray, funding: Optional[np.ndarray], signal_day: pd.Timestamp, side: int,
         atr: float, slip_pct: float, rule: Exit, notices: Optional[np.ndarray] = None,
         btc: Optional[np.ndarray] = None) -> Optional[Walk]:
    """Follow one position from the hour after ``signal_day``'s close; None when it cannot be opened."""
    ts = prices[:, 0]
    day0 = int((signal_day - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1))
    a = int(np.searchsorted(ts, day0 + DAY_MS))
    if a >= len(ts) or ts[a] >= day0 + DAY_MS + ENTRY_WITHIN_HOURS * HOUR_MS:
        return None
    b = int(np.searchsorted(ts, day0 + (1 + rule.days) * DAY_MS))
    settled = False
    if side < 0:
        if funding is None or not len(funding):
            return None
        f_ts = funding[:, 0]
        if np.searchsorted(f_ts, ts[a] + DAY_MS, side="right") == np.searchsorted(f_ts, ts[a]):
            return None                                     # no live perpetual at the entry
        end = int(np.searchsorted(ts, f_ts[-1] + PERP_SETTLE_MS, side="right"))
        if end < b:
            b, settled = end, True
    if b <= a:
        return None
    o, h, l, c = (np.asarray(prices[a:b, k]) for k in (1, 2, 3, 4))
    entry = float(o[0])
    span = b - a
    # gains as a share of the entry price, in the position's favour
    if side > 0:
        best, worst, opens, closes = (h - entry) / entry, (l - entry) / entry, (o - entry) / entry, (c - entry) / entry
    else:
        best, worst, opens, closes = (entry - l) / entry, (entry - h) / entry, (entry - o) / entry, (entry - c) / entry
    stop_dist = rule.stop_atr * atr / entry if rule.stop_atr and math.isfinite(atr) and atr > 0 else None
    stop = -stop_dist if stop_dist is not None else -np.inf
    if rule.stop_pct is not None:
        stop = max(stop, -rule.stop_pct)
    target = rule.target_r * stop_dist if stop_dist is not None and rule.target_r else np.inf
    liquidation = -1.0 / rule.leverage if side < 0 else -np.inf
    trail = np.full(span, -np.inf)
    if rule.keep is not None:
        peak = np.maximum.accumulate(np.r_[0.0, best[:-1]])           # best gain of the earlier hours
        trail = np.where(peak >= rule.keep_from, rule.keep * peak, -np.inf)
    floor = np.maximum(np.maximum(trail, stop), liquidation)
    hit_floor, hit_target = worst <= floor, best >= target
    kf = int(np.argmax(hit_floor)) if hit_floor.any() else span
    kt = int(np.argmax(hit_target)) if hit_target.any() else span
    ke, event = _first_event(prices, a, b, side, rule, notices, btc)
    if ke < span and ke <= min(kf, kt):
        k, gain, reason = ke, float(opens[ke]), f"event:{event}"
    elif kf < kt or (kf == kt < span and opens[kf] < target):
        k = kf
        gain = min(float(opens[k]), float(floor[k]))
        if floor[k] == liquidation and liquidation > -np.inf:
            reason = "liquidated"
        elif floor[k] == trail[k]:
            reason = "keep"
        else:
            reason = "stop"
    elif kt < span:
        k, gain, reason = kt, max(float(opens[kt]), target), "target"
    else:
        k, gain = span - 1, float(closes[-1])
        complete = ts[b - 1] >= day0 + (1 + rule.days) * DAY_MS - 2 * HOUR_MS
        reason = "settled" if settled else ("time" if complete else "ended")
    exit_price = entry * (1 + gain) if side > 0 else entry * (1 - gain)
    ret = _net(side, entry, exit_price, slip_pct)
    paid = 0.0
    if side < 0:
        f_ts, f_rate = funding[:, 0], funding[:, 1]
        cum = np.r_[0.0, np.cumsum(f_rate)]
        lo = np.searchsorted(f_ts, ts[a])
        paid = float(cum[np.searchsorted(f_ts, ts[a + k] + HOUR_MS - 1, side="right")] - cum[lo])
        ret += paid
    if side < 0 and rule.leverage >= 1.0:
        ret = max(ret, -1.0 / rule.leverage)             # isolated margin: the margin is all it can lose
    # marks: the net result if closed at each day's close while the position is open
    marks = {}
    hour_days = (ts[a:a + k + 1] // DAY_MS).astype(np.int64)
    last_of_day = np.flatnonzero(np.r_[hour_days[1:] != hour_days[:-1], True])
    for j in last_of_day[:-1]:                           # the exit day itself is booked as the exit
        price = float(c[j])
        marks[pd.Timestamp(int(hour_days[j]) * DAY_MS, unit="ms")] = _net(side, entry, price, slip_pct)
    worst_close = min([0.0, *marks.values()])
    return Walk(int(ts[a]), int(ts[a + k]) + HOUR_MS, float(ret), reason, marks, worst_close)


# -- the portfolio --------------------------------------------------------------------------

@dataclass(frozen=True)
class Rules:
    position_share: float = POSITION_SHARE
    max_positions: int = MAX_POSITIONS
    max_short_share: float = MAX_SHORT_SHARE
    drawdown_brake: float = DRAWDOWN_BRAKE


def run(orders: pd.DataFrame, days: pd.DatetimeIndex, market: Market, rules: Rules = Rules(),
        btc_daily_ret: Optional[pd.Series] = None, idle_in_btc: Optional[pd.Series] = None
        ) -> tuple[pd.Series, pd.DataFrame]:
    """(value at each day's close with open positions at market, closed positions).

    ``orders``: day, history, side, priority, atr, slip_pct, rule (an Exit), optional share, and
    optional group: orders of one group (a pair) open together or not at all.
    """
    by_day = {d: g.sort_values("priority", ascending=False) for d, g in orders.groupby("day")} if len(orders) else {}
    cash, best = START_USD, START_USD
    open_pos: list[dict] = []
    closed, values = [], []
    for day in days:
        day_end = int((day - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)) + DAY_MS
        still = []
        for p in open_pos:
            if p["walk"].exit_ts <= day_end:
                cash += p["size"] * (1.0 + p["walk"].ret)
                closed.append({"day": p["day"], "history": p["history"], "side": p["side"], "size": p["size"],
                               "ret": p["walk"].ret, "reason": p["walk"].reason, "worst": p["walk"].worst,
                               "pnl": p["size"] * p["walk"].ret, "closed": day})
            else:
                still.append(p)
        open_pos = still
        if idle_in_btc is not None and btc_daily_ret is not None and bool(idle_in_btc.get(day, False)):
            r = btc_daily_ret.get(day, 0.0)
            cash *= 1.0 + (r if math.isfinite(r) else 0.0)
        marked = sum(p["size"] * (1.0 + p["walk"].marks.get(day, 0.0)) for p in open_pos)
        value = cash + marked
        best = max(best, value)
        scale = 0.5 if value < best * (1.0 - rules.drawdown_brake) else 1.0
        held = {p["history"] for p in open_pos}
        short_out = sum(p["size"] for p in open_pos if p["side"] < 0)
        for unit in _units(by_day.get(day)):
            if len(open_pos) + len(unit) > rules.max_positions or any(o.history in held for o in unit):
                continue
            sizes = []
            for o in unit:
                share = getattr(o, "share", rules.position_share)
                sizes.append(value * (share if isinstance(share, float) and math.isfinite(share)
                                      else rules.position_share) * scale)
            new_short = sum(z for o, z in zip(unit, sizes) if o.side < 0)
            if sum(sizes) > cash or min(sizes) <= 0 or short_out + new_short > value * rules.max_short_share:
                continue
            walks = [walk(market.prices(o.history), market.funding(o.history) if o.side < 0 else None,
                          day, int(o.side), float(o.atr), float(o.slip_pct), o.rule,
                          market.notices(o.history) if o.rule.on_delist else None,
                          market.btc() if o.rule.btc_crash is not None else None) for o in unit]
            if any(w is None for w in walks):
                continue                     # a pair is both legs or nothing
            for o, z, w in zip(unit, sizes, walks):
                cash -= z
                open_pos.append({"day": day, "history": o.history, "side": int(o.side), "size": z, "walk": w})
                held.add(o.history)
            short_out += new_short
        values.append(cash + sum(p["size"] * (1.0 + p["walk"].marks.get(day, 0.0)) for p in open_pos))
    return pd.Series(values, index=days), pd.DataFrame(closed)


def _units(orders: Optional[pd.DataFrame]) -> list[list]:
    """A day's orders as units in priority order: a lone order, or all orders of one ``group`` (a pair)."""
    if orders is None or not len(orders):
        return []
    if "group" not in orders.columns:
        return [[o] for o in orders.itertuples(index=False)]
    units: dict = {}
    for i, o in enumerate(orders.itertuples(index=False)):
        key = o.group if isinstance(o.group, str) else f"_{i}"
        units.setdefault(key, []).append(o)
    return list(units.values())               # dicts keep insertion order: the best unit first


def order_inputs(rows: pd.DataFrame) -> pd.DataFrame:
    """What an order needs from the dataset: the signal day's ATR and the slippage of its turnover."""
    out = rows[["history", "day", "atr"]].assign(history=rows["history"].astype(str))
    out["slip_pct"] = slippage_pct(rows["turnover_30d"].to_numpy(dtype=float))
    return out
