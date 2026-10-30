"""Market regimes and the switcher (Vlad, 30.10.2026): use the strategy that earns most in the situation now.

A regime is read from BTC at an hour's close, from that hour and the ones before it only:
  trend   up / flat / down: the close above its 50-day average by more than BAND, below it by more,
          or in between;
  mood    nervous / calm: the week's hourly volatility above or below its median of the year before
          (the library's "nervous market", core/brain/events.py).
Six regimes; a new one counts after it held 6 hours in a row, so a boundary does not flicker. In each the factory (core/brain/factory.py) breeds specialists that trade only there,
and they compete with plain ways to sit through it: hold BTC, hold ETH, short BTC on 1x margin, or
stay in USDT. The switcher holds one $100 account: every hour the regime of the hour that just
closed names its champion, chosen on 2021-2024 only; a hold is closed when the regime changes, a
specialist's trades run to their own exits. Graded on 2025 and 2026 against the single best
strategy and against holding BTC.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from core.brain.dataset import FEE_PCT, FUTURES_FEE_PCT
from core.sandbox.market_store import HOUR_MS

BAND = 0.03
CONFIRM_HOURS = 6
TREND_HOURS = 1200
WEEK_HOURS = 168
YEAR_HOURS = 365 * 24
TRENDS = {"up": "рост", "flat": "боковик", "down": "падение"}
MOODS = {"calm": "спокойно", "nervous": "нервно"}
REGIMES = tuple(f"{t}|{m}" for t in TRENDS for m in MOODS)
HOLD_SLIP = 0.0005                   # BTC and ETH: the deepest books
SHARE, MAX_OPEN = 0.2, 5             # a specialist's trades, as in the factory's books


def ru(regime: str) -> str:
    t, m = regime.split("|")
    return f"{TRENDS[t]} · {MOODS[m]}"


def btc_regimes(prices: np.ndarray) -> pd.DataFrame:
    """Per hour of BTC (ts = the hour's start): trend, mood and regime as known at its close."""
    ts = prices[:, 0].astype(np.int64)
    full = np.arange(ts[0], ts[-1] + HOUR_MS, HOUR_MS, dtype=np.int64)
    c = pd.Series(prices[:, 4], index=ts).reindex(full)
    sma = c.rolling(TREND_HOURS, min_periods=TREND_HOURS // 2).mean()
    vol = np.log(c).diff().rolling(WEEK_HOURS, min_periods=120).std()
    nervous = vol > vol.rolling(YEAR_HOURS, min_periods=2000).median()
    trend = np.where(c > sma * (1 + BAND), "up", np.where(c < sma * (1 - BAND), "down", "flat"))
    out = pd.DataFrame({"ts": full, "close": c.to_numpy(), "trend": trend,
                        "mood": np.where(nervous.to_numpy(), "nervous", "calm")})
    known = sma.notna().to_numpy() & vol.notna().to_numpy() & c.notna().to_numpy()
    out["raw"] = np.where(known, out["trend"] + "|" + out["mood"], "")
    out["regime"] = confirmed(out["raw"].to_numpy())
    return out


def confirmed(raw: np.ndarray, hours: int = CONFIRM_HOURS) -> np.ndarray:
    """A new regime counts once it held ``hours`` closes in a row; until then the old one stays.
    Near a boundary the raw label flickers every few hours, and every flicker would cost a switch."""
    out = np.empty(len(raw), dtype=object)
    current, candidate, run = "", "", 0
    for i, x in enumerate(raw):
        if x == candidate:
            run += 1
        else:
            candidate, run = x, 1
        if x == "" or not current or (x != current and run >= hours):
            current = x
        out[i] = current
    return out.astype(str)


def at(regimes: pd.DataFrame, ts_ms: np.ndarray) -> np.ndarray:
    """The regime of the hours starting at ``ts_ms`` ('' where unknown)."""
    idx = np.searchsorted(regimes["ts"].to_numpy(), np.asarray(ts_ms, dtype=np.int64))
    idx = np.clip(idx, 0, len(regimes) - 1)
    hit = regimes["ts"].to_numpy()[idx] == np.asarray(ts_ms, dtype=np.int64)
    return np.where(hit, regimes["regime"].to_numpy()[idx], "")


# -- plain ways through a regime ---------------------------------------------------------------

def _hourly(asset: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """An asset's close on the regime grid, carried over missing hours."""
    s = pd.Series(asset[:, 4], index=asset[:, 0].astype(np.int64))
    return s.reindex(grid).ffill().to_numpy()


def hold_curve(regimes: pd.DataFrame, asset: np.ndarray, regime: Optional[str], side: int,
               funding: Optional[np.ndarray], start_ms: int, end_ms: int, start_usd: float = 100.0) -> np.ndarray:
    """Equity at every hour's close of holding ``asset`` (spot long or 1x short, rebalanced hourly) through every
    hour whose previous hour closed in ``regime`` (None: always), in USDT otherwise; fees at every switch."""
    grid = regimes["ts"].to_numpy()
    span = (grid >= start_ms) & (grid < end_ms)
    close = _hourly(asset, grid)
    ret = np.r_[0.0, close[1:] / close[:-1] - 1]
    ret = np.nan_to_num(ret)
    label = regimes["regime"].to_numpy()
    on = np.r_[False, (label[:-1] == regime) if regime else (label[:-1] != "")]
    on &= span
    fee = (FEE_PCT if side > 0 else FUTURES_FEE_PCT) / 100.0 + HOLD_SLIP
    step = np.where(on, 1 + side * ret, 1.0)
    switches = np.r_[on[0], on[1:] != on[:-1]] & span
    step = step * np.where(switches, 1 - fee, 1.0)
    if side < 0 and funding is not None and len(funding):
        paid = pd.Series(funding[:, 1], index=(funding[:, 0] // HOUR_MS * HOUR_MS).astype(np.int64))
        paid = paid.groupby(level=0).sum().reindex(grid).fillna(0.0).to_numpy()
        step = step * np.where(on, 1 + paid, 1.0)
    return start_usd * np.cumprod(step[span])


def curve_stats(curve: np.ndarray, start_usd: float = 100.0) -> dict:
    if not len(curve):
        return {"growth_pct": 0.0, "drawdown_pct": 0.0}
    peak = np.maximum.accumulate(np.r_[start_usd, curve])
    return {"growth_pct": 100 * (float(curve[-1]) / start_usd - 1),
            "drawdown_pct": 100 * float(((np.r_[start_usd, curve] - peak) / peak).min())}


# -- the switcher ---------------------------------------------------------------------------------

def switch_curve(regimes: pd.DataFrame, champions: dict, trades: dict, assets: dict, funding: dict,
                 start_ms: int, end_ms: int, start_usd: float = 100.0) -> tuple[np.ndarray, dict]:
    """One $100 account through [start, end): each hour the champion of the regime the previous hour closed
    in. ``champions``: regime -> {"kind": "cash"} | {"kind": "hold", "asset", "side"} | {"kind": "genome",
    "id"}; ``trades``: genome id -> its trades (signal_ts = entry hour, held, net, history, turnover).
    Holds are marked every hour, a specialist's trades at cost until they close. Returns the equity at
    every hour's close and what was done."""
    grid = regimes["ts"].to_numpy()
    idx = np.flatnonzero((grid >= start_ms) & (grid < end_ms))
    label = regimes["regime"].to_numpy()
    closes = {a: _hourly(p, grid) for a, p in assets.items()}
    paid = {}
    for a, f in funding.items():
        if f is not None and len(f):
            s = pd.Series(f[:, 1], index=(f[:, 0] // HOUR_MS * HOUR_MS).astype(np.int64))
            paid[a] = s.groupby(level=0).sum().reindex(grid).fillna(0.0).to_numpy()
    entries = {}
    for gid, t in trades.items():
        for r in t.sort_values(["signal_ts", "turnover"], ascending=[True, False], kind="stable").itertuples(index=False):
            entries.setdefault((gid, int(r.signal_ts)), []).append(r)
    cash, hold, hold_value, open_ = start_usd, None, 0.0, []
    curve, done = [], {"switches": 0, "trades": 0, "hold_hours": 0}
    for i in idx:
        ts = int(grid[i])
        for p in [p for p in open_ if p[0] <= ts]:
            cash += p[1] * (1 + p[2])
            open_.remove(p)
        champ = champions.get(label[i - 1], {"kind": "cash"}) if i > 0 else {"kind": "cash"}
        want = (champ["asset"], champ["side"]) if champ["kind"] == "hold" else None
        if hold != want:
            if hold is not None:
                fee = (FEE_PCT if hold[1] > 0 else FUTURES_FEE_PCT) / 100.0 + HOLD_SLIP
                cash += hold_value * (1 - fee)
                hold, hold_value = None, 0.0
            if want is not None:
                fee = (FEE_PCT if want[1] > 0 else FUTURES_FEE_PCT) / 100.0 + HOLD_SLIP
                hold, hold_value, cash = want, cash * (1 - fee), 0.0
            done["switches"] += 1
        if champ["kind"] == "genome":
            for r in entries.get((champ["id"], ts), []):
                equity = cash + hold_value + sum(p[1] for p in open_)
                size = SHARE * equity
                if len(open_) >= MAX_OPEN or size > cash or any(p[3] == r.history for p in open_):
                    continue
                cash -= size
                open_.append((ts + int(r.held) * HOUR_MS, size, float(r.net), r.history))
                done["trades"] += 1
        if hold is not None:
            c = closes[hold[0]]
            move = c[i] / c[i - 1] - 1 if i > 0 and math.isfinite(c[i]) and math.isfinite(c[i - 1]) else 0.0
            hold_value *= 1 + hold[1] * move
            if hold[1] < 0 and hold[0] in paid:
                hold_value *= 1 + paid[hold[0]][i]
            hold_value = max(hold_value, 0.0)
            done["hold_hours"] += 1
        curve.append(cash + hold_value + sum(p[1] for p in open_))
    for p in open_:
        cash += p[1] * (1 + p[2])
    if curve:
        curve[-1] = cash + hold_value
    return np.asarray(curve), done
