"""Market regimes and the switcher: use the strategy that earns most in the situation now.

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

from core.brain.dataset import DAY_MS, FEE_PCT, FUTURES_FEE_PCT
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


# plain ways through a regime

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


# the switcher

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


# the market's state beyond BTC (plan item 3)
#   breadth  the share of coins above their own week's average (hourly, every coin with prices)
#   alts     the share of altcoins that beat BTC over 30 days
#   funding  the median over perpetuals of their mean funding over the previous UTC day (a day late: known)
# Each also as a coarse state, which is what strategies and the selector see: wide/narrow, alts/btc,
# hot/neutral/cold. The same numbers come from the watcher's hour live (core/brain/watch.py).
BREADTH_HOURS = 168
ALT_HOURS = 720
FUND_HOT, FUND_COLD = 0.0002, 0.0                  # per payment: +0.02% is a crowd paying to be long
STATES = {"breadth": {"wide": "широкий рынок", "narrow": "узкий рынок"},
          "alts": {"alts": "сезон альтов", "btc": "сезон BTC"},
          "fund": {"hot": "фандинг горячий", "neutral": "фандинг обычный", "cold": "фандинг холодный"}}


def breadth_state(share: float) -> str:
    return "" if not np.isfinite(share) else ("wide" if share >= 0.5 else "narrow")


def alt_state(share: float) -> str:
    return "" if not np.isfinite(share) else ("alts" if share >= 0.5 else "btc")


def fund_state(level: float) -> str:
    if not np.isfinite(level):
        return ""
    return "hot" if level >= FUND_HOT else ("cold" if level < FUND_COLD else "neutral")


def coin_flags(closes: np.ndarray, btc_closes: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per hour of one coin (closes on a full hourly grid): above its week's average, and beat BTC over 30 days
    (NaN where unknown)."""
    c = pd.Series(closes)
    sma = c.rolling(BREADTH_HOURS, min_periods=BREADTH_HOURS // 2).mean()
    above = np.where(sma.notna() & c.notna(), (c > sma).astype(float), np.nan)
    beat = np.full(len(c), np.nan)
    if btc_closes is not None:
        r, rb = c / c.shift(ALT_HOURS) - 1, pd.Series(btc_closes) / pd.Series(btc_closes).shift(ALT_HOURS) - 1
        ok = r.notna() & rb.notna()
        beat = np.where(ok, (r > rb).astype(float), np.nan)
    return above, beat


def states_at(frames: list[np.ndarray], btc_closes: np.ndarray, fundings, hour_ms: int) -> dict:
    """The market's state live at the hour starting at ``hour_ms``: ``frames`` are every coin's closes on the
    hourly grid ending at that hour, ``btc_closes`` BTC's on the same grid."""
    above, beat = [], []
    for closes in frames:
        a, b = coin_flags(closes, btc_closes[-len(closes):] if len(btc_closes) >= len(closes) else None)
        if len(a) and np.isfinite(a[-1]):
            above.append(a[-1])
        if len(b) and np.isfinite(b[-1]):
            beat.append(b[-1])
    breadth = float(np.mean(above)) if len(above) >= 10 else float("nan")
    alts = float(np.mean(beat)) if len(beat) >= 10 else float("nan")
    level = float(funding_level(fundings, np.array([hour_ms], dtype=np.int64))[0])
    return {"breadth": breadth, "alts": alts, "funding": level, "breadth_state": breadth_state(breadth),
            "alt_state": alt_state(alts), "fund_state": fund_state(level)}


def market_states(market, histories=None) -> pd.DataFrame:
    """Per hour (ts = the hour's start, as known at its close): breadth, the alts' share, the funding level and
    their states, from every coin of the hourly cache and every perpetual's funding."""
    btc = market.btc()
    grid = np.arange(int(btc[0, 0]), int(btc[-1, 0]) + HOUR_MS, HOUR_MS, dtype=np.int64)
    btc_c = pd.Series(btc[:, 4], index=btc[:, 0].astype(np.int64)).reindex(grid).to_numpy()
    above_n, above_k = np.zeros(len(grid)), np.zeros(len(grid))
    beat_n, beat_k = np.zeros(len(grid)), np.zeros(len(grid))
    fund = {}
    for name in histories or market.base_of:
        p = market.prices(name)
        if not len(p):
            continue
        lo = int(np.searchsorted(grid, int(p[0, 0])))
        closes = pd.Series(p[:, 4], index=p[:, 0].astype(np.int64)).reindex(grid[lo:lo + int((p[-1, 0] - p[0, 0])
                                                                                            // HOUR_MS) + 1]).to_numpy()
        above, beat = coin_flags(closes, None if name == "BTC" else btc_c[lo:lo + len(closes)])
        sl = slice(lo, lo + len(closes))
        above_n[sl] += np.isfinite(above)
        above_k[sl] += np.nan_to_num(above)
        beat_n[sl] += np.isfinite(beat)
        beat_k[sl] += np.nan_to_num(beat)
        f = market.funding(name)
        if f is not None and len(f) and market.base_of.get(name, name) not in fund:
            fund[market.base_of.get(name, name)] = f
    out = pd.DataFrame({"ts": grid, "breadth": np.where(above_n >= 10, above_k / np.maximum(above_n, 1), np.nan),
                        "alts": np.where(beat_n >= 10, beat_k / np.maximum(beat_n, 1), np.nan)})
    out["funding"] = funding_level(fund.values(), grid)
    out["breadth_state"] = [breadth_state(x) for x in out["breadth"]]
    out["alt_state"] = [alt_state(x) for x in out["alts"]]
    out["fund_state"] = [fund_state(x) for x in out["funding"]]
    return out


def funding_level(fundings, grid: np.ndarray) -> np.ndarray:
    """Per hour: over perpetuals, the median of each one's mean payment on the previous UTC day."""
    rows = [pd.DataFrame({"day": (np.asarray(f)[:, 0] // DAY_MS).astype(np.int64), "rate": np.asarray(f)[:, 1]})
            for f in fundings if f is not None and len(f)]
    if not rows:
        return np.full(len(grid), np.nan)
    rows = [r.groupby("day")["rate"].mean() for r in rows]
    daily = pd.concat(rows, axis=1).median(axis=1)
    days = (grid // DAY_MS - 1).astype(np.int64)
    return daily.reindex(days).to_numpy()
