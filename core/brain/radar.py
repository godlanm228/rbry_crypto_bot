"""The listing radar (plan item 9): which new coins are worth a look, learned from past listings only.

The base it has to beat: a coin Binance lists, bought the day after, is down a month later 7
times of 10. A new listing here is a USDT pair whose first candle came within a week after Binance
announced it (2021 on), read from market.db with delisted coins too (a coin that died in a month is the
lesson, not noise). Known at the close of its first 24 hours:
  turnover   the first day's USDT turnover (log)
  day        the first day's move, its best high and how far it closed below it
  perp       a perpetual paying funding within two days of the start
  market     breadth, the alts' share, the funding level, BTC's 30 days and trend (core/brain/regime.py)
  crowd      how many other coins Binance listed in the 30 days before
The outcome: bought at the open of hour 25, held 7, 30 or 90 days, spot fees and slippage by the first
day's turnover. A ridge logistic regression of "up after 30 days" is refit every quarter on the listings
whose 30 days had ended before it and scores only that quarter's (walk-forward); the radar's pick is a
listing scored above the training scores' top third. Small data (a few hundred listings): a few
features and a strong ridge, not a forest.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from core.brain.dataset import DAY_MS, FEE_PCT, slippage_pct
from core.sandbox.market_store import HOUR_MS

START = "2021-01-01"
NEWS_DAYS = 7
HOLDS = (7, 30, 90)
FEATURES = ("log_turnover", "day_ret", "day_high", "off_high", "perp", "breadth", "alts", "funding", "btc_30d",
            "btc_up", "crowd")
WORDS = {"log_turnover": "оборот первого дня", "day_ret": "движение первого дня", "day_high": "лучший взлёт дня",
         "off_high": "откат от максимума дня", "perp": "фьючерс с первых дней", "breadth": "ширина рынка",
         "alts": "сезон альтов", "funding": "фандинг рынка", "btc_30d": "BTC за 30 дней", "btc_up": "тренд BTC",
         "crowd": "сколько листингов за месяц до"}
RIDGE = 3.0
REFIT = "QS"
PICK_QUANTILE = 2 / 3


def listing_row(hourly: pd.DataFrame, base: str, announced: int, funding: Optional[pd.DataFrame],
                states: pd.DataFrame, btc: pd.DataFrame, ended: bool = False) -> Optional[dict]:
    """A new listing's first day and what followed; None without a whole first day."""
    h = hourly.sort_values("ts").reset_index(drop=True)
    if len(h) < 25:
        return None
    first, o0 = int(h["ts"].iloc[0]), float(h["open"].iloc[0])
    day = h.iloc[:24]
    if int(day["ts"].iloc[-1]) - first > 30 * HOUR_MS or o0 <= 0:
        return None
    entry_ts = first + 24 * HOUR_MS
    a = int(np.searchsorted(h["ts"].to_numpy(), entry_ts))
    if a >= len(h):
        return None
    entry = float(h["open"].iloc[a])
    turnover = float(day["quote_volume"].sum())
    slip = float(slippage_pct(np.array([turnover]))[0]) / 100.0
    fee = FEE_PCT / 100.0
    out = {"base": base, "first_ts": first, "announced": announced, "turnover": turnover,
           "log_turnover": math.log10(max(turnover, 1.0)), "day_ret": float(day["close"].iloc[-1]) / o0 - 1,
           "day_high": float(day["high"].max()) / o0 - 1,
           "off_high": float(day["close"].iloc[-1]) / float(day["high"].max()) - 1,
           "perp": float(funding is not None and len(funding) and
                         ((funding["ts"] >= first) & (funding["ts"] <= first + 2 * DAY_MS)).any())}
    st = states[states["ts"] == entry_ts - HOUR_MS]
    for col in ("breadth", "alts", "funding"):
        out[col] = float(st[col].iloc[0]) if len(st) else float("nan")
    b = btc[btc["ts"] <= entry_ts - HOUR_MS]
    if len(b) > 720:
        out["btc_30d"] = float(b["close"].iloc[-1]) / float(b["close"].iloc[-721]) - 1
        out["btc_up"] = float(b["close"].iloc[-1] > b["close"].iloc[-1200:].mean())
    else:
        out["btc_30d"], out["btc_up"] = float("nan"), float("nan")
    ts = h["ts"].to_numpy()
    for days in HOLDS:
        target = entry_ts + days * DAY_MS
        if ts[-1] >= target - 2 * HOUR_MS:
            k = int(np.searchsorted(ts, target)) - 1
        elif ended:                                  # stopped trading first: out at its last close
            k = len(ts) - 1
        else:                                        # still trading, the days not over yet
            out[f"ret{days}"] = float("nan")
            continue
        exit_ = float(h["close"].iloc[max(k, a)])
        out[f"ret{days}"] = exit_ * (1 - slip) * (1 - fee) / (entry * (1 + slip) * (1 + fee)) - 1
    return out


def day_line(titles: list[str], model: Optional[dict]) -> str:
    """The daily radar: Binance's new listings of the day and what such listings did (radar-model.json)."""
    import re
    bases = sorted({b for t in titles for b in re.findall(r"\(([A-Z0-9]{2,15})\)", t)})
    base = (f"исторически через месяц в плюсе ~{10 * model['base_up']:.0f} из 10 (по {model['listings']} листингам), "
            f"фильтр первого дня не помог — радар предупреждает, а не советует") if model else ""
    if not bases:
        return "Радар: новых листингов Binance за сутки нет."
    return f"Радар: новые листинги Binance за сутки — {', '.join(bases)}" + (f"; {base}." if base else ".")


def fit(x: np.ndarray, y: np.ndarray, ridge: float = RIDGE, iters: int = 50) -> np.ndarray:
    """Ridge logistic regression by Newton's method; x without the intercept column."""
    xb = np.c_[np.ones(len(x)), x]
    w = np.zeros(xb.shape[1])
    penalty = np.r_[0.0, np.full(x.shape[1], ridge)]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-xb @ w))
        grad = xb.T @ (p - y) + penalty * w
        hess = (xb * (p * (1 - p))[:, None]).T @ xb + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def prob(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.c_[np.ones(len(x)), x] @ w))


def standardise(train: pd.DataFrame, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Features z-scored on the training listings (missing -> the training mean)."""
    mean = train[list(FEATURES)].mean()
    std = train[list(FEATURES)].std().replace(0, 1).fillna(1)
    z = lambda f: ((f[list(FEATURES)].fillna(mean) - mean) / std).to_numpy(dtype=float)  # noqa: E731
    return z(train), z(rows), {"mean": mean.to_dict(), "std": std.to_dict()}


def walk_forward(t: pd.DataFrame, start: str = START, hold: int = 30) -> pd.DataFrame:
    """Every listing from ``start`` on, scored by a model of the listings whose ``hold`` days had ended."""
    t = t[np.isfinite(t[f"ret{hold}"])].sort_values("first_ts").reset_index(drop=True)
    quarters = pd.date_range(start, pd.to_datetime(t["first_ts"].max(), unit="ms"), freq=REFIT)
    out = []
    for q in quarters:
        lo, hi = int(q.value // 10**6), int((q + pd.offsets.QuarterBegin(startingMonth=1)).value // 10**6)
        train = t[t["first_ts"] + (hold + 2) * DAY_MS < lo]
        test = t[(t["first_ts"] >= lo) & (t["first_ts"] < hi)]
        if len(train) < 40 or not len(test):
            continue
        xt, xs, _ = standardise(train, test)
        w = fit(xt, (train[f"ret{hold}"] > 0).to_numpy(dtype=float))
        bar = float(np.quantile(prob(w, xt), PICK_QUANTILE))
        out.append(test.assign(score=prob(w, xs), picked=prob(w, xs) >= bar))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
