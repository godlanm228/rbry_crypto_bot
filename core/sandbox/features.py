"""Situation vectors: what the market looked like at the close of a day.

Every value at day ``t`` uses candles up to and including ``t`` only - the
test suite recomputes rows on truncated history and requires them unchanged.
Next to the vector each row carries the live trade plan's own levels for that
day (stop, target, support, better entry), computed with the very functions
the  card uses, so the replay measures what the bot actually shows.

Bump FEATURE_VERSION whenever a definition changes: stored vectors of
different versions must never be compared.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from core.analysis.technical_analysis import atr as atr_series
from core.analysis.technical_analysis import rsi as rsi_series
from core.risk.trade_plan import (
    better_entry,
    place_stop,
    place_target,
    trend_of,
    verdict_for,
)
from core.sandbox.levels import support_resistance

FEATURE_VERSION = "f1"
WARMUP = 200          # SMA200 and the live plan's 200-candle window
LEVEL_WINDOW = 200    # closed candles the live plan feeds to the S/R detector
LEVEL_CAP_ATR = 10.0  # "no level nearby" is stored as 10 ATR away

FEATURES = (
    "ret_1d", "ret_7d", "ret_30d",          # log returns
    "atr_pct",                              # ATR(14) / close
    "vol_regime",                           # sigma 14d / sigma 90d
    "dist_sma20", "dist_sma50", "dist_sma200",  # (close - SMA) / ATR
    "sma50_slope",                          # SMA50 vs 10 days ago
    "rsi14",
    "range_pos_30",                         # 0 = 30d low, 1 = 30d high
    "dd_90",                                # close / 90d high - 1
    "volume_ratio",                         # log(volume / 20d average before)
    "support_dist", "resistance_dist",      # ATRs to the nearest level
    "btc_ret_30d", "btc_trend",             # BTC regime
    "rs_30d",                               # coin minus BTC, 30d
    "fear_greed", "fear_greed_7d",          # index / 100 and its 7d change
)


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def btc_context(btc: pd.DataFrame) -> pd.DataFrame:
    """Per-day BTC regime, indexed by day."""
    close = btc.set_index("day")["close"].astype(float)
    return pd.DataFrame({
        "btc_ret_30d": np.log(close / close.shift(30)),
        "btc_trend": close / _sma(close, 50) - 1.0,
    })


def fear_greed_context(fng: Optional[pd.DataFrame]) -> pd.DataFrame:
    if fng is None or fng.empty:
        return pd.DataFrame(columns=["fear_greed", "fear_greed_7d"])
    value = fng.set_index("day")["value"].astype(float).sort_index() / 100.0
    value = value[~value.index.duplicated(keep="last")]
    full = value.asfreq("D").ffill(limit=3)
    return pd.DataFrame({"fear_greed": full, "fear_greed_7d": full - full.shift(7)})


def build_situations(df: pd.DataFrame, btc: Optional[pd.DataFrame] = None,
                     fng: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per closed day with a full warm-up.

    ``df``: day, open, high, low, close, volume (oldest first, closed candles).
    """
    df = df.sort_values("day").reset_index(drop=True)
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    close = df["close"].astype(float)
    logc = np.log(close)
    ret1 = logc.diff()
    atr = atr_series(df["high"], df["low"], df["close"], period=14)
    sma20, sma50, sma200 = _sma(close, 20), _sma(close, 50), _sma(close, 200)
    vol = df["volume"].astype(float)

    out = pd.DataFrame({
        "day": df["day"],
        "close": close,
        "atr": atr,
        "ret_1d": ret1,
        "ret_7d": logc - logc.shift(7),
        "ret_30d": logc - logc.shift(30),
        "atr_pct": atr / close,
        "vol_regime": ret1.rolling(14).std() / ret1.rolling(90).std(),
        "dist_sma20": (close - sma20) / atr,
        "dist_sma50": (close - sma50) / atr,
        "dist_sma200": (close - sma200) / atr,
        "sma50_slope": sma50 / sma50.shift(10) - 1.0,
        "rsi14": rsi_series(close, 14),
        "range_pos_30": (close - df["low"].rolling(30).min())
                        / (df["high"].rolling(30).max() - df["low"].rolling(30).min()),
        "dd_90": close / df["high"].rolling(90).max() - 1.0,
        "volume_ratio": np.log(vol / vol.shift(1).rolling(20).mean()),
        "sma20": sma20,
        "sma50": sma50,
        "high20_prev": df["high"].shift(1).rolling(20).max(),
    })
    out = out.replace([np.inf, -np.inf], np.nan)

    plan_cols = {k: np.full(len(df), np.nan) for k in (
        "support", "resistance", "support_dist", "resistance_dist", "plan_stop", "plan_target",
        "plan_rr", "better_entry", "better_stop", "better_rr")}
    stop_basis = np.full(len(df), "", dtype=object)
    trend = np.full(len(df), "", dtype=object)
    verdict = np.full(len(df), "", dtype=object)
    atr_v = atr.to_numpy(dtype=float)
    for t in range(WARMUP - 1, len(df)):
        a, price = atr_v[t], c[t]
        if not (math.isfinite(a) and a > 0):
            continue
        lo = t - LEVEL_WINDOW + 1
        supports, resistances = support_resistance(h[lo:t + 1], l[lo:t + 1])
        below = [s for s in supports if s < price]
        above = [r for r in resistances if r > price]
        if below:
            plan_cols["support"][t] = max(below)
        if above:
            plan_cols["resistance"][t] = min(above)
        plan_cols["support_dist"][t] = min(LEVEL_CAP_ATR, (price - max(below)) / a) if below else LEVEL_CAP_ATR
        plan_cols["resistance_dist"][t] = min(LEVEL_CAP_ATR, (min(above) - price) / a) if above else LEVEL_CAP_ATR

        stop, basis, support = place_stop(price, a, supports)
        if stop <= 0:
            continue
        target, _ = place_target(price, a, stop, resistances)
        rr = (target - price) / (price - stop)
        tr = trend_of(price, c[max(0, t - 59):t + 1])
        v = verdict_for(rr, tr)
        plan_cols["plan_stop"][t], plan_cols["plan_target"][t], plan_cols["plan_rr"][t] = stop, target, rr
        stop_basis[t], trend[t], verdict[t] = basis, tr, v
        if v != "good":
            better = better_entry(price, a, support, target)
            if better is not None:
                plan_cols["better_entry"][t] = better.price
                plan_cols["better_stop"][t] = better.stop
                plan_cols["better_rr"][t] = better.reward_risk

    for k, v in plan_cols.items():
        out[k] = v
    out["stop_basis"], out["trend"], out["verdict"] = stop_basis, trend, verdict

    if btc is not None and not btc.empty:
        ctx = btc_context(btc)
        out = out.join(ctx, on="day")
    else:
        out["btc_ret_30d"], out["btc_trend"] = np.nan, np.nan
    out["rs_30d"] = out["ret_30d"] - out["btc_ret_30d"]
    fg = fear_greed_context(fng)
    out = out.join(fg, on="day") if not fg.empty else out.assign(fear_greed=np.nan, fear_greed_7d=np.nan)

    out["bar"] = np.arange(len(df))
    return out.iloc[WARMUP - 1:].reset_index(drop=True)


def vectors(situations: pd.DataFrame) -> np.ndarray:
    """The FEATURES columns as a float32 matrix (NaN where unknown)."""
    return situations.loc[:, list(FEATURES)].to_numpy(dtype=np.float32)

