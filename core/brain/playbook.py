"""Trading strategies built on the brain's daily places, written down before testing.

Every strategy reads one frame of scored coin-days (the exam's out-of-sample
places joined with the dataset's outcomes) and returns orders for the
portfolio. The market regime is fixed in advance and not tuned: BTC above its
50-day average is "up" (``btc_trend > 0``), otherwise "down".

Two ways to hold a position, both graded in the dataset after costs:
  hold   14 days, no stop (long: spot; short: the perpetual, with funding)
  trade  the fixed trade: stop 2 ATR, target twice the risk, 14 days at most
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.brain.dataset import TRADE_DAYS
from core.brain.portfolio import exit_days

PICKS = 3


@dataclass(frozen=True)
class Strategy:
    id: str
    title: str
    hypothesis: str
    rule: str
    orders: Callable[[pd.DataFrame], pd.DataFrame]
    idle_in_btc: Optional[Callable[[pd.DataFrame], pd.Series]] = None


def _pick(frame: pd.DataFrame, side: int, how: str, n: int = PICKS) -> pd.DataFrame:
    """Each day's n best (long) or n worst (short, only coins with a perpetual) by the brain's place."""
    if side > 0:
        ret, hours = ("hold_ret", None) if how == "hold" else ("ret_net", "hours")
        pool = frame[frame[ret].notna()]
        chosen = pool[pool.groupby("day")["place"].rank(ascending=False, method="first") <= n]
        priority = chosen["place"]
    else:
        ret, hours = ("short_hold_ret", None) if how == "hold" else ("short_ret_net", "short_hours")
        pool = frame[frame[ret].notna()]
        chosen = pool[pool.groupby("day")["place"].rank(ascending=True, method="first") <= n]
        priority = 1.0 - chosen["place"]
    exit_day = (chosen["day"] + pd.Timedelta(days=TRADE_DAYS + 1) if hours is None
                else exit_days(chosen["day"], chosen[hours]))
    return pd.DataFrame({"day": chosen["day"], "history": chosen["history"].astype(str), "side": side,
                         "priority": priority, "ret": chosen[ret].astype(float), "exit_day": exit_day})


def _up(frame: pd.DataFrame) -> pd.Series:
    return frame["btc_trend"] > 0


def _both(*parts: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def _idle_btc_when_up(frame: pd.DataFrame) -> pd.Series:
    """Hold idle money in BTC on a day when BTC was up at the *previous* close.

    The portfolio applies day d's BTC move (close d-1 to close d) on day d, so the
    decision may only use what was known at close d-1.
    """
    up = frame.groupby("day")["btc_trend"].first() > 0
    return up.shift(1, fill_value=False)


PLAYBOOK: tuple[Strategy, ...] = (
    Strategy("L1", "3 лучших каждый день, держать 2 недели",
             "лучшие по мозгу монеты сами по себе приносят деньги", "long top-3 by place, hold 14d, always",
             lambda f: _pick(f, +1, "hold")),
    Strategy("L2", "3 лучших, только когда BTC растёт, иначе USDT",
             "в падающем рынке лучше не покупать ничего", "long top-3 hold 14d when btc_trend > 0; idle USDT",
             lambda f: _pick(f[_up(f)], +1, "hold")),
    Strategy("L3", "3 лучших, когда BTC растёт; свободные деньги в BTC в рост, в USDT в падение",
             "в рост свободные деньги тоже должны работать", "L2 + idle money in BTC when btc_trend > 0",
             lambda f: _pick(f[_up(f)], +1, "hold"), _idle_btc_when_up),
    Strategy("S1", "шорт 3 худших, когда BTC падает, иначе USDT",
             "в падающем рынке худшие монеты падают сильнее всех", "short bottom-3 hold 14d when btc_trend <= 0",
             lambda f: _pick(f[~_up(f)], -1, "hold")),
    Strategy("LS1", "в рост — 3 лучших, в падение — шорт 3 худших",
             "мозг полезен в обе стороны: рынок решает только направление", "L2 + S1",
             lambda f: _both(_pick(f[_up(f)], +1, "hold"), _pick(f[~_up(f)], -1, "hold"))),
    Strategy("N1", "всегда: 3 лучших купить и 3 худших в шорт",
             "разница лучших и худших зарабатывает при любом рынке", "long top-3 + short bottom-3 hold 14d, always",
             lambda f: _both(_pick(f, +1, "hold"), _pick(f, -1, "hold"))),
    Strategy("L2T", "как L2, но со стопом и целью",
             "стоп и цель лучше, чем просто держать 2 недели", "L2 with the fixed trade (2 ATR stop, 2R target)",
             lambda f: _pick(f[_up(f)], +1, "trade")),
    Strategy("LS1T", "как LS1, но со стопами и целями",
             "стопы защищают и лонги, и шорты", "LS1 with the fixed trades",
             lambda f: _both(_pick(f[_up(f)], +1, "trade"), _pick(f[~_up(f)], -1, "trade"))),
)
BY_ID = {s.id: s for s in PLAYBOOK}


def scored_frame(pred: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Exam places joined with the dataset's outcomes and BTC's trend for the same coin-days."""
    cols = ["history", "day", "btc_trend", "hold_ret", "ret_net", "hours", "short_hold_ret", "short_ret_net",
            "short_hours"]
    left = pred[["history", "base", "day", "place"]].assign(history=pred["history"].astype(str))
    right = rows[cols].assign(history=rows["history"].astype(str))
    return left.merge(right, on=["history", "day"], how="inner")


def btc_series(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """BTC's close and daily return by day, from the dataset."""
    btc = rows[rows["base"].astype(str) == "BTC"].drop_duplicates("day").set_index("day").sort_index()
    close = btc["close"].astype(float)
    return close, close.pct_change().fillna(0.0).astype(float).replace([np.inf, -np.inf], 0.0)
