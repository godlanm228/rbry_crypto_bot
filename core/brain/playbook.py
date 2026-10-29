"""Trading strategies built on the brain's daily places, written down before testing.

Every strategy reads one frame of scored coin-days (the exam's out-of-sample
places joined with the dataset's outcomes) and returns orders for the
portfolio. The market regime is fixed in advance and not tuned: BTC above its
50-day average is "up" (``btc_trend > 0``), otherwise "down".

Ways to hold a position. The first portfolio books them as the dataset's
outcomes; simulator v2 walks them on the hourly prices with the matching
``Exit`` (there every short is on 1x isolated margin):
  hold   14 days, no stop (long: spot; short: the perpetual, with funding)
  trade  the fixed trade: stop 2 ATR, target twice the risk, 14 days at most
  cap    shorts only: the 14-day hold, closed once the coin is 30% above the entry
  liq    shorts only: the 14-day hold on 1x isolated margin, liquidated at +100% (loses at most its size)
  keep   simulator v2 only: the 14-day hold that is closed once it gave back 30% of its best gain
         (keeps 70% of the peak; active from +5%)

Alt season, also fixed in advance: more than half of the coins ahead of BTC
over the last 30 days (``beat_btc_30d > 0.5``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.brain.dataset import TRADE_DAYS, slippage_pct
from core.brain.portfolio import exit_days
from core.brain.simulator import Exit
from core.sandbox.evaluate import DEV_END, VALID_END

PICKS = 3
RISK_PER_TRADE = 0.02      # N4: a move of 2 ATR against a position costs 2% of the portfolio
MAX_SHARE = 0.15
EXITS = {"hold": Exit(days=TRADE_DAYS), "trade": Exit(days=TRADE_DAYS, stop_atr=2.0, target_r=2.0),
         "cap": Exit(days=TRADE_DAYS, stop_pct=0.30), "liq": Exit(days=TRADE_DAYS),
         "keep": Exit(days=TRADE_DAYS, keep=0.7, keep_from=0.05)}


@dataclass(frozen=True)
class Strategy:
    id: str
    title: str
    hypothesis: str
    rule: str
    orders: Callable[[pd.DataFrame], pd.DataFrame]
    idle_in_btc: Optional[Callable[[pd.DataFrame], pd.Series]] = None


def _pick(frame: pd.DataFrame, side: int, how: str, n: int = PICKS, sized: bool = False) -> pd.DataFrame:
    """Each day's n best (long) or n worst (short, only coins with a perpetual) by the brain's place."""
    if side > 0:
        ret, hours = ("hold_ret", None) if how in ("hold", "keep") else ("ret_net", "hours")
        pool = frame[frame[ret].notna()]
        chosen = pool[pool.groupby("day")["place"].rank(ascending=False, method="first") <= n]
        priority = chosen["place"]
    else:
        ret, hours = {"hold": ("short_hold_ret", None), "trade": ("short_ret_net", "short_hours"),
                      "cap": ("short_cap_ret", "short_cap_hours"), "liq": ("short_liq_ret", "short_liq_hours"),
                      "keep": ("short_liq_ret", None)}[how]
        pool = frame[frame[ret].notna()]
        chosen = pool[pool.groupby("day")["place"].rank(ascending=True, method="first") <= n]
        priority = 1.0 - chosen["place"]
    exit_day = (chosen["day"] + pd.Timedelta(days=TRADE_DAYS + 1) if hours is None
                else exit_days(chosen["day"], chosen[hours]))
    return pd.DataFrame({"day": chosen["day"], "history": chosen["history"].astype(str), "side": side,
                         "priority": priority, "ret": chosen[ret].astype(float), "exit_day": exit_day,
                         "share": (RISK_PER_TRADE / (2 * chosen["atr_pct"].astype(float))).clip(upper=MAX_SHARE)
                         if sized else np.nan,
                         "atr": chosen["atr"].astype(float), "slip_pct": chosen["slip_pct"].astype(float),
                         "rule": [EXITS[how]] * len(chosen)})


def _up(frame: pd.DataFrame) -> pd.Series:
    return frame["btc_trend"] > 0


def _alt_season(frame: pd.DataFrame) -> pd.Series:
    return frame["beat_btc_30d"] > 0.5


def _none(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["day", "history", "side", "priority", "ret", "exit_day", "share", "atr",
                                 "slip_pct", "rule"])


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
    # second batch, 28.10.2026: a baseline and protections for the shorts
    Strategy("B1", "только BTC, когда он выше 50-дневной средней, иначе USDT",
             "простая трендовая стратегия на BTC — планка для всех остальных", "no coins; idle money in BTC when up",
             _none, _idle_btc_when_up),
    Strategy("N2", "N1, но шорт закрывается, если монета выросла на 30%",
             "одна сделка не должна съедать больше трети позиции", "long top-3 hold + short bottom-3 capped at +30%",
             lambda f: _both(_pick(f, +1, "hold"), _pick(f, -1, "cap"))),
    Strategy("N3", "N1 без шортов в альтсезон",
             "в альтсезон худшие монеты тоже взлетают, шортить их нельзя",
             "N1, shorts only when beat_btc_30d <= 0.5",
             lambda f: _both(_pick(f, +1, "hold"), _pick(f[~_alt_season(f)], -1, "hold"))),
    Strategy("N4", "N1 с размером по резкости монеты",
             "одинаковый риск на сделку: резкие монеты меньшей позицией",
             "N1 with size = 2% / (2 x ATR%), at most 15%",
             lambda f: _both(_pick(f, +1, "hold", sized=True), _pick(f, -1, "hold", sized=True))),
    Strategy("LS2", "LS1 с защитой шортов на +30%",
             "защита шортов помогает и стратегии по режиму рынка", "L2 + S1 with shorts capped at +30%",
             lambda f: _both(_pick(f[_up(f)], +1, "hold"), _pick(f[~_up(f)], -1, "cap"))),
    # third batch, 28.10.2026
    Strategy("N5", "N1 с шортом как на бирже: маржа 1x, ликвидация при +100%",
             "реалистичный N1: шорт не может потерять больше своего размера",
             "long top-3 hold + short bottom-3 on 1x isolated margin",
             lambda f: _both(_pick(f, +1, "hold"), _pick(f, -1, "liq"))),
)
PLAYBOOK += (
    # fourth batch, 28.10.2026: Vlad's "keep at least 70% of what was earned" (simulator v2)
    Strategy("N1K", "N1, но сделка закрывается, отдав 30% от пика прибыли",
             "защита заработанного: не отдавать рынку больше трети пика", "N1 with the keep-70% trailing floor",
             lambda f: _both(_pick(f, +1, "keep"), _pick(f, -1, "keep"))),
    Strategy("N3K", "N3 с защитой 70% от пика", "то же для N3", "N3 with the keep-70% trailing floor",
             lambda f: _both(_pick(f, +1, "keep"), _pick(f[~_alt_season(f)], -1, "keep"))),
    Strategy("LS2K", "LS1 с защитой 70% от пика", "то же для стратегии по режиму рынка",
             "L2 + S1 with the keep-70% trailing floor",
             lambda f: _both(_pick(f[_up(f)], +1, "keep"), _pick(f[~_up(f)], -1, "keep"))),
)
BY_ID = {s.id: s for s in PLAYBOOK}

def _pairs(frame: pd.DataFrame, n: int = PICKS, calm_shorts: bool = False) -> pd.DataFrame:
    """Each day the k-th best coin long and the k-th worst short coin short, same size, as one unit.

    ``calm_shorts``: no short on the day's 20% most volatile coins (the squeezes that lose a lot).
    """
    shorts = frame[frame["short_hold_ret"].notna()]
    if calm_shorts:
        shorts = shorts[shorts.groupby("day")["atr_pct"].rank(pct=True) <= 0.8]
    longs = _pick(frame, +1, "hold", n)
    shorts = _pick(shorts, -1, "hold", n)
    longs["k"] = longs.groupby("day")["priority"].rank(ascending=False, method="first")
    shorts["k"] = shorts.groupby("day")["priority"].rank(ascending=False, method="first")
    both = longs.merge(shorts[["day", "k"]], on=["day", "k"])[["day", "k"]]
    out = pd.concat([longs.merge(both, on=["day", "k"]), shorts.merge(both, on=["day", "k"])], ignore_index=True)
    out["group"] = out["day"].dt.strftime("%Y%m%d") + "-" + out["k"].astype(int).astype(str)
    out["priority"] = -out["k"]                       # the best pair of the day first, its two legs together
    return out.sort_values(["day", "k", "side"], ascending=[True, True, False]).drop(columns="k")


PLAYBOOK += (
    # fifth batch, 28.10.2026: pairs (the trade's result depends on the order of coins, not the market)
    Strategy("P1", "пары: лучшая монета в лонг + худшая в шорт, поровну, 3 пары в день",
             "итог пары решает разница между монетами, которую мозг умеет предсказывать",
             "3 pairs a day, long k-th best + short k-th worst, equal size, 14-day hold, both legs or none",
             lambda f: _pairs(f)),
    Strategy("P2", "пары, но не в альтсезон", "в альтсезон взлетают и худшие монеты: пары не открывать",
             "P1 only when beat_btc_30d <= 0.5", lambda f: _pairs(f[~_alt_season(f)])),
    Strategy("P3", "пары не в альтсезон и без шорта самых резких монет",
             "отсеять только сделки, где можно много потерять: шорт 20% самых резких монет дня",
             "P2 with shorts only among the day's 80% calmest coins", lambda f: _pairs(f[~_alt_season(f)], calm_shorts=True)),
    Strategy("P4", "пары каждый день без шорта самых резких монет",
             "отсев риска большой потери без фильтра альтсезона", "P1 with shorts only among the 80% calmest coins",
             lambda f: _pairs(f, calm_shorts=True)),
)
BY_ID = {s.id: s for s in PLAYBOOK}

EVENTS = {
    "E1": ("делистинг", "Binance объявил делистинг монеты: выйти в следующий час", {"on_delist": True}),
    "E2": ("обвал BTC", "BTC упал на 10% за сутки: закрыть лонги", {"btc_crash": 0.10}),
    "E3": ("вынос шорта", "монета в шорте выросла на 50% за сутки: закрыть шорт", {"squeeze": 0.50}),
    "E4": ("обвал монеты", "монета в лонге упала на 30% за сутки: закрыть лонг", {"crash": 0.30}),
    "EA": ("все события", "все четыре правила сразу",
           {"on_delist": True, "btc_crash": 0.10, "squeeze": 0.50, "crash": 0.30}),
}


def _with_events(base: "Strategy", fields: dict):
    def orders(frame: pd.DataFrame) -> pd.DataFrame:
        out = base.orders(frame)
        out["rule"] = [replace(r, **fields) for r in out["rule"]]
        return out
    return orders


# sixth batch, 29.10.2026: early exits on events, each rule alone and all together, on pairs and on N1
PLAYBOOK += tuple(
    Strategy(f"{base}{eid}", f"{BY_ID[base].title} + ранний выход: {name}",
             f"досрочный выход по событию «{name}» лучше, чем держать до плана", f"{base} with {rule}",
             _with_events(BY_ID[base], fields))
    for base in ("P1", "N1") for eid, (name, rule, fields) in EVENTS.items())
BY_ID = {s.id: s for s in PLAYBOOK}

# blends of two strategies' money, rebalanced every day: (id, title, hypothesis, rule, (a, b), weight of a)
BLENDS = (
    ("C1", "половина денег в B1, половина в N5",
     "B1 зарабатывает в рост BTC, N5 — когда альты падают: вместе ровнее", "daily 50/50 of B1 and N5", ("B1", "N5"), 0.5),
)
BY_ID = {s.id: s for s in PLAYBOOK}


def scored_frame(pred: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Exam places joined with the dataset's outcomes and BTC's trend for the same coin-days."""
    cols = ["history", "day", "btc_trend", "beat_btc_30d", "atr_pct", "atr", "turnover_30d", "hold_ret", "ret_net", "hours",
            "short_hold_ret", "short_ret_net", "short_hours", "short_cap_ret", "short_cap_hours",
            "short_liq_ret", "short_liq_hours"]
    left = pred[["history", "base", "day", "place"]].assign(history=pred["history"].astype(str))
    right = rows[cols].assign(history=rows["history"].astype(str))
    out = left.merge(right, on=["history", "day"], how="inner")
    out["slip_pct"] = slippage_pct(out["turnover_30d"].to_numpy(dtype=float))
    return out


def btc_series(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """BTC's close and daily return by day, from the dataset."""
    btc = rows[rows["base"].astype(str) == "BTC"].drop_duplicates("day").set_index("day").sort_index()
    close = btc["close"].astype(float)
    return close, close.pct_change().fillna(0.0).astype(float).replace([np.inf, -np.inf], 0.0)


def periods(values: pd.Series) -> dict[str, pd.Series]:
    """2020–2024 to choose, 2025 to confirm; 2025 starts from the last 2024 value."""
    dev = values[values.index <= DEV_END]
    valid = values[(values.index > DEV_END) & (values.index <= VALID_END)]
    return {"2020–2024": dev, "2025": pd.concat([dev.iloc[-1:], valid]) if len(dev) else valid}
