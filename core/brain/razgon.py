"""«Разгон v2»: one model that grows by versions, each step graded by the same exam.

The owner on 13.11.2026: in two weeks he never saw a strategy develop. Ideas were tested and rejected, the
selector was retrained in a circle, and the "Разгон" account could not speed up by its build: $20 a trade on
spot, at most five, mostly in cash. What does earn on history and in 2025-2026 is the mass panic
(core/brain/exits.py: 2018-2024 8 of 10 trades in plus; the selector over 34,000 trades is a coin toss), so
the panic is the core and its paper rule is version v1.

A step changes ONE thing of the current version (leverage, how long it holds, how many coins of a moment).
The exam books the version as the factory books a genome: $100 in every year 2018-2026, the mass moments'
most liquid coins, entry 10 minutes after the signal, fees, slippage, funding; then bends history four ways
(core/brain/robust.py: an hour late, costs doubled, a fifth of the signals missed, without the best 5% of
trades). A step becomes the next version only if, against the current one,
  - it grows faster over 2018-2024 (compounded, a year);
  - it ends 2025 and 2026 not lower;
  - it holds the goal as the factory asks of a paper candidate: every year's drawdown within 50% (the owner:
    losing 50 of 100 is acceptable, not more), growth over 2018-2024, growth in 2025 and 2026;
  - it is no more fragile: every bent history the current version holds, the step holds too (v1 itself ends
    2025 flat when every entry comes an hour late, so "all four" would close the ladder before its first step).
Otherwise it stays on the ladder as a step that did not hold, with the reason in words. Every step is an idea
in the journal before its exam. 2025 and 2026 were looked at many times by now: the clean judge is the paper.

Leverage is isolated margin on the coin's perpetual: the trade's margin is its share of the account, the
position ``leverage`` times that; fees, slippage and funding are paid on the position; a fall of
1/leverage minus the maintenance margin takes the whole margin and no more. A coin without a live perpetual
at the signal (every coin before 2020) is bought on spot as before. A version may lever only while BTC is not
deep under its 50-day average (``lever_above``): that is where the liquidations of the first exam sat.

The second engine (14.11.2026). The first exam showed the money in trades 14% of the time: a panic is rare,
so the account mostly stands in dollars. A version with ``idle_sma`` keeps the free money in BTC while BTC's
daily close is above the average of its last ``idle_sma`` daily closes (decided at the UTC day's close, acted
on at the next hour's open) and in dollars otherwise. A panic trade is paid for by selling that much BTC, and
its proceeds go back into BTC; every such move pays the spot fee and slippage. The book then marks the account
at every hour's close, so a year's drawdown includes BTC's own falls. The bent histories bend this engine too
where they can: an hour late moves the switch an hour, doubled costs double BTC's; a missed signal and the
best trades are the panic's alone.

All of the free money in BTC was examined first: 2018-2024 +78% a year against +30%, 2025 and 2026 not lower,
the worst year's drawdown 45%. It was not accepted: with costs doubled, a fifth of the signals missed or
without the best trades a year's drawdown passes 50% (2021, 2022), which v1 holds. So a version may put only
``idle_share`` of the free money in BTC: a trade is paid for from the dollars first, BTC is sold only for
what they lack, and proceeds refill BTC up to its share.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.brain import exits as ex
from core.brain import factory as fa
from core.brain import robust as rb
from core.brain import selector as se
from core.brain.dataset import FEE_PCT, FUTURES_FEE_PCT
from core.sandbox.market_store import HOUR_MS

DAY_MS = 24 * HOUR_MS
BTC_SLIP = 0.0005                  # a side, on top of the spot fee: BTC's book is the deepest there is

RAZGON_DIR = Path(__file__).resolve().parents[2] / "runtime" / "learning" / "razgon"
LADDER = RAZGON_DIR / "razgon-versions.json"     # the night copies it to the VM under this name
GOAL = "razgon"
MASS = ex.MASS["mass4"]
MAINTENANCE = 0.02                 # of the position: Binance asks 0.4% for BTC up to 2.5% for thin perpetuals
DRAWS = 100
MISS, BEST = 0.2, 0.05


@dataclass(frozen=True)
class Version:
    picks: int = 3                 # coins of a mass moment, the most liquid first
    hours: int = 72                # the rest is closed after this at the latest
    half_pct: Optional[float] = 0.05   # half out at this gain, the rest's stop to the entry
    keep: Optional[float] = None   # the rest trails: keeps this share of its best gain once it reached keep_from
    keep_from: float = 0.05
    leverage: float = 1.0          # on coins with a live perpetual
    lever_above: Optional[float] = None    # ... and only while BTC stands above this against its 50-day average
    share: float = ex.SHARE        # of the account a trade
    max_open: int = ex.MAX_OPEN
    idle_sma: Optional[int] = None     # free money in BTC while it closes above its average of this many days
    idle_share: float = 1.0            # ... this part of it; the rest waits in dollars

    def rule(self) -> ex.Rule:
        return ex.Rule("RZ", "", self.hours, half_pct=self.half_pct, keep=self.keep, keep_from=self.keep_from)

    def words(self) -> str:
        days = f"{self.hours // 24} дн." if self.hours % 24 == 0 else f"{self.hours} ч"
        rest = (f"остаток держит {100 * self.keep:.0f}% лучшего роста и не ниже цены входа, до {days}"
                if self.keep else f"остаток — стоп на цену входа, закрыть через {days}")
        lev = "спот, без плеча"
        if self.leverage > 1:
            when = (f", только пока BTC не ниже {100 * self.lever_above:+.0f}% от своей 50-дневной средней, глубже — спот"
                    if self.lever_above is not None else "")
            lev = f"плечо {self.leverage:g}x (фьючерс; без фьючерса — спот{when})"
        part = "свободные деньги" if self.idle_share >= 1 else f"{100 * self.idle_share:.0f}% свободных денег"
        idle = (f"; {part} — в BTC, пока он выше своей {self.idle_sma}-дневной средней, иначе в долларах"
                if self.idle_sma else "")
        return (f"массовая паника (от {MASS} монет, нервный рынок): {self.picks} самых ликвидных; {lev}; "
                f"половину на +{100 * (self.half_pct or 0):.0f}%, {rest}; {100 * self.share:.0f}% счёта на сделку, "
                f"не больше {self.max_open} сразу{idle}")


V1 = Version()


@dataclass(frozen=True)
class Step:
    key: str
    title: str                     # what changes, for the owner
    why: str
    change: dict                   # the Version's fields it replaces


# the first steps, as proposed to the owner on 13.11.2026; a new idea is a new line here
STEPS = (
    Step("lev2", "плечо 2x на покупках паники",
         "перевес паники мал на сделку (+$0.37 на $20), но устойчив: плечо удваивает его; цена — ликвидация при "
         "падении монеты почти вдвое", {"leverage": 2.0}),
    Step("lev3", "плечо 3x на покупках паники",
         "если 2x держит просадку, 3x разгоняет ещё; ликвидация уже при −31%", {"leverage": 3.0}),
    Step("hold7", "держать до 7 дней, пока отскок продолжается",
         "через 3 дня остаток закрывается по времени, даже когда монета ещё растёт; остаток под скользящим "
         "выходом (держит половину лучшего роста) может забрать больше", {"hours": 168, "keep": 0.5}),
    Step("hold14", "держать до 14 дней, пока отскок продолжается",
         "то же, вдвое дольше", {"hours": 336, "keep": 0.5}),
    Step("picks5", "5 монет момента вместо 3",
         "тонкие монеты отскакивают не хуже ликвидных (LAB-20261113-5): больше монет — больше денег в деле",
         {"picks": 5}),
    # after the first five: 12 of the 13 liquidations of 2x on 2018-2024 came with BTC 8% or more under its
    # 50-day average (COVID, May 2021, LUNA); above it, 794 levered trades and one liquidation
    Step("lev2_calm", "плечо 2x, только пока BTC не провалился под 50-дневную среднюю на 8%",
         "плечо само по себе не прошло из-за каскадов ликвидаций, а каскады случаются, когда BTC уже глубоко "
         "под своей средней; выше неё паника — это просадка в растущем рынке, и плечо там удваивает перевес",
         {"leverage": 2.0, "lever_above": -0.08}),
    Step("lev3_calm", "плечо 3x, только пока BTC не провалился под 50-дневную среднюю на 8%",
         "если 2x при этом условии держит просадку, 3x разгоняет ещё", {"leverage": 3.0, "lever_above": -0.08}),
    # the second engine, 14.11.2026: the money sat in trades 14% of the time (2025: 11%, 2026: 9%)
    Step("idle_btc50", "свободные деньги — в BTC, пока он выше 50-дневной средней",
         "паника редка: 86% времени счёт стоит в долларах и не растёт. BTC выше своей средней — это растущий "
         "рынок; ниже неё деньги снова в долларах и ждут панику", {"idle_sma": 50}),
    Step("idle_btc100", "свободные деньги — в BTC, пока он выше 100-дневной средней",
         "то же с более медленной средней: меньше ложных переключений, позже вход и выход", {"idle_sma": 100}),
    Step("idle_btc200", "свободные деньги — в BTC, пока он выше 200-дневной средней",
         "самая медленная из принятых на рынке границ «бычий/медвежий»: переключений единицы в год",
         {"idle_sma": 200}),
    # all of the free money in BTC above its 50-day average was faster (+78% a year) and not lower in 2025-2026,
    # but under doubled costs, missed signals or without the best trades a year's drawdown passed 50%: a part
    # of the money instead, from the largest part down, so the first that holds is the one that stays
    Step("idle_btc50_75", "три четверти свободных денег — в BTC, пока он выше 50-дневной средней",
         "все свободные деньги в BTC разгоняют счёт (+78% в год против +30%), но при удвоенных комиссиях или "
         "пропуске сигналов просадка года уходит за 50%; меньшая доля режет просадку BTC пропорционально",
         {"idle_sma": 50, "idle_share": 0.75}),
    Step("idle_btc50_50", "половина свободных денег — в BTC, пока он выше 50-дневной средней",
         "то же с половиной: просадка от BTC вдвое меньше, сделка паники оплачивается из долларов без продажи BTC",
         {"idle_sma": 50, "idle_share": 0.5}),
    Step("idle_btc50_33", "треть свободных денег — в BTC, пока он выше 50-дневной средней",
         "самая осторожная доля: если и половина не держит просадку", {"idle_sma": 50, "idle_share": 1 / 3}),
)


# the trades of a version

def table(rows: pd.DataFrame, market) -> pd.DataFrame:
    """Every panic of a coin in a nervous market (exits.panic_trades), with the year of its signal."""
    p = ex.panic_trades(rows, market)
    p["year"] = pd.to_datetime(p["signal_ts"], unit="ms").dt.year.astype(int)
    p["perp"] = p["perp"].fillna(False).astype(bool)
    btc = se.btc_frame(market)[["ts", "btc_sma"]].rename(columns={"ts": "hour_ts"})     # at the signal hour's close
    return p.assign(hour_ts=p["signal_ts"] - HOUR_MS).merge(btc, on="hour_ts", how="left").drop(columns="hour_ts")


def picked(t: pd.DataFrame, v: Version) -> pd.DataFrame:
    """The coins a version takes: mass moments, the most liquid first."""
    top = (-t["turnover"]).groupby(t["signal_ts"]).rank(method="first") <= v.picks
    return t[top.to_numpy() & (t["moment_size"].to_numpy() >= MASS)]


def liquidation_fall(leverage: float) -> float:
    """The fall of the price that takes the whole margin of an isolated long."""
    return 1.0 / leverage - MAINTENANCE


def walk(prices: np.ndarray, funding: Optional[np.ndarray], signal_ts: int, v: Version, delta: float, slip: float,
         newest: int, levered: bool) -> Optional[dict]:
    """One buy of the panic under a version: net on the trade's margin, hours held, how it ended."""
    rule = v.rule()
    w = ex._window(prices, int(signal_ts), rule.hours, newest)
    if w is None:
        return None
    a, b = w
    o, h, l, c = (np.asarray(prices[a:b, k], dtype=float) for k in (1, 2, 3, 4))
    entry = float(o[0]) * (1 + delta)
    if not levered:
        exits, held, reason, worst = ex.walk_long(o, h, l, c, entry, None, rule)
        return {"net": ex.net_of(exits, entry, slip), "held": held, "reason": reason, "worst": worst, "lev": 1.0}
    exits, held, reason, worst = ex.walk_long(o, h, l, c, entry, liquidation_fall(v.leverage), rule)
    if reason == "stop":                       # the liquidation level, reached before any half was taken
        return {"net": -1.0, "held": held, "reason": "liquidated", "worst": worst, "lev": v.leverage}
    fee = FUTURES_FEE_PCT / 100.0
    gross = sum(s * p * (1 - slip) * (1 - fee) for s, p in exits) / (entry * (1 + slip) * (1 + fee)) - 1
    paid = 0.0
    if funding is not None and len(funding):   # a long pays the funding of the hours it is held (on all of it:
        f_ts = funding[:, 0]                   # the half sold earlier is not taken off, the cautious side)
        start, end = int(prices[a, 0]), int(prices[a, 0]) + held * HOUR_MS
        paid = float(np.sum(funding[np.searchsorted(f_ts, start, side="right"):
                                    np.searchsorted(f_ts, end, side="right"), 1]))
    return {"net": max(v.leverage * (gross - paid), -1.0), "held": held, "reason": reason, "worst": worst,
            "lev": v.leverage}


def trades(t: pd.DataFrame, v: Version, market, newest: int, delay: int = 0) -> pd.DataFrame:
    """Every trade of a version; ``delay``: entered that many hours late, at that hour's open."""
    out = []
    for r in picked(t, v).itertuples(index=False):
        calm = v.lever_above is None or float(getattr(r, "btc_sma", float("nan"))) > v.lever_above
        funding = market.funding(r.history) if v.leverage > 1 and r.perp and calm else None
        levered = funding is not None and len(funding) > 0
        res = walk(market.prices(r.history), funding, int(r.signal_ts) + delay * HOUR_MS, v,
                   0.0 if delay else float(r.delta), float(r.slip), newest, levered)
        if res is not None:
            out.append((r.history, int(r.signal_ts), float(r.turnover), int(r.year), float(r.slip), res["net"],
                        res["held"] + delay, res["reason"], res["worst"], res["lev"]))
    return pd.DataFrame(out, columns=["history", "signal_ts", "turnover", "year", "slip", "net", "held", "reason",
                                      "worst", "lev"])


# the second engine: free money in BTC by its trend

def idle_plan(prices: np.ndarray, days: int, delay: int = 0) -> dict:
    """BTC per hour and whether the free money sits in it. Decided at each UTC day's close (above the average
    of the last ``days`` daily closes), acted on from the next day's first hour; ``delay``: that many hours later."""
    ts = prices[:, 0].astype(np.int64)
    grid = np.arange(int(ts[0]), int(ts[-1]) + HOUR_MS, HOUR_MS, dtype=np.int64)
    close = pd.Series(np.asarray(prices[:, 4], dtype=float), index=ts).reindex(grid).ffill()
    open_ = pd.Series(np.asarray(prices[:, 1], dtype=float), index=ts).reindex(grid).fillna(close)   # a gap hour
    day = grid // DAY_MS
    daily = close.groupby(day).last()
    above = (daily > daily.rolling(days, min_periods=days).mean()).shift(1, fill_value=False)   # for the next day
    on = above.reindex(day).to_numpy(dtype=bool)
    if delay:
        on = np.concatenate([np.zeros(delay, dtype=bool), on[:-delay]])
    return {"ts": grid, "open": open_.to_numpy(), "close": close.to_numpy(), "on": on}


def book(t: pd.DataFrame, start_ms: int, end_ms: int, v: Version, plan: dict, cost_mult: float = 1.0,
         start_usd: float = ex.START_USD) -> dict:
    """exits.portfolio with the free money in BTC while ``plan`` says so, the account marked at every hour's
    close while it holds BTC. With a plan that is never on, the same numbers as exits.portfolio."""
    cost = cost_mult * (FEE_PCT / 100.0 + BTC_SLIP)
    ts, px_open, px_close, on = plan["ts"], plan["open"], plan["close"], plan["on"]
    a, b = int(np.searchsorted(ts, start_ms)), int(np.searchsorted(ts, end_ms))      # the span's hours [a, b)
    t = t[(t["signal_ts"] >= start_ms) & (t["signal_ts"] < end_ms)].sort_values(
        ["signal_ts", "turnover"], ascending=[True, False], kind="stable")
    rows = list(t.itertuples(index=False))
    flips = [h for h in range(a, b) if on[h] != (on[h - 1] if h > a else False)]
    events = sorted([(int(ts[h]), 0, h) for h in flips] + [(int(r.signal_ts), 1, i) for i, r in enumerate(rows)])
    state = {"cash": start_usd, "units": 0.0, "hour": a, "in_btc": 0.0, "holding": False}
    open_, curve, taken = [], [start_usd], 0

    def price(ms: int) -> float:
        """BTC at an hour's start: that hour's open; past the data, the last close."""
        i = int(np.searchsorted(ts, ms))
        return float(px_open[i]) if i < len(ts) and ts[i] == ms else float(px_close[min(i, len(ts)) - 1])

    def cost_held() -> float:
        return sum(q[1] for q in open_)

    def mark(until_ms: int) -> None:
        """The closes of the hours that ended by ``until_ms``, while BTC is held."""
        j = min(int(np.searchsorted(ts, until_ms)), b)
        if state["units"] > 0 and j > state["hour"]:
            btc = state["units"] * px_close[state["hour"]:j]
            total = state["cash"] + cost_held() + btc
            curve.extend(total.tolist())
            state["in_btc"] += float((btc / total).sum())
        state["hour"] = max(state["hour"], j)

    def into_btc(ms: int) -> None:
        """Dollars into BTC up to the version's share of the free money (never a sale: that is a trade's)."""
        px = price(ms)
        buy = state["cash"]
        if v.idle_share < 1:
            buy = min(buy, v.idle_share * (state["cash"] + state["units"] * px) - state["units"] * px)
        if buy > 0:
            state["units"] += buy * (1 - cost) / px
            state["cash"] -= buy

    def settle(until_ms: int) -> None:
        """Trades that ended by ``until_ms``, each at its own hour: the proceeds follow the plan."""
        for p in sorted(q for q in open_ if q[0] <= until_ms):
            mark(p[0])
            state["cash"] += p[1] * (1 + p[2])
            open_.remove(p)
            if state["holding"] and p[0] < end_ms:
                into_btc(p[0])
            curve.append(state["cash"] + cost_held() + state["units"] * price(min(p[0], end_ms)))

    for when, kind, what in events:
        settle(when)
        mark(when)
        if kind == 0:
            state["holding"] = bool(on[what])
            if state["holding"]:
                into_btc(when)
            else:
                state["cash"] += state["units"] * price(when) * (1 - cost)
                state["units"] = 0.0
            continue
        r = rows[what]
        if len(open_) >= v.max_open or any(q[3] == r.history for q in open_):
            continue
        px = price(when)
        size = v.share * (state["cash"] + cost_held() + state["units"] * px)
        if size <= 0:
            continue
        if size > state["cash"]:                # the trade is paid for with BTC
            sell = (size - state["cash"]) / (1 - cost)
            if sell > state["units"] * px:
                continue
            state["units"] -= sell / px
            state["cash"] = size
        state["cash"] -= size
        open_.append((int(r.signal_ts) + int(r.held) * HOUR_MS, size, float(r.net), r.history))
        taken += 1
    settle(end_ms)
    mark(end_ms)
    state["cash"] += state["units"] * price(end_ms)      # the year's end: BTC at its mark, no sale
    state["units"] = 0.0
    for p in sorted(open_):                               # trades still open then, as exits.portfolio counts them
        state["cash"] += p[1] * (1 + p[2])
        open_.remove(p)
        curve.append(state["cash"] + cost_held())
    c = np.asarray(curve)
    peak = np.maximum.accumulate(c)
    return {"end_usd": float(c[-1]), "growth_pct": 100 * (float(c[-1]) / start_usd - 1),
            "drawdown_pct": 100 * float(((c - peak) / peak).min()), "trades": taken,
            "btc_pct": 100 * state["in_btc"] / (b - a) if b > a else 0.0,
            "years": (end_ms - start_ms) / (365.25 * 86_400_000)}


# the exam

def books(tr: pd.DataFrame, v: Version, newest: int, plan: Optional[dict] = None, cost_mult: float = 1.0) -> dict:
    """$100 in every year, as the factory books a genome, with the version's share and places; ``plan``:
    the free money follows BTC's trend (idle_plan)."""
    years = {}
    for y in fa.YEARS:
        ty = tr[tr["year"] == y]
        span = fa.year_span(y, newest)
        b = (ex.portfolio(ty, *span, share=v.share, max_open=v.max_open) if plan is None
             else book(ty, *span, v, plan, cost_mult))
        years[str(y)] = {"n": int(len(ty)), "growth_pct": b["growth_pct"], "drawdown_pct": b["drawdown_pct"],
                         "btc_pct": b.get("btc_pct", 0.0),
                         "usd_per_100": 100 * float(ty["net"].mean()) if len(ty) else float("nan"),
                         # the share of the year's money-hours that sat in trades: places taken x hours x share
                         "busy_pct": (100 * b["trades"] * float(ty["held"].mean()) * v.share * HOUR_MS / (span[1] - span[0])
                                      if len(ty) and span[1] > span[0] else 0.0)}
    return {"years": years, "learn_n": int(tr["year"].isin(fa.LEARN_YEARS).sum())}


def plans(v: Version, market) -> tuple[Optional[dict], Optional[dict]]:
    """(the version's BTC plan, the same an hour late), or Nones for a version that keeps free money in dollars."""
    if not v.idle_sma:
        return None, None
    btc = market.btc()
    return idle_plan(btc, v.idle_sma), idle_plan(btc, v.idle_sma, delay=1)


def _brief(stats: dict) -> dict:
    """A book in the scoreboard's numbers."""
    ys = stats["years"]
    return {"learn_yearly_pct": round(rb.learn_growth(stats), 1),
            "learn_worst_drawdown_pct": round(min(ys[str(y)]["drawdown_pct"] for y in fa.LEARN_YEARS), 1),
            "losing_years": [y for y in fa.LEARN_YEARS if ys[str(y)]["growth_pct"] < 0],
            "busy_pct": round(float(np.mean([ys[str(y)].get("busy_pct", 0.0) for y in fa.LEARN_YEARS])), 0),
            "btc_pct": round(float(np.mean([ys[str(y)].get("btc_pct", 0.0) for y in fa.LEARN_YEARS])), 0),
            **{str(y): {"growth_pct": round(ys[str(y)]["growth_pct"], 1),
                        "drawdown_pct": round(ys[str(y)]["drawdown_pct"], 1), "n": ys[str(y)]["n"]}
               for y in fa.YEARS},
            "holds": bool(rb.holds(stats, GOAL))}


def exam(v: Version, t: pd.DataFrame, market, newest: int, seed: int = 5) -> dict:
    """A version's books as it is and bent four ways, and its trades in the reports' words."""
    base = trades(t, v, market, newest)
    plan, plan_late = plans(v, market)
    stats = books(base, v, newest, plan)
    fee = np.where(base["lev"] > 1, FUTURES_FEE_PCT, FEE_PCT) / 100.0
    costly = base.assign(net=np.maximum(base["net"] - 2 * (fee + base["slip"]) * base["lev"], -1.0))
    rng = np.random.default_rng(seed)
    draws = [books(base[rng.random(len(base)) >= MISS], v, newest, plan) for _ in range(DRAWS)]
    cut = base["net"].quantile(1 - BEST) if len(base) else 0.0
    bent = {"late": books(trades(t, v, market, newest, delay=1), v, newest, plan_late),
            "costs": books(costly, v, newest, plan, cost_mult=2.0), "missed": rb.bad_tenth(draws),
            "no_best": books(base[base["net"] < cut] if len(base) > 20 else base, v, newest, plan)}
    learn = base[base["year"].isin(fa.LEARN_YEARS)]
    out = {"version": asdict(v), "words": v.words(), **_brief(stats),
           "trades": int(len(base)), "levered_pct": round(100 * float((base["lev"] > 1).mean()), 0) if len(base) else 0.0,
           "liquidations": int((base["reason"] == "liquidated").sum()),
           "liquidations_by_year": {str(y): int(n) for y, n in
                                    base[base["reason"] == "liquidated"].groupby("year").size().items()},
           "up_of10": round(10 * float((learn["net"] > 0).mean()), 1) if len(learn) else None,
           "usd_per_20": round(20 * float(learn["net"].mean()), 2) if len(learn) else None,
           "worst_trade_pct": round(100 * float(base["net"].min()), 1) if len(base) else None,
           "held_hours": round(float(base["held"].mean()), 0) if len(base) else None,
           "bent": {name: _brief(s) for name, s in bent.items()}}
    out["robust"] = bool(out["holds"] and all(b["holds"] for b in out["bent"].values()))
    return out


def decide(candidate: dict, current: dict) -> tuple[bool, list[str]]:
    """(the step becomes the next version, why in words), by the rule in the module's docstring."""
    ok, why = True, []
    a, b = candidate["learn_yearly_pct"], current["learn_yearly_pct"]
    if a > b:
        why.append(f"2018–2024: {a:+.0f}% в год против {b:+.0f}% — быстрее")
    else:
        ok = False
        why.append(f"2018–2024: {a:+.0f}% в год против {b:+.0f}% — не быстрее")
    for y in fa.CHECK_YEARS:
        ca, cb = candidate[str(y)]["growth_pct"], current[str(y)]["growth_pct"]
        if ca < cb:
            ok = False
            why.append(f"{y}: счёт ${100 + ca:.0f} против ${100 + cb:.0f} — ниже")
        else:
            why.append(f"{y}: счёт ${100 + ca:.0f} против ${100 + cb:.0f} — не ниже")
    limit = -100 * fa.GOALS[GOAL]["drawdown"]
    deep = [str(y) for y in fa.YEARS if candidate[str(y)]["drawdown_pct"] < limit]
    if deep:
        ok = False
        worst = min(candidate[str(y)]["drawdown_pct"] for y in fa.YEARS)
        why.append(f"просадка до {worst:.0f}% ({', '.join(deep)}) — глубже допустимых {limit:.0f}%")
    elif not candidate["holds"]:
        ok = False
        why.append("не растёт в 2018–2024, в 2025 или в 2026")
    failed = [rb.MUTATIONS[name] for name, s in candidate["bent"].items()
              if current["bent"][name]["holds"] and not s["holds"]]
    if failed:
        ok = False
        why.append("стала хрупче: не держит то, что держит текущая версия — " + "; ".join(failed))
    elif not deep and candidate["holds"]:
        why.append("на прочность не хуже текущей версии")
    if candidate["liquidations"]:
        why.append(f"ликвидаций: {candidate['liquidations']} из {candidate['trades']} сделок")
    return ok, why


# the ladder

def load(path: Path = LADDER) -> dict:
    if not path.exists():
        return {"current": None, "versions": [], "steps": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save(ladder: dict, path: Path = LADDER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ladder, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def current_version(ladder: dict) -> Version:
    done = [x for x in ladder["versions"] if x["name"] == ladder["current"]]
    return Version(**done[0]["exam"]["version"]) if done else V1


def climb(ladder: dict, t: pd.DataFrame, market, newest: int, steps=STEPS, journal=None, source: str = "",
          today: Optional[str] = None) -> dict:
    """Examine v1 when the ladder is empty, then every step not decided yet against the version current at
    its turn; an accepted step becomes the next version. Returns the ladder (also saved by the caller)."""
    today = today or time.strftime("%Y-%m-%d")
    if not ladder["versions"]:
        ladder["versions"].append({"name": "v1", "title": "паника как на бумаге с 30.10.2026", "since": today,
                                   "exam": exam(V1, t, market, newest)})
        ladder["current"] = "v1"
    decided = {s["key"] for s in ladder["steps"]}
    for step in steps:
        if step.key in decided:
            continue
        now = [x for x in ladder["versions"] if x["name"] == ladder["current"]][0]
        v = replace(Version(**now["exam"]["version"]), **step.change)
        if journal is not None:
            journal.register(
                f"RZ2-{step.key}", f"Разгон v2, шаг «{step.title}» поверх {now['name']}", step.why,
                f"{v.words()}. Экзамен core/brain/razgon.py: книга $100 в каждом году 2018–2026; шаг принимается, "
                f"если против {now['name']}: рост за 2018–2024 в год выше; счёт 2025 и 2026 не ниже; просадка каждого "
                f"года в пределах 50% и рост в 2018–2024, 2025 и 2026; и не хрупче: каждый изгиб истории (час "
                f"опоздания, комиссии ×2, пропуск 20% сигналов, без лучших 5% сделок), который держит "
                f"{now['name']}, держит и шаг.")
        result = exam(v, t, market, newest)
        ok, why = decide(result, now["exam"])
        entry = {"key": step.key, "title": step.title, "why": step.why, "on": now["name"], "decided": today,
                 "accepted": ok, "reasons": why, "exam": result, "source": source}
        if ok:
            name = f"v{len(ladder['versions']) + 1}"
            ladder["versions"].append({"name": name, "title": step.title, "since": today, "exam": result,
                                       "from": now["name"], "step": step.key})
            ladder["current"] = entry["became"] = name
        ladder["steps"].append(entry)
        if journal is not None:
            for period, years in (("2018–2024", fa.LEARN_YEARS), ("2025", (2025,)), ("2026", (2026,))):
                journal.result(f"RZ2-{step.key}", period, {
                    "accepted": ok, "against": now["name"], "learn_yearly_pct": result["learn_yearly_pct"],
                    "years": {str(y): result[str(y)] for y in years}, "robust": result["robust"],
                    "liquidations": result["liquidations"], "reasons": why, "events": source})
    ladder["updated"] = time.strftime("%Y-%m-%d %H:%M")
    ladder["newest_hour"] = int(newest)
    return ladder
