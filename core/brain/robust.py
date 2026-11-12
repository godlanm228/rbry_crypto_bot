"""Robustness (plan item 1): does a strategy hold when history is bent, and is choosing by history luck?

A strategy's own trades are bent four ways and booked as the factory books them ($100 in every year,
$20 a trade, at most 5 open, core/brain/factory.py):
  late      the signal seen an hour late: entered at the next hour's open, no 10-minute edge
  costs     fees and slippage paid twice over
  missed    a fifth of the signals missed at random (100 draws; the bad tenth of them counts)
  no-best   without its best 5% of trades
A strategy is robust when every bent version still earns and keeps its goal's drawdown: over
2018-2024 it grows in all (compounded) with every year's drawdown within the goal, and it grows in
2025 and in 2026 within it. The factory's own rule (at most a quarter of the years losing) is a rule
for choosing and is shown beside it, not asked of a bent history: a strategy that earns in crashes
and waits in between loses a quiet year more as soon as costs double (, the panic).
Crises show its trades in known shocks (COVID, the China ban, LUNA, FTX, the yen unwind).

The factory's choice gets its own number, PBO (the probability of backtest overfitting, Bailey and
Lopez de Prado): for every split of the 7 learning years into 4 to choose on and 3 to check, the genome
the goal's score would choose: does it land in the upper half of all genomes on the 3 years it never
saw? The share of splits where it does not is the chance that choosing by history is luck; 50% is a
coin toss.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from core.brain import exits as ex
from core.brain import factory as fa
from core.brain.dataset import FEE_PCT, FUTURES_FEE_PCT
from core.sandbox.market_store import HOUR_MS

DRAWS = 100
MISS = 0.2
BEST = 0.05
CRISES = (("ковид", "2020-03-12", "2020-03-14"), ("запрет в Китае", "2021-05-18", "2021-05-21"),
          ("LUNA", "2022-05-07", "2022-05-14"), ("FTX", "2022-11-07", "2022-11-12"),
          ("иена", "2024-08-04", "2024-08-07"), ("10.10.2025", "2025-10-10", "2025-10-12"),
          ("взлом Binance", "2026-10-29", "2026-10-31"))
MUTATIONS = {"late": "вход на час позже", "costs": "комиссии ×2", "missed": "пропуск 20% сигналов (плохие 10%)",
             "no_best": "без лучших 5% сделок"}


def score(stats: dict, goal: str, years, how: str = "mean") -> float:
    """A goal's score on ``years``: every year within its drawdown, at most a quarter of them losing; "mean":
    the yearly growth, "worst": the worst year."""
    if stats.get("learn_n", 0) < fa.MIN_LEARN_TRADES or any(str(y) not in stats["years"] for y in years):
        return float("-inf")
    limit = -100 * fa.GOALS[goal]["drawdown"]
    ys = [stats["years"][str(y)] for y in years]
    if any(y["drawdown_pct"] < limit for y in ys) or sum(y["growth_pct"] < 0 for y in ys) > fa.allowed_losses(len(ys)):
        return float("-inf")
    if how == "worst":
        return min(y["growth_pct"] for y in ys)
    growth = np.prod([1 + y["growth_pct"] / 100 for y in ys])
    return 100 * (growth ** (1 / len(ys)) - 1) if growth > 0 else float("-inf")


def books(trades: pd.DataFrame, newest: int) -> dict:
    """Trades booked like a genome: a $100 portfolio in every year."""
    years = {}
    for y in fa.YEARS:
        ty = trades[trades["year"] == y]
        b = ex.portfolio(ty, *fa.year_span(y, newest))
        years[str(y)] = {"n": len(ty), "growth_pct": b["growth_pct"], "drawdown_pct": b["drawdown_pct"],
                         "usd_per_100": 100 * float(ty["net"].mean()) if len(ty) else float("nan")}
    learn = trades[trades["year"].isin(fa.LEARN_YEARS)]
    return {"years": years, "learn_n": len(learn)}


def mutations(g: fa.Genome, table: pd.DataFrame, market, newest: int, seed: int = 5) -> dict:
    """The genome's books as it is and bent four ways."""
    s = fa.chosen(g, table)
    base = fa.walk_all(s, g, market, newest)
    out = {"base": books(base, newest)}
    late = s.assign(signal_ts=s["signal_ts"] + HOUR_MS, delta=0.0)
    out["late"] = books(fa.walk_all(late, g, market, newest), newest)
    fee = (FEE_PCT if g.side > 0 else FUTURES_FEE_PCT) / 100.0
    slip = base.merge(s[["history", "signal_ts", "slip"]], on=["history", "signal_ts"], how="left")["slip"].fillna(0.001)
    out["costs"] = books(base.assign(net=base["net"] - 2 * (fee + slip.to_numpy())), newest)
    rng = np.random.default_rng(seed)
    draws = [books(base[rng.random(len(base)) >= MISS], newest) for _ in range(DRAWS)]
    out["missed"] = bad_tenth(draws)
    cut = base["net"].quantile(1 - BEST) if len(base) else 0.0
    out["no_best"] = books(base[base["net"] < cut] if len(base) > 20 else base, newest)
    out["trades"] = base
    return out


def bad_tenth(draws: list[dict]) -> dict:
    """Per year the 10th percentile of growth and drawdown over the draws (each on its own)."""
    years = {}
    for y in fa.YEARS:
        g = [d["years"][str(y)]["growth_pct"] for d in draws]
        dd = [d["years"][str(y)]["drawdown_pct"] for d in draws]
        n = [d["years"][str(y)]["n"] for d in draws]
        years[str(y)] = {"n": int(np.median(n)), "growth_pct": float(np.quantile(g, 0.1)),
                         "drawdown_pct": float(np.quantile(dd, 0.1)), "usd_per_100": float("nan")}
    return {"years": years, "learn_n": int(np.median([d["learn_n"] for d in draws]))}


def learn_growth(stats: dict) -> float:
    """2018-2024 compounded, a year's growth."""
    ys = [stats["years"][str(y)]["growth_pct"] for y in fa.LEARN_YEARS]
    total = float(np.prod([1 + g / 100 for g in ys]))
    return 100 * (total ** (1 / len(ys)) - 1) if total > 0 else -100.0


def holds(stats: dict, goal: str) -> bool:
    """Grows over 2018-2024 with every year's drawdown within the goal, and grows in 2025 and in 2026 within it."""
    limit = -100 * fa.GOALS[goal]["drawdown"]
    if any(stats["years"][str(y)]["drawdown_pct"] < limit for y in fa.LEARN_YEARS):
        return False
    return learn_growth(stats) > 0 and fa.holds(stats, goal)


def strict(stats: dict, goal: str) -> bool:
    """The factory's choosing rule as well: at most a quarter of the learning years losing."""
    return holds(stats, goal) and score(stats, goal, fa.LEARN_YEARS) > float("-inf")


def verdict(muts: dict, goal: str) -> tuple[bool, list[str]]:
    """(robust, the mutations it fails)."""
    failed = [name for name in MUTATIONS if not holds(muts[name], goal)]
    return not failed and holds(muts["base"], goal), failed


def crises(trades: pd.DataFrame) -> list[dict]:
    """Per known shock: the trades open during it and what they ended with."""
    out = []
    for name, a, b in CRISES:
        lo, hi = int(pd.Timestamp(a).value // 10**6), int(pd.Timestamp(b).value // 10**6)
        end = trades["signal_ts"] + trades["held"] * HOUR_MS
        t = trades[(trades["signal_ts"] < hi) & (end > lo)]
        out.append({"crisis": name, "trades": len(t),
                    "usd_per_100": 100 * float(t["net"].mean()) if len(t) else float("nan"),
                    "worst_pct": 100 * float(t["net"].min()) if len(t) else float("nan")})
    return out


def pbo(pool: dict, goal: str, years=fa.LEARN_YEARS, train: int = 4) -> dict:
    """The chance that the goal's choice on some learning years is luck on the others."""
    entries = [e for e in pool.values() if all(str(y) in e["years"] for y in years)]
    if len(entries) < 10:
        return {"pbo": float("nan"), "splits": 0, "median_rank": float("nan")}
    growth = {y: np.array([e["years"][str(y)]["growth_pct"] for e in entries]) for y in years}
    ranks = []
    for chosen_years in itertools.combinations(years, train):
        rest = [y for y in years if y not in chosen_years]
        scores = np.array([score(e, goal, chosen_years) for e in entries])
        if not np.isfinite(scores).any():
            continue
        best = int(np.nanargmax(np.where(np.isfinite(scores), scores, -np.inf)))
        test = np.prod([1 + growth[y] / 100 for y in rest], axis=0)
        ranks.append(float((test < test[best]).mean()))
    if not ranks:
        return {"pbo": float("nan"), "splits": 0, "median_rank": float("nan")}
    return {"pbo": float(np.mean(np.array(ranks) < 0.5)), "splits": len(ranks), "median_rank": float(np.median(ranks))}
