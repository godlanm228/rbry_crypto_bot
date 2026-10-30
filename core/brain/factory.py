"""The strategy factory (A): the bot builds and tests strategies itself, every night.

A strategy is a genome of the event library's bricks (core/brain/events.py):
  events      1-4 of the library's events
  side        buy on spot, or short a live perpetual on 1x isolated margin
  conditions  BTC's trend, a nervous market, the time of day, weekends, the coin's size: any or one value
  moment      how many coins must show the events in one hour (1: a lone coin is enough; 4+: a mass moment)
  pick        which coins of a moment: the most liquid or the most extreme, 1-3 of them, or all
  exit        hours held, a stop in usual daily moves or a fixed one, half out at a gain, a target,
              a trailing floor (core/brain/exits.py)
  squeeze     shorts only: skip a coin whose squeeze chance (the trade evaluator, the day before) is higher
  regime      trade only in one market regime (core/brain/regime.py): a specialist for that situation
A genome is booked year by year: each year a $100 portfolio from its first day ($20 a trade, at most
5 open) on hourly paths, entry 10 minutes after the signal, fees, slippage and a short's funding.

Two goals (Vlad, 30.10.2026): «Разгон» grows fastest with a drawdown within 50% in every year,
«Сейф» within 20%. The search is guided by 2018-2024 alone: every night new genomes are bred from the
best of each goal (crossover, mutation), a fifth are random (exploration), and genomes booked on older
data are booked again on the newest, the best and a random share of the rest: a loser of old data is
retried on new data, nothing is bad forever. Every genome ever booked is a try; the bar on a finalist's
2018-2024 edge over random entries (the same exit from random hours of the coin's month) rises with
them. A finalist that clears it and grows in 2025 and in 2026 within its goal's drawdown is a
candidate, and candidates go to the paper watcher, where the live market judges them. 2025-2026 never
guide the search, but the library's events were chosen after seeing them: only paper is clean.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

from core.brain import events as ev
from core.brain import exits as ex
from core.brain import regime as rg
from core.brain.dataset import DAY_MS, FUTURES_FEE_PCT
from core.sandbox.evaluate import t_cluster
from core.sandbox.market_store import HOUR_MS

FACTORY_DIR = Path(__file__).resolve().parents[2] / "runtime" / "learning" / "factory"
YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)
LEARN_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
CHECK_YEARS = (2025, 2026)
GOALS = {"razgon": {"title": "Разгон", "drawdown": 0.50}, "seif": {"title": "Сейф", "drawdown": 0.20}}
MIN_LEARN_TRADES = 40
MIN_CHECK_TRADES = 5
def allowed_losses(years: int) -> int:
    """How many losing years a goal forgives: a quarter, rounded (1 of 4, 2 of 7)."""
    return int(years / 4 + 0.5)
ALPHA = 0.05
BATCH = 300
EXPLORE = 0.2
RETRY = 100
FINALISTS = 8
MAX_PAPER = 6
PARENTS = 20

EVENT_IDS = tuple(e.id for e in ev.EVENTS)
# what the paper watcher can see live (core/brain/watch.py): no funding, crowd or listing feed there yet,
# and the trade evaluator runs on the laptop only
LIVE_EVENTS = frozenset(EVENT_IDS) - {"FUND-HIGH", "FUND-LOW", "CROWD-LONG", "CROWD-SHORT", "LISTING-24H"}
HOURS = (4, 24, 72, 168)
STOPS = (None, ("sigma", 2.0), ("sigma", 3.0), ("pct", 0.10), ("pct", 0.30))
HALVES = (None, 0.03, 0.05, 0.10)
TAKES = (None, 2.0, 3.0)
KEEPS = (None, 0.5)
MASSES = (1, 2, 4, 8, 11)
PICK_WAYS = ("liquid", "extreme", "all")
KS = (1, 2, 3)
SQUEEZES = (None, 0.2, 0.35)
REGIME_GENES = (None,) + rg.REGIMES
NICHES = tuple((goal, r) for goal in ("razgon", "seif") for r in REGIME_GENES)   # each breeds its own best
TABLE_VERSION = "v2"                 # v2: the regime of every event
DAYPARTS = (None, "ночь", "утро", "день", "вечер")
FLAGS = (None, True, False)

SHORT_RU = {
    "SPIKE": "свеча +3% на объёме", "DUMP": "свеча −3% на объёме", "BREAKOUT": "пробой максимума недели",
    "BREAKDOWN": "пробой минимума недели", "ABSORB-BUY": "объём без движения (покупки)",
    "ABSORB-SELL": "объём без движения (продажи)", "TAKER-BUY": "70% рыночных покупок",
    "TAKER-SELL": "70% рыночных продаж", "FUND-HIGH": "фандинг ≥ +0.1%", "FUND-LOW": "фандинг ≤ −0.1%",
    "OI-UP-UP": "OI +10% и цена +3%", "OI-UP-DOWN": "OI +10% и цена −3%", "OI-FLUSH": "волна ликвидаций",
    "CROWD-LONG": "толпа в лонгах", "CROWD-SHORT": "толпа в шортах", "STRONG-VS-BTC": "сильнее BTC",
    "CAPITULATION-TURN": "разворот после −15%", "FELL-WITH-BTC": "упала сильнее BTC",
    "LISTING-24H": "сутки после листинга"}


@dataclass(frozen=True)
class Genome:
    events: tuple
    side: int = 1
    btc: Optional[bool] = None
    nervous: Optional[bool] = None
    daypart: Optional[str] = None
    weekend: Optional[bool] = None
    big: Optional[bool] = None
    mass: int = 1
    pick: str = "liquid"
    k: int = 3
    hours: int = 24
    stop: Optional[tuple] = None          # ("sigma", usual daily moves) | ("pct", share)
    half_pct: Optional[float] = None
    take: Optional[float] = None          # in stop distances, with a sigma stop only
    keep: Optional[float] = None
    squeeze: Optional[float] = None       # shorts only
    regime: Optional[str] = None          # one of regime.REGIMES: trades only there

    @property
    def id(self) -> str:
        d = self.to_dict()
        if d["regime"] is None:
            del d["regime"]                   # genomes from before regimes keep their ids
        return "F-" + hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["events"] = sorted(d["events"])
        d["stop"] = list(d["stop"]) if d["stop"] else None
        return d

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        d = dict(d)
        d["events"] = tuple(sorted(d["events"]))
        d["stop"] = tuple(d["stop"]) if d.get("stop") else None
        return Genome(**d)

    def rule(self) -> ex.Rule:
        kind, value = self.stop if self.stop else (None, None)
        return ex.Rule(self.id, "", self.hours, stop=value if kind == "sigma" else None,
                       stop_pct=value if kind == "pct" else None, take=self.take, half_pct=self.half_pct,
                       keep=self.keep)


def normal(g: Genome) -> Genome:
    """The same strategy in one spelling: settings that cannot act are dropped."""
    events = tuple(sorted(set(g.events)))
    take = g.take if g.stop and g.stop[0] == "sigma" else None
    squeeze = g.squeeze if g.side < 0 else None
    mass = 1 if "LISTING-24H" in events else g.mass
    k = 3 if g.pick == "all" else g.k
    return replace(g, events=events, take=take, squeeze=squeeze, mass=mass, k=k)


def live_ready(g: Genome) -> bool:
    """The paper watcher can run it on live data."""
    return set(g.events) <= LIVE_EVENTS and g.squeeze is None


def random_genome(rng: np.random.Generator, regime: Optional[str] = None) -> Genome:
    n = int(rng.choice([1, 1, 2, 2, 3, 4]))
    pick = lambda xs: xs[int(rng.integers(len(xs)))]  # noqa: E731
    return normal(Genome(regime=regime,events=tuple(rng.choice(EVENT_IDS, size=n, replace=False)), side=int(pick((1, 1, -1))),
                         btc=pick(FLAGS), nervous=pick(FLAGS), daypart=pick(DAYPARTS) if rng.random() < 0.3 else None,
                         weekend=pick(FLAGS) if rng.random() < 0.3 else None, big=pick(FLAGS), mass=pick(MASSES),
                         pick=pick(PICK_WAYS), k=pick(KS), hours=pick(HOURS), stop=pick(STOPS), half_pct=pick(HALVES),
                         take=pick(TAKES), keep=pick(KEEPS) if rng.random() < 0.3 else None, squeeze=pick(SQUEEZES)))


GENES = {"btc": FLAGS, "nervous": FLAGS, "daypart": DAYPARTS, "weekend": FLAGS, "big": FLAGS, "mass": MASSES,
         "pick": PICK_WAYS, "k": KS, "hours": HOURS, "stop": STOPS, "half_pct": HALVES, "take": TAKES,
         "keep": KEEPS, "squeeze": SQUEEZES, "side": (1, -1), "regime": REGIME_GENES}


def mutate(g: Genome, rng: np.random.Generator, changes: int = 2) -> Genome:
    out = g
    for _ in range(changes):
        gene = str(rng.choice(list(GENES) + ["events"]))
        if gene == "events":
            evs = set(out.events)
            if len(evs) > 1 and rng.random() < 0.4:
                evs.discard(str(rng.choice(sorted(evs))))
            elif len(evs) < 4 and rng.random() < 0.5:
                evs.add(str(rng.choice(EVENT_IDS)))
            else:
                evs.discard(str(rng.choice(sorted(evs))))
                evs.add(str(rng.choice(EVENT_IDS)))
            out = replace(out, events=tuple(evs) or g.events)
        else:
            values = GENES[gene]
            out = replace(out, **{gene: values[int(rng.integers(len(values)))]})
    return normal(out)


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    da, db = a.to_dict(), b.to_dict()
    child = {k: (da[k] if rng.random() < 0.5 else db[k]) for k in da}
    return normal(Genome.from_dict(child))


def seeds() -> list[Genome]:
    """Where the factory starts: stage 2's panic candidates (30.10.2026)."""
    base = Genome(events=tuple(sorted(ex.PANIC_EVENTS)), side=1, nervous=True, mass=4, pick="liquid", k=3,
                  hours=72, half_pct=0.05)
    return [normal(base), normal(replace(base, stop=("pct", 0.30))), normal(replace(base, mass=11))]


def describe(g: Genome) -> str:
    """The genome in plain words."""
    parts = [("купить" if g.side > 0 else "шорт") + " после: " + " / ".join(SHORT_RU.get(e, e) for e in g.events)]
    cond = []
    if g.btc is not None:
        cond.append("BTC в росте" if g.btc else "BTC в падении")
    if g.nervous is not None:
        cond.append("рынок нервный" if g.nervous else "рынок спокойный")
    if g.daypart:
        cond.append(ev.CONDITIONS["daypart"][g.daypart])
    if g.weekend is not None:
        cond.append("выходные" if g.weekend else "будни")
    if g.big is not None:
        cond.append("крупные монеты" if g.big else "средние монеты")
    if g.regime:
        cond.append(f"режим «{rg.ru(g.regime)}»")
    if cond:
        parts.append("когда " + ", ".join(cond))
    if g.mass > 1:
        parts.append(f"момент от {g.mass} монет")
    parts.append("все монеты момента" if g.pick == "all" else
                 f"{g.k} самых {'ликвидных' if g.pick == 'liquid' else ('упавших' if g.side > 0 else 'выросших')}")
    exit_ = [f"держать {g.hours} ч"]
    if g.half_pct:
        exit_.append(f"половину на +{100 * g.half_pct:.0f}%, остаток без убытка")
    if g.stop:
        exit_.append(f"стоп {g.stop[1]:.0f} обычных хода" if g.stop[0] == "sigma" else f"стоп −{100 * g.stop[1]:.0f}%")
    if g.take:
        exit_.append(f"всё забрать на +{g.take:.0f} стопа")
    if g.keep:
        exit_.append(f"от +5% держать {100 * g.keep:.0f}% лучшего")
    if g.squeeze is not None:
        exit_.append(f"без монет с шансом выноса выше {100 * g.squeeze:.0f}%")
    parts.append(", ".join(exit_))
    return "; ".join(parts)


# -- the trades every genome picks from ----------------------------------------------------------

def trade_table(rows: pd.DataFrame, market, evaluator: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every library event with what a genome needs: conditions, liquidity, the 24-hour move, the usual
    daily move, the 10-minute reaction of its event, and the squeeze chance of the day before."""
    keep = ["event", "history", "base", "signal_ts", "daypart", "weekend", "btc", "nervous", "big", "perp", "slip",
            "turnover", "g4_0", "react4"]
    t = rows.loc[rows["history"].isin(set(market.base_of)), keep].copy()   # the hourly cache: 60+ days of trading
    react = {}
    for e, r in t.groupby("event"):
        g, x = r["g4_0"].to_numpy(dtype=float), r["react4"].to_numpy(dtype=float)
        delta = (1 + g) / (1 + g + x) - 1
        react[e] = float(np.nanmean(delta)) if np.isfinite(delta).any() else 0.0
    t["delta"] = t["event"].map(react)
    t = t.drop(columns=["g4_0", "react4"])
    fell, sigma = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for hist, idx in t.groupby("history").indices.items():
        prices = market.prices(hist)
        ts = prices[:, 0]
        for i in idx:
            a = int(np.searchsorted(ts, t["signal_ts"].iat[i]))
            if 25 <= a <= len(prices):
                fell[i] = prices[a - 1, 4] / prices[a - 25, 4] - 1
                sigma[i] = ex.daily_sigma(prices[:, 4], a)
    t["fell24"], t["sigma_day"] = fell, sigma
    t["year"] = pd.to_datetime(t["signal_ts"], unit="ms").dt.year
    t["regime"] = rg.at(rg.btc_regimes(market.btc()), t["signal_ts"].to_numpy() - HOUR_MS)   # the event's hour
    t["squeeze_chance"] = np.nan
    if evaluator is not None and len(evaluator):
        e = evaluator[["history", "day", "squeeze_chance"]].copy()
        e["history"] = e["history"].astype(str)
        day = pd.to_datetime((t["signal_ts"] // DAY_MS - 1) * DAY_MS, unit="ms")
        t = t.assign(day=day).merge(e, on=["history", "day"], how="left", suffixes=("_x", ""))
        t = t.drop(columns=["day", "squeeze_chance_x"])
    for c in ("weekend", "btc", "nervous", "big", "perp"):
        t[c] = t[c].astype(object)
    return t[np.isfinite(t["sigma_day"])].reset_index(drop=True)


def chosen(g: Genome, t: pd.DataFrame) -> pd.DataFrame:
    """The trades a genome takes: its events under its conditions, its moments, its picks."""
    m = t["event"].isin(g.events).to_numpy()
    for name in ("btc", "nervous", "weekend", "big"):
        value = getattr(g, name)
        if value is not None:
            m &= (t[name] == value).to_numpy()
    if g.daypart:
        m &= (t["daypart"] == g.daypart).to_numpy()
    if g.regime:
        m &= (t["regime"] == g.regime).to_numpy()
    if g.side < 0:
        m &= (t["perp"] == True).to_numpy()   # noqa: E712 (object column)
        if g.squeeze is not None:
            m &= (t["squeeze_chance"] <= g.squeeze).to_numpy()
    s = t[m].drop_duplicates(["history", "signal_ts"])
    if g.mass > 1:
        s = s[s.groupby("signal_ts")["history"].transform("size") >= g.mass]
    if g.pick != "all" and len(s):
        key = -s["turnover"] if g.pick == "liquid" else g.side * s["fell24"]
        s = s[(key.groupby(s["signal_ts"]).rank(method="first") <= g.k).to_numpy()]
    return s


# -- walking a trade --------------------------------------------------------------------------------

def walk(prices: np.ndarray, funding: Optional[np.ndarray], signal_ts: int, side: int, rule: ex.Rule,
         sigma_day: float, delta: float, slip: float, newest: int) -> Optional[tuple]:
    """(net, hours held, worst on the way) of one long or 1x short; None when not over by the newest data."""
    w = ex._window(prices, int(signal_ts), rule.hours, newest)
    if w is None:
        return None
    a, b = w
    o, h, l, c = (np.asarray(prices[a:b, k], dtype=float) for k in (1, 2, 3, 4))
    entry = float(o[0]) * (1 + delta)
    dist = ex._stop_dist(rule, sigma_day)
    if side > 0:
        exits, held, _, worst = ex.walk_long(o, h, l, c, entry, dist, rule)
        return ex.net_of(exits, entry, slip), held, worst
    # a short is a long on the mirrored path 2*entry - price: the same stops, halves and floors
    exits, held, _, worst = ex.walk_long(2 * entry - o, 2 * entry - l, 2 * entry - h, 2 * entry - c, entry, dist,
                                         rule)
    if float(np.max(h[:held])) >= 2 * entry:
        return -1.0, held, -1.0                                  # the margin is gone
    return net_short(exits, entry, slip, funding, int(prices[a, 0]), int(prices[a + held - 1, 0]) + HOUR_MS),         held, worst


def net_short(exits: list, entry: float, slip: float, funding: Optional[np.ndarray], start_ts: int,
              end_ts: int) -> float:
    """A 1x short from exits on its mirrored path (price 2*entry - p): fees and slippage both ways, the
    funding paid after the entry up to the exit; never below the margin."""
    fee = FUTURES_FEE_PCT / 100.0
    net = sum(s * ((1 - slip) * (1 - fee) - (2 * entry - p) / entry * (1 + slip) * (1 + fee)) for s, p in exits)
    if funding is not None and len(funding):
        f_ts = funding[:, 0]
        net += float(np.sum(funding[np.searchsorted(f_ts, start_ts, side="right"):
                                    np.searchsorted(f_ts, end_ts, side="right"), 1]))
    return max(net, -1.0)


def walk_all(s: pd.DataFrame, g: Genome, market, newest: int, memo: Optional[dict] = None) -> pd.DataFrame:
    rule = g.rule()
    sig = (g.side, g.hours, g.stop, g.half_pct, g.take, g.keep)
    out = []
    for r in s.itertuples(index=False):
        key = (r.history, int(r.signal_ts), sig)
        if memo is not None and key in memo:
            res = memo[key]
        else:
            res = walk(market.prices(r.history), market.funding(r.history) if g.side < 0 else None, r.signal_ts,
                       g.side, rule, r.sigma_day, r.delta, r.slip, newest)
            if memo is not None:
                memo[key] = res
        if res is not None:
            out.append((r.history, int(r.signal_ts), r.turnover, int(r.year), res[0], res[1], res[2]))
    return pd.DataFrame(out, columns=["history", "signal_ts", "turnover", "year", "net", "held", "worst"])


def twin_nets(trades: pd.DataFrame, g: Genome, t: pd.DataFrame, market, newest: int, seed: int = 11) -> np.ndarray:
    """Per trade: the same exit from 2 random hours of the coin's month (with that hour's usual move)."""
    rng = np.random.default_rng(seed)
    rule = g.rule()
    slips = t.drop_duplicates(["history", "signal_ts"]).set_index(["history", "signal_ts"])["slip"]
    out = np.full(len(trades), np.nan)
    for i, r in enumerate(trades.itertuples(index=False)):
        prices = market.prices(r.history)
        ts = prices[:, 0]
        first = pd.Timestamp(int(r.signal_ts), unit="ms").to_period("M").start_time
        lo = max(int(np.searchsorted(ts, first.value // 10**6)), ex.SIGMA_HOURS)
        hi = int(np.searchsorted(ts, (first + pd.offsets.MonthBegin(1)).value // 10**6))
        if hi <= lo:
            continue
        slip = float(slips.get((r.history, r.signal_ts), 0.001))
        nets = []
        for a in rng.integers(lo, hi, size=ex.TWINS):
            sigma = ex.daily_sigma(prices[:, 4], int(a))
            res = walk(prices, market.funding(r.history) if g.side < 0 else None, int(ts[a]), g.side, rule,
                       sigma, 0.0, slip, newest) if math.isfinite(sigma) else None
            if res is not None:
                nets.append(res[0])
        if nets:
            out[i] = float(np.mean(nets))
    return out


# -- booking a genome ------------------------------------------------------------------------------

def year_span(year: int, newest: int) -> tuple[int, int]:
    start = int(pd.Timestamp(f"{year}-01-01").value // 10**6)
    return start, min(int(pd.Timestamp(f"{year + 1}-01-01").value // 10**6), newest + HOUR_MS)


def book(g: Genome, t: pd.DataFrame, market, newest: int, memo: Optional[dict] = None) -> dict:
    """A genome's trades and its $100 portfolio in every year."""
    trades = walk_all(chosen(g, t), g, market, newest, memo)
    years = {}
    for y in YEARS:
        ty = trades[trades["year"] == y]
        b = ex.portfolio(ty, *year_span(y, newest))
        years[str(y)] = {"n": len(ty), "usd_per_100": 100 * float(ty["net"].mean()) if len(ty) else float("nan"),
                         "up_of10": 10 * float((ty["net"] > 0).mean()) if len(ty) else float("nan"),
                         "growth_pct": b["growth_pct"], "drawdown_pct": b["drawdown_pct"], "taken": b["trades"]}
    learn = trades[trades["year"].isin(LEARN_YEARS)]
    weeks = pd.to_datetime(learn["signal_ts"], unit="ms").dt.strftime("%G-W%V").to_numpy()
    return {"id": g.id, "genome": g.to_dict(), "years": years, "learn_n": len(learn),
            "learn_t": t_cluster(learn["net"].to_numpy(dtype=float), weeks) if len(learn) > 1 else float("nan")}


def fitness(stats: dict, goal: str) -> float:
    """The yearly growth over 2018-2024 if every learn year kept the goal's drawdown and at most a quarter of
    them lost; -inf otherwise."""
    limit = -100 * GOALS[goal]["drawdown"]
    if stats.get("learn_n", 0) < MIN_LEARN_TRADES or any(str(y) not in stats["years"] for y in LEARN_YEARS):
        return float("-inf")                 # booked before 2018-2020 were in: rebooked on the newest data first
    ys = [stats["years"][str(y)] for y in LEARN_YEARS]
    if any(y["drawdown_pct"] < limit for y in ys):
        return float("-inf")
    if sum(y["growth_pct"] < 0 for y in ys) > allowed_losses(len(ys)):
        return float("-inf")
    growth = np.prod([1 + y["growth_pct"] / 100 for y in ys])
    return 100 * (growth ** (1 / len(ys)) - 1) if growth > 0 else float("-inf")


def holds(stats: dict, goal: str) -> bool:
    """2025 and 2026, never seen by the search: growth within the goal's drawdown, enough trades."""
    limit = -100 * GOALS[goal]["drawdown"]
    return all(stats["years"][str(y)]["n"] >= MIN_CHECK_TRADES and stats["years"][str(y)]["growth_pct"] > 0
               and stats["years"][str(y)]["drawdown_pct"] >= limit for y in CHECK_YEARS)


def bar(tries: int) -> float:
    return NormalDist().inv_cdf(1 - ALPHA / (2 * max(tries, 1)))


# -- the pool and the night's breeding ----------------------------------------------------------------

def load_pool(path: Path) -> dict[str, dict]:
    """Every genome ever booked, its latest booking."""
    pool: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                pool[e["id"]] = e
    return pool


def append_pool(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False, default=_plain) + "\n")


def _plain(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def ranked(pool: dict[str, dict], goal: str) -> list[dict]:
    scored = [(fitness(e, goal), e) for e in pool.values()]
    return [e for f, e in sorted(scored, key=lambda x: -x[0]) if math.isfinite(f)]


def niche(pool: dict[str, dict], regime: Optional[str]) -> dict[str, dict]:
    return {i: e for i, e in pool.items() if e["genome"].get("regime") == regime}


def breed(pool: dict[str, dict], stamp: str, rng: np.random.Generator, batch: int = BATCH,
          retry: int = RETRY) -> tuple[list[Genome], list[Genome]]:
    """(new genomes, genomes booked on older data to book again). Every niche (a goal, and all markets or
    one regime) breeds from its own best, so each regime gets specialists of its own."""
    known = set(pool)
    fresh: list[Genome] = [g for g in seeds() if g.id not in known]
    parents = {(goal, r): [Genome.from_dict(e["genome"]) for e in ranked(niche(pool, r), goal)[:PARENTS]]
               for goal, r in NICHES}
    tries = 0
    while len(fresh) < batch and tries < batch * 20:
        tries += 1
        goal, regime = NICHES[int(rng.integers(len(NICHES)))]
        ps = parents[(goal, regime)]
        if not ps or rng.random() < EXPLORE:
            g = random_genome(rng, regime)
        elif len(ps) > 1 and rng.random() < 0.4:
            i, j = rng.choice(len(ps), size=2, replace=False)
            g = mutate(crossover(ps[i], ps[j], rng), rng, changes=1)
        else:
            g = mutate(ps[int(rng.integers(len(ps)))], rng, changes=int(rng.integers(1, 3)))
        if regime and g.regime != regime:
            g = normal(replace(g, regime=regime))          # a niche's child stays in its niche
        if g.id not in known:
            known.add(g.id)
            fresh.append(g)
    stale = [e for e in pool.values() if e.get("stamp") != stamp]
    best = {e["id"] for goal in GOALS for e in ranked({e["id"]: e for e in stale}, goal)[:retry // 2]}
    rest = [e["id"] for e in stale if e["id"] not in best]
    rng.shuffle(rest)
    again = [Genome.from_dict(pool[i]["genome"]) for i in list(best) + rest[:max(0, retry - len(best))]]
    return fresh, again
