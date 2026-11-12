"""The strategy factory's night (core/brain/factory.py): breed, book, judge, send survivors to paper.

    python scripts/brain_factory.py [--batch 300] [--retry 100] [--workers 3] [--seed N]

Reads the newest event library run, books new genomes (bred from the best of each goal, a fifth
random) and genomes booked on older data, appends every booking to runtime/learning/factory/pool.jsonl
(the record of every try), judges each goal's finalists against random entries and on 2025-2026, and
adds candidates to runtime/learning/factory/promoted.json for the paper watcher. Report next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.console import utf8_output  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import factory as fa  # noqa: E402
from core.brain import regime as rg  # noqa: E402
from core.brain import robust as rb  # noqa: E402
from dataclasses import replace  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402
from core.sandbox.evaluate import t_cluster  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
POOL = fa.FACTORY_DIR / "pool.jsonl"
PROMOTED = fa.FACTORY_DIR / "promoted.json"
SWITCHERS = fa.FACTORY_DIR / "switchers.json"
HOLDS = (("BTC", 1, "держать BTC"), ("ETH", 1, "держать ETH"), ("BTC", -1, "шорт BTC (1x)"))
_W: dict = {}


def _init(table: pd.DataFrame, newest: int) -> None:
    _W.update(table=table, newest=newest, market=sim.Market(), memo={})


def book_many(genomes: list[dict]) -> list[dict]:
    return [fa.book(fa.Genome.from_dict(g), _W["table"], _W["market"], _W["newest"], _W["memo"]) for g in genomes]


def table_for(source: Path, market) -> pd.DataFrame:
    """The trade table of one event library run, built once and kept next to the pool."""
    path = fa.FACTORY_DIR / f"table-{source.stem}-{fa.TABLE_VERSION}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    ev_path = BRAIN_DIR / "evaluator-exam.pkl"
    table = fa.trade_table(pd.read_pickle(source), market, pd.read_pickle(ev_path) if ev_path.exists() else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in fa.FACTORY_DIR.glob("table-*.pkl"):
        old.unlink()
    pd.to_pickle(table, path)
    return table


def judge(e: dict, goal: str, table: pd.DataFrame, market, newest: int, bar_t: float) -> dict:
    """A finalist against random entries in 2018-2024 and on 2025-2026."""
    g = fa.Genome.from_dict(e["genome"])
    trades = fa.walk_all(fa.chosen(g, table), g, market, newest)
    learn = trades[trades["year"].isin(fa.LEARN_YEARS)].reset_index(drop=True)
    twins = fa.twin_nets(learn, g, table, market, newest)
    excess = learn["net"].to_numpy(dtype=float) - twins
    ok = np.isfinite(excess)
    weeks = pd.to_datetime(learn["signal_ts"], unit="ms").dt.strftime("%G-W%V").to_numpy()
    t = t_cluster(excess[ok], weeks[ok]) if ok.sum() > 1 else float("nan")
    held = fa.holds(e, goal)
    if np.isfinite(t) and t >= bar_t and held:
        verdict = "КАНДИДАТ"
    elif held:
        verdict = "держится в 2025–2026, но ниже планки попыток"
    else:
        verdict = "не выдержал 2025–2026"
    return {"excess_pct": 100 * float(np.nanmean(excess)) if ok.any() else float("nan"), "t": t, "verdict": verdict}


def plain_ways(regimes: pd.DataFrame, market, newest: int) -> dict:
    """Per regime: hold BTC, hold ETH, short BTC through it (USDT otherwise), booked like a genome."""
    assets, funding = {"BTC": market.btc(), "ETH": market.prices("ETH")}, {"BTC": market.funding("BTC")}
    label = regimes["regime"].to_numpy()
    years_of = pd.to_datetime(regimes["ts"], unit="ms").dt.year.to_numpy()
    out = {}
    for r in rg.REGIMES:
        out[r] = []
        for asset, side, title in HOLDS:
            years = {}
            for y in fa.YEARS:
                s = rg.curve_stats(rg.hold_curve(regimes, assets[asset], r, side, funding.get(asset),
                                                 *fa.year_span(y, newest)))
                years[str(y)] = {**s, "n": int(((label == r) & (years_of == y)).sum()), "up_of10": float("nan")}
            out[r].append({"kind": "hold", "asset": asset, "side": side, "title": title, "learn_n": 10 ** 6,
                           "years": years})
    return out


def fit_on(stats: dict, goal: str, years: list[int], how: str) -> float:
    """A goal's score on some years: every year within its drawdown, at most a quarter of them losing;
    "mean": the yearly growth, "worst": the worst year (a choice that has to hold everywhere)."""
    if stats.get("learn_n", 0) < fa.MIN_LEARN_TRADES:
        return float("-inf")
    limit = -100 * fa.GOALS[goal]["drawdown"]
    if any(str(y) not in stats["years"] for y in years):
        return float("-inf")
    ys = [stats["years"][str(y)] for y in years]
    if any(y["drawdown_pct"] < limit for y in ys) or sum(y["growth_pct"] < 0 for y in ys) > fa.allowed_losses(len(ys)):
        return float("-inf")
    if how == "worst":
        return min(y["growth_pct"] for y in ys)
    growth = np.prod([1 + y["growth_pct"] / 100 for y in ys])
    return 100 * (growth ** (1 / len(ys)) - 1) if growth > 0 else float("-inf")


def panic_in(table: pd.DataFrame, market, newest: int) -> dict:
    """The panic (the factory's first seed, on paper since 30.10.2026) gated to each regime, booked."""
    out = {}
    for r in rg.REGIMES:
        g = fa.normal(replace(fa.seeds()[0], regime=r))
        out[r] = {**fa.book(g, table, market, newest), "genome": g.to_dict()}
    return out


def champions(goal: str, current: dict, ways: dict, years: list[int], how: str,
              panic: Optional[dict] = None) -> dict:
    """Per regime the best on ``years`` for the goal: a specialist, a plain way through, or USDT (0).
    With ``panic`` the only specialist is the panic in that regime: five choices, not hundreds."""
    out = {}
    for r in rg.REGIMES:
        best, best_f = {"kind": "cash", "title": "сидеть в USDT"}, 0.0
        for w in ways[r]:
            f = fit_on(w, goal, years, how)
            if f > best_f:
                best, best_f = {k: w[k] for k in ("kind", "asset", "side", "title")}, f
        for e in ([panic[r]] if panic else fa.niche(current, r).values()):
            f = fit_on(e, goal, years, how)
            if f > best_f:
                best, best_f = {"kind": "genome", "id": e["id"], "genome": e["genome"]}, f
        if best["kind"] == "genome":
            best["title"] = fa.describe(fa.Genome.from_dict(best["genome"]))
        out[r] = {**best, "fitness": best_f}
    return out


def single_best(goal: str, current: dict, years: list[int], how: str) -> Optional[dict]:
    """The one genome for all markets that the same choice on ``years`` would take."""
    scored = [(fit_on(e, goal, years, how), e) for e in current.values()]
    scored = [x for x in scored if np.isfinite(x[0])]
    return max(scored, key=lambda x: x[0])[1] if scored else None


def switcher_year(champs: dict, year_or_span, regimes: pd.DataFrame, table: pd.DataFrame, market,
                  newest: int, cache: dict) -> dict:
    """The switcher's $100 through one year (or a (start, end) span) with these champions."""
    trades = {}
    for c in champs.values():
        if c["kind"] == "genome" and c["id"] not in trades:
            if c["id"] not in cache:
                g = fa.Genome.from_dict(c["genome"])
                cache[c["id"]] = fa.walk_all(fa.chosen(g, table), g, market, newest)
            trades[c["id"]] = cache[c["id"]]
    span = fa.year_span(year_or_span, newest) if isinstance(year_or_span, int) else year_or_span
    curve, done = rg.switch_curve(regimes, champs, trades, {"BTC": market.btc(), "ETH": market.prices("ETH")},
                                  {"BTC": market.funding("BTC")}, *span)
    return {**rg.curve_stats(curve), **done}


TESTS = (2020, 2021, 2022, 2023, 2024, 2025, 2026)     # each chosen on the years before it


def learn_line(e: dict) -> str:
    ys = [e["years"][str(y)] for y in fa.LEARN_YEARS]
    worst = min(y["drawdown_pct"] for y in ys)
    trades = sum(y["n"] for y in ys)
    return f"{trades:,} сделок · худшая просадка {fmt(worst, 0)}%"


def year_cell(e: dict, y: int) -> str:
    x = e["years"][str(y)]
    return (f"{x['n']:,} · {fmt(x['up_of10'], 1)} из 10 · {fmt(x['growth_pct'], 0, True)}% · "
            f"просадка {fmt(x['drawdown_pct'], 0)}%")


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=fa.BATCH)
    parser.add_argument("--retry", type=int, default=fa.RETRY)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    source = sorted(BRAIN_DIR.glob("events-*.pkl"))[-1]
    stamp = source.stem
    market = sim.Market()
    newest = int(market.btc()[-1, 0])
    table = table_for(source, market)
    print(f"table {len(table):,} events from {source.name}: {time.time() - started:.0f} s", flush=True)
    pool = fa.load_pool(POOL)
    rng = np.random.default_rng(args.seed or int(time.time()))
    fresh, again = fa.breed(pool, stamp, rng, args.batch, args.retry)
    todo = [g.to_dict() for g in fresh + again]
    chunks = [todo[i:i + 10] for i in range(0, len(todo), 10)]
    at = time.strftime("%Y-%m-%d %H:%M")
    booked = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(table, newest)) as pool_ex:
        for k, res in enumerate(pool_ex.map(book_many, chunks), 1):
            booked += [{**r, "stamp": stamp, "at": at} for r in res]
            if k % 5 == 0 or k == len(chunks):
                print(f"  {len(booked)}/{len(todo)} booked: {time.time() - started:.0f} s", flush=True)
    fa.append_pool(POOL, booked)
    pool.update({e["id"]: e for e in booked})

    journal = Journal()
    journal.register("FACTORY", "фабрика стратегий: ночной отбор из кубиков библиотеки событий",
                     "сочетания событий, условий, выбора монет и выходов, которые никто не придумывал руками",
                     "genomes bred from the best of Разгон/Сейф on 2018-2024, a fifth random; candidates must beat "
                     "random entries past the tries bar and grow in 2025 and 2026 within the goal's drawdown")
    ev_tries = len({e["id"] for e in journal.entries() if e["event"] == "result" and e["id"].startswith("EV-")})
    tries = len(pool) + ev_tries
    bar_t = fa.bar(tries)
    current = {i: e for i, e in pool.items() if e.get("stamp") == stamp}
    finals = {goal: fa.ranked(current, goal)[:fa.FINALISTS] for goal in fa.GOALS}
    verdicts = {(goal, e["id"]): judge(e, goal, table, market, newest, bar_t)
                for goal, es in finals.items() for e in es}
    promoted = json.loads(PROMOTED.read_text(encoding="utf-8")) if PROMOTED.exists() else []
    on_paper = {p["id"] for p in promoted}
    added = []
    for (goal, gid), v in verdicts.items():
        live = fa.live_ready(fa.Genome.from_dict(pool[gid]["genome"]))
        if v["verdict"] == "КАНДИДАТ" and not live:
            v["verdict"] = "КАНДИДАТ, но вживую дозорный его пока не видит (фандинг, толпа, листинг или оценщик)"
        if v["verdict"] == "КАНДИДАТ" and gid not in on_paper:
            ok, failed = rb.verdict(rb.mutations(fa.Genome.from_dict(pool[gid]["genome"]), table, market, newest), goal)
            if not ok:              # plan item 1: only what holds when history is bent goes to paper
                v["verdict"] = "КАНДИДАТ, но хрупкий: " + (", ".join(rb.MUTATIONS[m] for m in failed) or "без искажений")
        if v["verdict"] == "КАНДИДАТ" and gid not in on_paper and len(promoted) < fa.MAX_PAPER:
            e = pool[gid]
            promoted.append({"id": gid, "goal": goal, "genome": e["genome"],
                             "title": fa.describe(fa.Genome.from_dict(e["genome"])), "promoted_at": at,
                             "years": e["years"], "excess_pct": v["excess_pct"], "t": v["t"], "stamp": stamp})
            on_paper.add(gid)
            added.append(gid)
    if added:
        PROMOTED.write_text(json.dumps(promoted, ensure_ascii=False, indent=1, default=fa._plain), encoding="utf-8")
    candidates = sum(v["verdict"].startswith("КАНДИДАТ") for v in verdicts.values())
    journal.result("FACTORY", at, {"tries": tries, "new": len(fresh), "rebooked": len(again), "bar_t": bar_t,
                                   "candidates": candidates, "promoted": added, "events": stamp})

    lines = [f"# Фабрика стратегий — {at}", "",
             f"За ночь: новых вариантов {len(fresh)}, перепроверено на свежих данных {len(again)}; всего попыток "
             f"{tries:,} (с библиотекой событий), поэтому планка t ≥ {bar_t:.2f}. Данные: {source.name}. Каждый вариант — "
             f"портфель $100 в каждом году отдельно ($20 на сделку, не больше 5 сразу), вход через 10 минут, "
             f"комиссии, проскальзывание, фандинг шорта. Поиск видит только 2018–2024; 2025 и 2026 — проверка, "
             f"но события библиотеки выбраны после того, как их видели, — чистый судья только бумага. "
             f"Прогон {time.time() - started:.0f} с.", ""]
    for goal, spec in fa.GOALS.items():
        lines += [f"## {spec['title']} (просадка не больше {100 * spec['drawdown']:.0f}% в каждом году)", "",
                  "| вариант | 2018–2024 в год | 2018–2024 | 2025: сделок · в плюс · рост · просадка | 2026 | "
                  "лучше случайного (t) | итог |", "|---|---|---|---|---|---|---|"]
        if not finals[goal]:
            lines.append("| ни один вариант не удержал просадку и 3 года из 4 без убытка | | | | | | |")
        for e in finals[goal]:
            v = verdicts[(goal, e["id"])]
            lines.append(f"| {fa.describe(fa.Genome.from_dict(e['genome']))} | {fmt(fa.fitness(e, goal), 0, True)}% | "
                         f"{learn_line(e)} | {year_cell(e, 2025)} | {year_cell(e, 2026)} | "
                         f"{fmt(v['excess_pct'], 2, True)}% (t {fmt(v['t'], 1)}) | {v['verdict']} |")
        lines.append("")
    regimes = rg.btc_regimes(market.btc())
    ways = plain_ways(regimes, market, newest)
    btc_always = {str(y): rg.curve_stats(rg.hold_curve(regimes, market.btc(), None, 1, None, *fa.year_span(y, newest)))
                  for y in fa.YEARS}
    label = regimes["regime"].to_numpy()
    recent = (regimes["ts"].to_numpy() >= fa.year_span(2025, newest)[0]) & (label != "")
    switchers, walks = {}, {}
    panic = panic_in(table, market, newest)
    for goal, spec in fa.GOALS.items():
        limit = -100 * spec["drawdown"]
        lines += [f"## Переключатель по режимам — {spec['title']}", "",
                  "Каждый час бот смотрит режим рынка (тренд BTC × спокойно/нервно, новый режим — после 6 часов "
                  "подряд) и включает чемпиона этого режима: специалиста фабрики, простой путь (держать BTC/ETH, "
                  "шорт BTC) или USDT. Держание закрывается при смене режима, сделки специалиста идут до своего "
                  "выхода. Проверка вперёд: чемпионов на каждый год выбирают только по годам до него. Два способа "
                  "выбора: «в среднем» — самый быстрый рост, «в худший год» — лучший худший год. 2020–2024 "
                  "наполовину честные (фабрика выводила варианты, видя их), 2025–2026 — честные.", "",
                  "| год | переключатель «в среднем» | переключатель «в худший год» | простой переключатель | "
                  "одна стратегия «в среднем» | одна «в худший год» | держать BTC |", "|---|---|---|---|---|---|---|"]
        rows = {}
        for y in TESTS:
            train = [x for x in fa.YEARS if x < y]
            for how in ("mean", "worst"):
                champs = champions(goal, current, ways, train, how)
                rows[(y, how)] = (switcher_year(champs, y, regimes, table, market, newest, walks), champs)
            champs = champions(goal, current, ways, train, "mean", panic)
            rows[(y, "simple")] = (switcher_year(champs, y, regimes, table, market, newest, walks), champs)
            for how in ("mean", "worst"):
                one = single_best(goal, current, train, how)
                rows[(y, how, "one")] = one["years"][str(y)] if one else None
            def cell(s):
                return f"{fmt(s['growth_pct'], 0, True)}% · {fmt(s['drawdown_pct'], 0)}%" if s else "—"
            lines.append(f"| {y} | {cell(rows[(y, 'mean')][0])} | {cell(rows[(y, 'worst')][0])} | "
                         f"{cell(rows[(y, 'simple')][0])} | "
                         f"{cell(rows[(y, 'mean', 'one')])} | {cell(rows[(y, 'worst', 'one')])} | "
                         f"{cell(btc_always[str(y)])} |")
        for how, title in (("mean", "в среднем"), ("worst", "в худший год"), ("simple", "простой")):
            champs = champions(goal, current, ways, list(fa.LEARN_YEARS), "mean" if how == "simple" else how,
                               panic if how == "simple" else None)
            since = switcher_year(champs, (fa.year_span(2025, newest)[0], newest + 3_600_000), regimes, table, market,
                                  newest, walks)
            held = all(rows[(y, how)][0]["growth_pct"] > 0 and rows[(y, how)][0]["drawdown_pct"] >= limit
                       for y in fa.CHECK_YEARS)
            switchers[f"{goal}-{how}"] = {"goal": goal, "how": how, "champions": champs, "since_2025": since,
                                          "tests": {str(y): rows[(y, how)][0] for y in TESTS}, "holds": held,
                                          "at": at, "stamp": stamp}
            lines += ["", f"Чемпионы «{title}» (по 2018–2024), с ними с 1.1.2025: ${100 + since['growth_pct']:.0f}, "
                          f"просадка {fmt(since['drawdown_pct'], 0)}% — "
                          f"**{'держится в 2025 и 2026' if held else 'в 2025–2026 не удержался'}**:", "",
                      "| режим | доля времени 2025–2026 | чемпион | балл на 2018–2024 |", "|---|---|---|---|"]
            for r in rg.REGIMES:
                c = champs[r]
                share = 100 * float((label[recent] == r).mean()) if recent.any() else float("nan")
                lines.append(f"| {rg.ru(r)} | {fmt(share, 0)}% | {c['title']} | {fmt(c['fitness'], 0, True)}% |")
        lines.append("")
    SWITCHERS.write_text(json.dumps(switchers, ensure_ascii=False, indent=1, default=fa._plain), encoding="utf-8")
    lines += ["## На бумаге", ""]
    if promoted:
        for p in promoted:
            lines.append(f"- {fa.GOALS[p['goal']]['title']}: {p['title']} (с {p['promoted_at']}"
                         + (", новый" if p["id"] in added else "") + ")")
    else:
        lines.append("- Пока никого: ни один финалист не прошёл планку попыток и 2025–2026.")
    report = fa.FACTORY_DIR / f"factory-{time.strftime('%Y%m%d-%H%M')}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
