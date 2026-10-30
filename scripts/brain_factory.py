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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import factory as fa  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402
from core.sandbox.evaluate import t_cluster  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
POOL = fa.FACTORY_DIR / "pool.jsonl"
PROMOTED = fa.FACTORY_DIR / "promoted.json"
_W: dict = {}


def _init(table: pd.DataFrame, newest: int) -> None:
    _W.update(table=table, newest=newest, market=sim.Market(), memo={})


def book_many(genomes: list[dict]) -> list[dict]:
    return [fa.book(fa.Genome.from_dict(g), _W["table"], _W["market"], _W["newest"], _W["memo"]) for g in genomes]


def table_for(source: Path, market) -> pd.DataFrame:
    """The trade table of one event library run, built once and kept next to the pool."""
    path = fa.FACTORY_DIR / f"table-{source.stem}.pkl"
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
    """A finalist against random entries in 2021-2024 and on 2025-2026."""
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
                     "genomes bred from the best of Разгон/Сейф on 2021-2024, a fifth random; candidates must beat "
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
             f"комиссии, проскальзывание, фандинг шорта. Поиск видит только 2021–2024; 2025 и 2026 — проверка, "
             f"но события библиотеки выбраны после того, как их видели, — чистый судья только бумага. "
             f"Прогон {time.time() - started:.0f} с.", ""]
    for goal, spec in fa.GOALS.items():
        lines += [f"## {spec['title']} (просадка не больше {100 * spec['drawdown']:.0f}% в каждом году)", "",
                  "| вариант | 2021–2024 в год | 2021–2024 | 2025: сделок · в плюс · рост · просадка | 2026 | "
                  "лучше случайного (t) | итог |", "|---|---|---|---|---|---|---|"]
        if not finals[goal]:
            lines.append("| ни один вариант не удержал просадку и 3 года из 4 без убытка | | | | | | |")
        for e in finals[goal]:
            v = verdicts[(goal, e["id"])]
            lines.append(f"| {fa.describe(fa.Genome.from_dict(e['genome']))} | {fmt(fa.fitness(e, goal), 0, True)}% | "
                         f"{learn_line(e)} | {year_cell(e, 2025)} | {year_cell(e, 2026)} | "
                         f"{fmt(v['excess_pct'], 2, True)}% (t {fmt(v['t'], 1)}) | {v['verdict']} |")
        lines.append("")
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
