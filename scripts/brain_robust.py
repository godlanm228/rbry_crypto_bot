"""Robustness of everything on paper (core/brain/robust.py): bend history, check the crises, take the fragile off.

    python scripts/brain_robust.py

Books the panic's two books and every factory candidate on paper as they are and bent four ways,
lists their trades in known shocks, gives the factory's choice its chance of being luck (PBO), and
takes the candidates that do not hold off runtime/learning/factory/promoted.json (to demoted.json,
with the reason); the night copies the list to the watcher. Report next to the factory's.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.brain import factory as fa  # noqa: E402
from core.brain import robust as rb  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
PROMOTED = fa.FACTORY_DIR / "promoted.json"
DEMOTED = fa.FACTORY_DIR / "demoted.json"
BUILT_IN = {"паника без стопа": fa.seeds()[0], "паника со стопом −30%": fa.seeds()[1]}


def cell(stats: dict, goal: str) -> str:
    y25, y26 = stats["years"]["2025"], stats["years"]["2026"]
    worst = min(stats["years"][str(y)]["drawdown_pct"] for y in fa.LEARN_YEARS)
    losing = sum(stats["years"][str(y)]["growth_pct"] < 0 for y in fa.LEARN_YEARS)
    mark = "" if rb.holds(stats, goal) else " ✗"
    return (f"{fmt(rb.learn_growth(stats), 0, True)}%/год (просадка до {fmt(worst, 0)}%, убыточных лет {losing}) · "
            f"{fmt(y25['growth_pct'], 0, True)}% · {fmt(y26['growth_pct'], 0, True)}%{mark}")


def main() -> int:
    started = time.time()
    source = sorted(BRAIN_DIR.glob("events-*.pkl"))[-1]
    table = pd.read_pickle(fa.FACTORY_DIR / f"table-{source.stem}-{fa.TABLE_VERSION}.pkl")
    market = sim.Market()
    newest = int(market.btc()[-1, 0])
    promoted = json.loads(PROMOTED.read_text(encoding="utf-8")) if PROMOTED.exists() else []
    strategies = [(title, g, "razgon", None) for title, g in BUILT_IN.items()]
    strategies += [(f"фабрика · {fa.GOALS[p['goal']]['title']} · {p['id'][-4:]}", fa.Genome.from_dict(p["genome"]),
                    p["goal"], p["id"]) for p in promoted if p["id"] not in {g.id for g in BUILT_IN.values()}]
    pool = {i: e for i, e in fa.load_pool(fa.FACTORY_DIR / "pool.jsonl").items() if e.get("stamp") == source.stem}
    lines = [f"# Прочность того, что на бумаге — {time.strftime('%Y-%m-%d %H:%M')}", "",
             "Каждая стратегия — как есть и в четырёх искажениях истории: вход на час позже, комиссии вдвое, пропуск "
             "20% сигналов (из 100 случайных пропусков берутся плохие 10%), без лучших 5% сделок. Прочная — если в "
             "каждом искажении зарабатывает за 2018–2024 в целом, просадка каждый год в пределах цели (Разгон 50%, "
             "Сейф 20%), и растёт в 2025 и 2026. «Строго» — ещё и правило отбора фабрики: убыточных лет из 7 не "
             "больше двух. Ячейка: 2018–2024 в год (худшая просадка, убыточных лет) · 2025 · 2026; ✗ — не держит.", "",
             "| стратегия | как есть | " + " | ".join(rb.MUTATIONS.values()) + " | итог |", "|" + "---|" * 7]
    crisis_rows, fragile = [], []
    journal = Journal()
    journal.register("ROBUST", "прочность стратегий на бумаге: искажения истории, кризисы, шанс подгонки",
                     "стратегия, которая ломается от часа задержки, двойных комиссий или пропуска сигналов, — подгонка",
                     "late +1h, costs x2, 20% of signals missed (bad tenth of 100), without the best 5%; goal on "
                     "2018-2024 and growth in 2025 and 2026 in each")
    for title, g, goal, pid in strategies:
        muts = rb.mutations(g, table, market, newest)
        ok, failed = rb.verdict(muts, goal)
        lines.append(f"| {title} | {cell(muts['base'], goal)} | "
                     + " | ".join(cell(muts[m], goal) for m in rb.MUTATIONS)
                     + " | " + ("прочная" + (", и строго" if all(rb.strict(muts[m], goal) for m in ("base", *rb.MUTATIONS))
                                              else "") if ok else
                                ("хрупкая: " + ", ".join(rb.MUTATIONS[m] for m in failed) if failed
                                 else "не держит и без искажений")) + " |")
        journal.result("ROBUST", f"{time.strftime('%Y-%m-%d')} {g.id}", {"robust": ok, "failed": failed, "goal": goal,
                                                                         "events": source.stem})
        crisis_rows.append((title, rb.crises(muts["trades"])))
        if not ok and pid:
            fragile.append((pid, failed or ["base"]))
    lines += ["", "## Кризисы: сделки, открытые во время шока", "",
              "| стратегия | " + " | ".join(c[0] for c in rb.CRISES) + " |", "|" + "---|" * (len(rb.CRISES) + 1)]
    for title, rows in crisis_rows:
        lines.append(f"| {title} | " + " | ".join(
            (f"{r['trades']} · {fmt(r['usd_per_100'], 1, True)} $ · худшая {fmt(r['worst_pct'], 0)}%" if r["trades"]
             else "—") for r in rows) + " |")
    lines += ["", "## Шанс, что выбор фабрики — удача (PBO)", "",
              "Для каждого из 35 способов разделить 7 лет обучения на 4 «выбрать» и 3 «проверить»: попадает ли "
              "выбранный по 4 годам вариант в верхнюю половину всех на 3 других. 50% — монетка.", ""]
    for goal, spec in fa.GOALS.items():
        p = rb.pbo(pool, goal)
        lines.append(f"- {spec['title']}: шанс удачи {fmt(100 * p['pbo'], 0)}% (разбиений {p['splits']}; выбранный "
                     f"в среднем лучше {fmt(100 * p['median_rank'], 0)}% вариантов на непроверенных годах).")
        journal.result("ROBUST", f"{time.strftime('%Y-%m-%d')} PBO {goal}", {**p, "events": source.stem})
    if fragile:
        keep = [p for p in promoted if p["id"] not in {f for f, _ in fragile}]
        gone = [{**p, "demoted_at": time.strftime("%Y-%m-%d %H:%M"), "why": dict(fragile)[p["id"]]}
                for p in promoted if p["id"] in {f for f, _ in fragile}]
        old = json.loads(DEMOTED.read_text(encoding="utf-8")) if DEMOTED.exists() else []
        DEMOTED.write_text(json.dumps(old + gone, ensure_ascii=False, indent=1, default=fa._plain), encoding="utf-8")
        PROMOTED.write_text(json.dumps(keep, ensure_ascii=False, indent=1, default=fa._plain), encoding="utf-8")
        lines += ["", "## Снято с бумаги", ""] + [f"- {p['title']} — {', '.join(rb.MUTATIONS.get(w, w) for w in p['why'])}"
                                                  for p in gone]
    lines.append(f"\nПрогон {time.time() - started:.0f} с.")
    report = fa.FACTORY_DIR / f"robust-{time.strftime('%Y%m%d-%H%M')}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
