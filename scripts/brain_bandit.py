"""The RL experiment's exam (core/brain/bandit.py): an agent learning trade by trade against the selector.

    python scripts/brain_bandit.py

Uses the selector's walked labels of the newest event library run (scripts/brain_selector.py builds
them), runs the agent through history from 2018 and books its picks from 2019 as a $100 portfolio
per year, against the panic and holding BTC. Report in runtime/learning/selector/bandit-<time>.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.console import utf8_output  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import bandit as bd  # noqa: E402
from core.brain import factory as fa  # noqa: E402
from core.brain import regime as rg  # noqa: E402
from core.brain import selector as se  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
OUT_DIR = ROOT / "runtime" / "learning" / "selector"
YEARS = (2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)


def main() -> int:
    utf8_output()
    started = time.time()
    source = sorted(BRAIN_DIR.glob("events-*.pkl"))[-1]
    market = sim.Market()
    newest = int(market.btc()[-1, 0])
    table = pd.read_pickle(fa.FACTORY_DIR / f"table-{source.stem}-{fa.TABLE_VERSION}.pkl")
    cands = se.candidates(table, market)
    labels = pd.read_pickle(OUT_DIR / f"labels-{source.stem}.pkl")
    f = se.frame(cands, labels)
    picks = bd.run(f, cands)
    books = se.books(picks, YEARS, newest)
    panic = fa.book(fa.seeds()[0], table, market, newest)["years"]
    regimes = rg.btc_regimes(market.btc())
    btc = {str(y): rg.curve_stats(rg.hold_curve(regimes, market.btc(), None, 1, None, *fa.year_span(y, newest)))
           for y in YEARS}
    total = float(np.prod([1 + books[str(y)]["growth_pct"] / 100 for y in YEARS]))
    lines = [f"# RL-эксперимент: агент учится сделка за сделкой — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Агент проходит историю с 2018 года по {len(cands):,} монето-часам библиотеки событий ({source.name}). "
             f"В каждом — грубая ситуация (тренд и нервность BTC, ширина рынка, сезон альтов, первое событие монеты) "
             f"и 8 действий модели выбора или «пропустить». Он выбирает по выборке из того, что знает о действии в "
             f"этой ситуации (чем меньше знает — тем шире разброс: так он пробует новое), берёт лучшее, если оно "
             f"выше нуля, и узнаёт итог сделки только когда она закрылась. Рынок показывает итоги и невыбранных "
             f"действий, и агент учится на всех. Счёт $100 в каждом году, $20 на сделку, не больше 5 сразу.", "",
             "| год | агент: сделок · в плюс из 10 · $ на $100 · рост · просадка | паника | держать BTC |",
             "|---|---|---|---|"]
    for y in YEARS:
        b, p = books[str(y)], panic[str(y)]
        lines.append(f"| {y} | {b['n']:,} · {fmt(b['up_of10'], 1)} · {fmt(b['usd_per_100'], 2, True)} · "
                     f"{fmt(b['growth_pct'], 0, True)}% · {fmt(b['drawdown_pct'], 0)}% | "
                     f"{fmt(p['growth_pct'], 0, True)}% · {fmt(p['drawdown_pct'], 0)}% | "
                     f"{fmt(btc[str(y)]['growth_pct'], 0, True)}% · {fmt(btc[str(y)]['drawdown_pct'], 0)}% |")
    lines += ["", f"Все годы вместе: {fmt(100 * (total ** (1 / len(YEARS)) - 1) if total > 0 else -100.0, 0, True)}% "
                  f"в год. Действия агента (2025–2026): "
              + ", ".join(f"{se.ACTION_RU.get(a, a)} {100 * v:.0f}%"
                          for a, v in picks[picks["year"] >= 2025]["action"].value_counts(normalize=True)
                          .head(4).items()) + ".", f"Прогон {time.time() - started:.0f} с."]
    journal = Journal()
    journal.register("RL-BANDIT", "RL-эксперимент: контекстный бандит, учится сделка за сделкой",
                     "агент, который учится на наградах закрытых сделок в грубых ситуациях, найдёт, что делать",
                     "Thompson sampling over 8 actions + skip per (trend, mood, breadth, alts, first event), backing "
                     "off to (event, action) and (action) under 30 trades; learns every action's net at its close")
    for y in YEARS:
        journal.result("RL-BANDIT", str(y), {**books[str(y)], "events": source.stem})
    report = OUT_DIR / f"bandit-{time.strftime('%Y%m%d-%H%M')}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
