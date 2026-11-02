"""Trade the brain's places as a portfolio, strategy by strategy, and write every result down.

    python scripts/brain_strategies.py [--only B1,N2]

Uses the newest exam (out-of-sample places, 12.2020 … 12.2025) and the dataset's
outcomes. Each strategy of core/brain/playbook.py is registered in the journal
before its first run; every run's numbers go into it, good or bad. Choosing is
done on 2020–2024; 2025 confirms. 2026 is booked only with an exam that covers
it (``brain.py exam --open-holdout``, opened for good): then the
run goes on to the last full day of prices and 2026 is its own period.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.brain import dataset as ds  # noqa: E402
from core.brain import portfolio as pf  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.playbook import (BLENDS, BY_ID, PLAYBOOK, btc_series, last_day, periods,  # noqa: E402
                                 scored_frame)
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated strategy ids; all by default")
    args = parser.parse_args()
    chosen = [s for s in PLAYBOOK if not args.only or s.id in args.only.split(",")]
    rows = ds.load_table(BRAIN_DIR / f"dataset-{ds.DATASET_VERSION}")
    exam = sorted(BRAIN_DIR.glob("exam-*.pkl"))[-1]
    pred = pd.read_pickle(exam)
    scored = scored_frame(pred, rows)
    btc_close, btc_ret = btc_series(rows)
    journal = Journal()
    for s in PLAYBOOK:
        journal.register(s.id, s.title, s.hypothesis, s.rule)
    for bid, title, hypothesis, rule, _, _ in BLENDS:
        journal.register(bid, title, hypothesis, rule)
    requested = set(args.only.split(",")) if args.only else {s.id for s in PLAYBOOK} | {b[0] for b in BLENDS}
    blends = [b for b in BLENDS if b[0] in requested]
    needed = {x for b in blends for x in b[4]}
    chosen += [BY_ID[x] for x in sorted(needed) if BY_ID[x] not in chosen]

    table, yearly, curves = [], {}, {}
    market = sim.Market()
    days = pd.date_range(scored["day"].min(), last_day(scored, market.btc()), freq="D")
    for s in chosen:
        orders = s.orders(scored)
        idle = s.idle_in_btc(scored) if s.idle_in_btc else None
        values, trades = sim.run(orders, days, market, sim.Rules(), btc_ret, idle)
        curves[s.id] = values
        if s.id not in requested:
            continue                     # only needed for a blend; its own result is already in the journal
        for name, v in periods(values).items():
            numbers = pf.summary(v, trades, btc_close)
            journal.result(s.id, name, {**numbers, "exam": exam.stem, "dataset": ds.DATASET_VERSION, "sim": "v2"})
            table.append({"id": s.id, "стратегия": s.title, "период": name, **numbers})
        yearly[s.id] = pf.by_year(values, trades, btc_close)
    no_trades = pd.DataFrame(columns=["closed", "side", "ret", "pnl"])
    for bid, title, _, _, (a, b), weight in blends:
        values = pf.blend(curves[a], curves[b], weight)
        for name, v in periods(values).items():
            numbers = pf.summary(v, no_trades, btc_close)
            journal.result(bid, name, {**numbers, "exam": exam.stem, "dataset": ds.DATASET_VERSION, "sim": "v2"})
            table.append({"id": bid, "стратегия": title, "период": name, **numbers})
        yearly[bid] = pf.by_year(values, no_trades, btc_close)

    lines = [f"# Стратегии на местах мозга — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Симулятор v2: позиции идут по часовым ценам, портфель оценивается по рынку каждый день. Экзамен {exam.stem}; ${pf.START_USD:,.0f} на старте, {pf.POSITION_SHARE:.0%} на позицию, "
             f"до {pf.MAX_POSITIONS} позиций, шорты до {pf.MAX_SHORT_SHARE:.0%} портфеля, после просадки "
             f"{pf.DRAWDOWN_BRAKE:.0%} позиции вдвое меньше. Идей в журнале с результатом: {journal.tried()}.", "",
             "| id | стратегия | период | итог, % | худшая просадка, % | BTC, % | BTC просадка, % | сделок | в плюс из 10 | "
             "$ на $100 | без 5 лучших, % | месяцев в плюс из 10 |", "|" + "---|" * 12]
    for r in table:
        lines.append(f"| {r['id']} | {r['стратегия']} | {r['период']} | {fmt(r.get('return_pct'), 1, True)} | "
                     f"{fmt(r.get('max_drawdown_pct'), 1)} | {fmt(r.get('btc_return_pct'), 1, True)} | "
                     f"{fmt(r.get('btc_max_drawdown_pct'), 1)} | {r.get('trades', 0)} | {fmt(r.get('win_of10'), 1)} | "
                     f"{fmt(r.get('usd_per_100'), 2, True)} | {fmt(r.get('without_best5_pct'), 1, True)} | "
                     f"{fmt(r.get('months_up_of10'), 1)} |")
    lines += ["", "## По годам (итог %, рядом BTC)", ""]
    years = sorted({y for t in yearly.values() for y in t["year"]})
    lines += ["| id | " + " | ".join(str(y) for y in years) + " |", "|" + "---|" * (len(years) + 1)]
    for sid, t in yearly.items():
        cells = {int(r.year): f"{fmt(r.return_pct, 0, True)} ({fmt(r.btc_return_pct, 0, True)})" for r in t.itertuples()}
        lines.append(f"| {sid} | " + " | ".join(cells.get(y, "—") for y in years) + " |")
    report = BRAIN_DIR / f"strategies-{time.strftime('%Y%m%d-%H%M')}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
