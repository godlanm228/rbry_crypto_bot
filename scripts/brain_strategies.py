"""Trade the brain's places as a portfolio, strategy by strategy, and write every result down.

    python scripts/brain_strategies.py

Uses the newest exam (out-of-sample places, 12.2020 … 12.2025) and the dataset's
outcomes. Each strategy of core/brain/playbook.py is registered in the journal
before its first run; every run's numbers go into it, good or bad. Choosing is
done on 2020–2024; 2025 confirms; 2026 is closed (positions still open on
31.12.2025 count at cost, so no 2026 price is ever booked).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.brain import dataset as ds  # noqa: E402
from core.brain import portfolio as pf  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.playbook import PLAYBOOK, btc_series, scored_frame  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402
from core.sandbox.evaluate import DEV_END, VALID_END  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"


def periods(values: pd.Series) -> dict[str, pd.Series]:
    dev = values[values.index <= DEV_END]
    valid = values[(values.index > DEV_END) & (values.index <= VALID_END)]
    return {"2020–2024": dev, "2025": pd.concat([dev.iloc[-1:], valid]) if len(dev) else valid}


def main() -> int:
    rows = pd.read_pickle(BRAIN_DIR / f"dataset-{ds.DATASET_VERSION}.pkl")
    exam = sorted(BRAIN_DIR.glob("exam-*.pkl"))[-1]
    pred = pd.read_pickle(exam)
    scored = scored_frame(pred, rows)
    btc_close, btc_ret = btc_series(rows)
    days = pd.date_range(scored["day"].min(), VALID_END, freq="D")
    journal = Journal()
    for s in PLAYBOOK:
        journal.register(s.id, s.title, s.hypothesis, s.rule)

    table, yearly = [], {}
    for s in PLAYBOOK:
        orders = s.orders(scored)
        idle = s.idle_in_btc(scored) if s.idle_in_btc else None
        values, trades = pf.run(orders, days, btc_ret, idle)
        for name, v in periods(values).items():
            numbers = pf.summary(v, trades, btc_close)
            journal.result(s.id, name, {**numbers, "exam": exam.stem})
            table.append({"id": s.id, "стратегия": s.title, "период": name, **numbers})
        yearly[s.id] = pf.by_year(values, trades, btc_close)

    lines = [f"# Стратегии на местах мозга — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Экзамен {exam.stem}; ${pf.START_USD:,.0f} на старте, {pf.POSITION_SHARE:.0%} на позицию, "
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
