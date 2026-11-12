"""Every strategy in many versions: is its result skill or the luck of one path?

    python scripts/brain_robustness.py [--only N1,C1] [--variants 100]

Same data as brain_strategies.py (the newest exam's places, dataset outcomes,
2020–2024 to choose, 2025 to confirm, 2026 with an exam that covers it). Versions differ
in start day, number of positions and missed signals (core/brain/robustness.py).
The spread of each strategy goes into the journal next to its single runs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.console import utf8_output  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import dataset as ds  # noqa: E402
from core.brain import portfolio as pf  # noqa: E402
from core.brain import robustness as rb  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.playbook import (BLENDS, BY_ID, PLAYBOOK, btc_series, last_day, periods,  # noqa: E402
                                 scored_frame)
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"


def walk_runner(market):
    """Simulator v2 with the rules of a version."""
    def run(orders, days, rules, btc_ret, idle):
        r = sim.Rules(position_share=rules.position_share, max_positions=rules.max_positions,
                      max_short_share=rules.max_short_share, drawdown_brake=rules.drawdown_brake)
        return sim.run(orders, days, market, r, btc_ret, idle)
    return run


def blend_variants(scored, a, b, weight, days, btc_ret, btc_close, specs, runner) -> pd.DataFrame:
    """A blend in each version: both strategies with the same changes, money split every day."""
    rows = []
    for k, spec in enumerate(specs):
        curves = []
        for strat in (a, b):
            orders = strat.orders(scored)
            idle = strat.idle_in_btc(scored) if strat.idle_in_btc else None
            rng = np.random.default_rng(spec["seed"])
            kept = rb._miss(orders, rng)
            rules = replace(pf.Rules(), max_positions=spec["positions"], position_share=1.0 / spec["positions"])
            start = days[spec["shift"]]
            values, _ = runner(kept[kept["day"] >= start] if len(kept) else kept, days[days >= start],
                               rules, btc_ret, idle)
            curves.append(values)
        values = pf.blend(curves[0], curves[1], weight)
        for name, v in periods(values).items():
            s = pf.summary(v, pd.DataFrame(columns=["closed", "side", "ret", "pnl"]), btc_close)
            rows.append({"variant": k, "period": name, **spec, "return_pct": s.get("return_pct"),
                         "max_drawdown_pct": s.get("max_drawdown_pct"), "trades": 0})
    return pd.DataFrame(rows)


_CACHE: dict = {}


def _data():
    """Each worker process loads the inputs once."""
    if not _CACHE:
        rows = ds.load_table(BRAIN_DIR / f"dataset-{ds.DATASET_VERSION}")
        exam = sorted(BRAIN_DIR.glob("exam-*.pkl"))[-1]
        scored = scored_frame(pd.read_pickle(exam), rows)
        btc_close, btc_ret = btc_series(rows)
        market = sim.Market()
        _CACHE.update(scored=scored, btc_close=btc_close, btc_ret=btc_ret, exam=exam.stem,
                      runner=walk_runner(market),
                      days=pd.date_range(scored["day"].min(), last_day(scored, market.btc()), freq="D"))
    return _CACHE


def spread_for(sid: str, n: int) -> tuple[str, str, pd.DataFrame]:
    d = _data()
    specs = rb.variants(n)
    blend = next((b for b in BLENDS if b[0] == sid), None)
    if blend:
        _, title, _, _, (a, b), weight = blend
        results = blend_variants(d["scored"], BY_ID[a], BY_ID[b], weight, d["days"], d["btc_ret"], d["btc_close"], specs,
                                 d["runner"])
    else:
        s = BY_ID[sid]
        title = s.title
        idle = s.idle_in_btc(d["scored"]) if s.idle_in_btc else None
        results = rb.run_variants(s.orders(d["scored"]), d["days"], periods, d["btc_ret"], idle, d["btc_close"], specs,
                                  d["runner"])
    return sid, title, rb.spread_of(results)


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--variants", type=int, default=rb.VARIANTS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    ids = [s.id for s in PLAYBOOK] + [b[0] for b in BLENDS]
    ids = [i for i in ids if not args.only or i in args.only.split(",")]
    journal = Journal()
    for s in PLAYBOOK:                    # an idea is written down before its first result, wherever that comes from
        journal.register(s.id, s.title, s.hypothesis, s.rule)
    for bid, title, hypothesis, rule, _, _ in BLENDS:
        journal.register(bid, title, hypothesis, rule)
    exam = sorted(BRAIN_DIR.glob("exam-*.pkl"))[-1].stem
    started = time.time()
    table = []
    with ProcessPoolExecutor(args.workers) as pool:
        for sid, title, spread in pool.map(spread_for, ids, [args.variants] * len(ids)):
            table.append((sid, title, spread))
            print(f"{sid}: {time.time() - started:.0f} s", flush=True)
    for sid, _, spread in table:
        for per, r in spread.iterrows():
            journal.result(sid, f"устойчивость {per}", {**r.to_dict(), "variants": args.variants,
                                                         "exam": exam, "dataset": ds.DATASET_VERSION, "sim": "v2"})
    lines = [f"# Устойчивость стратегий — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Симулятор v2 (часовой путь, портфель по рынку). {args.variants} вариантов каждой: старт сдвинут на 0–{rb.MAX_START_SHIFT} дней, "
             f"{'/'.join(map(str, rb.POSITIONS))} позиций, каждый {int(1 / rb.MISS)}-й сигнал пропущен. "
             f"Идей в журнале: {journal.tried()}.", "",
             "| id | стратегия | период | в плюсе, из 10 | обычный итог, % | плохие 10%, % | хорошие 10%, % | "
             "обычная просадка, % | плохая просадка, % |", "|" + "---|" * 9]
    for sid, title, spread in table:
        for per, r in spread.iterrows():
            lines.append(f"| {sid} | {title} | {per} | {fmt(r.up_of10, 1)} | {fmt(r.median_pct, 0, True)} | "
                         f"{fmt(r.bad_tenth_pct, 0, True)} | {fmt(r.good_tenth_pct, 0, True)} | "
                         f"{fmt(r.median_drawdown_pct, 0)} | {fmt(r.bad_drawdown_pct, 0)} |")
    report = BRAIN_DIR / f"robustness-{time.strftime('%Y%m%d-%H%M')}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
