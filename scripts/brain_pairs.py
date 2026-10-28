"""The pair assessor's exam, and pairs chosen by it, against pairs taken as they come.

    python scripts/brain_pairs.py [--variants 50]

1. 8 best x 8 worst candidate pairs a day (core/brain/pairs.py), the assessor's monthly
   exam from 2022: promised chance of profit against what came true;
2. PA1 / PA2 take each day up to 3 pairs whose expected result after costs is
   positive, best first (PA2 not in alt seasons); P1 / P2 are the same pairs
   without the assessor. All four in simulator v2, in many versions, over the
   assessor's own window (2022-2024 to choose, 2025 to confirm).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.brain import assessor as asr  # noqa: E402
from core.brain import dataset as ds  # noqa: E402
from core.brain import pairs as pr  # noqa: E402
from core.brain import robustness as rb  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.playbook import BY_ID, EXITS, btc_series, periods, scored_frame  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
logger = logging.getLogger("pairs")


def chosen_pairs(pred: pd.DataFrame, alt_season_off: bool) -> pd.DataFrame:
    p = pred[pred["size"] > 0]
    if alt_season_off:
        p = p[p["beat_btc_30d"] <= 0.5]
    return pr.choose(p.assign(priority=p["size"]))


def runner(market):
    def run(orders, days, rules, btc_ret, idle):
        r = sim.Rules(position_share=rules.position_share, max_positions=rules.max_positions,
                      max_short_share=rules.max_short_share, drawdown_brake=rules.drawdown_brake)
        return sim.run(orders, days, market, r, btc_ret, idle)
    return run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    journal = Journal()
    journal.register("PA1", "пары по оценщику: до 3 в день с ожидаемым плюсом после комиссий",
                     "отсеять только пары с плохим соотношением шансов и размеров, неуверенные выгодные оставить",
                     "8 best x 8 worst candidate pairs a day, keep calibrated size > 0, best 3 by it, no coin twice")
    journal.register("PA2", "PA1, но не в альтсезон", "то же, без альтсезона", "PA1 when beat_btc_30d <= 0.5")
    rows = ds.load_table(BRAIN_DIR / f"dataset-{ds.DATASET_VERSION}")
    exam = sorted(BRAIN_DIR.glob("exam-*.pkl"))[-1]
    brain_pred = pd.read_pickle(exam)
    cands = pr.candidates(brain_pred, rows)
    logger.info("pair candidates: %d", len(cands))
    pred = asr.run(cands, features=pr.FEATURES, on_month=lambda m, n: logger.info("pairs %s: %d to learn from", m, n))
    pred = pred.drop(columns=["base", "place", "atr", "turnover_30d", "side", "ret"]).merge(cands, on=["history", "day"])
    stamp = time.strftime("%Y%m%d-%H%M")
    pred.to_pickle(BRAIN_DIR / f"pairs-{stamp}.pkl")

    lines = [f"# Оценщик пар — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Кандидатов: {len(pred):,} пар (до {pr.CANDIDATES} в день). Экзамен как у оценщика сделки: каждый месяц "
             "учится только на прошлом, обещания калибруются на прошлых месяцах.", ""]
    for name, mask in (("2022–2024", pred["day"] <= "2024-12-31"), ("2025", pred["day"] > "2024-12-31")):
        lines += [f"## {name}: обещал против правды", "", "| обещанный шанс плюса | пар | обещал, из 10 | сбылось, из 10 | "
                  "$ на $100 | средний плюс | средний минус |", "|---|---|---|---|---|---|---|"]
        for r in asr.promise_table(pred, mask).reset_index().itertuples(index=False):
            lines.append(f"| {r.win} | {r.trades:,} | {fmt(r.promised_of10, 1)} | {fmt(r.real_of10, 1)} | "
                         f"{fmt(r.usd_per_100, 2, True)} | {fmt(r.avg_win_usd, 1, True)} | {fmt(r.avg_loss_usd, 1, True)} |")
        lines.append("")

    # the portfolio: pairs by the assessor against the same pairs as they come
    scored = scored_frame(brain_pred, rows)
    btc_close, btc_ret = btc_series(rows)
    del rows
    days = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    market = sim.Market()
    specs = rb.variants(args.variants)
    books = {"PA1": pr.orders(chosen_pairs(pred, False), EXITS["hold"]),
             "PA2": pr.orders(chosen_pairs(pred, True), EXITS["hold"]),
             "P1": BY_ID["P1"].orders(scored[scored["day"] >= "2022-01-01"]),
             "P2": BY_ID["P2"].orders(scored[scored["day"] >= "2022-01-01"])}

    def split(values):
        return {name: v for name, v in periods(values).items()} if values.index[0] < pd.Timestamp("2025-01-01") else {}

    lines += [f"## Портфель 2022–2025, {args.variants} вариантов", "",
              "| id | период | в плюсе, из 10 | обычный итог, % | плохие 10%, % | обычная просадка, % |", "|---|---|---|---|---|---|"]
    for sid, orders in books.items():
        res = rb.run_variants(orders, days, split, btc_ret, None, btc_close, specs, runner(market))
        spread = rb.spread_of(res)
        for per, r in spread.iterrows():
            label = "2022–2024" if per == "2020–2024" else per
            lines.append(f"| {sid} | {label} | {fmt(r.up_of10, 1)} | {fmt(r.median_pct, 0, True)} | "
                         f"{fmt(r.bad_tenth_pct, 0, True)} | {fmt(r.median_drawdown_pct, 0)} |")
            if sid.startswith("PA"):
                journal.result(sid, f"устойчивость {label}", {**r.to_dict(), "variants": args.variants, "sim": "v2",
                                                             "exam": exam.stem, "dataset": ds.DATASET_VERSION})
        logger.info("%s done", sid)
    report = BRAIN_DIR / f"pairs-{stamp}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
