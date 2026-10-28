"""The trade assessor's exam: does a promised "7 in 10 end in profit" come true?

    python scripts/brain_assessor.py

Writes runtime/learning/brain/assessor-<stamp>.pkl (every candidate with its
out-of-sample chance and typical result) and a plain report; the idea and its
numbers go into the journal.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.brain import assessor as asr  # noqa: E402
from core.brain import dataset as ds  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
logger = logging.getLogger("assessor")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    journal = Journal()
    journal.register("A1", "оценщик сделки: шанс закончить в плюсе и обычный размер",
                     "по тому, что известно на входе, можно заранее отличить сделки, которые обычно кончаются плюсом",
                     "LightGBM win/size on all coin-days x both sides, monthly walk-forward from 2022, isotonic per side")
    rows = ds.load_table(BRAIN_DIR / f"dataset-{ds.DATASET_VERSION}")
    exam = sorted(BRAIN_DIR.glob("exam-*.pkl"))[-1]
    frame = asr.candidates(pd.read_pickle(exam), rows)
    del rows
    logger.info("candidates: %d (%d long, %d short)", len(frame), (frame.side > 0).sum(), (frame.side < 0).sum())
    pred = asr.run(frame, on_month=lambda m, n: logger.info("assessor %s: %d rows to learn from", m, n))
    stamp = time.strftime("%Y%m%d-%H%M")
    pred.to_pickle(BRAIN_DIR / f"assessor-{stamp}.pkl")

    lines = [f"# Оценщик сделки — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Кандидатов на экзамене: {len(pred):,} (каждая монета каждый день, лонг и шорт). Каждый месяц оценщик "
             f"учится только на прошлом, обещания калибруются на прошлых месяцах экзамена. {time.time() - started:.0f} с.", ""]
    for name, mask in (("2022–2024", pred["day"] <= "2024-12-31"), ("2025", pred["day"] > "2024-12-31")):
        table = asr.promise_table(pred, mask).reset_index()
        lines += [f"## {name}: обещал против правды", "",
                  "| сторона | обещанный шанс плюса | сделок | обещал, из 10 | сбылось, из 10 | $ на $100 | средний плюс | средний минус |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in table.itertuples(index=False):
            lines.append(f"| {'лонг' if r.side > 0 else 'шорт'} | {r.win} | {r.trades:,} | {fmt(r.promised_of10, 1)} | "
                         f"{fmt(r.real_of10, 1)} | {fmt(r.usd_per_100, 2, True)} | {fmt(r.avg_win_usd, 1, True)} | "
                         f"{fmt(r.avg_loss_usd, 1, True)} |")
            journal_key = f"{name} {'long' if r.side > 0 else 'short'} {r.win}"
            journal.result("A1", journal_key, {"trades": int(r.trades), "promised_of10": r.promised_of10,
                                               "real_of10": r.real_of10, "usd_per_100": r.usd_per_100,
                                               "exam": exam.stem, "dataset": ds.DATASET_VERSION})
        lines.append("")
    report = BRAIN_DIR / f"assessor-{stamp}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
