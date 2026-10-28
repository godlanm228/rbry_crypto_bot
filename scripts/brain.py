"""The brain (step 2): build its textbook, run its exam.

    python scripts/brain.py dataset            # market.db -> runtime/learning/brain/dataset-<version>.pkl
    python scripts/brain.py exam               # walk-forward exam up to 2025, report in plain words
    python scripts/brain.py exam --open-holdout  # the same through 2026; every opening is logged

2026 stays closed unless asked: each look at it is written to learning.db's
holdout log, and its count goes into the report.
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

from core.brain import dataset, model  # noqa: E402
from core.brain import walkforward as wf  # noqa: E402
from core.sandbox.market_store import DEFAULT_DB_PATH, MarketStore  # noqa: E402
from core.sandbox.store import LearningStore  # noqa: E402

logger = logging.getLogger("brain")
BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"


def dataset_path(version: str = dataset.DATASET_VERSION) -> Path:
    return BRAIN_DIR / f"dataset-{version}.pkl"


def build_dataset(args) -> None:
    started = time.time()
    rows = dataset.build(MarketStore(args.db))
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    path = dataset_path()
    rows.to_pickle(path)
    labelled = rows["r_net"].notna()
    logger.info("dataset %s: %d rows (%d with an outcome), %d histories, %s … %s, %.0f MB, %.0f s",
                dataset.DATASET_VERSION, len(rows), labelled.sum(), rows["history"].nunique(),
                rows["day"].min().date(), rows["day"].max().date(), path.stat().st_size / 1e6, time.time() - started)


def load_dataset() -> pd.DataFrame:
    return pd.read_pickle(dataset_path())


def _table(frame: pd.DataFrame, columns: dict[str, tuple[str, int, bool]]) -> list[str]:
    """Markdown rows; ``columns``: source column -> (header, digits, signed)."""
    lines = ["| " + " | ".join(h for h, _, _ in columns.values()) + " |", "|" + "---|" * len(columns)]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(wf.fmt(r[c], d, s) if isinstance(r[c], float) else str(r[c])
                                       for c, (_, d, s) in columns.items()) + " |")
    return lines


SELECTION_COLUMNS = {
    "group": ("что берём", 0, False), "n": ("монето-дней", 0, False), "beat_btc_of10": ("обогнали BTC, из 10", 1, False),
    "vs_btc_usd": ("против BTC, $ на $100", 2, True), "hold_usd": ("купил и держал 2 нед., $ на $100", 2, True),
    "vs_others_usd": ("против всех монет, $", 2, True), "t_vs_others": ("t", 1, True),
    "trade_win_of10": ("сделка со стопом в плюс, из 10", 1, False),
}


def write_exam_report(path: Path, pred: pd.DataFrame, log: list[tuple[str, dict]], started: float,
                      holdout_opens: int) -> None:
    graded = wf.grade(pred)
    pred_p = pred.assign(period=wf.period(pred["day"]))
    lines = [f"# Экзамен мозга — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Данные {dataset.DATASET_VERSION}, модель {model.MODEL_VERSION}; монето-дней на экзамене: {len(pred):,} "
             f"({pred['base'].nunique()} монет, оборот от ${wf.MIN_TURNOVER / 1e6:.0f} млн в день). "
             f"Каждый месяц модель заново учится только на прошлом (зазор {wf.GAP_DAYS} дней) и сдаёт месяц, "
             f"которого не видела. dev = 2020–2024, valid = 2025, holdout = 2026 "
             f"({'вскрыт, раз всего: ' + str(holdout_opens) if holdout_opens else 'закрыт'}). "
             f"Прогон {time.time() - started:.0f} с.", "",
             "Главная цифра — «против всех монет»: насколько выбор мозга лучше любой монеты в те же месяцы "
             "(рынок у обоих одинаковый). t — надёжность, от 2 и выше уже не случайность.", ""]
    for per, title in (("dev", "2020–2024"), ("valid", "2025 — первый честный экзамен"),
                       ("holdout", "2026 — закрытый год")):
        if not (pred_p["period"] == per).any():
            continue
        lines += [f"## {title}", ""]
        lines += _table(graded[graded["period"] == per], SELECTION_COLUMNS)
        lines += ["", "По десяткам (0 = худшие 10% дня по оценке мозга, 9 = лучшие):", ""]
        dec = wf.deciles(pred_p[pred_p["period"] == per]).reset_index(names="decile")
        lines += _table(dec, {"decile": ("десяток", 0, False), "beat_btc_of10": ("обогнали BTC, из 10", 1, False),
                              "vs_btc_usd": ("против BTC, $ на $100", 2, True),
                              "hold_usd": ("купил и держал, $ на $100", 2, True), "trade_r": ("сделка, в рисках", 2, True),
                              "n": ("монето-дней", 0, False)})
        lines.append("")
    lines += ["## По годам: лучшие 10% против худших 10%", ""]
    lines += _table(wf.by_year(pred), {
        "year": ("год", 0, False), "coins_per_day": ("монет в день", 0, False),
        "all_beat_btc_of10": ("все: обогнали BTC", 1, False), "top_beat_btc_of10": ("лучшие: обогнали", 1, False),
        "bottom_beat_btc_of10": ("худшие: обогнали", 1, False), "all_vs_btc": ("все против BTC, $", 2, True),
        "top_vs_btc": ("лучшие, $", 2, True), "bottom_vs_btc": ("худшие, $", 2, True),
        "top_minus_bottom": ("разница, $", 2, True)})
    last = "holdout" if (pred_p["period"] == "holdout").any() else "valid"
    cal = wf.calibration_table(pred_p[pred_p["period"] == last])
    if not cal.empty:
        lines += ["", f"## Обещания против правды ({'2026' if last == 'holdout' else '2025'}): "
                      "шанс обогнать BTC за 2 недели", ""]
        lines += _table(cal.reset_index(drop=True), {"promised_of10": ("обещал, из 10", 1, False),
                                                     "real_of10": ("сбылось, из 10", 1, False), "n": ("случаев", 0, False)})
    if log:
        month, meta = log[-1]
        lines += ["", f"## На что смотрит мозг (модель {month}, {meta['trees']} деревьев, {meta['rows']:,} строк)", ""]
        lines += [f"- {name}: {share * 100:.0f}%" for name, share in meta["importance"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def exam(args) -> None:
    started = time.time()
    rows = load_dataset()
    log: list[tuple[str, dict]] = []

    def on_month(month: str, meta: dict) -> None:
        log.append((month, meta))
        logger.info("exam %s: %d rows, %d trees", month, meta["rows"], meta["trees"])

    stamp = time.strftime("%Y%m%d-%H%M")
    end, opens = wf.VALID_END, 0
    if args.open_holdout:
        end = str(rows.loc[rows["excess_14d"].notna(), "day"].max().date())
        opens = LearningStore().log_holdout(f"brain-{stamp}", f"brain {model.MODEL_VERSION} exam through {end}")
        logger.warning("holdout 2026 opened for the brain; openings so far: %d", opens)
    pred = wf.run(rows, refit_months=args.refit_months, end=end, on_month=on_month)
    pred.to_pickle(BRAIN_DIR / f"exam-{stamp}.pkl")
    report = BRAIN_DIR / f"exam-{stamp}.md"
    write_exam_report(report, pred, log, started, opens)
    print(report.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dataset")
    e = sub.add_parser("exam")
    e.add_argument("--refit-months", type=int, default=1)
    e.add_argument("--open-holdout", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "dataset":
        build_dataset(args)
    elif args.command == "exam":
        exam(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
