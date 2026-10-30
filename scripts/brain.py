"""The brain (step 2): its textbook, its exam, its week and its month.

    python scripts/brain.py dataset [--ranking-only]  # market.db -> runtime/learning/brain/dataset-<version>.pkl
    python scripts/brain.py exam                 # walk-forward exam up to 2025, report in plain words
    python scripts/brain.py exam --open-holdout  # the same through 2026; every opening is logged
    python scripts/brain.py train                # first champion; its record starts from the newest exam
    python scripts/brain.py weekly [--rebuild] [--notify]   # grade, refit, swap if the new one passes
    python scripts/brain.py audit [--month 2026-11] [--notify]
    python scripts/brain.py cache                # market.db -> the simulator's hourly prices (.npy per history)

2026 was opened for good on 30.10.2026 (C1's final exam); each look is still
written to learning.db's holdout log, and its count goes into the report. --notify sends a short note
to the owner (TELEGRAM_OWNER_ID of .env.test) through the test bot, nobody else.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.brain import dataset, model  # noqa: E402
from core.brain import walkforward as wf  # noqa: E402
from core.brain import weekly as wk  # noqa: E402
from core.notify import notify_owner  # noqa: E402
from core.sandbox.market_store import DEFAULT_DB_PATH, HOUR_MS, MarketStore  # noqa: E402
from core.sandbox.store import LearningStore  # noqa: E402

logger = logging.getLogger("brain")
BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
CHAMPION = BRAIN_DIR / "champion.pkl"
PREVIOUS = BRAIN_DIR / "champion-prev.pkl"
RECORD = BRAIN_DIR / "record.pkl"


def dataset_path(version: str = dataset.DATASET_VERSION) -> Path:
    return BRAIN_DIR / f"dataset-{version}"


def build_dataset(args) -> None:
    started = time.time()
    features = dataset.RANKING_FEATURES if args.ranking_only else dataset.FEATURES
    store = MarketStore(args.db)
    path = dataset_path()
    with store.fill_lock():                   # after the night's data refresh, never halfway through it
        newest = store.newest_hour()
        if getattr(args, "if_stale", False) and (path / "meta.json").exists() and newest is not None:
            built = dataset.load_table(path, ["day"])["day"].max()
            closed = pd.Timestamp((newest + HOUR_MS) // dataset.DAY_MS * dataset.DAY_MS - dataset.DAY_MS, unit="ms")
            if built >= closed:
                logger.info("dataset %s already has %s: not rebuilt", dataset.DATASET_VERSION, built.date())
                return
        rows = dataset.build(store, features=features)
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_table(rows, path)
    labelled = rows["r_net"].notna()
    logger.info("dataset %s: %d rows (%d with an outcome), %d histories, %s … %s, %.0f MB, %.0f s",
                dataset.DATASET_VERSION, len(rows), labelled.sum(), rows["history"].nunique(),
                rows["day"].min().date(), rows["day"].max().date(),
                sum(f.stat().st_size for f in path.glob("*.npy")) / 1e6, time.time() - started)


def load_dataset() -> pd.DataFrame:
    return dataset.load_table(dataset_path())


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
    if not holdout_opens:                 # the last days of 2025 are graded on 2026 prices: closed with it
        pred = pred[wf.period(pred["day"]) != "holdout"]
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


def _latest_exam() -> pd.DataFrame:
    exams = sorted(BRAIN_DIR.glob("exam-*.pkl"))
    if not exams:
        raise SystemExit("no exam yet: run `brain.py exam` first")
    pred = pd.read_pickle(exams[-1])
    return pred.assign(model=f"exam:{exams[-1].stem}")[wk.RECORD_COLUMNS]


def first_champion(args) -> None:
    """A champion on everything graded, its record seeded with the newest exam's months."""
    rows = load_dataset()
    record = _latest_exam()
    brain = wk.calibrate_from(record, model.train(rows, dataset.RANKING_FEATURES, wk.graded(rows)))
    brain.meta["model_id"] = f"{model.MODEL_VERSION}-{time.strftime('%Y-%m-%d')}"
    brain.save(CHAMPION)
    record.to_pickle(RECORD)
    logger.info("champion %s: trained until %s on %d rows, %d trees; record %d rows (%s … %s)",
                brain.meta["model_id"], brain.trained_until, brain.rows, brain.meta["trees"], len(record),
                record["day"].min().date(), record["day"].max().date())


def _usd(x: float) -> str:
    return "—" if x != x else f"{'+' if x >= 0 else '−'}${abs(x):.2f}"


def weekly(args) -> None:
    started = time.time()
    if args.rebuild:
        args.ranking_only = True
        build_dataset(args)              # saved to disk and dropped: the table is read back memory-mapped
    rows = load_dataset()
    champion = model.Brain.load(CHAMPION) if CHAMPION.exists() else None
    record = pd.read_pickle(RECORD) if RECORD.exists() else _latest_exam()
    old_id = champion.meta.get("model_id", "?") if champion else "-"
    new_id = f"{model.MODEL_VERSION}-{time.strftime('%Y-%m-%d')}"
    result = wk.week(rows, champion, record, old_id, new_id)
    if result.promoted and champion is not None:
        champion.save(PREVIOUS)
    result.champion.save(CHAMPION)
    result.record.to_pickle(RECORD)
    probe = result.probe
    summary = {"at": time.strftime("%Y-%m-%d %H:%M"), "promoted": result.promoted,
               "champion": result.champion.meta.get("model_id"), "trained_until": result.champion.trained_until,
               "added_to_record": result.added, "probe": probe, "agreement": result.agreement,
               "notes": result.notes, "seconds": round(time.time() - started)}
    (BRAIN_DIR / "weekly.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("weekly: %s", summary)
    text = ["🧠 Мозг: недельное обучение",
            (f"✅ Новая версия {new_id}" if result.promoted else f"⚠️ Оставил прежнюю версию {old_id}"),
            f"Последние {wk.PROBE_WEEKS} недель ({probe['from']} … {probe['to']}): лучшие 10% монет дня "
            f"против худших 10%: {_usd(probe['spread_usd'])} на $100 за 2 недели."]
    text += [f"• {n}" for n in result.notes]
    if result.added:
        text.append(f"В журнал добавлено {result.added:,} закрытых монето-дней.".replace(",", " "))
    if args.notify and not notify_owner("\n".join(text)):
        logger.warning("weekly: could not reach the owner on Telegram")


def audit(args) -> None:
    record = pd.read_pickle(RECORD)
    month = pd.Period(args.month, "M") if args.month else pd.Timestamp.now().to_period("M") - 1
    champion = model.Brain.load(CHAMPION) if CHAMPION.exists() else None
    text = wk.audit_text(record, month, champion)
    print(text)
    if args.notify and not notify_owner(text):
        logger.warning("audit: could not reach the owner on Telegram")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dataset")
    e = sub.add_parser("exam")
    sub.add_parser("train")
    w = sub.add_parser("weekly")
    w.add_argument("--rebuild", action="store_true")
    a = sub.add_parser("audit")
    a.add_argument("--month", default="")
    for parser_ in (w, a):
        parser_.add_argument("--notify", action="store_true")
    for parser_ in (d, w):
        parser_.add_argument("--ranking-only", action="store_true")
    d.add_argument("--if-stale", action="store_true", help="skip when the table already has the newest closed day")
    sub.add_parser("cache")
    e.add_argument("--refit-months", type=int, default=1)
    e.add_argument("--open-holdout", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "dataset":
        build_dataset(args)
    elif args.command == "exam":
        exam(args)
    elif args.command == "train":
        first_champion(args)
    elif args.command == "weekly":
        weekly(args)
    elif args.command == "audit":
        audit(args)
    elif args.command == "cache":
        from core.brain import simulator
        logger.info("hourly cache: %d histories", simulator.build_cache(MarketStore(args.db)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
