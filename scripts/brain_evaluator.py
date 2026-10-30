"""The trade evaluator's exam (core/brain/evaluator.py): does the squeeze risk say which shorts to leave alone?

    python scripts/brain_evaluator.py

Every out-of-sample coin-day of the newest brain exam gets a squeeze and a crash
chance from models that only saw earlier days. Graded: promises against what
happened, and the shorts N5 would take (the day's 3 worst) against shorts the
evaluator picks among the day's worst tenth, trade by trade. Ideas go into the
journal before their first result. 2026 is open since 30.10.2026, but the
evaluator was designed after its exam, so 2026 confirms nothing on its own.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import dataset as ds  # noqa: E402
from core.brain import evaluator as ev  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
PICKS = 3
IDEAS = (
    ("B-SQZ", "оценщик выноса: шорт 3 из худших 10% дня с самым низким шансом выноса",
     "сквизы предсказуемы по самой монете: шорт там, где выносит редко, сохраняет плюс худших и убирает ликвидации",
     f"short, 14 days on 1x margin, the 3 coins of the day's worst tenth (place <= 0.1) with the lowest out-of-sample "
     f"chance of +{ev.SQUEEZE:.0%} within 14 days"),
    ("B-SQZ3", "N5, но без шортов с шансом выноса выше 30%",
     "отсев только самых опасных шортов, остальное как у N5",
     "short the day's 3 worst among coins with squeeze chance <= 0.30"),
    ("B-CRASH", "лонг 3 из лучших 10% дня с самым низким шансом обвала",
     "обвал лонга тоже предсказуем по самой монете",
     f"long, 14-day hold, the 3 coins of the day's best tenth with the lowest chance of -{ev.CRASH:.0%} within 14 days"),
)


def period(days: pd.Series) -> pd.Series:
    return pd.Series(np.where(days <= "2024-12-31", "2022–2024", np.where(days <= "2025-12-31", "2025", "2026")),
                     index=days.index)


def book(chosen: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per period: $ per $100 per trade, trades in profit of 10, liquidations (or -30% losses) in 100."""
    g = chosen.groupby(period(chosen["day"]))
    worst = -1.0 if col == "short_liq_ret" else -0.3
    return pd.DataFrame({"trades": g.size(), "usd_per_100": 100 * g[col].mean(),
                         "up_of10": 10 * g[col].apply(lambda r: float((r > 0).mean())),
                         "ruin_of100": 100 * g[col].apply(lambda r: float((r <= worst).mean()))})


def pick(pool: pd.DataFrame, by: str, ascending: bool, n: int = PICKS) -> pd.DataFrame:
    return pool[pool.groupby("day")[by].rank(ascending=ascending, method="first") <= n]


def main() -> int:
    started = time.time()
    journal = Journal()
    for iid, title, hypothesis, rule in IDEAS:
        journal.register(iid, title, hypothesis, rule)
    exam_file = sorted(BRAIN_DIR.glob("exam-*.pkl"))[-1]
    exam = pd.read_pickle(exam_file)[["history", "day", "place"]]
    exam["history"] = exam["history"].astype(str)
    cols = ["history", "base", "day", "hold_ret", "short_liq_ret", *ds.RANKING_FEATURES]
    rows = ds.load_table(BRAIN_DIR / f"dataset-{ds.DATASET_VERSION}", list(dict.fromkeys(cols)))
    rows = rows.assign(history=rows["history"].astype(str)).merge(exam, on=["history", "day"], how="inner")
    print(f"coin-days {len(rows):,}: {time.time() - started:.0f} s", flush=True)
    exc = ev.excursions(sim.Market(), rows[["history", "day"]])
    frame = ev.labels(rows.merge(exc, on=["history", "day"], how="left"))
    print(f"excursions: {time.time() - started:.0f} s; squeeze rate {100 * frame['squeeze'].mean():.1f}%, "
          f"crash rate {100 * frame['crash'].mean():.1f}%", flush=True)
    end = str(frame.loc[frame["squeeze"].notna(), "day"].max().date())
    pred = ev.run(frame, end, on_refit=lambda m, n: print(f"  refit {m}: {n:,} rows, {time.time() - started:.0f} s",
                                                          flush=True))
    pred = pred.merge(frame[["history", "day", "base", "hold_ret", "short_liq_ret"]], on=["history", "day"])
    pd.to_pickle(pred, BRAIN_DIR / "evaluator-exam.pkl")

    shorts = pred[pred["short_liq_ret"].notna() & pred["squeeze_chance"].notna()]
    worst = shorts[shorts["place"] <= 0.1]
    longs = pred[pred["hold_ret"].notna() & pred["crash_chance"].notna()]
    best = longs[longs["place"] > 0.9]
    books = {
        "N5: шорт 3 худших": book(pick(shorts, "place", True), "short_liq_ret"),
        "N6: шорт 4–6 с конца": book(shorts[shorts.groupby("day")["place"].rank(method="first").between(4, 6)],
                                     "short_liq_ret"),
        "B-SQZ: 3 из худших 10% с низким выносом": book(pick(worst, "squeeze_chance", True), "short_liq_ret"),
        "B-SQZ3: 3 худших с выносом <= 30%": book(pick(shorts[shorts["squeeze_chance"] <= 0.3], "place", True),
                                                 "short_liq_ret"),
        "все худшие 10% (для сравнения)": book(worst, "short_liq_ret"),
        "L1: лонг 3 лучших": book(pick(longs, "place", False), "hold_ret"),
        "B-CRASH: 3 из лучших 10% с низким обвалом": book(pick(best, "crash_chance", True), "hold_ret"),
    }
    ids = {"B-SQZ: 3 из худших 10% с низким выносом": "B-SQZ", "B-SQZ3: 3 худших с выносом <= 30%": "B-SQZ3",
           "B-CRASH: 3 из лучших 10% с низким обвалом": "B-CRASH"}
    for name, t in books.items():
        if name in ids:
            for per, r in t.iterrows():
                journal.result(ids[name], per, {**r.to_dict(), "exam": exam_file.stem, "books": "trade by trade"})

    lines = [f"# Оценщик сделок (B) — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Монето-дней: {len(frame):,}; вынос = цена была на {ev.SQUEEZE:.0%} выше входа в какой-то час за 14 дней "
             f"(в среднем у {100 * frame['squeeze'].mean():.1f}% монето-дней), обвал = на {ev.CRASH:.0%} ниже "
             f"({100 * frame['crash'].mean():.1f}%). Модели учатся только на прошлом (каждые {ev.REFIT_MONTHS} мес.), "
             f"признаки только самой монеты и место у мозга. Экзамен мозга {exam_file.stem}. "
             f"Прогон {time.time() - started:.0f} с.", ""]
    for target, title in (("squeeze", "Шанс выноса"), ("crash", "Шанс обвала")):
        for per in ("2022–2024", "2025", "2026"):
            sub = pred[period(pred["day"]) == per]
            t = ev.promise_table(sub, target)
            lines += [f"## {title}: обещал против правды, {per}", "", "| обещал | монето-дней | обещал, % | сбылось, % |",
                      "|---|---|---|---|"]
            lines += [f"| {iv} | {int(r.coin_days):,} | {fmt(r.promised_pct, 1)} | {fmt(r.real_pct, 1)} |"
                      for iv, r in t.iterrows()]
            lines.append("")
    lines += ["## Сделка в сделку: $ на $100, в плюс из 10, разорительных из 100 (шорт: ликвидация; лонг: −30%)", "",
              "| что | период | сделок | $ на $100 | в плюс из 10 | разорительных из 100 |", "|---|---|---|---|---|---|"]
    for name, t in books.items():
        for per, r in t.iterrows():
            lines.append(f"| {name} | {per} | {int(r.trades):,} | {fmt(r.usd_per_100, 2, True)} | {fmt(r.up_of10, 1)} | "
                         f"{fmt(r.ruin_of100, 1)} |")
    n5 = pick(shorts, "place", True)
    liq = n5[(n5["short_liq_ret"] <= -1.0) & (n5["day"] >= "2026-01-01")].sort_values("day")
    lines += ["", "## Ликвидированные шорты N5 в 2026: что о них говорил оценщик заранее", "",
              "| день | монета | шанс выноса | почему (что подняло риск больше всего) |", "|---|---|---|---|"]
    for r in liq.itertuples():
        lines.append(f"| {r.day:%Y-%m-%d} | {r.base} | {100 * r.squeeze_chance:.0f}% | {ev.why(r.why_squeeze)} |")
    report = BRAIN_DIR / f"evaluator-{time.strftime('%Y%m%d-%H%M')}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
