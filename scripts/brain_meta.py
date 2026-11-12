"""The meta-brain's exam: which books to run in which market (core/brain/meta.py).

    python scripts/brain_meta.py

Books are booked trade by trade on the newest exam's places; every learner and
every fixed rule is registered in the journal before its first result, and
every result goes in, good or bad. 2022-2024 to choose, 2025 to confirm,
nothing from 2026.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.console import utf8_output  # noqa: E402

import pandas as pd  # noqa: E402

from core.brain import dataset as ds  # noqa: E402
from core.brain import meta  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.playbook import scored_frame  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
IDEAS = (
    ("M0-ALL", "все три корзины всегда: 3 лучших лонг, 3 худших шорт, BTC", "планка: без переключения",
     "long + short + btc books every day, a third each", ("rule", "all")),
    ("M0-PAIRS", "лучшие лонг + худшие шорт всегда (как N1/P1)", "планка: наше лучшее семейство",
     "long + short books every day", ("rule", "pairs")),
    ("M0-RULE", "правило: шорт не в альтсезон, BTC в рост", "планка: ручные правила режима (N3 + B1)",
     "long always, short when beat_btc_30d <= 0.5, btc when btc_trend > 0", ("rule", "rule")),
    ("M0-BTC", "только BTC в рост (B1)", "планка: тренд BTC", "btc book when btc_trend > 0", ("rule", "btc_trend")),
    ("M1-LGBM", "метамозг: деревья решают, какие корзины включить",
     "обстановка дня (режим, ширина, альтсезон, фон, недавняя точность мозга) подсказывает, какая корзина заработает",
     "LightGBM per book on the day's state, refit monthly, book on when predicted > 0", ("learn", "lgbm")),
    ("M2-RIDGE", "метамозг: простая линейная модель", "то же, но модель, которой труднее подогнаться под прошлое",
     "ridge per book on the standardised state, refit monthly", ("learn", "ridge")),
    ("M3-KNN", "метамозг: похожие дни из прошлого", "в похожей обстановке корзины ведут себя похоже",
     "mean outcome of the 60 most similar past days, refit monthly", ("learn", "knn")),
    # second batch,: the first three drowned in 28 features; per-year checks showed a few markers that kept
    # their sign every year (BTC's volume share, VIX, funding), so let the meta-brain pick such markers itself
    ("M4-MARKERS", "метамозг: только ключевые маркеры", "маркеры, державшие знак во всех прошлых полугодиях, "
     "предсказывают корзины лучше, чем все признаки сразу",
     "monthly: features with one sign of rank correlation in every past half-year, signed z-scores averaged, "
     "book on when the fitted outcome > 0", ("learn", "markers")),
    ("M5-TILT", "метамозг: маркеры меняют размер, а не выключают",
     "наклон размера по обстановке лучше выключения: не теряем доходность в непредсказуемые дни",
     "weights = M4 forecast / the book's usual outcome, clipped to 0..2", ("tilt", "markers")),
)


def main() -> int:
    utf8_output()
    started = time.time()
    rows = ds.load_table(BRAIN_DIR / f"dataset-{ds.DATASET_VERSION}")
    exam_file = sorted(BRAIN_DIR.glob("exam-*.pkl"))[-1]
    scored = scored_frame(pd.read_pickle(exam_file), rows)
    outcomes = meta.books(scored, rows)
    outcomes = meta.known_by(outcomes)                       # the last days of 2025 are graded on 2026 prices
    table = meta.state(rows, outcomes).join(outcomes, how="inner")
    del rows
    journal = Journal()
    for iid, title, hypothesis, rule, _ in IDEAS:
        journal.register(iid, title, hypothesis, rule)

    preds, results = {}, []
    for iid, title, _, _, (kind, name) in IDEAS:
        if kind == "learn":
            meta.MARKER_LOG.clear()
            preds[iid] = meta.exam(table, name)
            if name == "markers":
                markers = list(meta.MARKER_LOG)
            print(f"{iid}: {time.time() - started:.0f} s", flush=True)
    days = next(iter(preds.values())).index
    for iid, title, _, _, (kind, name) in IDEAS:
        on = (meta.learned_switches(preds[iid]) if kind == "learn"
              else meta.tilted(preds["M4-MARKERS"], outcomes) if kind == "tilt"
              else meta.rule_switches(table.loc[days], name))
        g, years, sv = meta.grade(on, outcomes), meta.by_year(on, outcomes), meta.switching_value(on, outcomes)
        for per, r in g.iterrows():
            journal.result(iid, per, {**r.to_dict(), "exam": exam_file.stem, "dataset": ds.DATASET_VERSION,
                                      "books": "thirds, trade by trade"})
        results.append((iid, title, g, years, sv))
    pd.to_pickle({"preds": preds, "table": table, "outcomes": outcomes}, BRAIN_DIR / "meta-exam.pkl")

    lines = [f"# Метамозг — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Корзины (сделка в сделку, по местам экзамена {exam_file.stem}): лонг 3 лучших, шорт 3 худших (1x), BTC; "
             "держать 14 дней; каждая включённая корзина получает треть денег, выключенная — в USDT. "
             f"Дней экзамена: {len(days)} ({days.min().date()} … {days.max().date()}). Идей в журнале: {journal.tried()}.", "",
             "| id | что | период | $ на $100 за сигнал | дней в плюс из 10 | плохие 10% дней, $ | лонг вкл. из 10 | "
             "шорт вкл. из 10 | BTC вкл. из 10 |", "|" + "---|" * 9]
    for iid, title, g, _, _ in results:
        for per, r in g.iterrows():
            lines.append(f"| {iid} | {title} | {per} | {fmt(r.usd_per_100, 2, True)} | {fmt(r.up_of10, 1)} | "
                         f"{fmt(r.bad_tenth, 1, True)} | {fmt(r.on_long, 1)} | {fmt(r.on_short, 1)} | {fmt(r.on_btc, 1)} |")
    years = sorted({y for _, _, _, t, _ in results for y in t.index})
    lines += ["", "## По годам ($ на $100 за сигнал)", "", "| id | " + " | ".join(map(str, years)) + " |",
              "|" + "---|" * (len(years) + 1)]
    for iid, _, _, t, _ in results:
        lines.append(f"| {iid} | " + " | ".join(fmt(t.get(y), 2, True) for y in years) + " |")
    lines += ["", "## Польза переключения: средний итог корзины в дни, когда она включена и выключена ($ на $100)", "",
              "| id | период | корзина | вкл. из 10 | когда вкл. | когда выкл. |", "|" + "---|" * 6]
    for iid, _, _, _, sv in results:
        if not iid.startswith(("M0-", "M5-")):
            for r in sv.itertuples():
                lines.append(f"| {iid} | {r.period} | {r.book} | {fmt(r.on_of10, 1)} | {fmt(r.when_on, 2, True)} | "
                             f"{fmt(r.when_off, 2, True)} |")
    lines += ["", "## Какие маркеры метамозг выбирал (M4), доля месяцев", ""]
    for book in meta.BOOKS:
        months = [m for _, bk, m in markers if bk == book]
        counts = pd.Series([f"{'+' if s > 0 else '−'}{f}" for m in months for f, s in m.items()]).value_counts()
        top = ", ".join(f"{k} {v / len(months):.0%}" for k, v in counts.head(8).items()) if len(counts) else "ничего"
        lines.append(f"- {book}: {top}")
    report = BRAIN_DIR / f"meta-{time.strftime('%Y%m%d-%H%M')}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
