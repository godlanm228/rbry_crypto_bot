"""Fear markers on short holds: do VIX, BTC's share of volume, funding and Fear & Greed act within days?

    python scripts/brain_fast.py

The meta-brain (scripts/brain_meta.py) found these markers keeping
their sign every year but too weak to switch 14-day books. Crowd markers often
act fast, so the same books are held 1 and 3 days (core/brain/meta.py
short_books, with costs) and the meta-brain learns from the fear markers only.
Every idea is in the journal before its first result. 2022-2024 to choose,
2025 to confirm, nothing from 2026.
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
from core.brain import meta  # noqa: E402
from core.brain.dataset import TRADE_DAYS  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.playbook import scored_frame  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
HOLDS = (1, 3)
FEAR_TEXT = "VIX, доля BTC в объёме, фандинг, индекс страха и жадности (и его среднее за 7 дней)"
IDEAS = tuple(x for h in HOLDS for x in (
    (f"F0-{h}D", f"все три корзины всегда, держать {h} дн.", "планка: короткое держание без подсказок",
     f"long + short + btc books held {h} day(s), every day, a third each", (h, "all")),
    (f"F1-{h}D", f"маркеры страха включают корзины, держать {h} дн.",
     "маркеры толпы действуют быстро: на 1–3 дня их подсказка сильнее, чем на 14",
     f"monthly: key markers among {', '.join(meta.FEAR)} (one sign in every past half-year), "
     f"book on when the fitted {h}-day outcome > 0", (h, "switch")),
    (f"F2-{h}D", f"маркеры страха меняют размер, держать {h} дн.",
     "наклон размера по маркерам страха лучше выключения",
     f"weights = F1-{h}D forecast / the book's usual {h}-day outcome, clipped to 0..2", (h, "tilt")),
))


def strength(st: pd.DataFrame, outcomes: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Per marker, book and hold: the rank correlation with the outcome in every year, and how steady its sign is."""
    rows = []
    for hold, out in outcomes.items():
        joined = st[list(meta.FEAR)].join(out, how="inner")
        for f in meta.FEAR:
            for b in meta.BOOKS:
                per_year = {}
                for year, g in joined.groupby(joined.index.year):
                    ok = g[f].notna() & g[b].notna()
                    if ok.sum() >= 90:
                        per_year[year] = g.loc[ok, f].rank().corr(g.loc[ok, b].rank())
                c = pd.Series(per_year, dtype=float)
                sign = np.sign(c.mean()) if len(c) else 0.0
                rows.append({"marker": f, "book": b, "hold": hold, "mean": c.mean(),
                             "steady": int((np.sign(c) == sign).sum()), "years": len(c),
                             **{int(y): v for y, v in per_year.items()}})
    return pd.DataFrame(rows)


def main() -> int:
    started = time.time()
    rows = ds.load_table(BRAIN_DIR / f"dataset-{ds.DATASET_VERSION}")
    exam_file = sorted(BRAIN_DIR.glob("exam-*.pkl"))[-1]
    scored = scored_frame(pd.read_pickle(exam_file), rows)
    outcomes = {TRADE_DAYS: meta.known_by(meta.books(scored, rows))}
    for h in HOLDS:
        outcomes[h] = meta.known_by(meta.short_books(scored, rows, h))
    st = meta.state(rows, outcomes[TRADE_DAYS])
    del rows
    journal = Journal()
    for iid, title, hypothesis, rule, _ in IDEAS:
        journal.register(iid, title, hypothesis, rule)

    preds, markers, results = {}, {}, []
    for h in HOLDS:
        meta.MARKER_LOG.clear()
        preds[h] = meta.exam(st.join(outcomes[h], how="inner"), "markers", features=meta.FEAR)
        markers[h] = list(meta.MARKER_LOG)
        print(f"{h} d: {time.time() - started:.0f} s", flush=True)
    for iid, title, _, _, (h, kind) in IDEAS:
        days, out = preds[h].index, outcomes[h]
        on = (meta.learned_switches(preds[h]) if kind == "switch"
              else meta.tilted(preds[h], out) if kind == "tilt"
              else meta.rule_switches(st.loc[days], "all"))
        g, years, sv = meta.grade(on, out), meta.by_year(on, out), meta.switching_value(on, out)
        for per, r in g.iterrows():
            journal.result(iid, per, {**r.to_dict(), "hold_days": h, "exam": exam_file.stem,
                                      "dataset": ds.DATASET_VERSION, "books": "thirds, trade by trade, with costs"})
        results.append((iid, title, h, g, years, sv))
    power = strength(st, outcomes)

    lines = [f"# Маркеры страха на коротком сроке — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Те же корзины, что в метамозге (по местам экзамена {exam_file.stem}): лонг 3 лучших, шорт 3 худших (1x), "
             "BTC, — но держать 1 или 3 дня, от закрытия дня сигнала до закрытия. С комиссиями, проскальзыванием и "
             f"фандингом. Метамозг каждый месяц выбирает из маркеров страха ({FEAR_TEXT}) те, что держали знак "
             f"во всех прошлых полугодиях. Идей в журнале: {journal.tried()}.", "",
             "| id | что | период | $ на $100 за сигнал | ≈ за 2 недели | дней в плюс из 10 | плохие 10% дней, $ | "
             "лонг вкл. из 10 | шорт вкл. из 10 | BTC вкл. из 10 |", "|" + "---|" * 10]
    for iid, title, h, g, _, _ in results:
        for per, r in g.iterrows():
            lines.append(f"| {iid} | {title} | {per} | {fmt(r.usd_per_100, 2, True)} | "
                         f"{fmt(r.usd_per_100 * TRADE_DAYS / h, 1, True)} | {fmt(r.up_of10, 1)} | "
                         f"{fmt(r.bad_tenth, 1, True)} | {fmt(r.on_long, 1)} | {fmt(r.on_short, 1)} | {fmt(r.on_btc, 1)} |")
    years = sorted({y for *_, t, _ in results for y in t.index})
    lines += ["", "## По годам ($ на $100 за сигнал)", "", "| id | " + " | ".join(map(str, years)) + " |",
              "|" + "---|" * (len(years) + 1)]
    for iid, _, _, _, t, _ in results:
        lines.append(f"| {iid} | " + " | ".join(fmt(t.get(y), 2, True) for y in years) + " |")
    lines += ["", "## Польза переключения (F1): средний итог корзины, когда она включена и выключена ($ на $100)", "",
              "| id | период | корзина | вкл. из 10 | когда вкл. | когда выкл. |", "|" + "---|" * 6]
    for iid, _, _, _, _, sv in results:
        if iid.startswith("F1-"):
            for r in sv.itertuples():
                lines.append(f"| {iid} | {r.period} | {r.book} | {fmt(r.on_of10, 1)} | {fmt(r.when_on, 2, True)} | "
                             f"{fmt(r.when_off, 2, True)} |")
    py = sorted(c for c in power.columns if isinstance(c, int))
    lines += ["", "## Сила маркеров по сроку: связь с итогом корзины по годам (−1…+1, 0 = никакой)", "",
              "| маркер | корзина | срок, дн. | средняя | тот же знак, лет | " + " | ".join(map(str, py)) + " |",
              "|" + "---|" * (5 + len(py))]
    for _, r in power.sort_values(["marker", "book", "hold"]).iterrows():
        lines.append(f"| {r['marker']} | {r['book']} | {r['hold']} | {fmt(r['mean'], 2, True)} | "
                     f"{r['steady']} из {r['years']} | " + " | ".join(fmt(r.get(y), 2, True) for y in py) + " |")
    lines += ["", "## Какие маркеры метамозг выбирал, доля месяцев", ""]
    for h in HOLDS:
        for book in meta.BOOKS:
            months = [m for _, bk, m in markers[h] if bk == book]
            counts = pd.Series([f"{'+' if s > 0 else '−'}{f}" for m in months for f, s in m.items()]).value_counts()
            top = ", ".join(f"{k} {v / len(months):.0%}" for k, v in counts.items()) if len(counts) else "ничего"
            lines.append(f"- {h} дн., {book}: {top}")
    report = BRAIN_DIR / f"fast-{time.strftime('%Y%m%d-%H%M')}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.to_pickle({"preds": preds, "outcomes": outcomes, "power": power}, BRAIN_DIR / "fast-exam.pkl")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
