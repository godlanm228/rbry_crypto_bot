"""The selector's exam (core/brain/selector.py): does a model that learns trade by trade pick the right trade?

    python scripts/brain_selector.py [--workers 3]

Builds every coin-hour of the newest event library run under every action, walks them on hourly
prices (labels cached per library run), refits the model every month from 2022 on the trades
closed before that month, and books the chosen trades as a $100 portfolio in every year, against
the panic on paper and holding BTC. Report and forecasts in runtime/learning/selector/.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import factory as fa  # noqa: E402
from core.brain import regime as rg  # noqa: E402
from core.brain import selector as se  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
OUT_DIR = ROOT / "runtime" / "learning" / "selector"
YEARS = (2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)
_W: dict = {}


def _init(newest: int) -> None:
    _W.update(market=sim.Market(), newest=newest)


def label_chunk(c: pd.DataFrame) -> pd.DataFrame:
    return se.label_rows(c, _W["market"], _W["newest"])


def twin_chunk(c: pd.DataFrame) -> pd.DataFrame:
    return se.twin_rows(c, _W["market"], _W["newest"])


def cached(path: Path, groups: list, fn, workers: int, newest: int) -> pd.DataFrame:
    if path.exists():
        return pd.read_pickle(path)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(newest,)) as pool:
        out = pd.concat(list(pool.map(fn, groups)), ignore_index=True)
    for old in path.parent.glob(path.name.split("-")[0] + "-*.pkl"):
        old.unlink()
    pd.to_pickle(out, path)
    return out


def growth(results: dict, years) -> float:
    """All years' $100 books compounded, a year's growth: how a model did walked forward."""
    total = float(np.prod([1 + results[str(y)]["growth_pct"] / 100 for y in years]))
    return 100 * (total ** (1 / len(years)) - 1) if total > 0 else -100.0


def table_of(source: Path, market) -> pd.DataFrame:
    path = fa.FACTORY_DIR / f"table-{source.stem}-{fa.TABLE_VERSION}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    ev_path = BRAIN_DIR / "evaluator-exam.pkl"
    table = fa.trade_table(pd.read_pickle(source), market, pd.read_pickle(ev_path) if ev_path.exists() else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(table, path)
    return table


def year_cell(b: dict) -> str:
    return (f"{b['n']:,} · {fmt(b['up_of10'], 1)} · {fmt(b['usd_per_100'], 2, True)} · "
            f"{fmt(b['growth_pct'], 0, True)}% · {fmt(b['drawdown_pct'], 0)}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--live-only", action="store_true", help="only train and save the paper watcher's model")
    args = parser.parse_args()
    started = time.time()
    source = sorted(BRAIN_DIR.glob("events-*.pkl"))[-1]
    market = sim.Market()
    newest = int(market.btc()[-1, 0])
    table = table_of(source, market)
    cands = se.candidates(table, market)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = [g for _, g in cands.groupby(cands["history"].map(lambda h: hash(h) % 60))]
    labels = cached(OUT_DIR / f"labels-{source.stem}.pkl", groups, label_chunk, args.workers, newest)
    twins = cached(OUT_DIR / f"twins-{source.stem}.pkl", groups, twin_chunk, args.workers, newest)
    print(f"{len(cands):,} coin-hours, {len(labels):,} actions: {time.time() - started:.0f} s", flush=True)
    f = se.frame(cands, labels, twins)
    f_live = se.frame(se.candidates(se.live_table(table), market), labels, twins)
    choice_path = OUT_DIR / "live-target.txt"            # which target won the last full exam
    live_target = choice_path.read_text().strip() if choice_path.exists() else "net"

    old = se.load()
    prior = float(old[1].get("prior", 0.0)) if old else 0.0

    def save_live(target: str, prior: float = prior) -> None:
        se.save(se.train(f_live, target=target), {
            "trained_at": time.strftime("%Y-%m-%d %H:%M"), "events": source.stem, "trades": len(f_live),
            "newest_signal": int(f_live["signal_ts"].max()), "features": list(se.FEATURE_SETS["live"]),
            "bar": se.LIVE_BAR, "target": target, "prior": prior})
        print(f"live model ({target}): {len(f_live):,} trades, {time.time() - started:.0f} s", flush=True)

    if args.live_only:
        save_live(live_target)
        return 0
    regimes = rg.btc_regimes(market.btc())
    btc = {str(y): rg.curve_stats(rg.hold_curve(regimes, market.btc(), None, 1, None, *fa.year_span(y, newest)))
           for y in YEARS}
    panic = fa.book(fa.seeds()[0], table, market, newest)["years"]
    journal = Journal()
    journal.register("SELECTOR", "модель выбора сделки: в каждый момент лучшее из 8 действий по прогнозу",
                     "модель, которая учится только на закрытых сделках, выбирает купить/шорт и выход по ситуации "
                     "лучше любой одной стратегии",
                     "LightGBM regression of every action's net, refit monthly from 2022 on trades closed before the "
                     "month; per coin-hour the best forecast if above a bar; $100 book per year, $20 a trade, 5 open")
    lines = [f"# Модель выбора сделки — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Кандидаты: {len(cands):,} монето-часов библиотеки событий ({source.name}), у каждого до 8 действий "
             f"(купить на споте или шорт 1x; выход через 4 ч, сутки, 3 дня с половиной на +5%, 3 дня со стопом 2 хода "
             f"и целью 2 стопа), всего {len(labels):,} сделок, вход через 10 минут, комиссии, проскальзывание, "
             f"фандинг. Модель каждый месяц переучивается на сделках, закрытых ДО этого месяца, и прогнозирует только "
             f"его: ни одна сделка не знает своего будущего, каждая следующая учится на всех прошлых. Берётся лучшее "
             f"действие монеты, если прогноз выше планки; счёт $100 в каждом году, $20 на сделку, не больше 5 сразу, "
             f"сначала лучшие прогнозы. Две модели: «полная» видит и движения всего рынка (BTC за час/сутки/неделю, "
             f"отрыв от средней, нервность), «монетная» — только монету, момент, событие и грубый режим (урок "
             f"оценщика A1: признаки всего рынка запоминают удачные месяцы), «живая» — монетная только на событиях, "
             f"которые дозорный видит вживую, и без шанса выноса: она и стоит на бумаге (обучена на всех "
             f"{len(f_live):,} закрытых сделках).", "",
             "Ячейка: сделок · в плюс из 10 · $ на $100 · рост счёта · просадка.", ""]
    walked, per_trade = {}, {}
    for name, features, target in [(n, fs, tg) for n, fs in se.FEATURE_SETS.items() for tg in ("net", "edge")
                                   if not (n == "full" and tg == "edge")]:
        pred = se.walk_forward(f_live if name == "live" else f, features=features, target=target,
                               on_refit=lambda m, n: print(f"  {name} refit {m:%Y-%m}: {n:,} closed trades, "
                                                           f"{time.time() - started:.0f} s", flush=True)
                               if m.month == 1 else None)
        importance = pred.attrs.get("importance", pd.Series(dtype=float))
        pd.to_pickle(pred, OUT_DIR / f"forecasts-{name}-{target}.pkl")
        if target == (live_target if name == "live" else "net"):     # the plain name: the target that is used
            pd.to_pickle(pred, OUT_DIR / f"forecasts-{name}.pkl")     # (the night researcher, 06.11: "edge" won it)
        chosen = {b: se.choose(pred, b) for b in se.BARS}
        results = {b: se.books(chosen[b], YEARS, newest) for b in se.BARS}
        for b in se.BARS:
            for y in YEARS:
                journal.result("SELECTOR", f"{name}, {y}, планка {b}", {**results[b][str(y)], "events": source.stem})
        walked[(name, target)] = growth(results[se.LIVE_BAR], YEARS)
        recent = [results[se.LIVE_BAR][y]["usd_per_100"] for y in ("2025", "2026")]
        per_trade[(name, target)] = float(np.nanmean(recent)) / 100 if np.isfinite(recent).any() else 0.0
        lines += [f"## Модель «{ {'full': 'полная', 'coin': 'монетная', 'live': 'живая'}[name] }», учится на "
                  f"«{'результат сделки' if target == 'net' else 'лучше случайного входа'}» — все годы вместе при планке "
                  f"+1%: {fmt(walked[(name, target)], 0, True)}% в год", "",
                  "| год | " + " | ".join(f"планка +{100 * b:.1f}%" for b in se.BARS) + " | паника (на бумаге) | "
                  "держать BTC |", "|" + "---|" * (len(se.BARS) + 3)]
        for y in YEARS:
            p = panic[str(y)]
            lines.append(f"| {y} | " + " | ".join(year_cell(results[b][str(y)]) for b in se.BARS)
                         + f" | {p['n']:,} · {fmt(p['up_of10'], 1)} · {fmt(p['usd_per_100'], 2, True)} · "
                         f"{fmt(p['growth_pct'], 0, True)}% · {fmt(p['drawdown_pct'], 0)}% | "
                         f"{fmt(btc[str(y)]['growth_pct'], 0, True)}% · {fmt(btc[str(y)]['drawdown_pct'], 0)}% |")
        lines += ["", "Обещал против правды, по годам (все действия): прогноз → сделок · обещал · вышло, %", "",
                  "| прогноз | " + " | ".join(str(y) for y in YEARS) + " |", "|" + "---|" * (len(YEARS) + 1)]
        tables = {y: se.promise_table(pred[pred["year"] == y]) for y in YEARS}
        for band in tables[YEARS[-1]].index.union(tables[YEARS[0]].index):
            cells = []
            for y in YEARS:
                r = tables[y].loc[band] if band in tables[y].index else None
                cells.append(f"{int(r.trades):,} · {fmt(r.forecast_pct, 1, True)} · {fmt(r.real_pct, 1, True)}"
                             if r is not None else "—")
            lines.append(f"| {band} | " + " | ".join(cells) + " |")
        pick = chosen[0.01]
        pick = pick[pick["year"] >= 2025]
        if len(pick):
            lines += ["", "Что выбирает (планка +1%, 2025–2026) по режиму рынка:", "",
                      "| режим | сделок | чаще всего | в плюс из 10 | $ на $100 |", "|---|---|---|---|---|"]
            for (trend, mood), g in pick.groupby(["trend", "mood"]):
                top = g["action"].value_counts(normalize=True).head(2)
                lines.append(f"| {rg.TRENDS.get(trend, trend)} · {rg.MOODS.get(mood, mood)} | {len(g):,} | "
                             + ", ".join(f"{se.ACTION_RU[a]} {100 * v:.0f}%" for a, v in top.items())
                             + f" | {fmt(10 * float((g['net'] > 0).mean()), 1)} | "
                             f"{fmt(100 * float(g['net'].mean()), 2, True)} |")
        lines += ["", "На что смотрит больше всего: "
                  + ", ".join(f"{se.WORDS.get(k, k)} {100 * v:.0f}%" for k, v in importance.head(8).items()) + ".", ""]
    best = max(("net", "edge"), key=lambda tg: walked.get(("live", tg), -100.0))
    choice_path.write_text(best)
    save_live(best, per_trade[("live", best)])    # the allocator's prior: a trade's mean in 2025-2026
    lines.append(f"На бумагу идёт живая модель, учится на «{'результат сделки' if best == 'net' else 'лучше случайного входа'}»"
                 f": за все годы вперёд {fmt(walked[('live', best)], 0, True)}% в год против "
                 f"{fmt(walked[('live', 'edge' if best == 'net' else 'net')], 0, True)}% у другой.")
    lines.append(f"Прогон {time.time() - started:.0f} с.")
    report = OUT_DIR / f"selector-{time.strftime('%Y%m%d-%H%M')}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
