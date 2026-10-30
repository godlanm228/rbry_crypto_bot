"""Stage 2 of the watcher (core/brain/exits.py): exits and picks for the panic buy, chosen on 2021-2024.

    python scripts/brain_panic.py [--events runtime/learning/brain/events-<time>.pkl] [--workers 3]

Reads the newest event library run, walks every panic trade under every exit
rule on hourly prices, and books each pick x exit as a $100 portfolio ($20 a
trade, at most 5 open). The pick and the exit are chosen on 2021-2024 alone:
the fastest growth whose drawdown stayed within 50%. Every pick x exit goes into
the journal before its first result (EV-PANIC-<pick>-<exit>), and the event
family's bar rises with them.
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

from core.brain import events as ev  # noqa: E402
from core.brain import exits as ex  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.dataset import DAY_MS  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
_TRADES: dict = {}


def idea_id(pick: str, rule: str) -> str:
    return f"EV-PANIC-{pick}-{rule}"


def _init(trades: pd.DataFrame, newest: int) -> None:
    _TRADES.update(trades=trades, newest=newest)


def run_one(rule_id: str) -> tuple[str, pd.DataFrame]:
    return rule_id, ex.run_rule(_TRADES["trades"], sim.Market(), ex.BY_RULE[rule_id], _TRADES["newest"])


def books(res: pd.DataFrame, rule: ex.Rule, newest: int) -> dict:
    """Per period: the trades' numbers and a $100 portfolio started at the period's beginning."""
    t = ex.tag(res, rule.hours)
    spans = {**ex.SPANS, ev.PERIODS[2]: (ev.CHECK_END_MS, newest + ev.HOUR_MS)}
    out = {}
    for p in ev.PERIODS:
        book = ex.portfolio(res, *spans[p])
        out[p] = {**ex.describe(t[t["period"] == p]), **book, "yearly_pct": ex.yearly(book["growth_pct"], book["years"])}
    return out


def verdict(parts: dict, bar_t: float) -> str:
    """The trades must beat random entries and pay in every period, and the $100 portfolio must grow in each
    with a drawdown within 50%: a mean per trade that a portfolio cannot collect is not money."""
    learn, check, late = (parts[p] for p in ev.PERIODS)
    if min(learn["n"], check["n"], late["n"]) < ev.MIN_CHECK:
        return "мало сделок"
    trades_ok = all(np.isfinite(p["excess_pct"]) and p["excess_pct"] > 0 and p["usd_per_100"] > 0
                    for p in (check, late))
    book_ok = all(p["growth_pct"] > 0 and p["drawdown_pct"] >= -100 * ex.MAX_DRAWDOWN for p in (learn, check, late))
    passed = np.isfinite(learn["t"]) and learn["t"] >= bar_t and learn["usd_per_100"] > 0
    if passed and trades_ok and book_ok:
        return "КАНДИДАТ"
    if passed and trades_ok:
        return "сделки в плюсе, но портфель $100 — нет (или просадка больше 50%)"
    if trades_ok and book_ok:
        return "наблюдать: держится в 2025–2026, но в 2021–2024 ниже планки"
    return "не держится"


def cell(p: dict) -> str:
    return (f"{p['n']:,} · {fmt(p['up_of10'], 1)} · {fmt(p['usd_per_100'], 2, True)} · "
            f"{fmt(p['moment_usd_per_100'], 2, True)} · {fmt(p['lost_p10_pct'], 1)}% · "
            f"{fmt(p['yearly_pct'], 0, True)}%/год · {fmt(p['drawdown_pct'], 0)}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    started = time.time()
    source = Path(args.events) if args.events else sorted(BRAIN_DIR.glob("events-*.pkl"))[-1]
    rows = pd.read_pickle(source)
    market = sim.Market()
    newest = int(market.btc()[-1, 0])
    trades = ex.panic_trades(rows, market)
    print(f"{len(trades):,} panic trades from {source.name}: {time.time() - started:.0f} s", flush=True)
    results = {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(trades, newest)) as pool:
        for rid, res in pool.map(run_one, [r.id for r in ex.RULES]):
            results[rid] = res.merge(trades[["history", "signal_ts", "fell24", "turnover", "moment_size"]],
                                     on=["history", "signal_ts"])
            print(f"  {rid}: {len(res):,} trades, {time.time() - started:.0f} s", flush=True)

    journal = Journal()
    for pick, pick_title in ex.SELECTIONS.items():
        for r in ex.RULES:
            journal.register(idea_id(pick, r.id), f"паника: {pick_title}; выход: {r.title}",
                             "после вынужденных продаж в нервном рынке отскок; правило выхода режет хвост убытков",
                             f"PANIC ({', '.join(ex.PANIC_EVENTS)}, nervous market only), long on spot, pick {pick}, "
                             f"exit {r.id}; $100 portfolio, {ex.SHARE:.0%} a trade, at most {ex.MAX_OPEN} open")
    mine = {idea_id(p, r.id) for p in ex.SELECTIONS for r in ex.RULES}
    tried = {e["id"] for e in journal.entries() if e["event"] == "result" and e["id"].startswith("EV-")} | mine
    bar_t = ev.bar(len(tried))
    table = []
    for pick in ex.SELECTIONS:
        for r in ex.RULES:
            res = results[r.id]
            parts = books(res[ex.picked(res, pick)], r, newest)
            v = verdict(parts, bar_t)
            table.append({"pick": pick, "rule": r.id, "verdict": v, "parts": parts})
            for p in ev.PERIODS:
                journal.result(idea_id(pick, r.id), p, {**parts[p], "verdict": v, "events": source.stem})
    learn = ev.PERIODS[0]
    allowed = [x for x in table if x["parts"][learn]["drawdown_pct"] >= -100 * ex.MAX_DRAWDOWN]
    chosen = max(allowed or table, key=lambda x: x["parts"][learn]["yearly_pct"])
    none_held = not allowed
    base = next(x for x in table if x["pick"] == "all" and x["rule"] == "H24")
    since = ex.portfolio(results[chosen["rule"]][ex.picked(results[chosen["rule"]], chosen["pick"])],
                         ev.LEARN_END_MS, newest + ev.HOUR_MS)

    rule = ex.BY_RULE[chosen["rule"]]
    lines = [f"# Этап 2 дозорного: выходы для «паники» — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"«Паника» = одно из событий {', '.join(ex.PANIC_EVENTS)} в нервном рынке, покупка на споте; "
             f"{len(trades):,} сигналов (монета-час) с 2021 года, источник {source.name}. Её собрали, уже видя "
             f"2025 и 2026, поэтому сама «паника» на истории больше не проверяется честно — это сделает бумажный "
             f"журнал. Здесь история выбирает только выход и какие монеты брать из одного момента, и выбирает по "
             f"2021–2024: самый быстрый рост портфеля при просадке не больше 50%. 2025 и 2026 показывают, держится ли "
             f"выбор. Сделка: вход через 10 минут после сигнала, комиссии и проскальзывание Binance, стоп в "
             f"«обычных ходах» монеты за день. Портфель: $100, на сделку {ex.SHARE:.0%} счёта, не больше "
             f"{ex.MAX_OPEN} сразу, просадка — по закрытым сделкам. «Лучше случайного» — то же правило выхода из "
             f"случайных часов той же монеты в том же месяце. Проверок в семье событий: {len(tried)}, планка "
             f"t ≥ {bar_t:.2f}. Прогон {time.time() - started:.0f} с.", "",
             f"## Выбор по 2021–2024: {ex.SELECTIONS[chosen['pick']]}, {rule.title}", "",
             ("Ни один вариант не удержал просадку портфеля в 50% даже в 2021–2024; выбран самый быстрый рост."
              if none_held else "Из вариантов с просадкой портфеля в 2021–2024 не больше 50% — самый быстрый рост."), "",
             "| период | сделок | моментов | в плюс из 10 | $ на $100 | $ на $100 по моментам | худшие 10% сделок | "
             "по пути худшие 10% | лучше случайного | $100 → | в год | просадка |", "|" + "---|" * 12]
    for p in ev.PERIODS:
        x = chosen["parts"][p]
        lines.append(f"| {p} | {x['n']:,} | {x['moments']:,} | {fmt(x['up_of10'], 1)} | "
                     f"{fmt(x['usd_per_100'], 2, True)} | {fmt(x['moment_usd_per_100'], 2, True)} | "
                     f"{fmt(x['lost_p10_pct'], 1)}% | {fmt(x['worst_p10_pct'], 1)}% | "
                     f"{fmt(x['excess_pct'], 2, True)}% (t {fmt(x['t'], 1)}) | ${x['end_usd']:.0f} | "
                     f"{fmt(x['yearly_pct'], 0, True)}% | {fmt(x['drawdown_pct'], 0)}% |")
    lines += ["", f"- Итог: **{chosen['verdict']}**.",
              f"- С $100 1 января 2025 по этим правилам: сейчас ${since['end_usd']:.0f}, худшая просадка "
              f"{fmt(since['drawdown_pct'], 0)}%, сделок {since['trades']}.",
              f"- Для сравнения «все монеты, держать сутки» (как в этапе 1): худшие 10% сделок "
              + ", ".join(f"{p} {fmt(base['parts'][p]['lost_p10_pct'], 1)}%" for p in ev.PERIODS)
              + "; по пути " + ", ".join(f"{fmt(base['parts'][p]['worst_p10_pct'], 1)}%" for p in ev.PERIODS) + ".",
              "", "## Все варианты", "",
              "Ячейка: сделок · в плюс из 10 · $ на $100 · $ на $100 по моментам · худшие 10% сделок · портфель в "
              "год · просадка.", "",
              "| монеты | выход | 2021–2024 | 2025 | 2026 | итог |", "|---|---|---|---|---|---|"]
    for x in table:
        lines.append(f"| {ex.SELECTIONS[x['pick']]} | {ex.BY_RULE[x['rule']].title} | "
                     + " | ".join(cell(x["parts"][p]) for p in ev.PERIODS) + f" | {x['verdict']} |")
    passed = [x for x in table if x["verdict"] == "КАНДИДАТ"]
    for x in passed:
        r = ex.BY_RULE[x["rule"]]
        mine = results[r.id][ex.picked(results[r.id], x["pick"])]
        recent = mine[mine["signal_ts"] >= ev.LEARN_END_MS]
        months = (newest - ev.LEARN_END_MS) / (30.44 * DAY_MS)
        hours = pd.to_datetime(recent["signal_ts"].drop_duplicates(), unit="ms", utc=True).dt.tz_convert(
            "Europe/Berlin").dt.hour
        lines += ["", f"## Кандидат: {ex.SELECTIONS[x['pick']]}, {r.title}", "",
                  f"- Моментов в 2025–2026: ~{recent['signal_ts'].nunique() / months:.1f} в месяц; ночью (00–08) "
                  f"{100 * float((hours < 8).mean()):.0f}% из них.",
                  f"- Худшая сделка за всё время: {fmt(100 * float(mine['net'].min()), 1)}%; закрытия: "
                  + ", ".join(f"{k} {100 * v:.0f}%" for k, v in mine["reason"].value_counts(normalize=True).items())
                  + ".", "", "| вход | 2021–2024 | 2025 | 2026 |", "|---|---|---|---|"]
        sub = trades[ex.picked(trades, x["pick"])]
        for d in (0, 1, 3, 6):
            res = mine if d == 0 else ex.run_rule(sub, market, r, newest, delay=d, twins=False)
            t = ex.tag(res, r.hours + d)
            lines.append(f"| {'через 10 минут' if d == 0 else f'через {d} ч'} | "
                         + " | ".join(f"{fmt(10 * float((t.loc[t['period'] == p, 'net'] > 0).mean()), 1)} из 10 · "
                                      f"{fmt(100 * float(t.loc[t['period'] == p, 'net'].mean()), 2, True)} $"
                                      for p in ev.PERIODS) + " |")
    reasons = results[chosen["rule"]][ex.picked(results[chosen["rule"]], chosen["pick"])]["reason"]
    lines += ["", "## Как закрывались сделки выбранного варианта", "",
              ", ".join(f"{k}: {100 * v:.0f}%" for k, v in reasons.value_counts(normalize=True).items()) + ".", ""]
    stamp = time.strftime("%Y%m%d-%H%M")
    report = BRAIN_DIR / f"panic-{stamp}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.to_pickle({"trades": trades, "results": results, "chosen": (chosen["pick"], chosen["rule"])},
                 BRAIN_DIR / f"panic-{stamp}.pkl")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
