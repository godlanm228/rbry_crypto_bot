"""The event library's exam (core/brain/events.py): which real-time events pay after fees and a reaction?

    python scripts/brain_events.py [--workers 3] [--symbols BTCUSDT,ETHUSDT]
    python scripts/brain_events.py --rows runtime/learning/brain/events-<time>.pkl   # the report again

Stage 1 of the watcher (Vlad, 30.10.2026): history only, nothing is sent. Every
(event, horizon) is written into the journal before its first result
(EV-<event>-<hours>h), and the bar rises with every pair ever tried. 2026 was
opened for the brain on 30.10.2026; these events are new and were written down
before their first test, so 2026 checks them, but the 2026 market was seen
before they were written.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import events as ev  # noqa: E402
from core.brain import onchain  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.dataset import DAY_MS  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402
from core.sandbox import archive  # noqa: E402
from core.sandbox.market_store import MarketStore  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
CHUNK = 12
STAKE = 20                                   # $ a signal, of Vlad's $100
SIDE = {1: "купить", -1: "продать (шорт)"}
_CTX: dict = {}


def idea_id(event: str, hours: int) -> str:
    return f"EV-{event}-{hours}h"


def context(store: MarketStore) -> dict:
    spot = {b for _, b, _ in store.symbols()}
    with store.connect() as conn:
        perpetuals = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM funding")]
    perp_of: dict[str, str] = {}
    for perp, base in sorted(archive.futures_pairs(perpetuals, spot).items(), key=lambda kv: kv[0] != kv[1] + "USDT"):
        perp_of.setdefault(base, perp)
    listings: dict[str, list[int]] = {}
    for title, ts in store.news(kind="listing")[["title", "published_ts"]].itertuples(index=False):
        for base in re.findall(r"\(([A-Z0-9]{2,15})\)", title):
            listings.setdefault(base, []).append(int(ts))
    btc = next(h for h in store.histories(min_days=60, symbols=["BTCUSDT"]) if h.name == "BTC")
    return {"db": str(store.db_path), "perp_of": perp_of, "futures": set(store.futures_tracked()),
            "five": set(store.pairs_with("5m")), "listings": listings, "btc": ev.btc_context(ev.grid(btc)),
            "btc_grid": ev.grid(btc)[["ts", "quote_volume"]], "external": onchain.all_events(store, sim.Market())}


def _init(ctx: dict) -> None:
    _CTX.update(ctx)


def listed(h, listings: dict) -> int | None:
    """The announcement of a new listing up to a week before the history's first candle, if there was one."""
    if h.name != h.base:
        return None
    first = int(h.hourly["ts"].iloc[0])
    hits = [t for t in listings.get(h.base, []) if first - ev.LISTING_NEWS_DAYS * DAY_MS <= t <= first + DAY_MS]
    return min(hits) if hits else None


def chunk_rows(chunk: tuple[str, list[str]]) -> pd.DataFrame:
    """One chunk's events: ("usdt", symbols) or ("btc", BTC-quoted pairs in dollars, plan item 5)."""
    kind, symbols = chunk
    store = MarketStore(Path(_CTX["db"]))
    out = []
    found = store.histories(min_days=7, symbols=symbols) if kind == "usdt" else store.btc_histories(7, symbols)
    for h in found:
        perp = _CTX["perp_of"].get(h.base)
        funding = store.funding(perp) if perp else None
        futures = store.futures_hourly(perp) if perp and perp in _CTX["futures"] else None
        five = None
        if h.symbol in _CTX["five"]:
            with store.connect() as conn:
                five = pd.read_sql_query("SELECT ts, open FROM klines WHERE symbol = ? AND tf = '5m' AND ts >= ?",
                                         conn, params=(h.symbol, ev.START_MS))
        rows = ev.coin_rows(h, funding, futures, five, _CTX["btc"],
                            listed(h, _CTX["listings"]) if kind == "usdt" else None,
                            _CTX["external"].get(h.base) if kind == "usdt" else None)
        if len(rows):
            out.append(rows)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def liquid_hours(btc_grid: pd.DataFrame) -> pd.Series:
    """BTC's share of a day's volume by Berlin hour, 2025-2026: when the market is deepest."""
    g = btc_grid[btc_grid["ts"] >= ev.LEARN_END_MS].dropna()
    hours = pd.to_datetime(g["ts"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin").dt.hour
    share = g["quote_volume"].groupby(hours.to_numpy()).mean()
    return (100 * share / share.sum()).sort_values(ascending=False)


def cells(r, p: str, full: bool) -> str:
    base = f"{fmt(r[f'{p}|up_of10'], 1)} · {fmt(r[f'{p}|usd_per_100'], 2, True)}"
    if full:
        return f"{int(r[f'{p}|n']):,} · {base} · t {fmt(r[f'{p}|t'], 1)}"
    return f"{int(r[f'{p}|n']):,} · {base}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--rows", default="", help="a saved run's events: only the report, no journal results")
    args = parser.parse_args()
    started = time.time()
    store = MarketStore()
    ctx = context(store)
    if args.rows:
        return report(pd.read_pickle(args.rows), ctx, Path(args.rows).stem.removeprefix("events-"), started,
                      record=False)
    symbols = sorted(set(store.pairs_with("1h")) & {s for s, _, _ in store.symbols()})
    if args.symbols:
        symbols = [s for s in symbols if s in args.symbols.split(",")]
    chunks = [("usdt", symbols[i:i + CHUNK]) for i in range(0, len(symbols), CHUNK)]
    if not args.symbols:
        btc_pairs = store.btc_pairs()
        chunks += [("btc", btc_pairs[i:i + CHUNK]) for i in range(0, len(btc_pairs), CHUNK)]
    print(f"{len(symbols)} pairs, {len(chunks)} chunks: {time.time() - started:.0f} s", flush=True)
    parts = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(ctx,)) as pool:
        for k, rows in enumerate(pool.map(chunk_rows, chunks), 1):
            parts.append(rows)
            if k % 10 == 0 or k == len(chunks):
                print(f"  {k}/{len(chunks)} chunks, {sum(len(p) for p in parts):,} events: "
                      f"{time.time() - started:.0f} s", flush=True)
    rows = pd.concat([p for p in parts if len(p)], ignore_index=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    pd.to_pickle(rows, BRAIN_DIR / f"events-{stamp}.pkl")
    return report(rows, ctx, stamp, started, record=True)


def report(rows: pd.DataFrame, ctx: dict, stamp: str, started: float, record: bool) -> int:
    journal = Journal()
    for e in ev.EVENTS:
        for hz in ev.HORIZONS:
            journal.register(idea_id(e.id, hz), f"событие: {e.title}; держать {hz} ч",
                             "после события цена идёт не так, как в случайный час той же монеты в том же месяце; "
                             "направление (купить или продать) — по знаку перевеса в 2018–2024",
                             f"{e.id}: signal at an hour's close, entry 10 minutes after the next hour's open, "
                             f"exit after {hz} hours; long on spot, short on a live perpetual at 1x")
    mine = {idea_id(e.id, hz) for e in ev.EVENTS for hz in ev.HORIZONS}
    tried = {e["id"] for e in journal.entries() if e["event"] == "result" and e["id"].startswith("EV-")} | mine
    table, conditional, trades = ev.study(rows, len(tried))
    for d in table.to_dict("records") if record else ():
        for p in ev.PERIODS:
            numbers = {k.split("|")[1]: v for k, v in d.items() if k.startswith(f"{p}|")}
            journal.result(idea_id(d["event"], d["hours"]), p, {**numbers, "side": d["side"], "verdict": d["verdict"],
                                                                "react_pct": d["react_pct"], "run": stamp})
    for d in conditional.to_dict("records") if record else ():
        for p in ev.PERIODS:
            numbers = {k.split("|")[1]: v for k, v in d.items() if k.startswith(f"{p}|")}
            journal.result(idea_id(d["event"], d["hours"]), f"{p}, только: {d['condition']}",
                           {**numbers, "side": d["side"], "verdict": d["verdict"], "run": stamp})

    newest = int(rows["signal_ts"].max())
    years = (newest - ev.START_MS) / (365.25 * DAY_MS)
    recent_months = (newest - ev.LEARN_END_MS) / (30.44 * DAY_MS)
    coins = rows["history"].nunique()
    lines = [f"# Библиотека событий (этап 1 дозорного) — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Проверено {len(rows):,} событий {len(ev.EVENTS)} видов на {coins} монетах, часовые свечи "
             f"с 01.01.2018 по {pd.Timestamp(newest, unit='ms'):%d.%m.%Y}. Сделка: вход через 10 минут после "
             f"сигнала (цена через 10 минут — по 5-минутным свечам, где они есть), выход через "
             f"{', '.join(map(str, ev.HORIZONS))} часа. Комиссии Binance, проскальзывание по обороту монеты, у шорта — "
             f"фьючерсная комиссия и фандинг, маржа 1x. Монеты только с оборотом от $5M в день (кроме листингов). "
             f"«Перевес» — насколько результат лучше входа в ту же монету в случайный час того же месяца. "
             f"Купить или продать — решают данные 2018–2024, 2025 и 2026 должны подтвердить. Проверок в семье "
             f"событий: {len(tried)}, поэтому планка t ≥ {ev.bar(len(tried)):.2f}, для события внутри одного условия "
             f"t ≥ {ev.bar(len(tried) * ev.COND_VALUES):.2f}. Ячейки: случаев · в плюс из 10 · $ на $100 за сделку "
             f"(· t). Открытый интерес и доли счетов в архиве только у 100 фьючерсов, крупнейших на день загрузки "
             f"(до 28.10.2026), — у событий OI-* и CROWD-* монеты-выжившие, их цифры могут быть лучше правды. "
             f"Прогон {time.time() - started:.0f} с.", ""]
    good = table[table["verdict"] == "КАНДИДАТ"]
    cgood = conditional[conditional["verdict"] == "КАНДИДАТ"] if len(conditional) else conditional
    watch = table[table["verdict"].str.startswith("наблюдать")]
    lines += ["## Коротко", ""]
    if len(good) or len(cgood):
        for r in good.itertuples(index=False):
            lines.append(f"- **{ev.BY_ID[r.event].title}** → {SIDE[r.side]}, держать {r.hours} ч")
        for r in cgood.itertuples(index=False):
            lines.append(f"- **{ev.BY_ID[r.event].title}** → {SIDE[r.side]}, держать {r.hours} ч, только когда "
                         f"{r.condition}")
    else:
        lines.append("- Ни одно событие не прошло планку и проверку 2025–2026 в плюсе.")
    lines += [f"- Кандидатов: {len(good)}, только при одном условии: {len(cgood)}, наблюдать: {len(watch)} "
              f"(и {int((conditional['verdict'] != 'КАНДИДАТ').sum()) if len(conditional) else 0} при условии).",
              "- Остальные: " + ", ".join(f"{v} — {n}" for v, n in
                                          table.loc[~table["verdict"].isin(["КАНДИДАТ"]) &
                                                    ~table["verdict"].str.startswith("наблюдать"),
                                                    "verdict"].value_counts().items()) + ".", ""]
    lines += ["## Все события", "", "| событие | держать | что делать | 2018–2024 | 2025 | 2026 | итог |",
              "|---|---|---|---|---|---|---|"]
    for d in table.to_dict("records"):
        lines.append(f"| {ev.BY_ID[d['event']].title} | {d['hours']} ч | {SIDE[d['side']]} | "
                     f"{cells(d, ev.PERIODS[0], True)} | {cells(d, ev.PERIODS[1], False)} | "
                     f"{cells(d, ev.PERIODS[2], False)} | {d['verdict']} |")
    if len(conditional):
        lines += ["", "## Работает только при условии", "",
                  "| событие | держать | что делать | условие | 2018–2024 | 2025 | 2026 | итог |",
                  "|---|---|---|---|---|---|---|---|"]
        for d in conditional.to_dict("records"):
            lines.append(f"| {ev.BY_ID[d['event']].title} | {d['hours']} ч | {SIDE[d['side']]} | {d['condition']} | "
                         f"{cells(d, ev.PERIODS[0], True)} | {cells(d, ev.PERIODS[1], False)} | "
                         f"{cells(d, ev.PERIODS[2], False)} | {d['verdict']} |")
    window = {"КАНДИДАТ": ([], []), "наблюдать": ([], [])}
    if trades:
        lines += ["", "## Кандидаты и «наблюдать» подробно (условные «наблюдать» — только в таблице выше)", ""]
    for (eid, hz, cond), t in trades.items():
        row = (table[(table["event"] == eid) & (table["hours"] == hz)].iloc[0])
        verdict = row["verdict"] if not cond else conditional[(conditional["event"] == eid) & (conditional["hours"] == hz)
                                                             & (conditional["condition"] == cond)].iloc[0]["verdict"]
        if cond and verdict != "КАНДИДАТ":
            continue
        label, vals = ev.urgency(t)
        kind = "КАНДИДАТ" if verdict == "КАНДИДАТ" else "наблюдать"
        wait = re.search(r"до (\d+) ч", label)
        window[kind][0].append(t)
        window[kind][1].append(int(wait.group(1)) if wait else 0)
        recent = t[t["period"] != ev.PERIODS[0]]
        react = row["react_pct"]
        lines += [f"### {ev.BY_ID[eid].title} → {SIDE[int(row['side'])]}, держать {hz} ч"
                  + (f" (только когда {cond})" if cond else ""), "",
                  f"- Как часто: ~{len(recent) / max(recent_months, 1):.1f} сигнала в месяц в 2025–2026.",
                  "- В плюс из 10 и $ на $100: " + "; ".join(
                      f"{p}: {fmt(10 * float((t.loc[t['period'] == p, 'net0'] > 0).mean()), 1)} · "
                      f"{fmt(100 * float(t.loc[t['period'] == p, 'net0'].mean()), 2, True)}" for p in ev.PERIODS) + ".",
                  f"- По пути: в худших 10% сделок цена уходила против тебя на "
                  f"{fmt(-100 * float(np.nanquantile(t['worst'], 0.1)), 1)}% и больше.",
                  f"- Реакция 10 минут: {fmt(react if row['side'] > 0 else -react, 2, True)}% к результату"
                  if np.isfinite(react) else "- Реакция 10 минут: не измерена (мало 5-минутных свечей).",
                  f"- Срочность: {label}; $ на $100: сразу {fmt(vals[0], 2, True)}, через 1 ч {fmt(vals[1], 2, True)}, "
                  f"3 ч {fmt(vals[3], 2, True)}, 6 ч {fmt(vals[6], 2, True)}.", "",
                  "| условие | случаев | в плюс из 10 | $ на $100 |", "|---|---|---|---|"]
        for c, values in ev.CONDITIONS.items():
            for value, name in values.items():
                sub = t[t[c] == value]
                if len(sub):
                    lines.append(f"| {name} | {len(sub):,} | {fmt(10 * float((sub['net0'] > 0).mean()), 1)} | "
                                 f"{fmt(100 * float(sub['net0'].mean()), 2, True)} |")
        lines.append("")
    lines += ["## Окно для сделок (время Берлина)", ""]
    for kind, title in (("КАНДИДАТ", "кандидаты"), ("наблюдать", "«наблюдать», если подтвердятся")):
        picked, waits = window[kind]
        if not picked:
            lines.append(f"- {title}: нет — окно считать не из чего.")
            continue
        win = ev.best_window(picked, waits, years)
        always = 100 * sum(float(t["net0"].sum()) for t in picked) / years
        lines.append(f"- {title}: какую долю пользы «всегда у телефона» собирает окно в 3 часа "
                     f"(сигналы до окна — с опозданием, если ждать можно):")
        for r in win.head(3).itertuples(index=False):
            share = 100 * r.usd_per_100_trade_sum_per_year / always if always > 0 else float("nan")
            lines.append(f"  - {r.start:02d}:00–{(r.start + 3) % 24:02d}:00 — {fmt(share, 0)}%")
    deep = liquid_hours(ctx["btc_grid"]).head(5)
    lines += ["- Где рынок глубже всего (доля дневного объёма BTC, 2025–2026): "
              + ", ".join(f"{int(h):02d}:00 — {v:.1f}%" for h, v in deep.items()) + ".", ""]
    report = BRAIN_DIR / f"events-{stamp}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
