"""The listing radar's exam (core/brain/radar.py): does a filter learned on past listings beat buying them all?

    python scripts/brain_radar.py

Builds every Binance listing since 2021 from market.db (delisted coins included), scores them walked
forward (each quarter by a model of the listings whose 30 days had ended before it), and sets the
picks against all listings by year; the model on every listing so far is saved for the daily radar.
Report in runtime/learning/brain/radar-<time>.md.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import radar as rd  # noqa: E402
from core.brain import regime as rg  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.dataset import DAY_MS  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.brain.walkforward import fmt  # noqa: E402
from core.sandbox import archive  # noqa: E402
from core.sandbox.market_store import MarketStore  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
MODEL = BRAIN_DIR / "radar-model.json"


def table(store: MarketStore, market) -> pd.DataFrame:
    listings: dict[str, list[int]] = {}
    for title, ts in store.news(kind="listing")[["title", "published_ts"]].itertuples(index=False):
        for base in re.findall(r"\(([A-Z0-9]{2,15})\)", title):
            listings.setdefault(base, []).append(int(ts))
    spot = {b for _, b, _ in store.symbols()}
    with store.connect() as conn:
        perps = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM funding")]
    perp_of = {b: p for p, b in archive.futures_pairs(perps, spot).items()}
    states = rg.market_states(market)
    btc = pd.DataFrame(market.btc()[:, [0, 4]], columns=["ts", "close"])
    start = int(pd.Timestamp(rd.START).value // 10**6)
    symbols = [s for s, b, _ in store.symbols() if b in listings]
    rows = []
    for h in store.histories(min_days=2, symbols=symbols):
        if h.name != h.base:
            continue
        first = int(h.hourly["ts"].iloc[0])
        hits = [t for t in listings.get(h.base, []) if first - rd.NEWS_DAYS * DAY_MS <= t <= first + DAY_MS]
        if not hits or first < start:
            continue
        perp = perp_of.get(h.base)
        r = rd.listing_row(h.hourly, h.base, min(hits), store.funding(perp) if perp else None, states, btc, h.ended)
        if r:
            rows.append(r)
    t = pd.DataFrame(rows).sort_values("first_ts").reset_index(drop=True)
    firsts = t["first_ts"].to_numpy()
    t["crowd"] = [int(((firsts >= f - 30 * DAY_MS) & (firsts < f)).sum()) for f in firsts]
    t["year"] = pd.to_datetime(t["first_ts"], unit="ms").dt.year
    return t


def stats(g: pd.DataFrame, hold: int = 30) -> str:
    r = g[f"ret{hold}"].dropna()
    if not len(r):
        return "—"
    return f"{len(r)} · {fmt(10 * float((r > 0).mean()), 1)} из 10 · {fmt(100 * float(r.median()), 0, True)}%"


def main() -> int:
    started = time.time()
    store, market = MarketStore(), sim.Market()
    t = table(store, market)
    pred = rd.walk_forward(t)
    lines = [f"# Радар листингов — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Листингов Binance с 2021 (вместе с удалёнными монетами): {len(t)}. Покупка в начале 25-го часа, "
             f"держать 7/30/90 дней, комиссии и проскальзывание. Фильтр — логистическая регрессия на том, что "
             f"известно к концу первых суток ({', '.join(rd.WORDS[f] for f in rd.FEATURES)}), каждый квартал учится "
             f"только на листингах, чьи 30 дней уже прошли; «взять» — оценка выше верхней трети обучения. Ячейка: "
             f"листингов · в плюсе через 30 дней из 10 · обычный (медианный) итог.", "",
             "| год | все листинги | взял бы радар | не взял |", "|---|---|---|---|"]
    for y, g in pred.groupby("year"):
        lines.append(f"| {y} | {stats(g)} | {stats(g[g['picked']])} | {stats(g[~g['picked']])} |")
    lines.append(f"| все | {stats(pred)} | {stats(pred[pred['picked']])} | {stats(pred[~pred['picked']])} |")
    lines += ["", "Все листинги без фильтра, по годам и срокам (в плюсе из 10 · медиана):", "",
              "| год | 7 дней | 30 дней | 90 дней |", "|---|---|---|---|"]
    for y, g in t.groupby("year"):
        lines.append(f"| {y} | " + " | ".join(stats(g, d) for d in rd.HOLDS) + " |")
    # the model on every listing whose 30 days are over, for the daily radar
    known = t[np.isfinite(t["ret30"])]
    xt, _, scale = rd.standardise(known, known)
    w = rd.fit(xt, (known["ret30"] > 0).to_numpy(dtype=float))
    MODEL.write_text(json.dumps({"weights": w.tolist(), "features": list(rd.FEATURES), "scale": scale,
                                 "bar": float(np.quantile(rd.prob(w, xt), rd.PICK_QUANTILE)),
                                 "base_up": float((known["ret30"] > 0).mean()), "listings": len(known),
                                 "trained_at": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    coef = sorted(zip(rd.FEATURES, w[1:]), key=lambda x: -abs(x[1]))
    lines += ["", "Что фильтр считает важным (на всех листингах; плюс — чаще в плюсе): "
              + ", ".join(f"{rd.WORDS[f]} {c:+.2f}" for f, c in coef[:6]) + ".", "",
              f"Прогон {time.time() - started:.0f} с."]
    journal = Journal()
    journal.register("RADAR", "радар листингов: фильтр первого дня против покупки всех листингов",
                     "по первым суткам и состоянию рынка можно отличить листинги, которые через месяц в плюсе",
                     "ridge logistic of ret30 > 0 on day-one and market features, refit quarterly, pick above the "
                     "training scores' top third")
    for y, g in pred.groupby("year"):
        journal.result("RADAR", str(y), {"all_up": float((g["ret30"] > 0).mean()),
                                         "picked_up": float((g.loc[g["picked"], "ret30"] > 0).mean()) if g["picked"].any()
                                         else None, "picked": int(g["picked"].sum()), "n": len(g)})
    report = BRAIN_DIR / f"radar-{time.strftime('%Y%m%d-%H%M')}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
