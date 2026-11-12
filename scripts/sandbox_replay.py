"""Learning sandbox, step L1: replay every rule over history and report.

    python scripts/sandbox_replay.py --update            # fetch/refresh data, then replay
    python scripts/sandbox_replay.py                     # replay on stored data
    python scripts/sandbox_replay.py --open-holdout plan_good,oversold   # the one look at 2026

Writes runtime/learning/learning.db and runtime/learning/report-<run>.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.console import utf8_output  # noqa: E402

import pandas as pd  # noqa: E402

from core.sandbox import data as sandbox_data  # noqa: E402
from core.sandbox.evaluate import (  # noqa: E402
    DEV_END, VALID_END, add_context, cluster_table, holdout_report, memory_filter, similar_situations_test,
    strategy_table)
from core.sandbox.features import FEATURE_VERSION, FEATURES  # noqa: E402
from core.sandbox.replay import all_situations, replay  # noqa: E402
from core.sandbox.store import LearningStore  # noqa: E402
from core.sandbox.strategies import BY_NAME, STRATEGY_VERSION  # noqa: E402

logger = logging.getLogger("sandbox")


def _fmt(x, digits=2):
    if isinstance(x, float):
        return "—" if not math.isfinite(x) else f"{x:.{digits}f}"
    return str(x)


def _count(x) -> int:
    return int(x) if x is not None and pd.notna(x) else 0


def write_report(path: Path, run_id: str, meta: dict, table: pd.DataFrame, knn: pd.DataFrame,
                 profile: pd.DataFrame, cells: pd.DataFrame, holdout_opens: int) -> None:
    lines = [f"# Песочница L1 — прогон {run_id}", "",
             f"Монет: {meta['coins']} · дней истории: {meta['first_day']} … {meta['last_day']} · "
             f"ситуаций: {meta['situations']:,} · сделок: {meta['trades']:,} (без наложений: {meta['clean_trades']:,})",
             f"Признаки {FEATURE_VERSION}, правила {STRATEGY_VERSION}. Обучение ≤ {DEV_END}, проверка ≤ {VALID_END}, "
             f"2026 закрыт (открывали: {holdout_opens} раз).", "",
             "## Правила: перевес над случайными днями", "",
             "Двойники — та же сделка (стоп, цель, срок в ATR) в случайные дни той же монеты и месяца; "
             "перевес = результат минус двойники, то есть только выбор дня.", "",
             "| правило | сделок dev/valid | в цель dev/valid | средний R dev/valid | двойники R dev/valid | "
             "перевес dev (t) | перевес valid (t) | вывод |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(
            f"| {r.strategy} | {r.dev_n}/{r.valid_n} | {_fmt(r.dev_target_rate)}/{_fmt(r.valid_target_rate)} | "
            f"{_fmt(r.dev_avg_r, 3)} / {_fmt(r.valid_avg_r, 3)} | {_fmt(r.dev_avg_twin_r, 3)} / {_fmt(r.valid_avg_twin_r, 3)} | "
            f"{_fmt(r.dev_avg_excess_r, 3)} ({_fmt(r.dev_t_excess, 1)}) | "
            f"{_fmt(r.valid_avg_excess_r, 3)} ({_fmt(r.valid_t_excess, 1)}) | {r.verdict} |")
    lines += ["", "По годам (средний R, без 2026): " + "; ".join(
        f"{r.strategy}: {r.by_year}" for _, r in table.iterrows()), ""]
    lines += ["## Похожие ситуации: предсказывает ли память 2022–2024 результат 2025", "",
              "Цель r_net — сам результат (с рынком), excess_r — перевес над двойниками (только момент входа).", "",
              "| правило | цель | память | k | сделок 2025 | ранговая корр. (t) | все | лучшие 20% (t) | худшие 20% | по квинтилям |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in knn.iterrows():
        lines.append(f"| {r.strategy} | {r.target} | {r.memory} | {r.k} | {r.n_valid} | {_fmt(r.spearman, 3)} ({_fmt(r.t_rho, 1)}) | "
                     f"{_fmt(r.avg_all, 3)} | {_fmt(r.avg_top20, 3)} ({_fmt(r.t_top20, 1)}) | "
                     f"{_fmt(r.avg_bottom20, 3)} | {r.quintiles} |")
    lines += ["", "## Типы ситуаций (кластеры по 2022–2024)", "",
              "Центры в стандартизованных единицах (0 = обычный день):", "",
              "| тип | доля | " + " | ".join(FEATURES) + " |", "|---" * (len(FEATURES) + 2) + "|"]
    for i, r in profile.iterrows():
        lines.append(f"| {i} | {_fmt(r.share_dev)} | " + " | ".join(_fmt(r[f], 1) for f in FEATURES) + " |")
    lines += ["", "Средний R по правилу и типу (dev → valid, n):", ""]
    for _, r in cells.iterrows():
        lines.append(f"- {r.strategy} · тип {int(r.cluster)}: {_fmt(r.get('dev_mean', float('nan')), 3)} "
                     f"(n={_count(r.get('dev_count'))}) → {_fmt(r.get('valid_mean', float('nan')), 3)} "
                     f"(n={_count(r.get('valid_count'))})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="refresh universe, candles and Fear & Greed")
    parser.add_argument("--coins", type=int, default=50)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--open-holdout", default="", help="comma-separated rules for the one look at 2026")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    store = LearningStore()
    if args.update or not store.universe():
        universe = sandbox_data.pick_universe(args.coins)
        store.set_universe(universe)
        logger.info("universe: %s", " ".join(u["symbol"] for u in universe))
        counts = sandbox_data.update_candles(store, universe)
        logger.info("candles fetched: %d coins, %d rows", len(counts), sum(counts.values()))
        logger.info("fear & greed rows: %d", sandbox_data.update_fear_greed(store))

    universe = store.universe()
    candles = {u["symbol"]: store.candles(u["symbol"]) for u in universe}
    candles = {s: f for s, f in candles.items() if not f.empty}
    btc, fng = candles.get("BTC"), store.fear_greed()

    started = time.time()
    situations = all_situations(candles, btc, fng)
    logger.info("situations: %d rows in %.0fs", len(situations), time.time() - started)
    trades = replay(candles, situations)
    logger.info("trades: %d in %.0fs", len(trades), time.time() - started)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M")
    t = add_context(trades)
    store.save_run(run_id, FEATURE_VERSION, STRATEGY_VERSION, {"coins": len(candles)})
    store.save_trades(run_id, t)
    store.save_situations(run_id, situations, list(FEATURES))

    if args.open_holdout:
        wanted = args.open_holdout.split(",")
        names = [n for n in wanted if n in BY_NAME]
        extra = [memory_filter(t, situations, n.split("+")[0]) for n in wanted
                 if n.endswith("+memory") and n.split("+")[0] in BY_NAME]
        result = holdout_report(t, names, store, run_id, extra)
        path = store.db_path.parent / f"holdout-{run_id}.json"
        path.write_text(result.to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
        cols = ["strategy", "n", "coins", "of10_target", "of10_stop", "of10_time", "r_per_10", "avg_r",
                "t", "avg_twin_r", "avg_excess_r", "t_excess", "holdout_opened_total"]
        print(result[[c for c in cols if c in result]].round(3).to_string())
        return 0

    table = strategy_table(t)
    knn = similar_situations_test(t, situations)
    profile, cells = cluster_table(t, situations)
    meta = {"coins": len(candles), "first_day": situations["day_str"].min(), "last_day": situations["day_str"].max(),
            "situations": len(situations), "trades": len(t), "clean_trades": int((t["overlap"] == 0).sum())}
    report = store.db_path.parent / f"report-{run_id}.md"
    write_report(report, run_id, meta, table, knn, profile, cells, store.holdout_opens())
    store.save_summary(run_id, {"meta": meta, "strategies": json.loads(table.to_json(orient="records")),
                                "similar": json.loads(knn.to_json(orient="records"))})
    print(report)
    print(table[["strategy", "dev_n", "valid_n", "dev_avg_r", "valid_avg_r", "dev_avg_excess_r",
                 "valid_avg_excess_r", "valid_t_excess", "verdict"]].to_string())
    print(knn.to_string() if not knn.empty else "no kNN rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
