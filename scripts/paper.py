"""The paper journal of C1 (core/brain/paper.py): decide every night, book on live prices, report on Sundays.

    python scripts/paper.py day [--notify] [--dry-run]   # the newest day's decision, then the book
    python scripts/paper.py report [--weekly] [--notify]  # the report now (Sunday's: --weekly)

Runs on the test VM after the dataset of the night (deploy/rbry-paper.service);
the Sunday report has its own timer, at a waking hour (deploy/rbry-paper-report.*).
--notify writes to the owner only (TELEGRAM_OWNER_ID of .env.test): the first
decision, Sunday's report, and anything that stops the journal. --dry-run
decides and books in memory and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RBRY_ENV_FILE", str(ROOT / ".env.test"))

from core.brain import dataset  # noqa: E402
from core.brain import paper  # noqa: E402
from core.brain import watch  # noqa: E402
from core.brain.model import Brain  # noqa: E402
from core.notify import notify_owner  # noqa: E402
from core.sandbox.market_store import DEFAULT_DB_PATH, MarketStore  # noqa: E402

logger = logging.getLogger("paper")
BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"
DECISIONS = paper.PAPER_DIR / "decisions.jsonl"
STATE = paper.PAPER_DIR / "state.json"
COLUMNS = ["history", "base", "day", "turnover_30d", "perp_listed", "funding_1d", "close", "sma50", "atr", "atr_pct",
           *dataset.RANKING_FEATURES]


def _send(text: str, notify: bool) -> None:
    print(text)
    if notify and not notify_owner(text):
        logger.warning("paper: could not reach the owner on Telegram")


def _book(store: MarketStore, decisions: list[dict]) -> paper.Book:
    b = paper.book(decisions, paper.LiveMarket(store), paper.live_until(store))
    tmp = STATE.with_suffix(".json.tmp")                # whole or not at all: the test bot may read it any moment
    tmp.write_text(json.dumps(paper.state(b, decisions), ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    tmp.replace(STATE)
    return b


def day(args) -> None:
    store = MarketStore(args.db)
    decisions = paper.load(DECISIONS)
    newest = paper.last_closed_day(paper.live_until(store))
    rows = dataset.load_table(BRAIN_DIR / f"dataset-{dataset.DATASET_VERSION}", list(dict.fromkeys(COLUMNS)))
    rows = rows[rows["day"] == rows["day"].max()]
    built = rows["day"].max()
    if built < newest:
        if not args.dry_run:
            _send(f"📒 Бумажный журнал: признаки за {newest:%d.%m} не собраны (последние за {built:%d.%m}) — "
                  "решения на этот день нет.", args.notify)
    elif not any(d["day"] == str(built.date()) for d in decisions):
        seen = watch.tail(watch.WATCH_DIR / "hours.jsonl", 1)       # the market as the watcher saw it last
        decision = paper.decide(rows, Brain.load(BRAIN_DIR / "champion.pkl"), int(time.time() * 1000),
                                seen[-1] if seen else None)
        if args.dry_run:
            print(paper.start_text(decision))
            trial = decisions + [decision]
            print(paper.report_text(paper.book(trial, paper.LiveMarket(store), paper.live_until(store)), trial,
                                    title="проба"))
            return
        paper.append(DECISIONS, decision)
        decisions.append(decision)
        logger.info("paper: decision for %s: %s", decision["day"],
                    ", ".join(f"{'+' if o['side'] > 0 else '-'}{o['base']}" for o in decision["orders"]))
        if len(decisions) == 1:
            _send(paper.start_text(decision), args.notify)
    if decisions and not args.dry_run:
        _book(store, decisions)


def report(args) -> None:
    store = MarketStore(args.db)
    decisions = paper.load(DECISIONS)
    if not decisions:
        print("no decisions yet")
        return
    b = _book(store, decisions)
    _send(paper.report_text(b, decisions, title="неделя" if args.weekly else "сейчас"), args.notify)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    d, r = sub.add_parser("day"), sub.add_parser("report")
    for p in (d, r):
        p.add_argument("--notify", action="store_true")
    d.add_argument("--dry-run", action="store_true")
    r.add_argument("--weekly", action="store_true")
    args = parser.parse_args()
    try:
        {"day": day, "report": report}[args.command](args)
    except Exception as exc:                     # the owner hears when the journal stops
        if getattr(args, "notify", False):
            notify_owner(f"📒 Бумажный журнал: сбой ({type(exc).__name__}: {str(exc)[:200]}). Смотрю.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
