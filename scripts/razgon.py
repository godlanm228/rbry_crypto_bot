"""«Разгон v2» (core/brain/razgon.py): examine the steps not decided yet, keep the ladder of versions.

    python scripts/razgon.py exam [--notify]    # v1 if the ladder is empty, then every new step; fresh numbers
    python scripts/razgon.py show               # the ladder in words

Runs on the laptop (the event library and the hourly cache are here); the night copies the ladder
(runtime/learning/razgon/razgon-versions.json) to the test VM, where the owner's site shows it. --notify tells the
owner, without sound, when a step was decided. A step that passes becomes the next version of the exam only:
what trades on paper changes with the owner's word, not by itself.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.brain import factory as fa  # noqa: E402
from core.brain import razgon as rz  # noqa: E402
from core.brain import simulator as sim  # noqa: E402
from core.brain.journal import Journal  # noqa: E402
from core.console import utf8_output  # noqa: E402

BRAIN_DIR = ROOT / "runtime" / "learning" / "brain"


def line(name: str, e: dict) -> str:
    return (f"{name}: {e['learn_yearly_pct']:+.0f}% в год за 2018–2024 (просадка до {e['learn_worst_drawdown_pct']:.0f}%), "
            f"2025 ${100 + e['2025']['growth_pct']:.0f}, 2026 ${100 + e['2026']['growth_pct']:.0f}; сделок {e['trades']}, "
            f"в плюс {e['up_of10']} из 10, ликвидаций {e['liquidations']}")


def text(ladder: dict) -> str:
    out = ["🚀 Разгон v2 — лестница версий (экзамен на истории, не бумага)"]
    names = {v["name"]: v for v in ladder["versions"]}
    out.append("Текущая " + line(ladder["current"], ladder.get("now", names[ladder["current"]])["exam"]
                                 if "now" in ladder else names[ladder["current"]]["exam"]))
    out.append(names[ladder["current"]]["exam"]["words"])
    for s in ladder["steps"]:
        mark = f"✅ принят → {s['became']}" if s["accepted"] else "✗ не принят"
        out.append(f"\n{mark}: «{s['title']}» поверх {s['on']} ({s['decided']})")
        out.append("  " + line("шаг", s["exam"]))
        out += [f"  · {w}" for w in s["reasons"]]
    return "\n".join(out)


def exam(args) -> int:
    started = time.time()
    source = sorted(BRAIN_DIR.glob("events-*.pkl"))[-1]
    market = sim.Market()
    newest = int(market.btc()[-1, 0])
    table = rz.table(pd.read_pickle(source), market)
    ladder = rz.load()
    before = len(ladder["steps"])
    ladder = rz.climb(ladder, table, market, newest, journal=Journal(), source=source.stem)
    # the current version on today's data: the scoreboard's first line moves with the market, a step's numbers
    # stay as they were when it was decided
    ladder["now"] = {"name": ladder["current"], "exam": rz.exam(rz.current_version(ladder), table, market, newest),
                     "at": time.strftime("%Y-%m-%d %H:%M"), "events": source.stem}
    rz.save(ladder)
    print(text(ladder))
    print(f"\n{source.name}, newest hour {pd.Timestamp(newest, unit='ms')}, {time.time() - started:.0f} s; "
          f"years {fa.YEARS[0]}–{fa.YEARS[-1]}", flush=True)
    fresh = ladder["steps"][before:]
    if args.notify and fresh:
        from core.notify import notify_owner

        lines = ["🚀 Разгон v2: экзамен шагов"]
        for s in fresh:
            lines.append(("✅ " if s["accepted"] else "✗ ") + f"«{s['title']}» поверх {s['on']}: "
                         + ("принят, теперь " + s["became"] if s["accepted"] else "не принят"))
            lines += [f"  · {w}" for w in s["reasons"][:4]]
        lines.append("Табло — на сайте, раздел «Развитие». На бумагу версия идёт только после твоего «да».")
        notify_owner("\n".join(lines), silent=True)
    return 0


def show(args) -> int:
    ladder = rz.load()
    if not ladder["versions"]:
        print("the ladder is empty: run `python scripts/razgon.py exam`")
        return 1
    print(text(ladder))
    return 0


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exam")
    e.add_argument("--notify", action="store_true")
    sub.add_parser("show")
    args = parser.parse_args()
    return {"exam": exam, "show": show}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
