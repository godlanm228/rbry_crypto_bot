"""The laptop's night: fresh data, then the heavy learning (00:00-07:00 Berlin).

    python scripts/night.py [--until 06:50] [--weekly] [--dry-run]

Windows Task Scheduler wakes the laptop at midnight (task RBRY-Night, see
deploy/night-task.ps1). Every night:
  data      market.db's update (candles, funding, macro)
  news      the day's headlines and Binance announcements (the VM keeps its own archive; until
            02.11.2026 the laptop collected none, and the night researcher and the hack review
            found no news after 28.10)
  cex       Binance's reserves by token (the on-chain flows' history)
  sentiment who is buying by the hour: Coinbase, Upbit, Deribit's volatility index, the won
  dataset   the brain's full dataset, when it lacks the newest closed day
  cache     the simulator's hourly prices
  factory   the strategy factory breeds and books 500 new genomes, rebooks stale ones
  robust    everything on paper bent four ways; the fragile come off the paper list
  razgon    «Разгон v2»: the current version on today's data, and the exam of every step not decided yet
  promote   its candidates and the ladder of versions to the test VM (on Sundays also the selector's live model)
The night to Sunday (or --weekly, or any night once the last weekly run is a
week old) adds the weekly learning: token unlocks, the brain's exam through the
newest data (2026 is open since 30.10.2026), the trade evaluator, the
robustness of the paper journal's candidates, the event library, the panic
exits, the selector with its gate, the listing radar, and the selector's live
model to the VM. Each step is its own
process, so memory comes back between them; a step that would not finish
before the morning deadline is skipped. Everything goes to
runtime/learning/night/; the owner hears only when a step fails.
"""

from __future__ import annotations

import argparse
from typing import Optional
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RBRY_ENV_FILE", str(ROOT / ".env.test"))

from core.console import utf8_output  # noqa: E402

from dotenv import dotenv_values  # noqa: E402

from core.notify import notify_owner  # noqa: E402

NIGHT_DIR = ROOT / "runtime" / "learning" / "night"
PY = sys.executable
CANDIDATES = "C1,C4,C5,B1,N6"
# the test VM's paper watcher reads the factory's candidates from here (copied over the existing ssh access);
# host and key live in the env file, not in the repo
_ENV = {**dotenv_values(os.environ["RBRY_ENV_FILE"]), **os.environ}
VM = _ENV.get("RBRY_TEST_VM", "")
VM_KEY = _ENV.get("RBRY_TEST_VM_KEY", "")
PROMOTED = "runtime/learning/factory/promoted.json"
LADDER = "runtime/learning/razgon/razgon-versions.json"      # «Разгон v2»: the owner's site shows it (web/pages.razgon)
# (name, command, minutes it may take): a step starts only when that much time is left
NIGHTLY = (
    ("data", [PY, "scripts/market_data.py", "update"], 90),
    ("fine", [PY, "scripts/market_data.py", "fine"], 30),          # every pair's 5-minute candles, a folder of their own
    ("news", [PY, "scripts/market_data.py", "news"], 5),
    ("cex", [PY, "scripts/market_data.py", "cex"], 5),
    ("sentiment", [PY, "scripts/market_data.py", "sentiment"], 10),
    ("dataset", [PY, "scripts/brain.py", "dataset", "--if-stale"], 45),
    ("cache", [PY, "scripts/brain.py", "cache"], 30),
    ("factory", [PY, "scripts/brain_factory.py", "--batch", "500"], 20),
    ("robust", [PY, "scripts/brain_robust.py"], 10),
    ("razgon", [PY, "scripts/razgon.py", "exam", "--notify"], 5),
    ("promote", ["scp", "-i", VM_KEY, "-o", "BatchMode=yes", PROMOTED, "runtime/learning/brain/radar-model.json",
                 LADDER, f"{VM}:rbry_crypto_bot/runtime/learning/watch/"], 5),
)
WEEKLY = (
    ("unlocks", [PY, "scripts/market_data.py", "unlocks"], 15),
    ("exam", [PY, "scripts/brain.py", "exam", "--open-holdout"], 20),
    ("evaluator", [PY, "scripts/brain_evaluator.py"], 20),
    ("robustness", [PY, "scripts/brain_robustness.py", "--only", CANDIDATES, "--variants", "100"], 45),
    ("events", [PY, "scripts/brain_events.py", "--workers", "3"], 30),
    ("panic", [PY, "scripts/brain_panic.py", "--workers", "3"], 15),
    ("selector", [PY, "scripts/brain_selector.py", "--workers", "3"], 20),
    ("radar", [PY, "scripts/brain_radar.py"], 5),
    # the week's live model to the watcher: copied next to it, then moved in, so a run never reads half a file
    ("push-selector", ["scp", "-i", VM_KEY, "-o", "BatchMode=yes", "runtime/learning/selector/live/model.txt",
                       "runtime/learning/selector/live/meta.json", f"{VM}:rbry_crypto_bot/runtime/learning/watch/"], 5),
    # a version that differs from the live one keeps the live one as selector-prev (the watcher's rollback)
    ("move-selector", ["ssh", "-i", VM_KEY, "-o", "BatchMode=yes", VM,
                       "cd rbry_crypto_bot/runtime/learning/watch && mkdir -p selector && "
                       "if [ -f selector/model.txt ] && ! cmp -s model.txt selector/model.txt; then "
                       "rm -rf selector-prev && cp -r selector selector-prev; fi && "
                       "mv model.txt meta.json selector/"], 2),
)
SUNDAY = 6


def keep_awake(on: bool) -> None:
    """No sleep while the night runs (Windows); the display may go dark."""
    if os.name == "nt":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def deadline(until: str, now: datetime) -> datetime:
    hh, mm = (int(x) for x in until.split(":"))
    end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return end if end > now else end + timedelta(days=1)


LAST_WEEKLY = "last-weekly.txt"
# heavy files a run writes under a new time stamp each time; every reader takes the newest (sorted(...)[-1]), the
# one before stays for a comparison, older ones only filled the disk (11.11.2026: ten old event libraries of 50 MB)
TIDY_FAMILIES = ("exam-", "events-", "panic-", "assessor-", "pairs-")
TIDY_KEEP = 2
_STAMPED = re.compile(r"\d{8}-\d{4}\.pkl$")


def tidy(brain_dir: Path, keep: int = TIDY_KEEP, dry_run: bool = False) -> tuple[int, int]:
    """Drop all but the newest ``keep`` time-stamped .pkl of each family in the brain's folder; the reports
    (.md), the journals and the files without a stamp are never touched. Returns (files, bytes) dropped."""
    files = size = 0
    for family in TIDY_FAMILIES:
        stamped = sorted(p for p in brain_dir.glob(f"{family}*.pkl") if _STAMPED.fullmatch(p.name[len(family):]))
        for old in stamped[:-keep] if keep else stamped:
            files += 1
            size += old.stat().st_size
            if not dry_run:
                old.unlink()
    return files, size


def weekly_due(now: datetime, night_dir: Path) -> bool:
    """Sunday's night, or any night when the last weekly run is a week old or more: a Sunday the laptop slept
    through (08.11.2026: asleep till 02:26) no longer costs the selector its week (the researcher's bug 3)."""
    if now.weekday() == SUNDAY:
        return True
    try:
        last = datetime.strptime((night_dir / LAST_WEEKLY).read_text(encoding="utf-8").strip(), "%Y-%m-%d")
    except (OSError, ValueError):
        return False                                   # never ran: wait for Sunday, as before
    return (now.date() - last.date()).days >= 7


def plan(now: datetime, weekly: bool, night_dir: Optional[Path] = None) -> tuple:
    due = weekly or (weekly_due(now, night_dir) if night_dir is not None else now.weekday() == SUNDAY)
    return NIGHTLY + (WEEKLY if due else ())


def run_step(name: str, cmd: list, log) -> tuple[bool, float]:
    started = time.time()
    log.write(f"\n== {name} {datetime.now():%Y-%m-%d %H:%M:%S}: {' '.join(cmd[1:])}\n")
    log.flush()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MALLOC_ARENA_MAX": "2"}
    code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    log.write(f"== {name}: exit {code}, {time.time() - started:.0f} s\n")
    log.flush()
    return code == 0, time.time() - started


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", default="06:50")
    parser.add_argument("--weekly", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    now = datetime.now()
    end = deadline(args.until, now)
    steps = plan(now, args.weekly, NIGHT_DIR)
    if args.dry_run:
        print(f"until {end:%Y-%m-%d %H:%M}:", ", ".join(n for n, _, _ in steps))
        dropped, freed = tidy(ROOT / "runtime" / "learning" / "brain", dry_run=True)
        print(f"tidy would drop {dropped} old files, {freed / 1e6:.0f} MB")
        return 0
    NIGHT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"started": f"{now:%Y-%m-%d %H:%M}", "until": f"{end:%H:%M}", "steps": {}}
    failed = []
    keep_awake(True)
    try:
        with (NIGHT_DIR / f"night-{now:%Y%m%d}.log").open("a", encoding="utf-8") as log:
            for name, cmd, minutes in steps:
                if datetime.now() + timedelta(minutes=minutes) > end:
                    summary["steps"][name] = "skipped: not enough night left"
                    log.write(f"\n== {name}: skipped, not enough night left\n")
                    continue
                ok, seconds = run_step(name, cmd, log)
                summary["steps"][name] = f"{'ok' if ok else 'FAILED'} in {seconds / 60:.0f} min"
                if not ok:
                    failed.append(name)
    finally:
        keep_awake(False)
    try:
        dropped, freed = tidy(ROOT / "runtime" / "learning" / "brain")
        summary["tidy"] = f"{dropped} old files, {freed / 1e6:.0f} MB"
    except OSError as exc:                             # a file in use: the next night takes it
        summary["tidy"] = f"skipped: {exc}"
    summary["finished"] = f"{datetime.now():%Y-%m-%d %H:%M}"
    (NIGHT_DIR / "last.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    weekly_names = {name for name, _, _ in WEEKLY}
    if any(name in weekly_names and not str(v).startswith("skipped") for name, v in summary["steps"].items()):
        (NIGHT_DIR / LAST_WEEKLY).write_text(f"{now:%Y-%m-%d}", encoding="utf-8")
    if failed:
        notify_owner(f"🌙 Ночь на ноуте: не прошли шаги {', '.join(failed)}. Лог: runtime/learning/night/"
                     f"night-{now:%Y%m%d}.log")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
