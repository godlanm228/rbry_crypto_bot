"""The laptop's night: fresh data, then the heavy learning (Vlad, 30.10.2026: 00:00-07:00 Berlin).

    python scripts/night.py [--until 06:50] [--weekly] [--dry-run]

Windows Task Scheduler wakes the laptop at midnight (task RBRY-Night, see
deploy/night-task.ps1). Every night:
  data      market.db's update (candles, funding, macro, news)
  dataset   the brain's full dataset, when it lacks the newest closed day
  cache     the simulator's hourly prices
The night to Sunday (or --weekly) adds the weekly learning: the brain's exam
through the newest data (2026 is open since 30.10.2026), the trade evaluator,
and the robustness of the paper journal's candidates. Each step is its own
process, so memory comes back between them; a step that would not finish
before the morning deadline is skipped. Everything goes to
runtime/learning/night/; the owner hears only when a step fails.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RBRY_ENV_FILE", str(ROOT / ".env.test"))

from core.notify import notify_owner  # noqa: E402

NIGHT_DIR = ROOT / "runtime" / "learning" / "night"
PY = sys.executable
CANDIDATES = "C1,C4,C5,B1,N6"
# (name, command, minutes it may take): a step starts only when that much time is left
NIGHTLY = (
    ("data", [PY, "scripts/market_data.py", "update"], 90),
    ("dataset", [PY, "scripts/brain.py", "dataset", "--if-stale"], 45),
    ("cache", [PY, "scripts/brain.py", "cache"], 30),
)
WEEKLY = (
    ("exam", [PY, "scripts/brain.py", "exam", "--open-holdout"], 20),
    ("evaluator", [PY, "scripts/brain_evaluator.py"], 20),
    ("robustness", [PY, "scripts/brain_robustness.py", "--only", CANDIDATES, "--variants", "100"], 45),
    ("events", [PY, "scripts/brain_events.py", "--workers", "3"], 30),
    ("panic", [PY, "scripts/brain_panic.py", "--workers", "3"], 15),
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


def plan(now: datetime, weekly: bool) -> tuple:
    return NIGHTLY + (WEEKLY if weekly or now.weekday() == SUNDAY else ())


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", default="06:50")
    parser.add_argument("--weekly", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    now = datetime.now()
    end = deadline(args.until, now)
    steps = plan(now, args.weekly)
    if args.dry_run:
        print(f"until {end:%Y-%m-%d %H:%M}:", ", ".join(n for n, _, _ in steps))
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
    summary["finished"] = f"{datetime.now():%Y-%m-%d %H:%M}"
    (NIGHT_DIR / "last.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if failed:
        notify_owner(f"🌙 Ночь на ноуте: не прошли шаги {', '.join(failed)}. Лог: runtime/learning/night/"
                     f"night-{now:%Y%m%d}.log")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
