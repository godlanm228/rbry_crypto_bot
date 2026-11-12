"""The morning check: did the night's work on the laptop run at all?

    python scripts/night_check.py [--dry-run]

Three jobs run on the laptop, none of them a service of the bot:
  RBRY-Night      the Windows task at 00:00 (scripts/night.py), writes runtime/learning/night/last.json
  researcher      the Claude app's task, every night about 01:15, appends to runtime/learning/lab/nights.jsonl
                  (a night skipped for the plan's limit too)
  analyst         the Claude app's task, Sundays 08:05, writes runtime/learning/brain/analyst-summary.txt
None of them says anything when it never starts: the laptop was off or asleep, the Claude app was closed,
or a hung run of a Claude task kept the next one from starting (: a run waited for a permission
for a whole day and the next night did not start). The Windows task RBRY-NightCheck
(deploy/night-check-task.ps1) runs this every day at 09:30, later if the laptop slept: each job that wrote
nothing since its day's midnight is one message to the owner.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RBRY_ENV_FILE", str(ROOT / ".env.test"))

from core.console import utf8_output  # noqa: E402

from core.notify import notify_owner  # noqa: E402

LEARNING = ROOT / "runtime" / "learning"
# (name for the owner, the file every run writes, the weekday it runs or None for every day, a Claude task)
JOBS = (
    ("ночной расчёт (RBRY-Night, 00:00)", LEARNING / "night" / "last.json", None, False),
    ("ночной исследователь Claude (01:15)", LEARNING / "lab" / "nights.jsonl", None, True),
    ("недельный аналитик Claude (вс 08:05)", LEARNING / "brain" / "analyst-summary.txt", 6, True),
)
MORNING = 9      # before 09:00 the night may still be running: the day checked is yesterday


def checked_day(now: datetime) -> datetime:
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return day if now.hour >= MORNING else day - timedelta(days=1)


def missing(now: datetime, jobs=JOBS) -> list[tuple[str, bool]]:
    """The jobs due on the checked day whose file was not written since that day's midnight."""
    day = checked_day(now)
    return [(name, claude) for name, path, weekday, claude in jobs
            if (weekday is None or day.weekday() == weekday)
            and not (path.exists() and path.stat().st_mtime >= day.timestamp())]


def claude_open() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq claude.exe", "/NH"], capture_output=True, text=True)
    return "claude.exe" in out.stdout.lower()


def message(day: datetime, gone: list[tuple[str, bool]], app_open: bool) -> str:
    lines = [f"⚠️ Ночь {day:%d.%m}: не отработало", *(f"• {name}" for name, _ in gone)]
    if any(not claude for _, claude in gone):
        lines.append("Ночной расчёт не запускался: ноут, видимо, был выключен.")
    if any(claude for _, claude in gone):
        lines.append("Задачи Claude идут, только когда ноут включён и приложение Claude открыто."
                     if not app_open else
                     "Приложение Claude открыто: скорее всего, прошлый запуск завис и держит очередь.")
    lines.append("Напиши Claude — он проверит и перезапустит.")
    return "\n".join(lines)


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    now = datetime.now()
    gone = missing(now)
    if not gone:
        print(f"night {checked_day(now):%d.%m}: everything ran")
        return 0
    text = message(checked_day(now), gone, claude_open())
    if args.dry_run:
        print(text)
        return 0
    return 0 if notify_owner(text) else 1


if __name__ == "__main__":
    raise SystemExit(main())
