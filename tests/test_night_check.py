"""The morning check of the laptop's night: which jobs wrote nothing, and what the owner is told."""

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "night_check.py"
_spec = importlib.util.spec_from_file_location("night_check", _PATH)
nc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = nc
_spec.loader.exec_module(nc)


def _touch(path: Path, when: datetime) -> Path:
    path.write_text("x", encoding="utf-8")
    os.utime(path, (when.timestamp(), when.timestamp()))
    return path


def _jobs(tmp_path):
    return (("night", tmp_path / "last.json", None, False),
            ("researcher", tmp_path / "nights.jsonl", None, True),
            ("analyst", tmp_path / "analyst.txt", 6, True))


def test_before_nine_the_day_checked_is_yesterday():
    assert nc.checked_day(datetime(2026, 11, 2, 9, 30)) == datetime(2026, 11, 2)
    assert nc.checked_day(datetime(2026, 11, 3, 0, 40)) == datetime(2026, 11, 2)


def test_a_file_written_before_the_days_midnight_counts_as_missing(tmp_path):
    jobs = _jobs(tmp_path)
    _touch(jobs[0][1], datetime(2026, 11, 2, 0, 21))
    _touch(jobs[1][1], datetime(2026, 11, 1, 3, 24))          # last night's, not tonight's
    # Monday: the analyst is not due; a file that does not exist at all is missing too
    assert nc.missing(datetime(2026, 11, 2, 9, 30), jobs) == [("researcher", True)]
    _touch(jobs[1][1], datetime(2026, 11, 2, 2, 40))
    assert nc.missing(datetime(2026, 11, 2, 9, 30), jobs) == []


def test_the_analyst_is_checked_on_sundays_only(tmp_path):
    jobs = _jobs(tmp_path)
    sunday = datetime(2026, 11, 8, 9, 30)
    _touch(jobs[0][1], datetime(2026, 11, 8, 0, 30))
    _touch(jobs[1][1], datetime(2026, 11, 8, 2, 0))
    assert nc.missing(sunday, jobs) == [("analyst", True)]
    # the check ran late, after Sunday's midnight: still Sunday's analyst that is looked for
    assert nc.missing(datetime(2026, 11, 9, 0, 30), jobs) == [("analyst", True)]


def test_the_message_names_the_jobs_and_the_likely_cause():
    day = datetime(2026, 11, 2)
    closed = nc.message(day, [("researcher", True)], app_open=False)
    assert closed.startswith("⚠️ Ночь 02.11") and "• researcher" in closed and "приложение Claude открыто" in closed
    assert "завис" not in closed
    assert "завис" in nc.message(day, [("researcher", True)], app_open=True)
    laptop = nc.message(day, [("night", False)], app_open=True)
    assert "выключен" in laptop and "Claude открыто" not in laptop
