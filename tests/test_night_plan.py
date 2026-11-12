"""The laptop's night: the weekly steps on Sunday, and on the first night after a Sunday the laptop slept through
(08.11.2026: asleep till 02:26; the researcher's bug 3)."""

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("night", ROOT / "scripts" / "night.py")
night = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(night)
WEEKLY = {n for n, _, _ in night.WEEKLY}


def names(steps):
    return {n for n, _, _ in steps}


def test_sunday_runs_the_weekly_steps(tmp_path):
    assert WEEKLY <= names(night.plan(datetime(2026, 11, 8, 0, 0), False, tmp_path))


def test_a_missed_sunday_is_caught_up_the_next_night(tmp_path):
    (tmp_path / night.LAST_WEEKLY).write_text("2026-11-01", encoding="utf-8")
    assert WEEKLY <= names(night.plan(datetime(2026, 11, 9, 0, 0), False, tmp_path))      # Monday, 8 days later
    (tmp_path / night.LAST_WEEKLY).write_text("2026-11-08", encoding="utf-8")
    assert not (WEEKLY & names(night.plan(datetime(2026, 11, 9, 0, 0), False, tmp_path)))  # done yesterday
    assert not (WEEKLY & names(night.plan(datetime(2026, 11, 10, 0, 0), False, tmp_path / "none")))  # never ran


def test_tidy_keeps_the_two_newest_of_a_family_and_never_a_report_or_an_unstamped_file(tmp_path):
    names = [f"events-2026110{d}-0100.pkl" for d in (1, 2, 3, 4)] + ["exam-20261101-0017.pkl",
             "events-20261101-0100.md", "evaluator-exam.pkl", "record.pkl", "champion.pkl", "meta-exam.pkl",
             "panic-20261030-1420.pkl", "panic-20261101-0300.pkl", "panic-20261108-0300.pkl", "events-notes.pkl"]
    for name in names:
        (tmp_path / name).write_bytes(b"x" * 10)
    assert night.tidy(tmp_path, dry_run=True) == (3, 30) and len(list(tmp_path.iterdir())) == len(names)
    assert night.tidy(tmp_path) == (3, 30)
    left = sorted(p.name for p in tmp_path.iterdir())
    assert "events-20261101-0100.pkl" not in left and "events-20261102-0100.pkl" not in left
    assert "panic-20261030-1420.pkl" not in left
    for kept in ("events-20261103-0100.pkl", "events-20261104-0100.pkl", "exam-20261101-0017.pkl", "events-notes.pkl",
                 "events-20261101-0100.md", "evaluator-exam.pkl", "record.pkl", "champion.pkl", "meta-exam.pkl"):
        assert kept in left
    assert night.tidy(tmp_path) == (0, 0)
