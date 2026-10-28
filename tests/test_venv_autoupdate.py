"""The weekly library updater: version rules, checks and the owner's report."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "deploy" / "autoupdate" / "venv_autoupdate.py"
_spec = importlib.util.spec_from_file_location("venv_autoupdate", _PATH)
au = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = au      # dataclasses look their module up
_spec.loader.exec_module(au)


def test_a_new_major_is_never_allowed():
    assert au.upper_bound("pandas", "2.2.3") == "3"
    assert au.upper_bound("python-telegram-bot", "22.8") == "23"
    # 0.x: the minor is the major (httpx 0.28 -> 0.29 may break)
    assert au.upper_bound("httpx", "0.28.1") == "0.29"
    # versioned by date, or tooling: nothing to guard
    assert au.upper_bound("certifi", "2026.10.23") is None
    assert au.upper_bound("pip", "26.2.2") is None
    assert au.upper_bound("weird", "dev") is None


def test_constraints_keep_versions_inside_the_major_and_never_go_back():
    pkgs = {"pandas": ("pandas", "2.2.3"), "certifi": ("certifi", "2026.10.23"), "pip": ("pip", "26.0")}
    assert au.constraints(pkgs) == ["certifi>=2026.10.23", "pandas>=2.2.3,<3"]


def test_major_jump_detection():
    assert au.is_major_jump("pandas", "2.3.3", "3.0.1")
    assert not au.is_major_jump("pandas", "2.2.3", "2.3.3")
    assert au.is_major_jump("httpx", "0.28.1", "0.29.0")
    assert not au.is_major_jump("certifi", "2025.1.1", "2026.10.23")


def test_diff_lists_changes_in_plain_words_without_pip():
    old = {"pandas": ("pandas", "2.2.3"), "pip": ("pip", "25.0"), "gone": ("gone", "1.0")}
    new = {"pandas": ("pandas", "2.3.3"), "pip": ("pip", "26.0"), "fresh": ("fresh", "0.1")}
    assert au.diff(old, new) == ["pandas 2.2.3 → 2.3.3", "+ fresh 0.1 (новая зависимость)",
                                 "− gone 1.0 (больше не нужна)"]
    assert au.diff(old, {**old, "pip": ("pip", "26.0")}) == []


def test_junit_result_counts_failures_errors_and_skips(tmp_path):
    xml = tmp_path / "r.xml"
    xml.write_text(
        '<testsuites><testsuite>'
        '<testcase classname="tests.a" name="ok"/>'
        '<testcase classname="tests.a" name="bad"><failure message="x"/></testcase>'
        '<testcase classname="tests.b" name="boom"><error message="y"/></testcase>'
        '<testcase classname="tests.b" name="later"><skipped/></testcase>'
        '</testsuite></testsuites>')
    assert au.junit_result(xml) == (4, 1, {"tests.a::bad", "tests.b::boom"})
    assert au.junit_result(tmp_path / "missing.xml") == (0, 0, {au.NO_RESULTS})


def test_only_new_code_errors_count_after_a_restart():
    before = ["[2026-10-27 10:00:01] ERROR job: TypeError: bad operand 12",
              "telegram.error.NetworkError: Bad Gateway"]
    after = ["[2026-10-28 04:01:02] ERROR job: TypeError: bad operand 7",     # known, other numbers
             "telegram.error.NetworkError: Bad Gateway",                       # network, never counts
             "AttributeError: module 'pandas' has no attribute 'old_api'"]     # new -> counts
    assert au.new_code_errors(before, after) == ["AttributeError: module 'pandas' has no attribute 'old_api'"]


def test_report_is_silent_when_nothing_happened():
    assert au.report("rbry-bot", [au.Outcome("RBRY", "same"), au.Outcome("A1", "skipped")]) is None


def test_report_says_what_changed_and_what_was_checked():
    text = au.report("rbry-bot", [
        au.Outcome("RBRY test bot", "updated", changes=["pandas 2.2.3 → 2.3.3"],
                   checks=["совместимость ✅", "тесты: 404 прошли ✅"]),
        au.Outcome("Redmond hub", "blocked", reason="с новыми версиями падают тесты (1): tests.x::y"),
        au.Outcome("A1 hunter", "rolled_back", reason="служба падала и перезапускалась",
                   majors=[("oci", "2.187.0", "3.0.0")]),
    ])
    assert text.startswith("🧰 Библиотеки · сервер rbry-bot")
    assert "🔄 RBRY test bot: обновил 1\n• pandas 2.2.3 → 2.3.3" in text
    assert "⏸ Redmond hub: не обновлял — с новыми версиями падают тесты" in text
    assert "⏪ A1 hunter: откатил обновление — служба падала и перезапускалась\nВернул прежние версии" in text
    assert "• oci 3.0.0 (сейчас 2.187.0) — A1 hunter" in text


def test_read_kv_handles_quotes_and_comments(tmp_path):
    f = tmp_path / "x.conf"
    f.write_text('# c\nENV_FILE=/a/.env\nexport TOKEN="abc"\nCHAT=\'1,2\'\n')
    assert au.read_kv(f) == {"ENV_FILE": "/a/.env", "TOKEN": "abc", "CHAT": "1,2"}


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_swap_and_restore_keep_a_working_way_back(tmp_path):
    venv, store = tmp_path / ".venv", tmp_path / "store"
    venv.mkdir()
    (venv / "marker").write_text("legacy")
    store.mkdir()
    first, second = store / "20261101", store / "20261108"
    first.mkdir()
    second.mkdir()

    prev, legacy = au.swap(venv, first, store, "20261101")
    assert legacy and venv.is_symlink() and (prev / "marker").read_text() == "legacy"
    au.restore(venv, prev, legacy)
    assert not venv.is_symlink() and (venv / "marker").read_text() == "legacy"

    au.swap(venv, first, store, "20261101")
    prev, legacy = au.swap(venv, second, store, "20261108")
    assert not legacy and prev == first and venv.resolve() == second.resolve()
    au.restore(venv, prev, legacy)
    assert venv.resolve() == first.resolve()

    au.prune(store, {first})
    assert sorted(p.name for p in store.iterdir()) == ["20261101"]
