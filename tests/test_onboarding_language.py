"""First-contact language comes from Telegram; callback values are validated."""

import pytest

import modules.user_prefs as prefs
from core.market_signal import classify_price_momentum


@pytest.fixture
def isolated_prefs(tmp_path, monkeypatch):
    monkeypatch.setattr(prefs, "PREFS_FILE", tmp_path / "user_prefs.json")
    monkeypatch.setattr(prefs, "PREFS_DB_FILE", tmp_path / "bot_state.db")
    monkeypatch.setattr(prefs, "_cache", None)
    monkeypatch.setattr(prefs, "_cache_mtime", None)
    monkeypatch.setattr(prefs, "_db", None)
    monkeypatch.setattr(prefs, "_db_path", None)
    return prefs


@pytest.mark.parametrize("code, expected", [
    ("en", "en"), ("en-GB", "en"), ("de", "de"), ("ru", "ru"), ("uk", "uk"),
    ("fr", "en"), (None, "en"),
])
def test_language_from_telegram(code, expected):
    assert prefs.language_from_telegram(code) == expected


def test_first_contact_uses_client_language_and_never_overrides(isolated_prefs):
    isolated_prefs.init_profile_language(1, "de")
    assert isolated_prefs.get_language(1) == "de"
    isolated_prefs.set_language(1, "uk")
    isolated_prefs.init_profile_language(1, "en")  # later /start
    assert isolated_prefs.get_language(1) == "uk"


def test_forged_callback_values_are_ignored(isolated_prefs):
    isolated_prefs.init_profile_language(2, "en")
    isolated_prefs.set_language(2, "zz<script>")
    isolated_prefs.set_timezone(2, "../../etc/passwd")
    assert isolated_prefs.get_language(2) == "en"
    assert isolated_prefs.get_timezone(2) is None


@pytest.mark.parametrize("day, week, expected", [
    (2.5, -20.0, "analysis.signal_hold"),   # bounce inside a falling week
    (-3.0, 6.0, "analysis.signal_hold"),    # dip inside a rising week
    (3.0, 6.0, "analysis.signal_buy"),
    (-3.0, -6.0, "analysis.signal_sell"),
    (6.0, 11.0, "analysis.signal_strong_buy"),
    (float("nan"), 1.0, "analysis.signal_unavailable"),
])
def test_momentum_label_is_symmetric(day, week, expected):
    assert classify_price_momentum(day, week).translation_key == expected
