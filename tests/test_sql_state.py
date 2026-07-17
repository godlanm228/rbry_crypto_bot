import json
import os
import sqlite3

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

from core.access.access_manager import AccessManager, AccessTier
from core.analytics.user_analytics import UserAnalytics


def test_access_manager_persists_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import config

    monkeypatch.setattr(config, "OWNER_ID", 123456)

    first = AccessManager()
    first.grant_access(123456, 777, "premium", 30)

    second = AccessManager()

    assert second.get_user_tier(777) == AccessTier.PREMIUM
    assert (tmp_path / "data" / "bot_state.db").exists()


def test_access_manager_migrates_legacy_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import config

    monkeypatch.setattr(config, "OWNER_ID", 123456)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "access_control.json").write_text(
        json.dumps(
            {
                "users": {
                    "777": {
                        "tier": "beta",
                        "granted_date": 1000.0,
                        "expires_date": None,
                        "granted_by": 123456,
                    }
                },
                "access_log": [
                    {
                        "timestamp": 1000.0,
                        "action": "grant",
                        "admin_id": 123456,
                        "target_user_id": 777,
                        "tier": "beta",
                        "days": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    access = AccessManager()

    assert access.get_user_tier(777) == AccessTier.BETA
    assert access.users["access_log"][0]["action"] == "grant"


def test_user_prefs_persist_to_sqlite(tmp_path, monkeypatch):
    import modules.user_prefs as prefs

    monkeypatch.setattr(prefs, "PREFS_FILE", tmp_path / "user_prefs.json")
    monkeypatch.setattr(prefs, "PREFS_DB_FILE", tmp_path / "bot_state.db")
    monkeypatch.setattr(prefs, "_cache", None)
    monkeypatch.setattr(prefs, "_cache_mtime", None)
    monkeypatch.setattr(prefs, "_db", None)
    monkeypatch.setattr(prefs, "_db_path", None)

    prefs.set_language(777, "de")
    prefs.set_timezone(777, "UTC")
    prefs.set_show_explanations(777, False)

    monkeypatch.setattr(prefs, "_cache", None)
    monkeypatch.setattr(prefs, "_db", None)
    monkeypatch.setattr(prefs, "_db_path", None)

    assert prefs.get_language(777) == "de"
    assert prefs.get_timezone(777) == "UTC"
    assert prefs.get_show_explanations(777) is False


def test_user_analytics_persists_to_sqlite(tmp_path):
    db_path = tmp_path / "bot_state.db"

    first = UserAnalytics(db_path=db_path)
    first.register_user(777, username="alice", first_name="Alice", language_code="en")
    first.log_activity(777, "check_news", {"coin": "BTC"})

    second = UserAnalytics(db_path=db_path)
    stats = second.get_usage_statistics()
    activity = second.get_user_activity(777)

    assert stats["total_users"] == 1
    assert stats["total_requests"] == 1
    assert stats["action_counts"]["check_news"] == 1
    assert activity[0]["details"] == {"coin": "BTC"}


def test_access_managers_do_not_overwrite_other_writers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import config

    monkeypatch.setattr(config, "OWNER_ID", 123456)
    first = AccessManager()
    first.grant_access(123456, 701, "premium", 30)

    stale = AccessManager()
    first.grant_access(123456, 702, "beta", 30)
    stale.grant_access(123456, 703, "premium", 30)

    reloaded = AccessManager()
    try:
        assert reloaded.get_user_tier(701) == AccessTier.PREMIUM
        assert reloaded.get_user_tier(702) == AccessTier.BETA
        assert reloaded.get_user_tier(703) == AccessTier.PREMIUM
    finally:
        first.close()
        stale.close()
        reloaded.close()


def test_access_manager_rejects_negative_duration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import config

    monkeypatch.setattr(config, "OWNER_ID", 123456)
    access = AccessManager()
    try:
        result = access.grant_access(123456, 777, "premium", -1)
        assert "non-negative integer" in result
        assert access.get_user_tier(777) == AccessTier.FREE
    finally:
        access.close()


def test_user_prefs_update_does_not_delete_profile_created_by_another_writer(
    tmp_path, monkeypatch
):
    import modules.user_prefs as prefs

    db_path = tmp_path / "bot_state.db"
    monkeypatch.setattr(prefs, "PREFS_FILE", tmp_path / "user_prefs.json")
    monkeypatch.setattr(prefs, "PREFS_DB_FILE", db_path)
    monkeypatch.setattr(prefs, "_cache", None)
    monkeypatch.setattr(prefs, "_cache_mtime", None)
    monkeypatch.setattr(prefs, "_db", None)
    monkeypatch.setattr(prefs, "_db_path", None)

    prefs.set_language(777, "de")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_prefs (
                user_id, tz, lang, show_explanations, ai_debug, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (888, "UTC", "uk", 1, 0, 1.0),
        )

    prefs.set_timezone(777, "Europe/Berlin")

    assert prefs.get_language(888) == "uk"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT lang FROM user_prefs WHERE user_id = 888"
        ).fetchone()[0] == "uk"


def test_user_analytics_log_activity_registers_missing_user(tmp_path):
    analytics = UserAnalytics(db_path=tmp_path / "bot_state.db")
    try:
        analytics.log_activity(999, "orphan_safe")
        stats = analytics.get_usage_statistics()

        assert stats["total_users"] == 1
        assert stats["total_requests"] == 1
        assert stats["action_counts"] == {"orphan_safe": 1}
    finally:
        analytics.close()
