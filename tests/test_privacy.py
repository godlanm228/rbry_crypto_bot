"""/delete_me erases a user's data and leaves other users alone."""

import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

import modules.user_prefs as prefs
from core.analytics.user_analytics import get_user_analytics
from core.portfolio_db import get_portfolio_manager
from modules.privacy import delete_user_data


def test_delete_user_data_erases_only_that_user(tmp_path, monkeypatch):
    monkeypatch.setattr(prefs, "PREFS_FILE", tmp_path / "user_prefs.json")
    monkeypatch.setattr(prefs, "PREFS_DB_FILE", tmp_path / "bot_state.db")
    monkeypatch.setattr(prefs, "_cache", None)
    monkeypatch.setattr(prefs, "_cache_mtime", None)
    monkeypatch.setattr(prefs, "_db", None)
    monkeypatch.setattr(prefs, "_db_path", None)

    pm = get_portfolio_manager()
    analytics = get_user_analytics()
    for uid in (9001, 9002):
        pm.add_entry(uid, "BTC", 0.1, 60000)
        analytics.register_user(user_id=uid, username=f"u{uid}", first_name="U", language_code="en")
        analytics.log_activity(uid, "portfolio")
        prefs.init_profile_language(uid, "de")

    delete_user_data(9001)

    assert pm.load_portfolio(9001) == {}
    assert "BTC" in pm.load_portfolio(9002)
    conn = analytics.db.conn()
    assert conn.execute("SELECT COUNT(*) FROM analytics_users WHERE user_id = 9001").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM analytics_activity WHERE user_id = 9001").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM analytics_users WHERE user_id = 9002").fetchone()[0] == 1
    assert prefs._fetch_profile(9001) is None
    assert prefs.get_language(9002) == "de"
