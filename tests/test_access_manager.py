import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

from core.access.access_manager import AccessManager


def test_grant_access_extend_preserves_remaining_days(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import config

    monkeypatch.setattr(config, "OWNER_ID", 123456)

    access = AccessManager()
    access.grant_access(123456, 777, "premium", 30)
    first_expiry = access.users["users"]["777"]["expires_date"]

    access.grant_access(123456, 777, "premium", 30, extend=True)
    extended_expiry = access.users["users"]["777"]["expires_date"]

    assert 29 * 86400 < extended_expiry - first_expiry < 31 * 86400
