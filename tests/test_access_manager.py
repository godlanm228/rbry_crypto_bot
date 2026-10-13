import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

from core.access.access_manager import AccessManager, AccessTier


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


def test_tier_policy_is_consistent_for_features_coins_and_portfolio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import config

    monkeypatch.setattr(config, "OWNER_ID", 123456)
    access = AccessManager()
    try:
        assert access.get_user_tier(777) == AccessTier.FREE
        assert access.has_feature(777, "portfolio") is True
        assert access.has_feature(777, "top10_analysis") is False
        assert access.check_coin_limit(777, "solana")[0] is True
        assert access.check_coin_limit(777, "ripple")[0] is False
        coin_decision = access.evaluate_coin_limit(777, "ripple")
        assert coin_decision.code == "coin_not_available"
        assert coin_decision.params["allowed"] == ("bitcoin", "ethereum", "solana")
        assert access.check_portfolio_limit(
            777,
            ["bitcoin", "ethereum", "solana"],
            "bitcoin",
        )[0] is True
        assert access.check_portfolio_limit(
            777,
            ["bitcoin", "ethereum", "solana"],
            "ripple",
        )[0] is False

        access.grant_access(123456, 777, "premium", 30)
        assert access.has_feature(777, "portfolio") is True
        assert access.has_feature(777, "news") is True
        assert access.has_feature(777, "top10_analysis") is True
        assert access.check_coin_limit(777, "ripple")[0] is True

        access.record_request(777)
        cooldown_decision = access.evaluate_access(777, "basic_analysis")
        assert cooldown_decision.allowed is False
        assert cooldown_decision.code == "cooldown"
    finally:
        access.close()


def test_paid_grant_idempotency_does_not_extend_twice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import config

    monkeypatch.setattr(config, "OWNER_ID", 123456)
    access = AccessManager()
    try:
        access.grant_access(
            123456,
            777,
            "premium",
            30,
            extend=True,
            idempotency_key="telegram-stars:charge-1",
        )
        first_expiry = access.users["users"]["777"]["expires_date"]

        access.grant_access(
            123456,
            777,
            "premium",
            30,
            extend=True,
            idempotency_key="telegram-stars:charge-1",
        )

        assert access.users["users"]["777"]["expires_date"] == first_expiry
        log_count = access.db.conn().execute(
            "SELECT COUNT(*) FROM access_log WHERE target_user_id = ?",
            (777,),
        ).fetchone()[0]
        assert log_count == 1
    finally:
        access.close()


def test_distinct_paid_grants_extend_from_durable_expiry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import config

    monkeypatch.setattr(config, "OWNER_ID", 123456)
    db_path = tmp_path / "shared.db"
    first = AccessManager(db_path=db_path)
    stale_second = AccessManager(db_path=db_path)
    try:
        first.grant_access(
            123456, 777, "premium", 30, extend=True,
            idempotency_key="telegram-stars:charge-1",
        )
        first_expiry = first.users["users"]["777"]["expires_date"]

        stale_second.grant_access(
            123456, 777, "premium", 30, extend=True,
            idempotency_key="telegram-stars:charge-2",
        )
        second_expiry = stale_second.users["users"]["777"]["expires_date"]

        assert 29 * 86400 < second_expiry - first_expiry < 31 * 86400
    finally:
        first.close()
        stale_second.close()


def test_paid_month_never_cuts_permanent_access(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import config

    monkeypatch.setattr(config, "OWNER_ID", 123456)
    access = AccessManager()
    access.grant_access(123456, 888, "beta", 0)  # beta lifetime
    assert access.has_permanent_access(888)

    access.grant_access(123456, 888, "premium", 30, extend=True)  # paid month

    record = access.users["users"]["888"]
    assert record["tier"] == "beta"
    assert record["expires_date"] is None
    assert access.has_permanent_access(888)
    # The durable row agrees with memory.
    assert access._reload_user(888)["expires_date"] is None
