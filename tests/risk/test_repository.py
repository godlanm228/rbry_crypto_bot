import sqlite3

import pytest

from core.risk.models import RiskProfile
from core.risk.repository import RiskProfileRepository


@pytest.fixture
def repo(tmp_path):
    return RiskProfileRepository(db_path=tmp_path / "bot_state.db")


class TestRoundtrip:
    def test_save_and_get(self, repo):
        saved = repo.save(
            RiskProfile(
                user_id=7,
                account_currency="eur",
                risk_per_trade_pct=1.5,
                max_single_coin_pct=30.0,
                daily_loss_limit_pct=5.0,
                fee_pct=0.1,
            )
        )
        loaded = repo.get(7)
        assert loaded is not None
        assert loaded.account_currency == "EUR"
        assert loaded.risk_per_trade_pct == 1.5
        assert loaded.max_single_coin_pct == 30.0
        assert loaded.daily_loss_limit_pct == 5.0
        assert loaded.fee_pct == 0.1
        assert loaded.slippage_pct is None
        assert loaded.updated_at == pytest.approx(saved.updated_at)

    def test_upsert_overwrites(self, repo):
        repo.save(RiskProfile(user_id=7, risk_per_trade_pct=1.0))
        repo.save(RiskProfile(user_id=7, risk_per_trade_pct=2.0))
        loaded = repo.get(7)
        assert loaded.risk_per_trade_pct == 2.0

    def test_missing_returns_none(self, repo):
        assert repo.get(999) is None

    def test_get_or_default_not_persisted(self, repo):
        profile = repo.get_or_default(5)
        assert profile == RiskProfile.conservative_default(5)
        assert repo.get(5) is None

    def test_shares_bot_state_db_file(self, tmp_path):
        db_path = tmp_path / "bot_state.db"
        RiskProfileRepository(db_path=db_path)
        tables = {
            row[0]
            for row in sqlite3.connect(str(db_path)).execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "risk_profiles" in tables


class TestCorruptDataFailsLoudly:
    def test_out_of_range_stored_value_raises(self, repo):
        repo.save(RiskProfile(user_id=7))
        with sqlite3.connect(str(repo.db_path)) as conn:
            conn.execute("UPDATE risk_profiles SET risk_per_trade_pct = 50.0 WHERE user_id = 7")
        with pytest.raises(ValueError):
            repo.get(7)
