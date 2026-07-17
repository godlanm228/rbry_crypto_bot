import pytest

from core.risk.models import MetricsUnavailable, RiskProfile


class TestRiskProfileValidation:
    def test_conservative_default(self):
        profile = RiskProfile.conservative_default(42)
        assert profile.user_id == 42
        assert profile.account_currency == "EUR"
        assert profile.risk_per_trade_pct == 1.0
        assert profile.max_single_coin_pct == 25.0
        assert profile.max_speculative_pct == 15.0
        assert profile.max_stablecoin_issuer_pct == 50.0
        # Cost inputs must stay explicitly unconfigured, not default to zero.
        assert profile.fee_pct is None
        assert profile.slippage_pct is None
        assert profile.daily_loss_limit_pct is None

    def test_currency_normalized(self):
        assert RiskProfile(user_id=1, account_currency="eur").account_currency == "EUR"

    @pytest.mark.parametrize("value", [0.0, -1.0, 2.01, 100.0, float("nan"), float("inf")])
    def test_risk_per_trade_out_of_bounds(self, value):
        with pytest.raises(ValueError):
            RiskProfile(user_id=1, risk_per_trade_pct=value)

    def test_risk_per_trade_cap_is_two_percent(self):
        assert RiskProfile(user_id=1, risk_per_trade_pct=2.0).risk_per_trade_pct == 2.0

    @pytest.mark.parametrize("field", ["max_single_coin_pct", "max_stablecoin_issuer_pct"])
    @pytest.mark.parametrize("value", [0.0, -5.0, 101.0])
    def test_limit_bounds(self, field, value):
        with pytest.raises(ValueError):
            RiskProfile(user_id=1, **{field: value})

    def test_zero_speculative_bucket_allowed(self):
        assert RiskProfile(user_id=1, max_speculative_pct=0.0).max_speculative_pct == 0.0

    @pytest.mark.parametrize("value", [-0.1, 5.1])
    def test_fee_bounds(self, value):
        with pytest.raises(ValueError):
            RiskProfile(user_id=1, fee_pct=value)

    def test_zero_fee_must_be_explicit_not_default(self):
        assert RiskProfile(user_id=1, fee_pct=0.0).fee_pct == 0.0

    @pytest.mark.parametrize("user_id", [0, -3, "7"])
    def test_invalid_user_id(self, user_id):
        with pytest.raises(ValueError):
            RiskProfile(user_id=user_id)

    def test_invalid_currency(self):
        with pytest.raises(ValueError):
            RiskProfile(user_id=1, account_currency="€")


class TestMetricsUnavailable:
    def test_requires_reasons(self):
        with pytest.raises(ValueError):
            MetricsUnavailable(reasons=())

    def test_carries_reasons(self):
        block = MetricsUnavailable(reasons=("insufficient history",))
        assert block.reasons == ("insufficient history",)
