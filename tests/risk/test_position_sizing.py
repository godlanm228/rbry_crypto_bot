"""Position sizing: the EUR 100 plan fixture, caps, and rejections.

Plan fixture: account EUR 100, risk 1% → budget EUR 1; valid stop 5% →
raw maximum position EUR 20 before fees and portfolio caps.
"""

import pytest

from core.risk.models import DrawdownState, PositionSizeRequest, RiskProfile
from core.risk.position_sizing import size_position


def _profile(fee=0.0, slippage=0.0, **kwargs):
    return RiskProfile(user_id=1, fee_pct=fee, slippage_pct=slippage, **kwargs)


def _request(**kwargs):
    defaults = dict(symbol="ADA", account_value=100.0, stop_distance_pct=5.0)
    defaults.update(kwargs)
    return PositionSizeRequest(**defaults)


class TestEur100Fixture:
    def test_raw_formula_without_costs(self):
        result = size_position(_profile(), _request())
        assert result.approved
        assert result.risk_budget == pytest.approx(1.0)
        assert result.position_value == pytest.approx(20.0)
        assert result.caps_applied == ()

    def test_fees_and_slippage_shrink_the_size(self):
        # 2 * 0.1% fee + 0.1% slippage = 0.3% → loss at stop 5.3%.
        result = size_position(_profile(fee=0.1, slippage=0.1), _request())
        assert result.approved
        assert result.effective_loss_pct == pytest.approx(5.3)
        assert result.position_value == pytest.approx(1.0 / 0.053)

    def test_min_notional_invalidates_executable_size(self):
        result = size_position(
            _profile(), _request(min_notional=25.0)
        )
        assert not result.approved
        assert any("below_min_notional" in r for r in result.reject_reasons)

    def test_aggressive_2pct_cap(self):
        result = size_position(
            _profile(risk_per_trade_pct=2.0), _request()
        )
        assert result.risk_budget == pytest.approx(2.0)
        assert result.position_value == pytest.approx(25.0)  # capped by 25% concentration
        assert any("single_coin_concentration" in c for c in result.caps_applied)


class TestHardRejections:
    def test_no_stop_distance_means_no_size(self):
        result = size_position(_profile(), _request(stop_distance_pct=None))
        assert not result.approved
        assert "no_stop_distance" in result.reject_reasons
        assert result.position_value == 0.0

    def test_zero_or_negative_stop_rejected(self):
        for bad in (0.0, -3.0):
            result = size_position(_profile(), _request(stop_distance_pct=bad))
            assert not result.approved

    def test_unconfigured_costs_are_rejected_not_assumed_zero(self):
        profile = RiskProfile(user_id=1)  # fee/slippage stay None
        result = size_position(profile, _request())
        assert not result.approved
        assert "cost_inputs_not_configured" in result.reject_reasons

    def test_invalid_account_value(self):
        result = size_position(_profile(), _request(account_value=0.0))
        assert not result.approved
        assert "account_value_invalid" in result.reject_reasons


class TestCaps:
    def test_concentration_headroom(self):
        # Existing ADA position EUR 20; 25% cap on EUR 100 leaves EUR 5.
        result = size_position(
            _profile(), _request(existing_position_value=20.0)
        )
        assert result.approved
        assert result.position_value == pytest.approx(5.0)
        assert any("single_coin_concentration" in c for c in result.caps_applied)

    def test_exhausted_headroom_rejects(self):
        result = size_position(
            _profile(), _request(existing_position_value=25.0)
        )
        assert not result.approved
        assert "no_headroom_under_caps" in result.reject_reasons

    def test_correlated_cluster_cap(self):
        # Cluster already at 35% of the account; 40% cap leaves EUR 5.
        result = size_position(
            _profile(), _request(cluster_exposure_pct=35.0)
        )
        assert result.position_value == pytest.approx(5.0)
        assert any("correlated_cluster" in c for c in result.caps_applied)

    def test_available_capital_cap(self):
        result = size_position(
            _profile(), _request(available_capital=3.0)
        )
        assert result.position_value == pytest.approx(3.0)
        assert "available_capital" in result.caps_applied

    def test_unknown_cluster_is_noted_not_guessed(self):
        result = size_position(_profile(), _request())
        assert "cluster_exposure_unknown" in result.notes
        assert "liquidity_cap_not_evaluated" in result.notes


class TestDrawdownProtocol:
    def test_caution_halves_risk_budget(self):
        result = size_position(
            _profile(), _request(), drawdown_state=DrawdownState.CAUTION
        )
        assert result.approved
        assert result.risk_budget == pytest.approx(0.5)
        assert result.position_value == pytest.approx(10.0)

    def test_risk_freeze_blocks_speculative_entries(self):
        result = size_position(
            _profile(), _request(is_speculative=True),
            drawdown_state=DrawdownState.RISK_FREEZE,
        )
        assert not result.approved
        assert any("new_entry_blocked" in r for r in result.reject_reasons)

    def test_risk_freeze_quarters_non_speculative(self):
        result = size_position(
            _profile(), _request(), drawdown_state=DrawdownState.RISK_FREEZE
        )
        assert result.approved
        assert result.risk_budget == pytest.approx(0.25)
        assert result.position_value == pytest.approx(5.0)

    def test_defensive_blocks_all_new_entries(self):
        result = size_position(
            _profile(), _request(), drawdown_state=DrawdownState.DEFENSIVE
        )
        assert not result.approved


class TestVolatilityRegime:
    def test_high_volatility_halves_budget(self):
        result = size_position(
            _profile(), _request(volatility_regime="high")
        )
        assert result.risk_budget == pytest.approx(0.5)
        assert result.position_value == pytest.approx(10.0)

    def test_unknown_regime_noted_not_assumed_low(self):
        result = size_position(_profile(), _request(volatility_regime=None))
        assert result.risk_budget == pytest.approx(1.0)
        assert "volatility_regime_unknown" in result.notes
