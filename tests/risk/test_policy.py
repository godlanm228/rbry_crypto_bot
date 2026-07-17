"""Risk policy: drawdown protocol, limits, allowed actions."""

import pytest

from core.portfolio_valuation import value_portfolio
from core.risk.exposure import build_portfolio_exposure
from core.risk.models import ActionClass, DrawdownState, RiskProfile, Severity
from core.risk.policy import (
    allowed_actions,
    drawdown_state_for,
    evaluate_limits,
    new_entry_allowed,
)


def _exposure(portfolio, markets, **kwargs):
    return build_portfolio_exposure(
        value_portfolio(portfolio, markets, coin_id_resolver=str.lower), **kwargs
    )


class TestDrawdownStateMachine:
    @pytest.mark.parametrize("drawdown,expected", [
        (0.0, DrawdownState.NORMAL),
        (4.99, DrawdownState.NORMAL),
        (5.0, DrawdownState.CAUTION),
        (9.99, DrawdownState.CAUTION),
        (10.0, DrawdownState.RISK_FREEZE),
        (14.99, DrawdownState.RISK_FREEZE),
        (15.0, DrawdownState.CONCENTRATION_REVIEW),
        (20.0, DrawdownState.DEFENSIVE),
        (55.0, DrawdownState.DEFENSIVE),
    ])
    def test_thresholds(self, drawdown, expected):
        assert drawdown_state_for(drawdown) == expected

    def test_new_entry_gate(self):
        assert new_entry_allowed(DrawdownState.NORMAL, is_speculative=True)
        assert new_entry_allowed(DrawdownState.CAUTION, is_speculative=True)
        assert not new_entry_allowed(DrawdownState.RISK_FREEZE, is_speculative=True)
        assert new_entry_allowed(DrawdownState.RISK_FREEZE, is_speculative=False)
        assert not new_entry_allowed(DrawdownState.DEFENSIVE, is_speculative=False)

    def test_defensive_allows_only_risk_reducing_actions(self):
        actions = allowed_actions(DrawdownState.DEFENSIVE)
        assert ActionClass.CONSIDER_ENTRY not in actions
        assert ActionClass.DE_RISK_IMMEDIATELY in actions
        assert ActionClass.REDUCE_SIZE in actions

    def test_normal_allows_entries(self):
        assert ActionClass.CONSIDER_ENTRY in allowed_actions(DrawdownState.NORMAL)


class TestConcentrationLimits:
    def test_hard_breach_above_40_pct(self):
        # BTC 50% and ETH 41.7% both exceed the 40% hard-warning line.
        exposure = _exposure(
            {
                "BTC": {"amount": 0.001, "buy_price": 50000.0},
                "ETH": {"amount": 0.02, "buy_price": 2000.0},
                "USDT": {"amount": 10.0, "buy_price": 1.0},
            },
            {
                "btc": {"current_price": 60000.0},
                "eth": {"current_price": 2500.0},
                "usdt": {"current_price": 1.0},
            },
        )
        breaches = evaluate_limits(RiskProfile(user_id=1), exposure)
        single = {b.subject: b for b in breaches if b.code == "single_coin_concentration"}
        assert single["BTC"].severity == Severity.HIGH
        assert single["BTC"].suggested_action == ActionClass.REDUCE_SIZE
        assert single["ETH"].severity == Severity.HIGH
        assert "USDT" not in single  # stablecoins have their own bucket limit

    def test_soft_breach_between_limit_and_40(self):
        # Three coins at 30/35/35 percent with a 25% limit: all warnings.
        exposure = _exposure(
            {
                "AAA": {"amount": 30.0, "buy_price": 1.0},
                "BBB": {"amount": 35.0, "buy_price": 1.0},
                "CCC": {"amount": 35.0, "buy_price": 1.0},
            },
            {
                "aaa": {"current_price": 1.0},
                "bbb": {"current_price": 1.0},
                "ccc": {"current_price": 1.0},
            },
        )
        breaches = evaluate_limits(RiskProfile(user_id=1), exposure)
        singles = [b for b in breaches if b.code == "single_coin_concentration"]
        assert len(singles) == 3
        assert all(b.severity == Severity.WARNING for b in singles)
        assert all(b.suggested_action == ActionClass.DO_NOT_ADD for b in singles)

    def test_no_breach_within_limits(self):
        exposure = _exposure(
            {
                "AAA": {"amount": 20.0, "buy_price": 1.0},
                "BBB": {"amount": 20.0, "buy_price": 1.0},
                "CCC": {"amount": 20.0, "buy_price": 1.0},
                "DDD": {"amount": 20.0, "buy_price": 1.0},
                "EEE": {"amount": 20.0, "buy_price": 1.0},
            },
            {s.lower(): {"current_price": 1.0} for s in ("AAA", "BBB", "CCC", "DDD", "EEE")},
        )
        breaches = evaluate_limits(RiskProfile(user_id=1), exposure)
        assert not [b for b in breaches if b.code == "single_coin_concentration"]


class TestBucketAndClusterLimits:
    def test_stablecoin_bucket_breach(self):
        exposure = _exposure(
            {
                "USDT": {"amount": 60.0, "buy_price": 1.0},
                "BTC": {"amount": 0.001, "buy_price": 40000.0},
            },
            {
                "usdt": {"current_price": 1.0},
                "btc": {"current_price": 40000.0},
            },
        )
        breaches = evaluate_limits(RiskProfile(user_id=1), exposure)
        bucket = [b for b in breaches if b.code == "stablecoin_bucket"]
        assert len(bucket) == 1
        assert bucket[0].observed_pct == pytest.approx(60.0)
        assert bucket[0].suggested_action == ActionClass.REBALANCE

    def test_correlated_cluster_breach(self):
        correlation = {
            "BTC": {"BTC": 1.0, "ETH": 0.9},
            "ETH": {"BTC": 0.9, "ETH": 1.0},
        }
        exposure = _exposure(
            {
                "BTC": {"amount": 0.001, "buy_price": 50000.0},
                "ETH": {"amount": 0.02, "buy_price": 2000.0},
                "USDT": {"amount": 10.0, "buy_price": 1.0},
            },
            {
                "btc": {"current_price": 60000.0},
                "eth": {"current_price": 2500.0},
                "usdt": {"current_price": 1.0},
            },
            correlation=correlation,
        )
        breaches = evaluate_limits(RiskProfile(user_id=1), exposure)
        clusters = [b for b in breaches if b.code == "correlated_cluster"]
        assert len(clusters) == 1
        # BTC+ETH = 91.67% of known value > 40% * 1.25 → HIGH.
        assert clusters[0].severity == Severity.HIGH
        assert clusters[0].subject == "cluster:BTC+ETH"

    def test_drawdown_breach_from_underwater_portfolio(self):
        # Value 80 vs cost 100 → 20% cost-basis drawdown → DEFENSIVE.
        exposure = _exposure(
            {
                "BTC": {"amount": 0.001, "buy_price": 50000.0},
                "ETH": {"amount": 0.02, "buy_price": 2000.0},
                "USDT": {"amount": 10.0, "buy_price": 1.0},
            },
            {
                "btc": {"current_price": 40000.0},
                "eth": {"current_price": 1500.0},
                "usdt": {"current_price": 1.0},
            },
        )
        breaches = evaluate_limits(RiskProfile(user_id=1), exposure)
        drawdown = [b for b in breaches if b.code == "drawdown_protocol"]
        assert len(drawdown) == 1
        assert drawdown[0].severity == Severity.CRITICAL
        assert drawdown[0].suggested_action == ActionClass.DE_RISK_IMMEDIATELY
        assert drawdown[0].observed_pct == pytest.approx(20.0)
