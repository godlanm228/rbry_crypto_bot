"""Hand-computed fixtures for the pure risk mathematics.

Two-asset fixture, weights 0.5/0.5:

    A: [0.10, -0.05, 0.02, -0.02, 0.05]
    B: [0.04, -0.01, 0.03, -0.03, 0.02]

Portfolio returns: [0.07, -0.03, 0.025, -0.025, 0.035], mean 0.015.
Sample variance = 0.00715 / 4 = 0.0017875 -> vol = 4.227860...%.
VaR/ES at 95%, n=5: k=1 -> worst return -0.03 -> 3%.
Parametric VaR = 1.6449 * 0.04227860 - 0.015 = 0.05454432... -> 5.454432%.
Wealth index peaks at 1.07 after t1; worst trough is t4:
    W4 = 1.07 * 0.97 * 1.025 * 0.975 = 1.0372513125
    max drawdown = 1 - W4/1.07 = 3.060625%; final W5 is a new peak -> current 0.
Downside deviation = sqrt((0.03^2 + 0.025^2) / 5) = 1.746425...%.
Covariances (ddof=1): var_A=0.00345, var_B=0.00085, cov=0.001425.
Risk contribution A = 0.5*(0.5*0.00345+0.5*0.001425)/0.0017875 = 68.1818...%.
Correlation(A,B) = 0.001425/sqrt(0.00345*0.00085) = 0.832139...
"""

import math

import pytest

from core.risk.metrics import (
    compute_correlation_matrix,
    compute_portfolio_metrics,
    realized_volatility,
    returns_from_prices,
)
from core.risk.models import MetricsUnavailable, RiskMetrics


RETURNS_A = (0.10, -0.05, 0.02, -0.02, 0.05)
RETURNS_B = (0.04, -0.01, 0.03, -0.03, 0.02)
SERIES = {"AAA": RETURNS_A, "BBB": RETURNS_B}
WEIGHTS = {"AAA": 0.5, "BBB": 0.5}


class TestReturnsFromPrices:
    def test_simple_returns(self):
        assert returns_from_prices([100.0, 110.0, 99.0]) == pytest.approx((0.10, -0.10))

    @pytest.mark.parametrize("prices", [[100.0], [], [100.0, 0.0], [100.0, -5.0], [100.0, float("nan")]])
    def test_rejects_bad_input(self, prices):
        with pytest.raises(ValueError):
            returns_from_prices(prices)


class TestPortfolioMetrics:
    def _metrics(self) -> RiskMetrics:
        result = compute_portfolio_metrics(SERIES, WEIGHTS, min_observations=5)
        assert isinstance(result, RiskMetrics)
        return result

    def test_volatility(self):
        assert self._metrics().portfolio_volatility_pct == pytest.approx(
            math.sqrt(0.0017875) * 100, rel=1e-9
        )

    def test_historical_var_and_es(self):
        metrics = self._metrics()
        assert metrics.historical_var_pct == pytest.approx(3.0)
        assert metrics.historical_es_pct == pytest.approx(3.0)

    def test_parametric_var(self):
        expected = (1.6449 * math.sqrt(0.0017875) - 0.015) * 100
        assert self._metrics().parametric_var_pct == pytest.approx(expected, rel=1e-9)

    def test_drawdowns(self):
        metrics = self._metrics()
        expected_max = (1 - (1.07 * 0.97 * 1.025 * 0.975) / 1.07) * 100
        assert metrics.max_drawdown_pct == pytest.approx(expected_max, rel=1e-9)
        assert metrics.current_drawdown_pct == pytest.approx(0.0, abs=1e-12)

    def test_downside_deviation(self):
        expected = math.sqrt((0.03**2 + 0.025**2) / 5) * 100
        assert self._metrics().downside_deviation_pct == pytest.approx(expected, rel=1e-9)

    def test_risk_contributions_sum_to_100_and_match_hand_calc(self):
        contributions = self._metrics().risk_contribution_pct
        assert contributions["AAA"] == pytest.approx(0.00121875 / 0.0017875 * 100, rel=1e-9)
        assert contributions["BBB"] == pytest.approx(0.00056875 / 0.0017875 * 100, rel=1e-9)
        assert sum(contributions.values()) == pytest.approx(100.0)

    def test_metadata(self):
        metrics = self._metrics()
        assert metrics.observations == 5
        assert metrics.interval == "daily"
        assert metrics.confidence == 0.95


class TestCoverageGates:
    def test_insufficient_history_blocks(self):
        result = compute_portfolio_metrics(SERIES, WEIGHTS, min_observations=20)
        assert isinstance(result, MetricsUnavailable)
        assert any("insufficient history" in r for r in result.reasons)

    def test_missing_position_history_blocks(self):
        result = compute_portfolio_metrics(
            {"AAA": RETURNS_A}, {"AAA": 0.5, "ZZZ": 0.5}, min_observations=5
        )
        assert isinstance(result, MetricsUnavailable)
        assert any("ZZZ" in r for r in result.reasons)

    def test_misaligned_series_blocks(self):
        result = compute_portfolio_metrics(
            {"AAA": RETURNS_A, "BBB": RETURNS_B[:4]}, WEIGHTS, min_observations=4
        )
        assert isinstance(result, MetricsUnavailable)
        assert any("not aligned" in r for r in result.reasons)

    def test_bad_weights_raise_instead_of_block(self):
        with pytest.raises(ValueError):
            compute_portfolio_metrics(SERIES, {"AAA": 0.5, "BBB": 0.4}, min_observations=5)
        with pytest.raises(ValueError):
            compute_portfolio_metrics(SERIES, {"AAA": 1.5, "BBB": -0.5}, min_observations=5)

    def test_unsupported_confidence_raises(self):
        with pytest.raises(ValueError):
            compute_portfolio_metrics(SERIES, WEIGHTS, confidence=0.42, min_observations=5)


class TestCorrelationMatrix:
    def test_hand_computed_correlation(self):
        matrix = compute_correlation_matrix(SERIES, min_observations=5)
        assert isinstance(matrix, dict)
        expected = 0.001425 / math.sqrt(0.00345 * 0.00085)
        assert matrix["AAA"]["BBB"] == pytest.approx(expected, rel=1e-9)
        assert matrix["BBB"]["AAA"] == pytest.approx(expected, rel=1e-9)
        assert matrix["AAA"]["AAA"] == 1.0

    def test_zero_variance_symbol_excluded(self):
        series = dict(SERIES, FLAT=(0.0, 0.0, 0.0, 0.0, 0.0))
        matrix = compute_correlation_matrix(series, min_observations=5)
        assert isinstance(matrix, dict)
        assert "FLAT" not in matrix
        assert "FLAT" not in matrix["AAA"]

    def test_thin_history_blocks(self):
        result = compute_correlation_matrix(SERIES, min_observations=20)
        assert isinstance(result, MetricsUnavailable)


class TestRealizedVolatility:
    def test_matches_sample_stddev(self):
        assert realized_volatility(RETURNS_A) == pytest.approx(math.sqrt(0.00345), rel=1e-9)

    def test_needs_two_returns(self):
        with pytest.raises(ValueError):
            realized_volatility([0.01])
