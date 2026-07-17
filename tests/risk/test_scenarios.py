"""Stress scenarios over the R1 fixture portfolio.

Base fixture (all prices known): BTC value 60, ETH 50, USDT 10 → known 120.

Hand-computed expectations:
- btc_-20: BTC -20% = 12, ETH -22% = 11, USDT 0 → loss 23 → 19.1667%.
- stable_depeg_-5: 10 * 5% = 0.5 → 0.41667%.
- correlation_panic: (60+50) * 25% = 27.5 → 22.9167%.
- largest_holding_-40: BTC 60 * 40% = 24 → 20%.
- reverse stress -10%: non-stable share 110/120 → 10 / (110/120) = 10.9091%.
"""

import pytest

from core.portfolio_valuation import value_portfolio
from core.risk.exposure import build_portfolio_exposure
from core.risk.scenarios import classify_symbol, reverse_stress, run_scenarios

PORTFOLIO = {
    "BTC": {"amount": 0.001, "buy_price": 50000.0},
    "ETH": {"amount": 0.02, "buy_price": 2000.0},
    "USDT": {"amount": 10.0, "buy_price": 1.0},
}
MARKETS = {
    "btc": {"current_price": 60000.0},
    "eth": {"current_price": 2500.0},
    "usdt": {"current_price": 1.0},
}


def _exposure(portfolio=PORTFOLIO, markets=MARKETS):
    return build_portfolio_exposure(
        value_portfolio(portfolio, markets, coin_id_resolver=str.lower)
    )


def _result(report, key):
    return next(r for r in report.results if r.scenario_key == key)


class TestClassification:
    def test_classes(self):
        assert classify_symbol("BTC") == "btc"
        assert classify_symbol("ETH") == "eth"
        assert classify_symbol("USDT") == "stablecoin"
        assert classify_symbol("ADA") == "altcoin"


class TestShockScenarios:
    def test_btc_minus_20_hand_computed(self):
        result = _result(run_scenarios(_exposure()), "btc_-20")
        assert result.loss_value == pytest.approx(23.0)
        assert result.loss_pct == pytest.approx(23.0 / 120.0 * 100.0)
        assert result.contributors[0] == ("BTC", pytest.approx(12.0))
        assert result.contributors[1] == ("ETH", pytest.approx(11.0))

    def test_stable_depeg_5(self):
        result = _result(run_scenarios(_exposure()), "stable_depeg_-5")
        assert result.loss_value == pytest.approx(0.5)
        assert result.loss_pct == pytest.approx(0.5 / 120.0 * 100.0)

    def test_correlation_panic_spares_stablecoins(self):
        result = _result(run_scenarios(_exposure()), "correlation_panic")
        assert result.loss_value == pytest.approx(27.5)
        assert all(symbol != "USDT" for symbol, _ in result.contributors)

    def test_largest_holding_is_btc(self):
        result = _result(run_scenarios(_exposure()), "largest_holding_-40")
        assert result.loss_value == pytest.approx(24.0)
        assert result.loss_pct == pytest.approx(20.0)
        assert result.contributors == (("BTC", pytest.approx(24.0)),)

    def test_liquidity_haircut_hits_only_altcoins(self):
        no_alts = _result(run_scenarios(_exposure()), "liquidity_haircut")
        assert no_alts.loss_value == pytest.approx(0.0)

        portfolio = dict(PORTFOLIO, ADA={"amount": 100.0, "buy_price": 0.25})
        markets = dict(MARKETS, ada={"current_price": 0.30})  # value 30
        with_alt = _result(run_scenarios(_exposure(portfolio, markets)), "liquidity_haircut")
        assert with_alt.loss_value == pytest.approx(30.0 * 0.20)

    def test_scenario_count_and_config_version(self):
        report = run_scenarios(_exposure())
        assert report.is_available
        assert len(report.results) == 12  # 11 shock sets + largest holding
        assert report.config_version


class TestHonestGaps:
    def test_missing_price_symbols_reported_not_ignored(self):
        portfolio = dict(PORTFOLIO, XYZ={"amount": 5.0, "buy_price": 2.0})
        report = run_scenarios(_exposure(portfolio, MARKETS))
        assert report.excluded_symbols == ("XYZ",)
        assert all(r.excluded_symbols == ("XYZ",) for r in report.results)

    def test_no_priced_holdings_is_unavailable(self):
        report = run_scenarios(_exposure(PORTFOLIO, {}))
        assert not report.is_available
        assert report.results == ()
        assert "no_priced_holdings" in report.unavailable_reasons


class TestReverseStress:
    def test_hand_computed_uniform_shock(self):
        results = reverse_stress(_exposure())
        by_target = {r.target_loss_pct: r for r in results}
        share = 110.0 / 120.0
        assert by_target[10.0].required_shock_pct == pytest.approx(10.0 / share)
        assert by_target[20.0].required_shock_pct == pytest.approx(20.0 / share)
        assert by_target[30.0].required_shock_pct == pytest.approx(30.0 / share)

    def test_stable_only_portfolio_cannot_reach_targets(self):
        portfolio = {"USDT": {"amount": 100.0, "buy_price": 1.0}}
        markets = {"usdt": {"current_price": 1.0}}
        results = reverse_stress(_exposure(portfolio, markets))
        assert all(r.required_shock_pct is None for r in results)
        assert all(r.reason == "no_priced_non_stable_holdings" for r in results)

    def test_unreachable_target_reported(self):
        # Non-stable share 10/110: -30% portfolio would need a 330% shock.
        portfolio = {
            "USDT": {"amount": 100.0, "buy_price": 1.0},
            "ADA": {"amount": 10.0, "buy_price": 1.0},
        }
        markets = {"usdt": {"current_price": 1.0}, "ada": {"current_price": 1.0}}
        results = reverse_stress(_exposure(portfolio, markets))
        by_target = {r.target_loss_pct: r for r in results}
        assert by_target[30.0].required_shock_pct is None
        assert by_target[30.0].reason == "unreachable_via_non_stable_holdings"
