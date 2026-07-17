"""Exposure fixtures.

Base portfolio (all prices known):
    BTC: 0.001 @ 50000 -> cost 50, price 60000 -> value 60
    ETH: 0.02 @ 2000 -> cost 40, price 2500 -> value 50
    USDT: 10 @ 1 -> cost 10, price 1 -> value 10
Total known value 120: weights 50% / 41.666...% / 8.333...%.
top1 = 50, top3 = 100, stablecoin = 8.333...%.
"""

import pytest

from core.portfolio_valuation import value_portfolio
from core.risk.exposure import build_portfolio_exposure


def _valuation(portfolio, markets):
    return value_portfolio(portfolio, markets, coin_id_resolver=str.lower)


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


class TestCompletePortfolio:
    def _exposure(self, **kwargs):
        return build_portfolio_exposure(_valuation(PORTFOLIO, MARKETS), **kwargs)

    def test_weights(self):
        by_symbol = {p.symbol: p for p in self._exposure().positions}
        assert by_symbol["BTC"].weight_pct == pytest.approx(50.0)
        assert by_symbol["ETH"].weight_pct == pytest.approx(50 / 120 * 100)
        assert by_symbol["USDT"].weight_pct == pytest.approx(10 / 120 * 100)

    def test_concentration(self):
        exposure = self._exposure()
        assert exposure.top1_pct == pytest.approx(50.0)
        assert exposure.top3_pct == pytest.approx(100.0)
        assert exposure.top5_pct == pytest.approx(100.0)

    def test_stablecoin_share(self):
        assert self._exposure().stablecoin_pct == pytest.approx(10 / 120 * 100)

    def test_pnl(self):
        by_symbol = {p.symbol: p for p in self._exposure().positions}
        assert by_symbol["BTC"].unrealized_pnl == pytest.approx(10.0)
        assert by_symbol["BTC"].unrealized_pnl_pct == pytest.approx(20.0)
        assert by_symbol["USDT"].unrealized_pnl == pytest.approx(0.0)

    def test_profit_means_no_cost_drawdown(self):
        assert self._exposure().drawdown_from_cost_pct == pytest.approx(0.0)

    def test_complete(self):
        exposure = self._exposure()
        assert exposure.is_complete
        assert exposure.missing_symbols == ()

    def test_clusters_from_correlation(self):
        correlation = {
            "BTC": {"BTC": 1.0, "ETH": 0.85, "USDT": 0.01},
            "ETH": {"BTC": 0.85, "ETH": 1.0, "USDT": 0.02},
            "USDT": {"BTC": 0.01, "ETH": 0.02, "USDT": 1.0},
        }
        exposure = self._exposure(correlation=correlation, correlation_threshold=0.8)
        assert len(exposure.clusters) == 1
        cluster = exposure.clusters[0]
        assert cluster.symbols == ("BTC", "ETH")
        assert cluster.weight_pct == pytest.approx(50.0 + 50 / 120 * 100)

    def test_no_clusters_without_correlation(self):
        assert self._exposure().clusters == ()


class TestMissingPrices:
    def test_missing_price_is_surfaced_not_guessed(self):
        portfolio = dict(PORTFOLIO, XYZ={"amount": 5.0, "buy_price": 2.0})
        exposure = build_portfolio_exposure(_valuation(portfolio, MARKETS))
        assert not exposure.is_complete
        assert exposure.missing_symbols == ("XYZ",)
        assert exposure.missing_cost == pytest.approx(10.0)
        xyz = next(p for p in exposure.positions if p.symbol == "XYZ")
        assert xyz.value is None
        assert xyz.weight_pct is None
        assert not xyz.price_known
        # Known positions keep weights over the known value only.
        btc = next(p for p in exposure.positions if p.symbol == "BTC")
        assert btc.weight_pct == pytest.approx(50.0)

    def test_all_prices_missing(self):
        exposure = build_portfolio_exposure(_valuation(PORTFOLIO, {}))
        assert exposure.known_value == 0.0
        assert exposure.top1_pct is None
        assert exposure.stablecoin_pct is None
        assert exposure.clusters == ()


class TestCostDrawdown:
    def test_underwater_portfolio(self):
        markets = {
            "btc": {"current_price": 40000.0},  # value 40 vs cost 50
            "eth": {"current_price": 1500.0},   # value 30 vs cost 40
            "usdt": {"current_price": 1.0},     # value 10 vs cost 10
        }
        exposure = build_portfolio_exposure(_valuation(PORTFOLIO, markets))
        # 80 versus 100 invested -> 20% under cost basis.
        assert exposure.drawdown_from_cost_pct == pytest.approx(20.0)
