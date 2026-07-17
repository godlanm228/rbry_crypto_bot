from core.portfolio_valuation import value_portfolio


def test_partial_market_data_never_treats_missing_position_as_zero():
    portfolio = {
        "BTC": {"amount": 1.0, "buy_price": 100.0},
        "ETH": {"amount": 2.0, "buy_price": 50.0},
    }
    markets = {"bitcoin": {"current_price": 120.0}}

    valuation = value_portfolio(
        portfolio,
        markets,
        coin_id_resolver=lambda symbol: {"BTC": "bitcoin", "ETH": "ethereum"}[symbol],
    )

    assert valuation.is_complete is False
    assert valuation.total_pnl is None
    assert valuation.total_pnl_pct is None
    assert valuation.known_current_value == 120.0
    assert valuation.known_cost == 100.0
    assert valuation.missing_cost == 100.0
    assert valuation.missing_symbols == ("ETH",)


def test_complete_market_data_calculates_total_pnl():
    portfolio = {
        "BTC": {"amount": 1.0, "buy_price": 100.0},
        "ETH": {"amount": 2.0, "buy_price": 50.0},
    }
    markets = {
        "bitcoin": {"current_price": 120.0},
        "ethereum": {"current_price": 40.0},
    }

    valuation = value_portfolio(
        portfolio,
        markets,
        coin_id_resolver=lambda symbol: {"BTC": "bitcoin", "ETH": "ethereum"}[symbol],
    )

    assert valuation.is_complete is True
    assert valuation.total_invested == 200.0
    assert valuation.total_current_value == 200.0
    assert valuation.total_pnl == 0.0
    assert valuation.total_pnl_pct == 0.0
