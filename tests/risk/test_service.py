"""Risk panel facade on hand-computed fixtures."""

import math

import pytest

from core.risk.models import DrawdownState
from core.risk.service import STYLES, RiskStyleStore, build_risk_panel, size_trade

RESOLVE = {"BTC": "bitcoin", "SOL": "solana", "USDT": "tether"}.get
PORTFOLIO = {
    "BTC": {"amount": 0.001, "buy_price": 50_000.0},   # cost 50, value 60
    "SOL": {"amount": 1.0, "buy_price": 120.0},        # cost 120, value 100
}
MARKETS = {"bitcoin": {"current_price": 60_000.0}, "solana": {"current_price": 100.0}}


def _prices(start, daily, n=91):
    return [start * (1 + daily) ** i * (1 + 0.02 * math.sin(i)) for i in range(n)]


HISTORY = {"bitcoin": _prices(50_000, 0.001), "solana": _prices(100, 0.002)}


def test_panel_numbers_for_a_two_coin_portfolio():
    panel = build_risk_panel(PORTFOLIO, MARKETS, HISTORY, STYLES["balanced"], coin_id_resolver=RESOLVE)
    assert panel.value == pytest.approx(160.0)
    assert panel.cost == pytest.approx(170.0)
    assert panel.drawdown_pct == pytest.approx(10 / 170 * 100)        # 5.88 %
    assert panel.state == DrawdownState.CAUTION                        # balanced: from -5 %
    assert panel.next_state == DrawdownState.RISK_FREEZE and panel.next_state_at_pct == pytest.approx(10.0)
    sol = next(c for c in panel.concentration if c.symbol == "SOL")
    btc = next(c for c in panel.concentration if c.symbol == "BTC")
    assert sol.weight_pct == pytest.approx(62.5) and sol.over_limit         # altcoin limit 40 %
    assert btc.weight_pct == pytest.approx(37.5) and not btc.over_limit     # BTC/ETH limit 70 %
    assert panel.var_pct is not None and panel.var_value == pytest.approx(160 * panel.var_pct / 100)
    assert panel.worst is not None and panel.worst.loss_value > 0
    assert panel.risk_budget == pytest.approx(3.2)                      # 2 % of $160


def test_careful_style_raises_alarms_earlier():
    careful = build_risk_panel(PORTFOLIO, MARKETS, HISTORY, STYLES["careful"], coin_id_resolver=RESOLVE)
    assert careful.state == DrawdownState.CAUTION              # starts at -3 %
    assert careful.next_state_at_pct == pytest.approx(6.0)     # freeze at -6 % instead of -10 %
    assert next(c for c in careful.concentration if c.symbol == "SOL").limit_pct == 30.0


def test_missing_history_means_no_var_not_a_guess():
    panel = build_risk_panel(PORTFOLIO, MARKETS, {"bitcoin": HISTORY["bitcoin"]}, STYLES["balanced"],
                             coin_id_resolver=RESOLVE)
    assert panel.var_pct is None and panel.var_value is None


def test_size_trade_example_from_the_explanation():
    # $100 account, balanced (2 % = $2), SOL entry 105, stop 100, take 115.
    sizing = size_trade(user_id=1, style=STYLES["balanced"], symbol="SOL", entry=105, stop=100,
                        take=115, account_value=100)
    assert sizing.approved
    effective = (5 / 105 * 100) + 2 * 0.10 + 0.05                 # stop + round-trip costs, %
    assert sizing.position_value == pytest.approx(2 / (effective / 100))   # ≈ $39.9
    assert sizing.loss_at_stop == pytest.approx(2.0)
    assert sizing.reward_risk == pytest.approx(2.0)
    assert sizing.coins == pytest.approx(sizing.position_value / 105)


def test_stop_above_entry_and_tiny_orders_are_refused():
    above = size_trade(user_id=1, style=STYLES["balanced"], symbol="SOL", entry=100, stop=105,
                       take=None, account_value=100)
    assert not above.approved and above.reject_reasons == ("stop_not_below_entry",)
    tiny = size_trade(user_id=1, style=STYLES["balanced"], symbol="SOL", entry=100, stop=50,
                      take=None, account_value=100)              # $2 / 50 % ≈ $3.98 < $5 minimum
    assert not tiny.approved and any("min_notional" in r for r in tiny.reject_reasons)


def test_style_store_roundtrip(tmp_path):
    store = RiskStyleStore(tmp_path / "state.db")
    assert store.get(5) is None
    store.set(5, "bold")
    assert store.get(5).key == "bold"
    with pytest.raises(ValueError):
        store.set(5, "yolo")
