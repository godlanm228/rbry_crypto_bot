"""Risk panel facade: turns a user's portfolio into the numbers shown under 🛡 Risk.

Pure computation over already-fetched data (portfolio rows, market rows,
daily price history); handlers render the result. Every number here comes
from the verified risk engine (exposure, metrics, scenarios, sizing,
policy); this module only chooses the user's limits and wires the parts.

Personal limits come from a risk *style* the user picks, not from one rule
for everyone: the same 2% means $2 on a $100 account and $200 on $10,000,
and a careful holder wants alarms earlier than a bold one.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from core.portfolio_valuation import value_portfolio
from core.risk.exposure import DEFAULT_STABLECOIN_SYMBOLS, build_portfolio_exposure
from core.risk.metrics import compute_portfolio_metrics, returns_from_prices
from core.risk.models import (
    DrawdownState,
    PositionSizeRequest,
    RiskMetrics,
    RiskProfile,
    ScenarioReport,
    StressResult,
)
from core.risk.policy import DRAWDOWN_THRESHOLDS, drawdown_state_for
from core.risk.position_sizing import size_position
from core.risk.repository import DEFAULT_STATE_DB_PATH
from core.risk.scenarios import run_scenarios

SERVICE_VERSION = "1.0.0"

# Binance spot taker fee and a typical slippage, per side (percent).
DEFAULT_FEE_PCT = 0.10
DEFAULT_SLIPPAGE_PCT = 0.05
# Smallest order Binance spot accepts on most USDT pairs.
MIN_NOTIONAL_USD = 5.0
MAJOR_SYMBOLS = frozenset({"BTC", "ETH"})
VAR_LOOKBACK_DAYS = 90


@dataclass(frozen=True)
class RiskStyle:
    key: str
    risk_per_trade_pct: float      # loss accepted when a trade's stop is hit
    alt_limit_pct: float           # warning above this share for one altcoin
    major_limit_pct: float         # same for BTC / ETH (they swing less)
    drawdown_scale: float          # 1.0 = protocol at -5/-10/-15/-20 %


STYLES: dict[str, RiskStyle] = {
    "careful": RiskStyle("careful", 1.0, 30.0, 60.0, 0.6),    # -3/-6/-9/-12 %
    "balanced": RiskStyle("balanced", 2.0, 40.0, 70.0, 1.0),  # -5/-10/-15/-20 %
    "bold": RiskStyle("bold", 2.0, 60.0, 85.0, 1.5),          # -7.5/-15/-22.5/-30 %
}


def style_thresholds(style: RiskStyle) -> dict[DrawdownState, float]:
    """Drawdown (percent) at which each protocol state starts for this style."""
    return {state: threshold * style.drawdown_scale for threshold, state in DRAWDOWN_THRESHOLDS}


class RiskStyleStore:
    """The chosen style per user, in the shared state database."""

    def __init__(self, db_path: Path = DEFAULT_STATE_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS risk_styles ("
                " user_id INTEGER PRIMARY KEY, style TEXT NOT NULL, updated_at REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def get(self, user_id: int) -> Optional[RiskStyle]:
        with self._connect() as conn:
            row = conn.execute("SELECT style FROM risk_styles WHERE user_id = ?", (int(user_id),)).fetchone()
        return STYLES.get(row[0]) if row else None

    def set(self, user_id: int, key: str) -> RiskStyle:
        if key not in STYLES:
            raise ValueError(f"unknown risk style {key!r}")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO risk_styles (user_id, style, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET style = excluded.style, updated_at = excluded.updated_at",
                (int(user_id), key, time.time()),
            )
        return STYLES[key]

    def delete_user(self, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM risk_styles WHERE user_id = ?", (int(user_id),))


@dataclass(frozen=True)
class ConcentrationLine:
    symbol: str
    weight_pct: float
    limit_pct: float

    @property
    def over_limit(self) -> bool:
        return self.weight_pct > self.limit_pct


@dataclass(frozen=True)
class RiskPanel:
    style: RiskStyle
    value: float
    cost: float
    pnl_pct: Optional[float]
    missing_symbols: tuple[str, ...]
    concentration: tuple[ConcentrationLine, ...]
    stablecoin_pct: Optional[float]
    drawdown_pct: float
    state: DrawdownState
    next_state: Optional[DrawdownState]
    next_state_at_pct: Optional[float]
    var_pct: Optional[float]
    var_value: Optional[float]
    worst: Optional[StressResult]
    scenarios: ScenarioReport

    @property
    def risk_budget(self) -> float:
        return self.value * self.style.risk_per_trade_pct / 100.0


def limit_for(style: RiskStyle, symbol: str) -> float:
    return style.major_limit_pct if symbol.upper() in MAJOR_SYMBOLS else style.alt_limit_pct


def _daily_var(weights_pct: Mapping[str, float], prices_by_symbol: Mapping[str, list]) -> Optional[RiskMetrics]:
    """Historical one-day VaR over the last 90 daily returns, or None."""
    returns: dict[str, tuple[float, ...]] = {}
    for symbol in weights_pct:
        prices = prices_by_symbol.get(symbol)
        if symbol.upper() in DEFAULT_STABLECOIN_SYMBOLS and not prices:
            continue  # filled with zero returns below
        if not prices or len(prices) < 22:
            return None
        try:
            returns[symbol] = returns_from_prices(prices[-(VAR_LOOKBACK_DAYS + 1):])
        except ValueError:
            return None
    if not returns:
        return None
    length = min(len(series) for series in returns.values())
    aligned = {symbol: series[-length:] for symbol, series in returns.items()}
    for symbol in weights_pct:
        aligned.setdefault(symbol, (0.0,) * length)  # a dollar stablecoin barely moves
    total = math.fsum(weights_pct.values())
    weights = {symbol: weight / total for symbol, weight in weights_pct.items()}
    result = compute_portfolio_metrics(aligned, weights, min_observations=20)
    return result if isinstance(result, RiskMetrics) else None


def build_risk_panel(
    portfolio: Mapping[str, Mapping],
    markets: Mapping[str, Mapping],
    prices_by_coin_id: Mapping[str, list],
    style: RiskStyle,
    *,
    coin_id_resolver: Callable[[str], str],
) -> RiskPanel:
    valuation = value_portfolio(portfolio, markets, coin_id_resolver=coin_id_resolver)
    exposure = build_portfolio_exposure(valuation)
    value = exposure.known_value
    cost = exposure.known_cost

    weights = {p.symbol: p.weight_pct for p in exposure.positions if p.weight_pct}
    concentration = tuple(
        ConcentrationLine(symbol, weight, limit_for(style, symbol))
        for symbol, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:5]
    )

    drawdown = exposure.drawdown_from_cost_pct or 0.0
    state = drawdown_state_for(drawdown / style.drawdown_scale)
    thresholds = sorted(style_thresholds(style).items(), key=lambda item: item[1])
    upcoming = [(s, at) for s, at in thresholds if at > drawdown]
    next_state, next_at = upcoming[0] if upcoming else (None, None)

    prices_by_symbol = {
        symbol: prices_by_coin_id.get(coin_id_resolver(symbol)) for symbol in weights
    }
    metrics = _daily_var(weights, prices_by_symbol) if weights else None
    var_pct = metrics.historical_var_pct if metrics else None

    scenarios = run_scenarios(exposure)
    worst = max(scenarios.results, key=lambda r: r.loss_value, default=None) if scenarios.results else None

    return RiskPanel(
        style=style,
        value=value,
        cost=cost,
        pnl_pct=((value / cost - 1.0) * 100.0) if cost > 0 else None,
        missing_symbols=exposure.missing_symbols,
        concentration=concentration,
        stablecoin_pct=exposure.stablecoin_pct,
        drawdown_pct=drawdown,
        state=state,
        next_state=next_state,
        next_state_at_pct=next_at,
        var_pct=var_pct,
        var_value=(value * var_pct / 100.0) if var_pct is not None else None,
        worst=worst,
        scenarios=scenarios,
    )


@dataclass(frozen=True)
class TradeSizing:
    approved: bool
    reject_reasons: tuple[str, ...]
    account_value: float
    risk_budget: float
    position_value: float
    coins: float
    loss_at_stop: float
    stop_distance_pct: float
    reward_risk: Optional[float]
    profit_at_take: Optional[float]
    caps_applied: tuple[str, ...]


def size_trade(
    *,
    user_id: int,
    style: RiskStyle,
    symbol: str,
    entry: float,
    stop: float,
    take: Optional[float],
    account_value: float,
    existing_position_value: float = 0.0,
    state: DrawdownState = DrawdownState.NORMAL,
) -> TradeSizing:
    """How much to buy so that hitting the stop costs the style's risk budget."""
    if not (math.isfinite(entry) and math.isfinite(stop) and entry > 0 and stop > 0):
        raise ValueError("prices must be positive numbers")
    if stop >= entry:
        return TradeSizing(False, ("stop_not_below_entry",), account_value, 0.0, 0.0, 0.0, 0.0,
                           0.0, None, None, ())
    stop_pct = (entry - stop) / entry * 100.0
    profile = RiskProfile(
        user_id=int(user_id),
        account_currency="USD",
        risk_per_trade_pct=style.risk_per_trade_pct,
        max_single_coin_pct=limit_for(style, symbol),
        fee_pct=DEFAULT_FEE_PCT,
        slippage_pct=DEFAULT_SLIPPAGE_PCT,
    )
    result = size_position(
        profile,
        PositionSizeRequest(
            symbol=symbol.upper(),
            account_value=account_value,
            stop_distance_pct=stop_pct,
            existing_position_value=existing_position_value,
            min_notional=MIN_NOTIONAL_USD,
        ),
        drawdown_state=state,
    )
    value = result.position_value
    loss = value * result.effective_loss_pct / 100.0
    reward_risk = profit = None
    if take is not None and take > entry:
        reward_risk = (take - entry) / (entry - stop)
        costs = value * (2 * DEFAULT_FEE_PCT + DEFAULT_SLIPPAGE_PCT) / 100.0
        profit = value * (take - entry) / entry - costs
    return TradeSizing(
        approved=result.approved,
        reject_reasons=result.reject_reasons,
        account_value=account_value,
        risk_budget=result.risk_budget,
        position_value=value,
        coins=value / entry if value else 0.0,
        loss_at_stop=loss,
        stop_distance_pct=stop_pct,
        reward_risk=reward_risk,
        profit_at_take=profit,
        caps_applied=result.caps_applied,
    )
