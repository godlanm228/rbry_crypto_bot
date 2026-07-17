"""Pure portfolio valuation with explicit data coverage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional


@dataclass(frozen=True)
class PositionValuation:
    symbol: str
    amount: float
    entry_price: float
    current_price: Optional[float]

    @property
    def cost(self) -> float:
        return self.amount * self.entry_price

    @property
    def current_value(self) -> Optional[float]:
        return self.amount * self.current_price if self.current_price is not None else None


@dataclass(frozen=True)
class PortfolioValuation:
    positions: tuple[PositionValuation, ...]
    known_cost: float
    missing_cost: float
    known_current_value: float
    missing_symbols: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing_symbols

    @property
    def total_invested(self) -> float:
        return self.known_cost + self.missing_cost

    @property
    def total_current_value(self) -> Optional[float]:
        return self.known_current_value if self.is_complete else None

    @property
    def total_pnl(self) -> Optional[float]:
        if not self.is_complete:
            return None
        return self.known_current_value - self.total_invested

    @property
    def total_pnl_pct(self) -> Optional[float]:
        pnl = self.total_pnl
        if pnl is None or self.total_invested <= 0:
            return None
        return pnl / self.total_invested * 100.0


def _finite_positive(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def value_portfolio(
    portfolio: Mapping[str, Mapping],
    markets: Mapping[str, Mapping],
    *,
    coin_id_resolver: Callable[[str], str],
) -> PortfolioValuation:
    positions = []
    known_cost = 0.0
    missing_cost = 0.0
    known_current_value = 0.0
    missing_symbols = []

    for raw_symbol, data in portfolio.items():
        symbol = str(raw_symbol).upper()
        amount = _finite_positive(data.get("amount")) or 0.0
        entry = _finite_positive(data.get("buy_price", data.get("price"))) or 0.0
        coin_id = coin_id_resolver(symbol)
        market_row = markets.get(coin_id, {}) if markets else {}
        current = _finite_positive(market_row.get("current_price"))
        position = PositionValuation(symbol, amount, entry, current)
        positions.append(position)

        if current is None:
            missing_cost += position.cost
            missing_symbols.append(symbol)
        else:
            known_cost += position.cost
            known_current_value += position.current_value or 0.0

    return PortfolioValuation(
        positions=tuple(positions),
        known_cost=known_cost,
        missing_cost=missing_cost,
        known_current_value=known_current_value,
        missing_symbols=tuple(missing_symbols),
    )
