"""Portfolio exposure: what the user actually owns, with explicit coverage.

Builds on :mod:`core.portfolio_valuation` - this module never fetches prices
or resolves coins itself. Weights are computed over the value of positions
with known prices; missing-price positions are surfaced, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from core.portfolio_valuation import PortfolioValuation


DEFAULT_STABLECOIN_SYMBOLS = frozenset({
    "USDT", "USDC", "DAI", "TUSD", "FDUSD", "USDE", "PYUSD",
    "USDD", "GUSD", "USDP", "EURC", "EURT",
})


@dataclass(frozen=True)
class PositionExposure:
    symbol: str
    cost: float
    value: Optional[float]
    weight_pct: Optional[float]
    unrealized_pnl: Optional[float]
    unrealized_pnl_pct: Optional[float]
    is_stablecoin: bool

    @property
    def price_known(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class ClusterExposure:
    """Combined weight of positions whose returns move together."""

    symbols: tuple[str, ...]
    weight_pct: float


@dataclass(frozen=True)
class PortfolioExposure:
    positions: tuple[PositionExposure, ...]
    known_value: float
    known_cost: float
    missing_cost: float
    missing_symbols: tuple[str, ...]
    top1_pct: Optional[float]
    top3_pct: Optional[float]
    top5_pct: Optional[float]
    stablecoin_pct: Optional[float]
    drawdown_from_cost_pct: Optional[float]
    clusters: tuple[ClusterExposure, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing_symbols


def _cluster_symbols(
    weights_pct: Mapping[str, float],
    correlation: Mapping[str, Mapping[str, float]],
    threshold: float,
) -> tuple[ClusterExposure, ...]:
    """Union-find over positive-correlation links at/above the threshold."""
    symbols = [s for s in weights_pct if s in correlation]
    parent = {s: s for s in symbols}

    def find(s: str) -> str:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1:]:
            corr = correlation.get(sym_a, {}).get(sym_b)
            if corr is not None and corr >= threshold:
                parent[find(sym_a)] = find(sym_b)

    groups: dict[str, list[str]] = {}
    for symbol in symbols:
        groups.setdefault(find(symbol), []).append(symbol)

    clusters = [
        ClusterExposure(
            symbols=tuple(sorted(members)),
            weight_pct=sum(weights_pct[s] for s in members),
        )
        for members in groups.values()
        if len(members) >= 2
    ]
    clusters.sort(key=lambda c: c.weight_pct, reverse=True)
    return tuple(clusters)


def build_portfolio_exposure(
    valuation: PortfolioValuation,
    *,
    stablecoin_symbols: frozenset[str] = DEFAULT_STABLECOIN_SYMBOLS,
    correlation: Optional[Mapping[str, Mapping[str, float]]] = None,
    correlation_threshold: float = 0.8,
) -> PortfolioExposure:
    known_value = valuation.known_current_value
    weights_pct: dict[str, float] = {}

    positions = []
    for pos in valuation.positions:
        value = pos.current_value
        weight = None
        pnl = None
        pnl_pct = None
        if value is not None and known_value > 0:
            weight = value / known_value * 100.0
            weights_pct[pos.symbol] = weight
        if value is not None:
            pnl = value - pos.cost
            pnl_pct = pnl / pos.cost * 100.0 if pos.cost > 0 else None
        positions.append(
            PositionExposure(
                symbol=pos.symbol,
                cost=pos.cost,
                value=value,
                weight_pct=weight,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
                is_stablecoin=pos.symbol in stablecoin_symbols,
            )
        )

    top1 = top3 = top5 = stablecoin_pct = None
    if weights_pct:
        ranked = sorted(weights_pct.values(), reverse=True)
        top1 = ranked[0]
        top3 = sum(ranked[:3])
        top5 = sum(ranked[:5])
        stablecoin_pct = sum(
            w for s, w in weights_pct.items() if s in stablecoin_symbols
        )

    drawdown_from_cost = None
    if valuation.known_cost > 0:
        shortfall = 1.0 - known_value / valuation.known_cost
        drawdown_from_cost = max(0.0, shortfall) * 100.0

    clusters: tuple[ClusterExposure, ...] = ()
    if correlation and weights_pct:
        clusters = _cluster_symbols(weights_pct, correlation, correlation_threshold)

    return PortfolioExposure(
        positions=tuple(positions),
        known_value=known_value,
        known_cost=valuation.known_cost,
        missing_cost=valuation.missing_cost,
        missing_symbols=valuation.missing_symbols,
        top1_pct=top1,
        top3_pct=top3,
        top5_pct=top5,
        stablecoin_pct=stablecoin_pct,
        drawdown_from_cost_pct=drawdown_from_cost,
        clusters=clusters,
    )
