"""Institutional-style portfolio risk engine.

R1 scope: domain models, profile storage, exposure and metrics engines.
Scenarios, position sizing, policy, and the service facade arrive with their
own engines (R2/R3) so models are never defined ahead of verified behavior.
"""

from core.risk.models import (
    ActionClass,
    MetricsUnavailable,
    RiskMetrics,
    RiskProfile,
    Severity,
)
from core.risk.exposure import (
    ClusterExposure,
    DEFAULT_STABLECOIN_SYMBOLS,
    PortfolioExposure,
    PositionExposure,
    build_portfolio_exposure,
)
from core.risk.metrics import (
    compute_correlation_matrix,
    compute_portfolio_metrics,
    realized_volatility,
    returns_from_prices,
)
from core.risk.repository import RiskProfileRepository

__all__ = [
    "ActionClass",
    "ClusterExposure",
    "DEFAULT_STABLECOIN_SYMBOLS",
    "MetricsUnavailable",
    "PortfolioExposure",
    "PositionExposure",
    "RiskMetrics",
    "RiskProfile",
    "RiskProfileRepository",
    "Severity",
    "build_portfolio_exposure",
    "compute_correlation_matrix",
    "compute_portfolio_metrics",
    "realized_volatility",
    "returns_from_prices",
]
