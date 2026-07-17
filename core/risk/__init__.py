"""Institutional-style portfolio risk engine.

R1: domain models, profile storage, exposure and metrics engines.
R2: stress scenarios, position sizing, and the versioned risk policy
(drawdown protocol, limits, allowed actions). The service facade and the
Telegram surface arrive with R3.
"""

from core.risk.models import (
    ActionClass,
    DrawdownState,
    LimitBreach,
    MetricsUnavailable,
    PositionSizeRequest,
    PositionSizeResult,
    ReverseStressResult,
    RiskMetrics,
    RiskProfile,
    ScenarioReport,
    Severity,
    StressResult,
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
from core.risk.policy import (
    POLICY_VERSION,
    allowed_actions,
    drawdown_state_for,
    evaluate_limits,
    new_entry_allowed,
)
from core.risk.position_sizing import size_position
from core.risk.repository import RiskProfileRepository
from core.risk.scenarios import (
    SCENARIO_CONFIG_VERSION,
    classify_symbol,
    reverse_stress,
    run_scenarios,
)

__all__ = [
    "ActionClass",
    "ClusterExposure",
    "DEFAULT_STABLECOIN_SYMBOLS",
    "DrawdownState",
    "LimitBreach",
    "MetricsUnavailable",
    "POLICY_VERSION",
    "PortfolioExposure",
    "PositionExposure",
    "PositionSizeRequest",
    "PositionSizeResult",
    "ReverseStressResult",
    "RiskMetrics",
    "RiskProfile",
    "RiskProfileRepository",
    "SCENARIO_CONFIG_VERSION",
    "ScenarioReport",
    "Severity",
    "StressResult",
    "allowed_actions",
    "build_portfolio_exposure",
    "classify_symbol",
    "compute_correlation_matrix",
    "compute_portfolio_metrics",
    "drawdown_state_for",
    "evaluate_limits",
    "new_entry_allowed",
    "realized_volatility",
    "returns_from_prices",
    "reverse_stress",
    "run_scenarios",
    "size_position",
]
