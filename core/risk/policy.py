"""Versioned risk policy: drawdown protocol, limits, allowed actions.

This module centralizes the rules that used to be scattered `if/else`
candidates: handlers and renderers must translate these results, never
re-derive them.
"""

from __future__ import annotations

from core.risk.exposure import PortfolioExposure
from core.risk.models import (
    ActionClass,
    DrawdownState,
    LimitBreach,
    RiskProfile,
    Severity,
)

POLICY_VERSION = "1.0.0"

# Drawdown protocol thresholds (percent below cost basis / peak).
DRAWDOWN_THRESHOLDS = (
    (20.0, DrawdownState.DEFENSIVE),
    (15.0, DrawdownState.CONCENTRATION_REVIEW),
    (10.0, DrawdownState.RISK_FREEZE),
    (5.0, DrawdownState.CAUTION),
)

# Multiplier applied to the per-trade risk budget for NEW entries.
RISK_BUDGET_MULTIPLIER = {
    DrawdownState.NORMAL: 1.0,
    DrawdownState.CAUTION: 0.5,
    DrawdownState.RISK_FREEZE: 0.25,
    DrawdownState.CONCENTRATION_REVIEW: 0.25,
    DrawdownState.DEFENSIVE: 0.0,
}

# High volatility regime halves the risk budget; unknown regime is left
# unchanged but reported, never silently treated as low volatility.
HIGH_VOLATILITY_RISK_MULTIPLIER = 0.5

# Concentration hard-warning line and correlated-cluster cap (policy-level;
# per-user overrides can arrive with a later profile migration).
SINGLE_COIN_HARD_WARNING_PCT = 40.0
MAX_CLUSTER_PCT = 40.0

ALLOWED_ACTIONS = {
    DrawdownState.NORMAL: (
        ActionClass.NO_ACTION, ActionClass.MONITOR, ActionClass.WAIT_FOR_CONFIRMATION,
        ActionClass.CONSIDER_ENTRY, ActionClass.HOLD, ActionClass.REDUCE_SIZE,
        ActionClass.DO_NOT_ADD, ActionClass.REBALANCE, ActionClass.DE_RISK_IMMEDIATELY,
    ),
    DrawdownState.CAUTION: (
        ActionClass.MONITOR, ActionClass.WAIT_FOR_CONFIRMATION,
        ActionClass.CONSIDER_ENTRY, ActionClass.HOLD, ActionClass.REDUCE_SIZE,
        ActionClass.DO_NOT_ADD, ActionClass.REBALANCE, ActionClass.DE_RISK_IMMEDIATELY,
    ),
    DrawdownState.RISK_FREEZE: (
        ActionClass.MONITOR, ActionClass.HOLD, ActionClass.REDUCE_SIZE,
        ActionClass.DO_NOT_ADD, ActionClass.REBALANCE, ActionClass.DE_RISK_IMMEDIATELY,
    ),
    DrawdownState.CONCENTRATION_REVIEW: (
        ActionClass.MONITOR, ActionClass.HOLD, ActionClass.REDUCE_SIZE,
        ActionClass.DO_NOT_ADD, ActionClass.REBALANCE, ActionClass.DE_RISK_IMMEDIATELY,
    ),
    DrawdownState.DEFENSIVE: (
        ActionClass.MONITOR, ActionClass.REDUCE_SIZE, ActionClass.DO_NOT_ADD,
        ActionClass.DE_RISK_IMMEDIATELY,
    ),
}

_DRAWDOWN_SEVERITY = {
    DrawdownState.CAUTION: Severity.WARNING,
    DrawdownState.RISK_FREEZE: Severity.HIGH,
    DrawdownState.CONCENTRATION_REVIEW: Severity.HIGH,
    DrawdownState.DEFENSIVE: Severity.CRITICAL,
}
_DRAWDOWN_ACTION = {
    DrawdownState.CAUTION: ActionClass.REDUCE_SIZE,
    DrawdownState.RISK_FREEZE: ActionClass.DO_NOT_ADD,
    DrawdownState.CONCENTRATION_REVIEW: ActionClass.REBALANCE,
    DrawdownState.DEFENSIVE: ActionClass.DE_RISK_IMMEDIATELY,
}
_DRAWDOWN_THRESHOLD = {state: threshold for threshold, state in DRAWDOWN_THRESHOLDS}


# Float dust from valuation arithmetic (e.g. 19.999999999999996) must not
# soften the protocol state at an exact threshold.
_THRESHOLD_EPSILON = 1e-9


def drawdown_state_for(drawdown_pct: float) -> DrawdownState:
    for threshold, state in DRAWDOWN_THRESHOLDS:
        if drawdown_pct >= threshold - _THRESHOLD_EPSILON:
            return state
    return DrawdownState.NORMAL


def new_entry_allowed(state: DrawdownState, *, is_speculative: bool) -> bool:
    """The drawdown protocol's gate for opening NEW positions."""
    if state == DrawdownState.DEFENSIVE:
        return False
    if is_speculative and state in (
        DrawdownState.RISK_FREEZE, DrawdownState.CONCENTRATION_REVIEW
    ):
        return False
    return True


def allowed_actions(state: DrawdownState) -> tuple[ActionClass, ...]:
    return ALLOWED_ACTIONS[state]


def evaluate_limits(profile: RiskProfile, exposure: PortfolioExposure) -> tuple[LimitBreach, ...]:
    """Check the exposure against the user's limits and policy caps.

    Note: the speculative-bucket limit needs the coin taxonomy (Stage 4 of
    the risk plan) and is intentionally NOT evaluated here - evaluating it
    with a guessed classification would be invented data.
    """
    breaches = []

    for position in exposure.positions:
        weight = position.weight_pct
        if weight is None or position.is_stablecoin:
            continue
        if weight > profile.max_single_coin_pct:
            hard = weight > SINGLE_COIN_HARD_WARNING_PCT
            breaches.append(LimitBreach(
                code="single_coin_concentration",
                severity=Severity.HIGH if hard else Severity.WARNING,
                subject=position.symbol,
                observed_pct=weight,
                limit_pct=profile.max_single_coin_pct,
                suggested_action=ActionClass.REDUCE_SIZE if hard else ActionClass.DO_NOT_ADD,
            ))

    if exposure.stablecoin_pct is not None and \
            exposure.stablecoin_pct > profile.max_stablecoin_issuer_pct:
        breaches.append(LimitBreach(
            code="stablecoin_bucket",
            severity=Severity.WARNING,
            subject="stablecoins",
            observed_pct=exposure.stablecoin_pct,
            limit_pct=profile.max_stablecoin_issuer_pct,
            suggested_action=ActionClass.REBALANCE,
        ))

    for cluster in exposure.clusters:
        if cluster.weight_pct > MAX_CLUSTER_PCT:
            hard = cluster.weight_pct > MAX_CLUSTER_PCT * 1.25
            breaches.append(LimitBreach(
                code="correlated_cluster",
                severity=Severity.HIGH if hard else Severity.WARNING,
                subject="cluster:" + "+".join(cluster.symbols),
                observed_pct=cluster.weight_pct,
                limit_pct=MAX_CLUSTER_PCT,
                suggested_action=ActionClass.REDUCE_SIZE if hard else ActionClass.DO_NOT_ADD,
            ))

    drawdown = exposure.drawdown_from_cost_pct
    if drawdown is not None:
        state = drawdown_state_for(drawdown)
        if state != DrawdownState.NORMAL:
            breaches.append(LimitBreach(
                code="drawdown_protocol",
                severity=_DRAWDOWN_SEVERITY[state],
                subject="portfolio",
                observed_pct=drawdown,
                limit_pct=_DRAWDOWN_THRESHOLD[state],
                suggested_action=_DRAWDOWN_ACTION[state],
            ))

    return tuple(breaches)
