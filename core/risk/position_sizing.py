"""Position sizing: risk budget over stop distance, capped by policy.

Base formula (documented invariant):

    position_value = risk_budget / effective_loss_pct

where ``risk_budget = account_value * risk_per_trade_pct * state and
volatility multipliers`` and ``effective_loss_pct`` is the stop distance
plus round-trip fees and slippage - the loss you actually take when the
stop fires, not the theoretical distance.

Hard honesty rules:

- no stop distance -> no position size (reject, not a default stop);
- fees/slippage not configured in the profile -> reject: assuming zero
  costs on a small account is exactly how €100 dies by a thousand cuts;
- the drawdown protocol gates new entries before any math runs;
- every cap that reduced the size is reported, every rejection has a
  machine-readable reason.

Leverage is off by default and outside this MVP.
"""

from __future__ import annotations

import math

from core.risk.models import (
    DrawdownState,
    PositionSizeRequest,
    PositionSizeResult,
    RiskProfile,
)
from core.risk.policy import (
    HIGH_VOLATILITY_RISK_MULTIPLIER,
    MAX_CLUSTER_PCT,
    RISK_BUDGET_MULTIPLIER,
    new_entry_allowed,
)

SIZING_VERSION = "1.0.0"


def _reject(reasons: list[str], *, risk_budget: float = 0.0,
            effective_loss_pct: float = 0.0,
            notes: tuple[str, ...] = ()) -> PositionSizeResult:
    return PositionSizeResult(
        approved=False,
        position_value=0.0,
        risk_budget=risk_budget,
        effective_loss_pct=effective_loss_pct,
        caps_applied=(),
        notes=notes,
        reject_reasons=tuple(reasons),
    )


def size_position(
    profile: RiskProfile,
    request: PositionSizeRequest,
    *,
    drawdown_state: DrawdownState = DrawdownState.NORMAL,
) -> PositionSizeResult:
    notes: list[str] = []

    # gates before any math
    if request.account_value is None or not math.isfinite(request.account_value) \
            or request.account_value <= 0:
        return _reject(["account_value_invalid"])

    stop = request.stop_distance_pct
    if stop is None or not math.isfinite(stop) or stop <= 0:
        return _reject(["no_stop_distance"])

    if profile.fee_pct is None or profile.slippage_pct is None:
        # Required inputs that materially affect sizing must be configured
        # or the trade is not sizable - costs are never assumed to be zero.
        return _reject(["cost_inputs_not_configured"])

    if not new_entry_allowed(drawdown_state, is_speculative=request.is_speculative):
        return _reject([f"new_entry_blocked_by_drawdown_state:{drawdown_state.value}"])

    # risk budget
    multiplier = RISK_BUDGET_MULTIPLIER[drawdown_state]
    if multiplier <= 0:
        return _reject([f"risk_budget_zero_in_state:{drawdown_state.value}"])
    if drawdown_state != DrawdownState.NORMAL:
        notes.append(f"risk_budget_multiplier:{multiplier}({drawdown_state.value})")

    if request.volatility_regime == "high":
        multiplier *= HIGH_VOLATILITY_RISK_MULTIPLIER
        notes.append(f"high_volatility_multiplier:{HIGH_VOLATILITY_RISK_MULTIPLIER}")
    elif request.volatility_regime is None:
        notes.append("volatility_regime_unknown")

    risk_budget = request.account_value * profile.risk_per_trade_pct / 100.0 * multiplier

    # loss at stop includes execution costs
    round_trip_cost_pct = 2.0 * profile.fee_pct + profile.slippage_pct
    effective_loss_pct = stop + round_trip_cost_pct
    if round_trip_cost_pct > 0:
        notes.append(f"round_trip_cost_pct:{round_trip_cost_pct:.3f}")

    raw_size = risk_budget / (effective_loss_pct / 100.0)

    # caps
    caps: list[str] = []
    size = raw_size

    concentration_headroom = (
        request.account_value * profile.max_single_coin_pct / 100.0
        - request.existing_position_value
    )
    if size > concentration_headroom:
        size = concentration_headroom
        caps.append(f"single_coin_concentration({profile.max_single_coin_pct:.0f}%)")

    if request.cluster_exposure_pct is not None:
        cluster_headroom = (
            (MAX_CLUSTER_PCT - request.cluster_exposure_pct) / 100.0
            * request.account_value
        )
        if size > cluster_headroom:
            size = cluster_headroom
            caps.append(f"correlated_cluster({MAX_CLUSTER_PCT:.0f}%)")
    else:
        notes.append("cluster_exposure_unknown")

    if request.available_capital is not None and size > request.available_capital:
        size = request.available_capital
        caps.append("available_capital")

    # Liquidity cap requires liquidity data (Stage 4); not evaluated here.
    notes.append("liquidity_cap_not_evaluated")

    # final validity
    if size <= 0:
        return _reject(
            ["no_headroom_under_caps"],
            risk_budget=risk_budget,
            effective_loss_pct=effective_loss_pct,
            notes=tuple(notes),
        )
    if request.min_notional is not None and size < request.min_notional:
        return _reject(
            [f"below_min_notional({request.min_notional:g})"],
            risk_budget=risk_budget,
            effective_loss_pct=effective_loss_pct,
            notes=tuple(notes),
        )

    return PositionSizeResult(
        approved=True,
        position_value=size,
        risk_budget=risk_budget,
        effective_loss_pct=effective_loss_pct,
        caps_applied=tuple(caps),
        notes=tuple(notes),
    )
