"""Deterministic stress scenarios over the current portfolio exposure.

Versioned assumptions (SCENARIO_CONFIG_VERSION). Betas are deliberately
simple, documented multipliers — not fitted parameters; historical VaR/ES
in metrics.py provide the data-driven counterpart. Positions without a
known price are excluded and REPORTED, never treated as loss-free.

Asset classes for shocks: BTC, ETH, stablecoins (registry set), everything
else is an altcoin. Losses are positive numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from core.risk.exposure import DEFAULT_STABLECOIN_SYMBOLS, PortfolioExposure
from core.risk.models import ReverseStressResult, ScenarioReport, StressResult

SCENARIO_CONFIG_VERSION = "1.0.0"

ASSET_BTC = "btc"
ASSET_ETH = "eth"
ASSET_STABLE = "stablecoin"
ASSET_ALT = "altcoin"


def classify_symbol(symbol: str,
                    stablecoins: frozenset[str] = DEFAULT_STABLECOIN_SYMBOLS) -> str:
    upper = symbol.upper()
    if upper == "BTC":
        return ASSET_BTC
    if upper == "ETH":
        return ASSET_ETH
    if upper in stablecoins:
        return ASSET_STABLE
    return ASSET_ALT


@dataclass(frozen=True)
class ShockSet:
    """Percent shocks per asset class; negative means price drop."""

    key: str
    description: str
    btc: float = 0.0
    eth: float = 0.0
    alt: float = 0.0
    stable: float = 0.0

    def shock_for(self, asset_class: str) -> float:
        return {
            ASSET_BTC: self.btc,
            ASSET_ETH: self.eth,
            ASSET_ALT: self.alt,
            ASSET_STABLE: self.stable,
        }[asset_class]


# Documented beta assumptions:
# - BTC crash drags ETH at 1.1x and altcoins at 1.3x;
# - ETH crash drags BTC at 0.3x and altcoins at 0.6x;
# - panic pushes non-stable correlations toward 0.9 → uniform shock.
SCENARIOS: tuple[ShockSet, ...] = (
    ShockSet("btc_-10", "BTC -10% (ETH -11%, alts -13%)", btc=-10, eth=-11, alt=-13),
    ShockSet("btc_-20", "BTC -20% (ETH -22%, alts -26%)", btc=-20, eth=-22, alt=-26),
    ShockSet("btc_-30", "BTC -30% (ETH -33%, alts -39%)", btc=-30, eth=-33, alt=-39),
    ShockSet("eth_-15", "ETH -15% (BTC -4.5%, alts -9%)", eth=-15, btc=-4.5, alt=-9),
    ShockSet("eth_-25", "ETH -25% (BTC -7.5%, alts -15%)", eth=-25, btc=-7.5, alt=-15),
    ShockSet("altcoin_beta_crash", "Altcoins -35%, ETH -20%, BTC -15%",
             alt=-35, eth=-20, btc=-15),
    ShockSet("correlation_panic", "Correlations to 0.9: all non-stables -25%",
             btc=-25, eth=-25, alt=-25),
    ShockSet("stable_depeg_-2", "Stablecoins -2%", stable=-2),
    ShockSet("stable_depeg_-5", "Stablecoins -5%", stable=-5),
    ShockSet("stable_depeg_-10", "Stablecoins -10%", stable=-10),
    ShockSet("liquidity_haircut", "Altcoin liquidity/slippage haircut -20%", alt=-20),
)

LARGEST_HOLDING_SHOCK_PCT = -40.0
REVERSE_STRESS_TARGETS_PCT = (10.0, 20.0, 30.0)


def _priced_positions(exposure: PortfolioExposure):
    return [p for p in exposure.positions if p.value is not None and p.value > 0]


def _apply_shock_set(exposure: PortfolioExposure, shocks: ShockSet) -> StressResult:
    losses = []
    for position in _priced_positions(exposure):
        shock = shocks.shock_for(classify_symbol(position.symbol))
        if shock < 0:
            losses.append((position.symbol, position.value * (-shock) / 100.0))
    total_loss = sum(loss for _, loss in losses)
    contributors = tuple(sorted(losses, key=lambda item: item[1], reverse=True)[:3])
    return StressResult(
        scenario_key=shocks.key,
        description=shocks.description,
        loss_value=total_loss,
        loss_pct=total_loss / exposure.known_value * 100.0,
        contributors=contributors,
        excluded_symbols=exposure.missing_symbols,
    )


def _largest_holding_scenario(exposure: PortfolioExposure) -> Optional[StressResult]:
    non_stable = [
        p for p in _priced_positions(exposure)
        if classify_symbol(p.symbol) != ASSET_STABLE
    ]
    if not non_stable:
        return None
    biggest = max(non_stable, key=lambda p: p.value)
    loss = biggest.value * (-LARGEST_HOLDING_SHOCK_PCT) / 100.0
    return StressResult(
        scenario_key="largest_holding_-40",
        description=f"Largest holding {biggest.symbol} -40%",
        loss_value=loss,
        loss_pct=loss / exposure.known_value * 100.0,
        contributors=((biggest.symbol, loss),),
        excluded_symbols=exposure.missing_symbols,
    )


def reverse_stress(
    exposure: PortfolioExposure,
    targets_pct: Sequence[float] = REVERSE_STRESS_TARGETS_PCT,
) -> tuple[ReverseStressResult, ...]:
    """Smallest uniform shock on non-stable holdings for each target loss.

    With non-stable share s of known value, a uniform shock x produces a
    portfolio loss of x*s, so x = target / s. A shock above 100% means the
    target loss is unreachable through non-stable holdings alone.
    """
    known = exposure.known_value
    non_stable_value = sum(
        p.value for p in _priced_positions(exposure)
        if classify_symbol(p.symbol) != ASSET_STABLE
    )
    results = []
    for target in targets_pct:
        if known <= 0 or non_stable_value <= 0:
            results.append(ReverseStressResult(
                target_loss_pct=target,
                required_shock_pct=None,
                reason="no_priced_non_stable_holdings",
            ))
            continue
        share = non_stable_value / known
        required = target / share
        if required > 100.0:
            results.append(ReverseStressResult(
                target_loss_pct=target,
                required_shock_pct=None,
                reason="unreachable_via_non_stable_holdings",
            ))
        else:
            results.append(ReverseStressResult(
                target_loss_pct=target,
                required_shock_pct=required,
            ))
    return tuple(results)


def run_scenarios(exposure: PortfolioExposure) -> ScenarioReport:
    """All deterministic scenarios over currently priced holdings."""
    if exposure.known_value <= 0:
        return ScenarioReport(
            results=(),
            reverse=(),
            excluded_symbols=exposure.missing_symbols,
            unavailable_reasons=("no_priced_holdings",),
            config_version=SCENARIO_CONFIG_VERSION,
        )

    results = [_apply_shock_set(exposure, shocks) for shocks in SCENARIOS]
    largest = _largest_holding_scenario(exposure)
    if largest is not None:
        results.append(largest)

    return ScenarioReport(
        results=tuple(results),
        reverse=reverse_stress(exposure),
        excluded_symbols=exposure.missing_symbols,
        unavailable_reasons=(),
        config_version=SCENARIO_CONFIG_VERSION,
    )
