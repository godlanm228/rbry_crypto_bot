"""Immutable domain models for the risk engine.

Unit conventions, fixed for the whole package:

- fields suffixed ``_pct`` are percentages (``5.0`` means 5%);
- raw return series are fractions (``0.05`` means 5%) and only appear in
  function arguments, never in these public models;
- VaR and expected shortfall are reported as positive loss percentages.

Experience mode (beginner/advanced) is presentation state and intentionally
does not belong to :class:`RiskProfile`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional


class ActionClass(Enum):
    NO_ACTION = "no_action"
    MONITOR = "monitor"
    WAIT_FOR_CONFIRMATION = "wait_for_confirmation"
    CONSIDER_ENTRY = "consider_entry"
    HOLD = "hold"
    REDUCE_SIZE = "reduce_size"
    DO_NOT_ADD = "do_not_add"
    REBALANCE = "rebalance"
    DE_RISK_IMMEDIATELY = "de_risk_immediately"


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


def _require_range(name: str, value: float, low: float, high: float, *, low_open: bool = True) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if low_open:
        if not low < value <= high:
            raise ValueError(f"{name} must be in ({low}, {high}], got {value}")
    elif not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")


@dataclass(frozen=True)
class RiskProfile:
    """A user's risk policy inputs.

    ``fee_pct`` and ``slippage_pct`` stay ``None`` until the user configures
    them: position sizing must treat them as explicitly unavailable rather
    than assume zero cost.
    """

    user_id: int
    account_currency: str = "EUR"
    risk_per_trade_pct: float = 1.0
    max_single_coin_pct: float = 25.0
    max_speculative_pct: float = 15.0
    max_stablecoin_issuer_pct: float = 50.0
    daily_loss_limit_pct: Optional[float] = None
    weekly_loss_limit_pct: Optional[float] = None
    fee_pct: Optional[float] = None
    slippage_pct: Optional[float] = None
    updated_at: Optional[float] = None

    # The 2% ceiling implements the documented "1% conservative, 2%
    # aggressive cap" policy; it is a domain invariant, not a suggestion.
    MAX_RISK_PER_TRADE_PCT = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, int) or self.user_id <= 0:
            raise ValueError(f"user_id must be a positive integer, got {self.user_id!r}")
        currency = str(self.account_currency).strip().upper()
        if not 3 <= len(currency) <= 5 or not currency.isalpha():
            raise ValueError(f"account_currency looks invalid: {self.account_currency!r}")
        object.__setattr__(self, "account_currency", currency)

        _require_range("risk_per_trade_pct", self.risk_per_trade_pct, 0.0, self.MAX_RISK_PER_TRADE_PCT)
        _require_range("max_single_coin_pct", self.max_single_coin_pct, 0.0, 100.0)
        _require_range("max_speculative_pct", self.max_speculative_pct, 0.0, 100.0, low_open=False)
        _require_range("max_stablecoin_issuer_pct", self.max_stablecoin_issuer_pct, 0.0, 100.0)
        for name in ("daily_loss_limit_pct", "weekly_loss_limit_pct"):
            value = getattr(self, name)
            if value is not None:
                _require_range(name, value, 0.0, 100.0)
        for name in ("fee_pct", "slippage_pct"):
            value = getattr(self, name)
            if value is not None:
                _require_range(name, value, 0.0, 5.0, low_open=False)

    @classmethod
    def conservative_default(cls, user_id: int) -> "RiskProfile":
        """Owner-safe defaults; fee/slippage stay unconfigured on purpose."""
        return cls(user_id=user_id)


@dataclass(frozen=True)
class MetricsUnavailable:
    """Explicit refusal to compute metrics over insufficient data."""

    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ValueError("MetricsUnavailable requires at least one reason")


@dataclass(frozen=True)
class RiskMetrics:
    """Portfolio risk metrics over one aligned return history.

    All values assume a constant-mix portfolio at the supplied weights over
    the observed period ("interval" is the return period, e.g. ``daily``).
    ``historical_var_pct`` / ``historical_es_pct`` / ``parametric_var_pct``
    are positive loss percentages at ``confidence``.
    """

    observations: int
    interval: str
    confidence: float
    portfolio_volatility_pct: float
    historical_var_pct: float
    historical_es_pct: float
    parametric_var_pct: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    downside_deviation_pct: float
    risk_contribution_pct: Mapping[str, float]
