"""Pure portfolio risk mathematics.

No network, no storage, no defaults for missing data: callers pass aligned
return series and either get metrics or an explicit
:class:`~core.risk.models.MetricsUnavailable` with reasons.

Conventions (documented so fixtures can be hand-checked):

- returns are simple fractional returns ``p_t / p_{t-1} - 1``;
- volatility is the sample standard deviation (ddof=1) of portfolio returns;
- historical VaR at confidence ``c`` over ``n`` observations is the k-th
  worst portfolio return with ``k = max(1, ceil(n * (1 - c)))``, reported as
  a positive loss percentage; expected shortfall is the mean of those k
  worst returns;
- parametric VaR is ``z * sigma - mean`` with a fixed z-score table, shown
  only as a cross-check next to the historical figure;
- drawdowns come from the wealth index of compounded portfolio returns
  (constant-mix approximation at the supplied weights);
- downside deviation uses target 0 and divides by n (population form).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence, Union

from core.risk.models import MetricsUnavailable, RiskMetrics


# One-sided z-scores for the supported confidence levels.
_Z_SCORES = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}

_WEIGHT_SUM_TOLERANCE = 1e-6


def returns_from_prices(prices: Sequence[float]) -> tuple[float, ...]:
    """Simple returns from a chronological price series.

    Raises ``ValueError`` on non-finite or non-positive prices - garbage
    input must fail loudly instead of producing a plausible-looking series.
    """
    cleaned = []
    for value in prices:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"prices must be finite and positive, got {value!r}")
        cleaned.append(number)
    if len(cleaned) < 2:
        raise ValueError("need at least two prices to compute returns")
    return tuple(cleaned[i] / cleaned[i - 1] - 1.0 for i in range(1, len(cleaned)))


def realized_volatility(returns: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1) of one return series."""
    if len(returns) < 2:
        raise ValueError("need at least two returns for volatility")
    mean = math.fsum(returns) / len(returns)
    variance = math.fsum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance)


def _covariance(a: Sequence[float], b: Sequence[float]) -> float:
    mean_a = math.fsum(a) / len(a)
    mean_b = math.fsum(b) / len(b)
    return math.fsum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / (len(a) - 1)


def _validate_series(
    returns_by_symbol: Mapping[str, Sequence[float]],
    min_observations: int,
) -> tuple[int, tuple[str, ...]]:
    """Return (observation count, reasons); reasons non-empty means refuse."""
    reasons = []
    if not returns_by_symbol:
        return 0, ("no return series supplied",)

    lengths = {symbol: len(series) for symbol, series in returns_by_symbol.items()}
    distinct = set(lengths.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{s}={n}" for s, n in sorted(lengths.items()))
        reasons.append(f"return series are not aligned: {detail}")
    observations = min(distinct) if distinct else 0
    if observations < min_observations:
        reasons.append(
            f"insufficient history: {observations} observations, need {min_observations}"
        )
    for symbol, series in returns_by_symbol.items():
        if any(not math.isfinite(r) for r in series):
            reasons.append(f"non-finite return in series {symbol}")
    return observations, tuple(reasons)


def compute_correlation_matrix(
    returns_by_symbol: Mapping[str, Sequence[float]],
    *,
    min_observations: int = 20,
) -> Union[dict[str, dict[str, float]], MetricsUnavailable]:
    """Pearson correlations for every symbol pair, or an explicit refusal.

    Symbols with zero variance are excluded from the result (correlation is
    undefined for them) - callers see their absence, not a fake value.
    """
    _, reasons = _validate_series(returns_by_symbol, min_observations)
    if reasons:
        return MetricsUnavailable(reasons)

    variances = {
        symbol: _covariance(series, series)
        for symbol, series in returns_by_symbol.items()
    }
    symbols = [s for s in returns_by_symbol if variances[s] > 0]
    matrix: dict[str, dict[str, float]] = {s: {} for s in symbols}
    for i, sym_a in enumerate(symbols):
        matrix[sym_a][sym_a] = 1.0
        for sym_b in symbols[i + 1:]:
            corr = _covariance(
                returns_by_symbol[sym_a], returns_by_symbol[sym_b]
            ) / math.sqrt(variances[sym_a] * variances[sym_b])
            matrix[sym_a][sym_b] = corr
            matrix[sym_b][sym_a] = corr
    return matrix


def compute_portfolio_metrics(
    returns_by_symbol: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    *,
    confidence: float = 0.95,
    min_observations: int = 20,
    interval: str = "daily",
) -> Union[RiskMetrics, MetricsUnavailable]:
    """Portfolio metrics over aligned series, or an explicit refusal.

    Data problems (missing series, misalignment, thin history) produce
    ``MetricsUnavailable``. Caller mistakes (bad weights, unsupported
    confidence) raise ``ValueError``.
    """
    if confidence not in _Z_SCORES:
        raise ValueError(f"confidence must be one of {sorted(_Z_SCORES)}, got {confidence}")
    if not weights:
        raise ValueError("weights must not be empty")
    for symbol, weight in weights.items():
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {symbol} must be finite and >= 0, got {weight}")
    if abs(math.fsum(weights.values()) - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"weights must sum to 1, got {math.fsum(weights.values())}")

    missing = sorted(set(weights) - set(returns_by_symbol))
    if missing:
        return MetricsUnavailable(
            tuple(f"no return history for held position {s}" for s in missing)
        )

    relevant = {symbol: returns_by_symbol[symbol] for symbol in weights}
    observations, reasons = _validate_series(relevant, min_observations)
    if reasons:
        return MetricsUnavailable(reasons)

    ordered = sorted(weights)
    portfolio_returns = [
        math.fsum(weights[s] * relevant[s][t] for s in ordered)
        for t in range(observations)
    ]

    mean = math.fsum(portfolio_returns) / observations
    volatility = realized_volatility(portfolio_returns)

    k = max(1, math.ceil(observations * (1.0 - confidence)))
    worst = sorted(portfolio_returns)[:k]
    historical_var = -worst[-1]
    historical_es = -math.fsum(worst) / k
    parametric_var = _Z_SCORES[confidence] * volatility - mean

    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    current_drawdown = 0.0
    for r in portfolio_returns:
        wealth *= 1.0 + r
        peak = max(peak, wealth)
        current_drawdown = 1.0 - wealth / peak
        max_drawdown = max(max_drawdown, current_drawdown)

    downside = math.sqrt(
        math.fsum(min(r, 0.0) ** 2 for r in portfolio_returns) / observations
    )

    portfolio_variance = _covariance(portfolio_returns, portfolio_returns)
    contributions: dict[str, float] = {}
    if portfolio_variance > 0:
        for symbol in ordered:
            marginal = math.fsum(
                weights[other] * _covariance(relevant[symbol], relevant[other])
                for other in ordered
            )
            contributions[symbol] = weights[symbol] * marginal / portfolio_variance * 100.0

    return RiskMetrics(
        observations=observations,
        interval=interval,
        confidence=confidence,
        portfolio_volatility_pct=volatility * 100.0,
        historical_var_pct=historical_var * 100.0,
        historical_es_pct=historical_es * 100.0,
        parametric_var_pct=parametric_var * 100.0,
        max_drawdown_pct=max_drawdown * 100.0,
        current_drawdown_pct=current_drawdown * 100.0,
        downside_deviation_pct=downside * 100.0,
        risk_contribution_pct=contributions,
    )
