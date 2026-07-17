"""Localized presentation for structured access-policy decisions."""

from __future__ import annotations

from core.access.access_manager import AccessDecision
from locales.translator import t


_DECISION_KEYS = {
    "feature_requires_premium": "errors.feature_requires_premium",
    "cooldown": "errors.access_cooldown",
    "coin_not_available": "errors.coin_not_available",
    "portfolio_limit": "errors.portfolio_limit",
}


def format_access_decision(decision: AccessDecision, user_id: int) -> str:
    if decision.allowed:
        return ""
    key = _DECISION_KEYS.get(decision.code, "tier.access_denied")
    params = dict(decision.params)
    if isinstance(params.get("allowed"), (tuple, list)):
        params["allowed"] = ", ".join(params["allowed"])
    return t(key, user_id=user_id, **params)
