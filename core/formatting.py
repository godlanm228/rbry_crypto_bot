"""User-facing number formatting shared by handlers, alerts and analysis."""

import math


def format_price(price) -> str:
    """USD price with precision that fits its magnitude ($0.00001234, $83,440.00)."""
    try:
        price = float(price)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(price) or price < 0:
        return "n/a"
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.2f}"
    elif price >= 0.01:
        return f"${price:.4f}"
    elif price >= 0.0001:
        return f"${price:.6f}"
    else:
        return f"${price:.8f}"
