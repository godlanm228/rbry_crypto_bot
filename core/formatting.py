"""User-facing number formatting shared by handlers, alerts and analysis."""

import html
import math
import re


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


_BOLD = re.compile(r"\*\*(.+?)\*\*")
# Italic only for _word-delimited_ spans: "/add_entry ... /my_tier" must stay
# plain text (the old pattern italicised everything between two underscores).
_ITALIC = re.compile(r"(?<![\w/])_(?=\S)(.+?)(?<=\S)_(?!\w)")


def as_html(text: str) -> str:
    """Escape for Telegram HTML and turn **bold** / _italic_ into tags."""
    txt = html.escape(str(text))
    txt = _BOLD.sub(r"<b>\1</b>", txt)
    return _ITALIC.sub(r"<i>\1</i>", txt)


def plain(text: str) -> str:
    """Escape for Telegram HTML and drop markdown markers (caller adds tags)."""
    txt = html.escape(str(text))
    txt = _BOLD.sub(r"\1", txt)
    return _ITALIC.sub(r"\1", txt)
