"""Shared Telegram HTML helpers and price formatting."""

from core.formatting import as_html, format_price, plain


def test_command_names_are_not_italicised():
    text = "Use /add_entry COIN AMOUNT PRICE or /my_tier"
    assert "<i>" not in as_html(text)


def test_word_delimited_italic_and_bold():
    assert as_html("_As of 12:00 UTC_") == "<i>As of 12:00 UTC</i>"
    assert as_html("**Tier:** PREMIUM") == "<b>Tier:</b> PREMIUM"


def test_markup_is_escaped_before_tags():
    assert as_html("<script>**x**") == "&lt;script&gt;<b>x</b>"


def test_plain_drops_markers_inside_code_built_tags():
    assert plain("💼 **PORTFOLIO:**") == "💼 PORTFOLIO:"
    assert plain("_As of 12:00_") == "As of 12:00"


def test_format_price_keeps_significant_digits():
    assert format_price(0.00001234) == "$0.00001234"
    assert format_price(0.2364) == "$0.2364"
    assert format_price(83440) == "$83,440.00"
    assert format_price(float("nan")) == "n/a"
