"""The /analyze card names the coin, says when data is loading and why the verdict."""

from types import SimpleNamespace

from core.analyzer_full import build_analysis_card


def _assessment(direction="sideways", components=()):
    return SimpleNamespace(
        is_available=True, scenario="neutral", direction=direction, risk_level="high",
        raw_analysis=SimpleNamespace(components=components), market_state=None,
    )


def _card(**overrides):
    kwargs = dict(
        coin_id="avalanche-2",
        market_row={"name": "Avalanche", "symbol": "avax",
                    "price_change_percentage_24h_in_currency": 1.8,
                    "price_change_percentage_7d_in_currency": -9.3},
        current_price=9.57, tech_indicators={}, assessment=_assessment(),
        history_ready=False, amount=None, avg_price=None, user_id=None,
    )
    kwargs.update(overrides)
    return "\n".join(build_analysis_card(**kwargs))


def test_title_uses_name_and_ticker_not_the_internal_id():
    text = _card()
    assert "Avalanche (AVAX)" in text
    assert "AVALANCHE-2" not in text
    assert "+1.8%" in text and "-9.3%" in text


def test_loading_history_is_said_out_loud():
    assert "⏳" in _card(history_ready=False)
    assert "⏳" not in _card(history_ready=True, tech_indicators={"rsi": 43.2})


def test_sideways_verdict_names_weak_signals_and_the_strongest_one():
    momentum = SimpleNamespace(name="momentum", available=True, value=-0.5, weight=0.27)
    text = _card(assessment=_assessment(components=(momentum,)))
    assert "слабые" in text or "weak" in text
