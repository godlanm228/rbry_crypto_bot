from core.market_signal import classify_price_momentum


def test_missing_changes_are_unavailable_not_fake_hold_confidence():
    signal = classify_price_momentum(None, 4.0)

    assert signal.direction == "unavailable"
    assert signal.strength == "unavailable"
    assert signal.translation_key == "analysis.signal_unavailable"


def test_strong_move_is_a_rule_class_not_a_probability():
    signal = classify_price_momentum(6.0, 11.0)

    assert signal.direction == "up"
    assert signal.strength == "strong"
    assert not hasattr(signal, "confidence")
