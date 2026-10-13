"""News tone is reported only when a sentiment backend actually ran."""

from types import SimpleNamespace

from core.ai_analytics import ai_analyzer


def test_news_tone_is_none_without_backend(monkeypatch):
    stub = SimpleNamespace(status={"enabled": True, "available": False})
    monkeypatch.setattr(ai_analyzer, "finbert", stub)
    # The old code answered "neutral" here, even for "Bitcoin crashes 20%".
    assert ai_analyzer.analyze_news("Bitcoin crashes 20% after exchange hack") is None


def test_news_tone_label_from_available_backend(monkeypatch):
    stub = SimpleNamespace(
        status={"enabled": True, "available": True},
        analyze_sync=lambda text: ("negative", 0.93, "stub"),
    )
    monkeypatch.setattr(ai_analyzer, "finbert", stub)
    assert ai_analyzer.analyze_news("Bitcoin crashes 20%") == "negative"


def test_low_confidence_backend_label_is_neutral(monkeypatch):
    stub = SimpleNamespace(
        status={"enabled": True, "available": True},
        analyze_sync=lambda text: ("positive", 0.55, "stub"),
    )
    monkeypatch.setattr(ai_analyzer, "finbert", stub)
    assert ai_analyzer.analyze_news("ETF inflows") == "neutral"
