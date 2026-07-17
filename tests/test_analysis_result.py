import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")
from core.ai_analytics import RealAIAnalyzer
from core.analysis_result import AnalysisResult
from core.analyzer_full import analyze_coin_full


def _legacy_analyzer():
    analyzer = RealAIAnalyzer()
    analyzer.synthesis_engine = None
    analyzer.finbert = None
    return analyzer


def test_assessment_without_evidence_is_unavailable_not_a_fake_hold():
    result = asyncio.run(
        _legacy_analyzer().assess_growth_potential("bitcoin", 50_000.0, {}, {})
    )

    assert result.scenario == "unavailable"
    assert result.rule_score is None
    assert result.risk_level == "unknown"


def test_rule_assessment_has_no_probability_or_price_target():
    result = asyncio.run(
        _legacy_analyzer().assess_growth_potential(
            "bitcoin",
            50_000.0,
            {"rsi": 25},
            {"price_change_percentage_24h": 12, "patterns": []},
        )
    )

    assert result.method == "rule_based"
    assert result.rule_score == 8
    assert result.scenario == "positive"
    assert not hasattr(result, "confidence")
    assert not hasattr(result, "target_price")


def test_public_analysis_does_not_claim_ai_forecast_or_price_target(monkeypatch):
    async def assessment(*args, **kwargs):
        return AnalysisResult(
            scenario="positive",
            risk_level="medium",
            method="rule_based",
            rule_score=8,
            evidence=("rsi_oversold",),
        )

    monkeypatch.setattr(
        "core.analyzer_full.ai_analyzer.assess_growth_potential",
        assessment,
    )
    context = SimpleNamespace(application=SimpleNamespace(bot_data={}))

    text = asyncio.run(
        analyze_coin_full(
            "bitcoin",
            50_000.0,
            {"amount": 0.1, "buy_price": 45_000.0},
            {"price_change_percentage_24h": 3.0},
            context,
            user_id=123456,
            show_explanations=False,
        )
    )

    assert "AI Прогноз" not in text
    assert "Целевая цена" not in text
    assert "не вероятность" in text


def test_synthesis_can_be_disabled_by_tier_policy(monkeypatch):
    analyzer = _legacy_analyzer()
    analyzer.synthesis_engine = object()

    async def synthesized(coin_id):
        return AnalysisResult(
            scenario="positive",
            risk_level="medium",
            method="multi_factor_synthesis",
        )

    monkeypatch.setattr(analyzer, "assess_growth_potential_with_synthesis", synthesized)
    result = asyncio.run(
        analyzer.assess_growth_potential(
            "bitcoin",
            50_000.0,
            {"rsi": 50},
            {"price_change_percentage_24h": 1.0},
            allow_synthesis=False,
        )
    )

    assert result.method == "rule_based"


def test_synthesis_contract_uses_real_engine_fields_without_legacy_targets():
    analyzer = _legacy_analyzer()
    analyzer.synthesis_engine = SimpleNamespace()
    analyzer.market_state_builder = SimpleNamespace(
        build_market_state=AsyncMock(return_value=object())
    )
    analyzer.synthesis_engine.synthesize = AsyncMock(
        return_value=SimpleNamespace(
            signal_type="buy",
            risk_level="Medium",
            component_scores={"sentiment": 0.7},
            unified_reasoning="RSI and momentum agree",
        )
    )

    result = asyncio.run(analyzer.assess_growth_potential_with_synthesis("bitcoin"))

    assert result.method == "multi_factor_synthesis"
    assert result.scenario == "positive"
    assert result.evidence == ("RSI and momentum agree",)
    assert not hasattr(result, "target_price")


def test_missing_market_price_returns_updating_message():
    context = SimpleNamespace(application=SimpleNamespace(bot_data={}))

    text = asyncio.run(
        analyze_coin_full(
            "bitcoin",
            None,
            {},
            {},
            context,
            user_id=123456,
            show_explanations=False,
        )
    )

    assert "обновляются" in text
    assert "$0.00" not in text
