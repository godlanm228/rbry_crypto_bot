import os
import asyncio
import importlib


def _setup_env():
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
    os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")


def test_import_handlers_smoke():
    _setup_env()
    mod = importlib.import_module("modules.handlers")
    assert hasattr(mod, "portfolio_command")


def test_format_price_smoke():
    _setup_env()
    handlers = importlib.import_module("modules.handlers")
    assert handlers.format_price(1234.56).startswith("$")
    assert handlers.format_price(0.001).startswith("$")
    assert handlers.format_price(float("nan")) == "n/a"
    assert handlers.format_price(None) == "n/a"


def test_technical_indicators_smoke():
    import pandas as pd
    from core.analysis.technical_analysis import rsi, macd

    prices = pd.Series([100 + i for i in range(40)])
    r = rsi(prices, 14)
    macd_line, signal_line, _ = macd(prices)

    assert not r.empty
    assert not macd_line.empty
    assert len(macd_line) == len(signal_line)


def test_ai_legacy_assessment_smoke():
    _setup_env()
    ai_mod = importlib.import_module("core.ai_analytics")
    analyzer = ai_mod.RealAIAnalyzer()
    analyzer.synthesis_engine = None

    result = asyncio.run(
        analyzer.assess_growth_potential(
            coin_id="bitcoin",
            current_price=50000.0,
            tech_indicators={"rsi": 25},
            market_data={"price_change_percentage_24h": 12, "patterns": []},
        )
    )

    assert result.method == "rule_based"
    assert 0 <= result.rule_score <= 10
    assert result.scenario in ("positive", "neutral", "defensive")
    assert not hasattr(result, "target_price")
    assert not hasattr(result, "confidence")

