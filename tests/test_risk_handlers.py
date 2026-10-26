"""🛡 Risk flow: style first, then the panel; the calculator answers in dollars."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import modules.risk_handlers as rh
from core.risk.service import STYLES, RiskStyleStore, build_risk_panel

RESOLVE = {"BTC": "bitcoin", "ETH": "ethereum"}.get
PORTFOLIO = {"BTC": {"amount": 0.06, "buy_price": 81945.0}, "ETH": {"amount": 0.53, "buy_price": 2400.78}}
MARKETS = {"bitcoin": {"current_price": 80_000.0}, "ethereum": {"current_price": 2_350.0}}


@pytest.fixture
def risk_env(tmp_path, monkeypatch):
    monkeypatch.setattr(rh, "_store", RiskStyleStore(tmp_path / "state.db"))

    async def fake_panel(context, user_id, style):
        panel = build_risk_panel(PORTFOLIO, MARKETS, {}, style, coin_id_resolver=RESOLVE)
        return PORTFOLIO, MARKETS, panel

    monkeypatch.setattr(rh, "_panel", fake_panel)
    return SimpleNamespace(user_data={}, application=None)


def _query(data, qid):
    return SimpleNamespace(id=qid, data=data, from_user=SimpleNamespace(id=42),
                           answer=AsyncMock(), edit_message_text=AsyncMock())


def _press(data, context, qid):
    query = _query(data, qid)
    asyncio.run(rh.risk_callback(SimpleNamespace(callback_query=query), context))
    return query.edit_message_text.await_args.args[0]


def test_first_visit_shows_the_panel_with_the_default_style(risk_env):
    panel = _press("risk", risk_env, "q-risk-1")
    assert "⚖️" in panel and "$" in panel           # balanced until the user picks
    assert "BTC" in panel                            # BTC is over its 70% limit here


def test_style_picker_marks_the_current_style(risk_env):
    picker = _press("risk_style:careful", risk_env, "q-risk-2")
    assert "🛡" in picker and "⚖️" in picker and "🚀" in picker
    careful_block = picker.split(chr(10) * 2)[1]
    assert "✅" in careful_block


def test_stress_screen_lists_only_scenarios_that_hit(risk_env):
    _press("risk_style:careful", risk_env, "q-risk-3")
    stress = _press("risk_stress", risk_env, "q-risk-4")
    assert "$0.00000000" not in stress
    assert "BTC" in stress


def test_calculator_reply_with_deposit(risk_env):
    _press("risk_size", risk_env, "q-risk-5")
    assert risk_env.user_data.get("awaiting_risk_size")
    message = SimpleNamespace(text="SOL 105 100 115 $100", reply_text=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=42), message=message)
    asyncio.run(rh.handle_risk_text(update, risk_env))
    text = message.reply_text.await_args.args[0]
    assert "$39.90" in text and "$2.00" in text
    assert "awaiting_risk_size" not in risk_env.user_data


def test_calculator_rejects_stop_above_entry(risk_env):
    message = SimpleNamespace(text="SOL 100 105 $100", reply_text=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=42), message=message)
    asyncio.run(rh.handle_risk_text(update, risk_env))
    text = message.reply_text.await_args.args[0]
    assert "$" not in text.split("\n", 1)[1]        # no size offered


@pytest.mark.parametrize("raw, ok", [("SOL 105 100", True), ("SOL 105 100 115 $300", True),
                                     ("SOL 105", False), ("SOL abc 100", False)])
def test_parse_size_request(raw, ok):
    assert (rh._parse_size_request(raw) is not None) == ok
