"""🛡 Risk: style picker, risk panel, stress test and position-size calculator."""

from __future__ import annotations

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from core.formatting import format_price, plain
from core.portfolio_db import get_portfolio_manager
from core.risk.service import STYLES, RiskStyleStore, build_risk_panel, size_trade, style_thresholds
from core.risk.models import DrawdownState
from core.utils import parse_positive_number, resolve_coin_id
from locales.translator import t
from modules.callback_utils import answer_once

logger = logging.getLogger(__name__)

STYLE_ICONS = {"careful": "🛡", "balanced": "⚖️", "bold": "🚀"}
_store: RiskStyleStore | None = None


def _styles() -> RiskStyleStore:
    global _store
    if _store is None:
        _store = RiskStyleStore()
    return _store


def _tr(key: str, user_id, **params) -> str:
    return plain(t(key, user_id=user_id, **params))


async def _panel(context, user_id: int, style):
    from modules.handlers import _get_market_rows

    portfolio = get_portfolio_manager().load_portfolio(user_id)
    ids = [resolve_coin_id(symbol) for symbol in portfolio]
    markets = {}
    if ids:
        markets, _ts = await _get_market_rows(context.application, ids)
    history = context.application.bot_data.get("history_snapshot") if context.application else None
    latest = history.get_latest() if history else None
    prices = {cid: list(df["price"]) for cid, df in latest.by_coin.items()} if latest else {}
    panel = build_risk_panel(portfolio, markets, prices, style, coin_id_resolver=resolve_coin_id)
    return portfolio, markets, panel


def scenario_text(result, user_id) -> str:
    """Localized scenario name; the engine's own description as a fallback."""
    if result.scenario_key == "largest_holding_-40":
        symbol = result.contributors[0][0] if result.contributors else "?"
        return _tr("risk.scenario_largest_holding", user_id, symbol=symbol)
    key = "risk.scenario_" + result.scenario_key.replace("-", "m")
    text = t(key, user_id=user_id)
    return plain(text) if text != f"[{key}]" else plain(result.description)


def _style_name(style, user_id) -> str:
    return f"{STYLE_ICONS[style.key]} {_tr(f'risk.style_{style.key}', user_id)}"


def render_panel(panel, user_id) -> str:
    lines = [f"<b>{_tr('risk.title', user_id)}</b> · {_style_name(panel.style, user_id)}"]
    if panel.value <= 0:
        lines += ["", _tr("risk.empty", user_id)]
        return "\n".join(lines)
    pnl = f" ({panel.pnl_pct:+.1f}%)" if panel.pnl_pct is not None else ""
    lines.append(_tr("risk.value_line", user_id, value=format_price(panel.value), cost=format_price(panel.cost)) + pnl)

    lines += ["", f"<b>{_tr('risk.concentration_title', user_id)}</b>"]
    for line in panel.concentration:
        mark = "⚠️" if line.over_limit else "✅"
        lines.append(f"{mark} {plain(line.symbol)} {line.weight_pct:.0f}% "
                     + _tr("risk.limit_suffix", user_id, limit=f"{line.limit_pct:.0f}"))
    if panel.stablecoin_pct is not None:
        lines.append(_tr("risk.stables", user_id, pct=f"{panel.stablecoin_pct:.0f}"))

    lines += ["", f"<b>{_tr('risk.drawdown_title', user_id)}</b>"]
    lines.append(_tr("risk.drawdown_line", user_id, pct=f"{panel.drawdown_pct:.1f}",
                     state=t(f"risk.state_{panel.state.value}", user_id=user_id)))
    lines.append(_tr(f"risk.state_hint_{panel.state.value}", user_id))
    if panel.next_state is not None:
        lines.append(_tr("risk.next_state", user_id, pct=f"{panel.next_state_at_pct:.1f}",
                         state=t(f"risk.state_{panel.next_state.value}", user_id=user_id)))

    lines.append("")
    if panel.var_value is not None:
        lines.append(_tr("risk.var_line", user_id, value=format_price(panel.var_value), pct=f"{panel.var_pct:.1f}"))
    else:
        lines.append(_tr("risk.var_unavailable", user_id))
    if panel.worst is not None:
        lines.append(_tr("risk.worst_line", user_id, scenario=scenario_text(panel.worst, user_id),
                         value=format_price(panel.worst.loss_value), pct=f"{panel.worst.loss_pct:.0f}"))
    lines += ["", _tr("risk.rule_line", user_id, pct=f"{panel.style.risk_per_trade_pct:g}",
                      value=format_price(panel.risk_budget))]
    if panel.missing_symbols:
        lines.append(_tr("risk.missing_prices", user_id, coins=", ".join(panel.missing_symbols)))
    return "\n".join(lines)


def _panel_keyboard(user_id) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t("risk.button_size", user_id=user_id), callback_data="risk_size"),
         InlineKeyboardButton(t("risk.button_stress", user_id=user_id), callback_data="risk_stress")],
        [InlineKeyboardButton(t("risk.button_style", user_id=user_id), callback_data="risk_style"),
         InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")],
    ])


def render_style_picker(user_id, value: float) -> tuple[str, InlineKeyboardMarkup]:
    lines = [f"<b>{_tr('risk.style_question', user_id)}</b>", ""]
    for key, style in STYLES.items():
        caution = style_thresholds(style)[DrawdownState.CAUTION]
        budget = f" (= {format_price(value * style.risk_per_trade_pct / 100)})" if value > 0 else ""
        lines.append(f"{_style_name(style, user_id)}: " + _tr(f"risk.style_{key}_desc", user_id))
        lines.append("   " + _tr("risk.style_numbers", user_id, pct=f"{style.risk_per_trade_pct:g}", budget=budget,
                                   caution=f"{caution:g}", alt=f"{style.alt_limit_pct:.0f}",
                                   major=f"{style.major_limit_pct:.0f}"))
    lines += ["", _tr("risk.style_change_later", user_id)]
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton(_style_name(style, user_id), callback_data=f"risk_style:{key}")]
         for key, style in STYLES.items()]
    )
    return "\n".join(lines), keyboard


def render_stress(panel, user_id) -> str:
    lines = [f"<b>{_tr('risk.stress_title', user_id)}</b>", _tr("risk.stress_intro", user_id), ""]
    # Scenarios that do not touch this portfolio (e.g. stablecoin depeg with no
    # stablecoins) are left out instead of listing "-$0.00".
    hits = [r for r in panel.scenarios.results if r.loss_value >= 0.01]
    for result in sorted(hits, key=lambda r: r.loss_value, reverse=True):
        lines.append(f"• {scenario_text(result, user_id)}: −{format_price(result.loss_value)} (−{result.loss_pct:.1f}%)")
    reverse = [r for r in panel.scenarios.reverse if r.required_shock_pct is not None]
    # Without stablecoins the reverse stress is trivially "-10% needs -10%".
    if reverse and (panel.stablecoin_pct or 0) > 0:
        lines += ["", f"<b>{_tr('risk.reverse_title', user_id)}</b>"]
        for r in reverse:
            lines.append(_tr("risk.reverse_line", user_id, target=f"{r.target_loss_pct:.0f}",
                             shock=f"{abs(r.required_shock_pct):.0f}"))
    return "\n".join(lines)


async def _send(update: Update, text: str, keyboard) -> None:
    query = update.callback_query
    if query is not None:
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    else:
        await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


async def _show_panel_or_picker(update: Update, context, user_id: int) -> None:
    style = _styles().get(user_id)
    if style is None:
        _p, _m, panel = await _panel(context, user_id, STYLES["balanced"])
        text, keyboard = render_style_picker(user_id, panel.value)
    else:
        _p, _m, panel = await _panel(context, user_id, style)
        text, keyboard = render_panel(panel, user_id), _panel_keyboard(user_id)
    await _send(update, text, keyboard)


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _show_panel_or_picker(update, context, update.effective_user.id)


async def risk_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    context.user_data.pop("awaiting_add", None)
    context.user_data.pop("awaiting_coin_input", None)
    if data != "risk_size":
        context.user_data.pop("awaiting_risk_size", None)
    try:
        if data == "risk":
            await _show_panel_or_picker(update, context, user_id)
        elif data == "risk_style":
            _p, _m, panel = await _panel(context, user_id, _styles().get(user_id) or STYLES["balanced"])
            text, keyboard = render_style_picker(user_id, panel.value)
            await _send(update, text, keyboard)
        elif data.startswith("risk_style:"):
            key = data.split(":", 1)[1]
            if key in STYLES:
                _styles().set(user_id, key)
                await answer_once(query, t("risk.style_saved", user_id=user_id))
            await _show_panel_or_picker(update, context, user_id)
        elif data == "risk_stress":
            style = _styles().get(user_id) or STYLES["balanced"]
            _p, _m, panel = await _panel(context, user_id, style)
            text = render_stress(panel, user_id) if panel.value > 0 else _tr("risk.empty", user_id)
            back = InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="risk")]])
            await _send(update, text, back)
        elif data == "risk_size":
            context.user_data["awaiting_risk_size"] = True
            back = InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="risk")]])
            await _send(update, _tr("risk.size_prompt", user_id), back)
    finally:
        await answer_once(query)


def _parse_size_request(text: str):
    """'SOL 105 100 [115] [$300]' -> (symbol, entry, stop, take, deposit) or None."""
    tokens = (text or "").split()
    if len(tokens) < 3:
        return None
    symbol = tokens[0].upper()
    deposit = None
    numbers = []
    for token in tokens[1:]:
        if token.startswith("$"):
            deposit = parse_positive_number(token[1:])
            if deposit is None:
                return None
            continue
        value = parse_positive_number(token)
        if value is None:
            return None
        numbers.append(value)
    if len(numbers) not in (2, 3):
        return None
    entry, stop = numbers[0], numbers[1]
    take = numbers[2] if len(numbers) == 3 else None
    return symbol, entry, stop, take, deposit


def render_sizing(symbol, entry, stop, take, sizing, user_id) -> str:
    lines = [f"📐 <b>{plain(symbol)}</b>: " + _tr("risk.size_head", user_id, entry=format_price(entry),
                                                    stop=format_price(stop), dist=f"{sizing.stop_distance_pct:.1f}")]
    if not sizing.approved:
        reason = sizing.reject_reasons[0] if sizing.reject_reasons else "unknown"
        key = reason.split("(")[0].split(":")[0]
        lines.append(_tr(f"risk.size_reject_{key}", user_id) if t(f"risk.size_reject_{key}", user_id=user_id)
                     != f"[risk.size_reject_{key}]" else _tr("risk.size_reject_unknown", user_id))
        return "\n".join(lines)
    lines.append(_tr("risk.size_buy", user_id, value=format_price(sizing.position_value),
                     coins=f"{sizing.coins:.6g}", symbol=symbol))
    lines.append(_tr("risk.size_loss", user_id, loss=format_price(sizing.loss_at_stop),
                     account=format_price(sizing.account_value)))
    if sizing.profit_at_take is not None:
        lines.append(_tr("risk.size_take", user_id, take=format_price(take), profit=format_price(sizing.profit_at_take),
                         rr=f"{sizing.reward_risk:.1f}"))
    if any("single_coin" in cap for cap in sizing.caps_applied):
        lines.append(_tr("risk.size_capped", user_id))
    return "\n".join(lines)


async def handle_risk_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply to the calculator prompt."""
    user_id = update.effective_user.id
    parsed = _parse_size_request(update.message.text)
    if parsed is None:
        await update.message.reply_text(_tr("risk.size_format_hint", user_id), parse_mode=ParseMode.HTML)
        return
    symbol, entry, stop, take, deposit = parsed
    style = _styles().get(user_id) or STYLES["balanced"]
    portfolio, markets, panel = await _panel(context, user_id, style)
    account = deposit or panel.value
    if account <= 0:
        await update.message.reply_text(_tr("risk.size_need_deposit", user_id), parse_mode=ParseMode.HTML)
        return
    held = portfolio.get(symbol) or {}
    price = (markets.get(resolve_coin_id(symbol)) or {}).get("current_price")
    existing = float(held.get("amount") or 0) * float(price or 0) if not deposit else 0.0
    sizing = size_trade(user_id=user_id, style=style, symbol=symbol, entry=entry, stop=stop, take=take,
                        account_value=account, existing_position_value=existing,
                        state=panel.state if not deposit else DrawdownState.NORMAL)
    context.user_data.pop("awaiting_risk_size", None)
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(t("risk.button_size_again", user_id=user_id), callback_data="risk_size"),
        InlineKeyboardButton(t("risk.button_panel", user_id=user_id), callback_data="risk"),
    ]])
    await update.message.reply_text(render_sizing(symbol, entry, stop, take, sizing, user_id),
                                    parse_mode=ParseMode.HTML, reply_markup=keyboard)
