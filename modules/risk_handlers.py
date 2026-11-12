""" Risk: one casual screen; style and the manual calculator under ."""

from __future__ import annotations

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from core.formatting import format_amount, format_price, plain
from core.portfolio_db import get_portfolio_manager
from core.risk.service import STYLES, RiskStyleStore, build_risk_panel, size_trade, style_thresholds
from core.risk.models import DrawdownState
from core.utils import parse_positive_number, resolve_coin_id
from locales.translator import t
from modules.callback_utils import answer_once

logger = logging.getLogger(__name__)

STYLE_ICONS = {"careful": "🛡", "balanced": "⚖️", "bold": "🚀"}
DEFAULT_STYLE = STYLES["balanced"]
STRESS_LEVELS = (("btc_-10", "🌧", 10), ("btc_-20", "⛈", 20), ("btc_-30", "🌪", 30))
_store: RiskStyleStore | None = None


def _styles() -> RiskStyleStore:
    global _store
    if _store is None:
        _store = RiskStyleStore()
    return _store


def style_of(user_id: int):
    """The user's style; balanced until they pick one."""
    return _styles().get(user_id) or DEFAULT_STYLE


def _tr(key: str, user_id, **params) -> str:
    return plain(t(key, user_id=user_id, **params))


def money(value: float) -> str:
    """Whole dollars from $100 up: a risk screen is read, not audited."""
    return f"${value:,.0f}" if abs(value) >= 100 else format_price(value)


def pct(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


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


def style_name(style, user_id) -> str:
    return f"{STYLE_ICONS[style.key]} {_tr(f'risk.style_{style.key}', user_id)}"


def light_line(panel, user_id) -> str:
    """Traffic light for the drawdown state, plus where the next step starts."""
    state = panel.state.value
    if panel.state == DrawdownState.NORMAL:
        return _tr("risk.light_normal", user_id, next=pct(panel.next_state_at_pct or 0))
    line = _tr(f"risk.light_{state}", user_id, dd=pct(panel.drawdown_pct))
    if panel.next_state is not None:
        line += " " + _tr("risk.light_next", user_id, next=pct(panel.next_state_at_pct))
    return line


def render_panel(panel, user_id) -> str:
    lines = [f"🛡 <b>{_tr('risk.title', user_id)}</b> · {style_name(panel.style, user_id)}", ""]
    if panel.value <= 0:
        lines.append(_tr("risk.empty", user_id))
        return "\n".join(lines)
    if panel.pnl_pct is not None:
        sign = "+" if panel.pnl_pct >= 0 else "−"
        lines.append(_tr("risk.portfolio_line", user_id, value=money(panel.value),
                         pnl=f"{sign}{pct(abs(panel.pnl_pct))}%"))
    else:
        lines.append(_tr("risk.portfolio_line_nopnl", user_id, value=money(panel.value)))
    lines += ["", light_line(panel, user_id)]
    over = [line for line in panel.concentration if line.over_limit][:2]
    for line in over:
        lines.append(_tr("risk.over_limit", user_id, symbol=line.symbol, weight=f"{line.weight_pct:.0f}",
                         limit=f"{line.limit_pct:.0f}"))
    if not over and len(panel.concentration) > 1:
        lines.append(_tr("risk.shares_ok", user_id))
    lines.append("")
    if panel.var_value is not None:
        lines.append(_tr("risk.bad_day", user_id, value=money(panel.var_value)))
    else:
        lines.append(_tr("risk.bad_day_unavailable", user_id))
    lines.append(_tr("risk.budget_line", user_id, value=money(panel.risk_budget)))
    if panel.missing_symbols:
        lines.append(_tr("risk.missing_prices", user_id, coins=", ".join(panel.missing_symbols)))
    return "\n".join(lines)


def _panel_keyboard(user_id) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t("risk.button_plan", user_id=user_id), callback_data="tp")],
        [InlineKeyboardButton(t("risk.button_stress", user_id=user_id), callback_data="risk_stress")],
        [InlineKeyboardButton(t("risk.button_advanced", user_id=user_id), callback_data="risk_adv"),
         InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")],
    ])


def _back(user_id, target: str = "risk") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data=target)]])


def render_stress(panel, user_id) -> str:
    by_key = {r.scenario_key: r for r in panel.scenarios.results}
    lines = [f"<b>{_tr('risk.stress_title', user_id)}</b>",
             _tr("risk.stress_now", user_id, value=money(panel.value)), ""]
    for key, icon, drop in STRESS_LEVELS:
        result = by_key.get(key)
        if result is None or result.loss_value < 0.01:
            continue
        lines.append(_tr("risk.stress_line", user_id, icon=icon, drop=drop, loss=money(result.loss_value),
                         left=money(panel.value - result.loss_value)))
    lines.append(_tr("risk.stress_alts_note", user_id))
    largest = by_key.get("largest_holding_-40")
    if largest is not None and largest.contributors and panel.concentration:
        top = panel.concentration[0]
        lines += ["", _tr("risk.stress_weak_spot", user_id, symbol=top.symbol, weight=f"{top.weight_pct:.0f}",
                          loss=money(largest.loss_value))]
    depeg = by_key.get("stable_depeg_-5")
    if (panel.stablecoin_pct or 0) > 0 and depeg is not None and depeg.loss_value >= 0.01:
        lines.append(_tr("risk.stress_stables", user_id, loss=money(depeg.loss_value)))
    return "\n".join(lines)


def render_advanced(panel, user_id) -> tuple[str, InlineKeyboardMarkup]:
    style = panel.style
    thresholds = style_thresholds(style)
    lines = [f"<b>{_tr('risk.advanced_title', user_id)}</b> · {style_name(style, user_id)}", ""]
    if panel.value > 0:
        lines.append(_tr("risk.adv_loss", user_id, value=money(panel.risk_budget),
                         pct=pct(style.risk_per_trade_pct)))
    else:
        lines.append(_tr("risk.adv_loss_nobal", user_id, pct=pct(style.risk_per_trade_pct)))
    lines.append(_tr("risk.adv_limits", user_id, alt=f"{style.alt_limit_pct:.0f}", major=f"{style.major_limit_pct:.0f}"))
    lines.append(_tr("risk.adv_brakes", user_id, caution=pct(thresholds[DrawdownState.CAUTION]),
                     defensive=pct(thresholds[DrawdownState.DEFENSIVE])))
    lines += ["", _tr("risk.adv_manual_hint", user_id)]
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(t("risk.button_style", user_id=user_id), callback_data="risk_style")],
        [InlineKeyboardButton(t("risk.button_manual", user_id=user_id), callback_data="risk_size")],
        [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="risk")],
    ])
    return "\n".join(lines), keyboard


def render_style_picker(user_id, value: float, current) -> tuple[str, InlineKeyboardMarkup]:
    lines = [f"<b>{_tr('risk.style_title', user_id)}</b>"]
    for key, style in STYLES.items():
        caution = style_thresholds(style)[DrawdownState.CAUTION]
        name = f"{STYLE_ICONS[key]} <b>{_tr(f'risk.style_{key}', user_id).upper()}</b>"
        if current is not None and current.key == key:
            name += "  " + _tr("risk.style_current", user_id)
        lines += ["", name, _tr(f"risk.style_{key}_desc", user_id)]
        if value > 0:
            lines.append(_tr("risk.style_loss", user_id, value=money(value * style.risk_per_trade_pct / 100)))
        else:
            lines.append(_tr("risk.style_loss_pct", user_id, pct=pct(style.risk_per_trade_pct)))
        lines.append(_tr("risk.style_limits", user_id, alt=f"{style.alt_limit_pct:.0f}", caution=pct(caution)))
    buttons = []
    for key, style in STYLES.items():
        label = style_name(style, user_id)
        if current is not None and current.key == key:
            label = "✅ " + label
        buttons.append([InlineKeyboardButton(label, callback_data=f"risk_style:{key}")])
    buttons.append([InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="risk_adv")])
    return "\n".join(lines), InlineKeyboardMarkup(buttons)


async def _send(update: Update, text: str, keyboard) -> None:
    query = update.callback_query
    if query is not None:
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    else:
        await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


async def _show_panel(update: Update, context, user_id: int) -> None:
    _p, _m, panel = await _panel(context, user_id, style_of(user_id))
    await _send(update, render_panel(panel, user_id), _panel_keyboard(user_id))


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _clear_inputs(context)
    await _show_panel(update, context, update.effective_user.id)


def _clear_inputs(context) -> None:
    for key in ("awaiting_add", "awaiting_coin_input", "awaiting_risk_size", "awaiting_plan_coin"):
        context.user_data.pop(key, None)


async def risk_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    _clear_inputs(context)
    try:
        if data == "risk":
            await _show_panel(update, context, user_id)
        elif data == "risk_adv":
            _p, _m, panel = await _panel(context, user_id, style_of(user_id))
            await _send(update, *render_advanced(panel, user_id))
        elif data == "risk_style":
            _p, _m, panel = await _panel(context, user_id, style_of(user_id))
            await _send(update, *render_style_picker(user_id, panel.value, _styles().get(user_id)))
        elif data.startswith("risk_style:"):
            key = data.split(":", 1)[1]
            if key in STYLES:
                _styles().set(user_id, key)
                await answer_once(query, t("risk.style_saved", user_id=user_id))
            _p, _m, panel = await _panel(context, user_id, style_of(user_id))
            await _send(update, *render_style_picker(user_id, panel.value, _styles().get(user_id)))
        elif data == "risk_stress":
            _p, _m, panel = await _panel(context, user_id, style_of(user_id))
            text = render_stress(panel, user_id) if panel.value > 0 else _tr("risk.empty", user_id)
            await _send(update, text, _back(user_id))
        elif data == "risk_size":
            context.user_data["awaiting_risk_size"] = True
            await _send(update, _tr("risk.size_prompt", user_id), _back(user_id, "risk_adv"))
    finally:
        await answer_once(query)


async def warn_concentration(update: Update, context, user_id: int, symbol: str) -> None:
    """After a coin is added: say so when its share now exceeds the style's limit."""
    try:
        style = style_of(user_id)
        _p, _m, panel = await _panel(context, user_id, style)
    except Exception:  # noqa: BLE001 - a warning must never break adding a coin
        logger.warning("Concentration check after add failed", exc_info=True)
        return
    line = next((c for c in panel.concentration if c.symbol == symbol.upper()), None)
    if line is None or not line.over_limit:
        return
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton(t("risk.button_panel", user_id=user_id),
                                                           callback_data="risk")]])
    await update.message.reply_text(
        _tr("risk.notice_concentration", user_id, symbol=line.symbol, weight=f"{line.weight_pct:.0f}",
            style=style_name(style, user_id), limit=f"{line.limit_pct:.0f}"),
        parse_mode=ParseMode.HTML, reply_markup=keyboard)


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


def sizing_problem(sizing, user_id) -> str:
    reason = sizing.reject_reasons[0] if sizing.reject_reasons else "unknown"
    key = f"risk.size_reject_{reason.split('(')[0].split(':')[0]}"
    return _tr(key, user_id) if t(key, user_id=user_id) != f"[{key}]" else _tr("risk.size_reject_unknown", user_id)


def render_sizing(symbol, entry, stop, take, sizing, user_id) -> str:
    lines = [f"✏️ <b>{plain(symbol)}</b>: " + _tr("risk.size_head", user_id, entry=format_price(entry),
                                                    stop=format_price(stop), dist=f"{sizing.stop_distance_pct:.1f}")]
    if not sizing.approved:
        lines.append(sizing_problem(sizing, user_id))
        return "\n".join(lines)
    lines.append(_tr("risk.size_buy", user_id, value=format_price(sizing.position_value),
                     coins=format_amount(sizing.coins), symbol=symbol))
    lines.append(_tr("risk.size_loss", user_id, loss=format_price(sizing.loss_at_stop),
                     account=format_price(sizing.account_value)))
    if sizing.profit_at_take is not None:
        lines.append(_tr("risk.size_take", user_id, take=format_price(take), profit=format_price(sizing.profit_at_take),
                         rr=f"{sizing.reward_risk:.1f}"))
    if any("single_coin" in cap for cap in sizing.caps_applied):
        lines.append(_tr("risk.size_capped", user_id))
    return "\n".join(lines)


async def handle_risk_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply to the manual calculator prompt."""
    user_id = update.effective_user.id
    parsed = _parse_size_request(update.message.text)
    if parsed is None:
        await update.message.reply_text(_tr("risk.size_format_hint", user_id), parse_mode=ParseMode.HTML)
        return
    symbol, entry, stop, take, deposit = parsed
    style = style_of(user_id)
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
