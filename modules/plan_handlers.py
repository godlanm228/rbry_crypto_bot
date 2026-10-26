"""🎯 Trade plan in Telegram, entry-level watches and drawdown notices.

The plan card is the arithmetic of a trade the user is considering (see
core/risk/trade_plan.py): it never tells them to buy. Two background jobs
live here too because they speak the same language: a watch fires when a
coin reaches the better entry the plan suggested, and a notice goes out when
the portfolio's drawdown moves to another protocol state.
"""

from __future__ import annotations

import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

import config
from core.api.binance_client import SYMBOL_MAP, get_klines
from core.formatting import format_price, plain
from core.portfolio_db import get_portfolio_manager
from core.risk.exposure import DEFAULT_STABLECOIN_SYMBOLS
from core.risk.models import DrawdownState
from core.risk.trade_plan import (
    PlanUnavailable,
    build_trade_plan,
    candles_from_frame,
    candles_from_klines,
    pick_alternatives,
)
from core.risk.watch import EntryWatchStore, RiskNoticeStore
from core.utils import resolve_coin_id
from locales.translator import t
from modules.callback_utils import answer_once
from modules.risk_handlers import _panel, _tr, light_line, money, pct, style_of, _clear_inputs

logger = logging.getLogger(__name__)

PICKER_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "TON", "DOGE", "ADA", "LINK"]
EXAMPLE_ACCOUNT = 1000.0         # sizing shown for an empty portfolio
KLINES_TTL = 600
NOTICE_REPEAT_SECONDS = 12 * 3600
_SEVERITY = {state: i for i, state in enumerate(DrawdownState)}
_klines_cache: dict[str, tuple[float, object]] = {}
_watches: EntryWatchStore | None = None
_notices: RiskNoticeStore | None = None


def _watch_store() -> EntryWatchStore:
    global _watches
    if _watches is None:
        _watches = EntryWatchStore()
    return _watches


def _notice_store() -> RiskNoticeStore:
    global _notices
    if _notices is None:
        _notices = RiskNoticeStore()
    return _notices


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _history(context):
    service = context.application.bot_data.get("history_snapshot") if context.application else None
    return service.get_latest() if service else None


async def _candles_for(context, coin_id: str, symbol: str):
    """Snapshot candles when they carry highs/lows, else Binance daily klines."""
    latest = _history(context)
    frame = latest.by_coin.get(coin_id) if latest else None
    candles = candles_from_frame(frame)
    if candles is not None and len(candles.closes) >= 60:
        return candles, None
    pair = SYMBOL_MAP.get(coin_id) or f"{symbol.upper()}USDT"
    cached = _klines_cache.get(pair)
    if cached and time.time() - cached[0] < KLINES_TTL:
        rows = cached[1]
    else:
        rows = await get_klines(pair, "1d", limit=201)
        _klines_cache[pair] = (time.time(), rows)
    live = rows[-1]["close"] if rows else None
    return candles_from_klines(rows), live


async def _market_rows(context, coin_ids):
    from modules.handlers import _get_market_rows

    rows, _ts = await _get_market_rows(context.application, coin_ids)
    return rows


async def build_plan_for(context, user_id: int, symbol: str, *, with_alternatives: bool = True):
    """(plan, alternatives, account_is_example). Raises PlanUnavailable."""
    symbol = symbol.upper()
    coin_id = resolve_coin_id(symbol)
    if not coin_id:
        raise PlanUnavailable("unknown")
    style = style_of(user_id)
    portfolio, markets, panel = await _panel(context, user_id, style)
    rows = await _market_rows(context, [coin_id])
    candles, live = await _candles_for(context, coin_id, symbol)
    price = (rows.get(coin_id) or {}).get("current_price") or live
    if not price:
        raise PlanUnavailable("price" if candles else "history")
    example = panel.value <= 0
    account = EXAMPLE_ACCOUNT if example else panel.value
    held = portfolio.get(symbol) or {}
    existing = 0.0 if example else float(held.get("amount") or 0) * float(price)
    state = DrawdownState.NORMAL if example else panel.state
    plan = build_trade_plan(symbol, float(price), candles, user_id=user_id, style=style,
                            account_value=account, existing_position_value=existing, state=state)
    alternatives = []
    if with_alternatives:
        alternatives = _alternatives(context, user_id, plan, style, account, state)
    return plan, alternatives, example


def _alternatives(context, user_id, plan, style, account, state):
    """Good setups among the coins whose candles are already in the snapshot."""
    from modules.handlers import _fresh_market_snapshot

    latest = _history(context)
    snapshot = _fresh_market_snapshot(context.application)
    markets = getattr(snapshot, "markets", None)
    if not latest or not markets:
        return []
    plans = []
    for coin_id, frame in latest.by_coin.items():
        row = markets.get(coin_id) or {}
        symbol = str(row.get("symbol") or "").upper()
        price = row.get("current_price")
        if not symbol or not price or symbol == plan.symbol or symbol in DEFAULT_STABLECOIN_SYMBOLS:
            continue
        try:
            plans.append(build_trade_plan(symbol, float(price), candles_from_frame(frame), user_id=user_id,
                                          style=style, account_value=account, state=state))
        except (PlanUnavailable, ValueError):
            continue
    return pick_alternatives(plans, plan.symbol)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _tenths(chance: float) -> int:
    return max(0, min(10, round(chance * 10)))


def render_plan(plan, alternatives, user_id, example: bool = False) -> str:
    p = plan
    lines = [f"🎯 <b>{plain(p.symbol)}</b> · {format_price(p.price)}",
             _tr(f"plan.trend_{p.trend}", user_id), "", f"<b>{_tr('plan.if_now', user_id)}</b>"]
    stop_pct = (p.price - p.stop) / p.price * 100
    if p.stop_basis == "support":
        lines.append(_tr("plan.stop_support", user_id, stop=format_price(p.stop), pct=pct(stop_pct),
                         support=format_price(p.support)))
    else:
        lines.append(_tr("plan.stop_volatility", user_id, stop=format_price(p.stop), pct=pct(stop_pct)))
    target_pct = (p.target - p.price) / p.price * 100
    lines.append(_tr(f"plan.target_{p.target_basis}", user_id, target=format_price(p.target), pct=pct(target_pct)))

    s = p.sizing
    if s.approved:
        key = "plan.buy_example" if example else "plan.buy"
        lines.append(_tr(key, user_id, value=money(s.position_value), coins=f"{s.coins:.4g}", symbol=p.symbol,
                         account=money(s.account_value)))
        if any("single_coin" in cap for cap in s.caps_applied):
            lines.append(_tr("plan.capped", user_id))
        lines.append(_tr("plan.outcomes", user_id, loss=money(s.loss_at_stop), profit=money(s.profit_at_take or 0)))
    else:
        from modules.risk_handlers import sizing_problem
        lines.append(sizing_problem(s, user_id))
    lines.append(_tr("plan.odds", user_id, rr=f"{p.reward_risk:.1f}", n=_tenths(p.hit_chance)))

    if p.week is not None:
        lines += ["", _tr("plan.week_title", user_id, symbol=p.symbol),
                  _tr("plan.week_up", user_id, price=format_price(p.week.high)),
                  _tr("plan.week_mid", user_id, low=format_price(p.week.low), high=format_price(p.week.high)),
                  _tr("plan.week_down", user_id, price=format_price(p.week.low))]

    against = p.trend == "down"
    verdict_key = f"plan.verdict_{p.verdict}" + ("_trend" if against and p.verdict != "good" else "")
    lines += ["", _tr(verdict_key, user_id)]
    if p.better is not None:
        lines.append(_tr("plan.better", user_id, price=format_price(p.better.price),
                         rr=f"{p.better.reward_risk:.1f}", n=_tenths(p.better.hit_chance)))
    if alternatives:
        listed = " · ".join(f"{a.symbol} ({a.reward_risk:.1f})" for a in alternatives)
        lines.append(_tr("plan.alternatives", user_id, list=listed))
    if example:
        lines += ["", _tr("plan.empty_hint", user_id)]
    return "\n".join(lines)


def plan_keyboard(plan, alternatives, user_id) -> InlineKeyboardMarkup:
    rows = []
    if plan.better is not None:
        level = f"{plan.better.price:.8g}"
        rows.append([InlineKeyboardButton(t("plan.button_watch", user_id=user_id,
                                            price=format_price(plan.better.price)),
                                          callback_data=f"tp_watch:{plan.symbol}:{level}")])
    if alternatives:
        rows.append([InlineKeyboardButton(t("plan.button_for", user_id=user_id, symbol=a.symbol),
                                          callback_data=f"tp:{a.symbol}") for a in alternatives])
    rows.append([InlineKeyboardButton(t("plan.button_other_coin", user_id=user_id), callback_data="tp"),
                 InlineKeyboardButton(t("risk.button_panel", user_id=user_id), callback_data="risk")])
    return InlineKeyboardMarkup(rows)


def picker(user_id) -> tuple[str, InlineKeyboardMarkup]:
    held = list(get_portfolio_manager().load_portfolio(user_id))[:6]
    coins = held + [c for c in PICKER_COINS if c not in held]
    rows = [[InlineKeyboardButton(c, callback_data=f"tp:{c}") for c in coins[i:i + 3]]
            for i in range(0, len(coins), 3)]
    rows.append([InlineKeyboardButton(t("plan.button_other", user_id=user_id), callback_data="tp_other"),
                 InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="risk")])
    text = f"<b>{_tr('plan.picker_title', user_id)}</b>\n{_tr('plan.picker_text', user_id)}"
    return text, InlineKeyboardMarkup(rows)


def unavailable_text(reason: str, symbol: str, user_id) -> str:
    key = {"unknown": "plan.unknown_coin", "price": "plan.unavailable_price"}.get(reason, "plan.unavailable_history")
    return _tr(key, user_id, symbol=symbol.upper())


async def _plan_message(context, user_id, symbol):
    try:
        plan, alternatives, example = await build_plan_for(context, user_id, symbol)
    except PlanUnavailable as exc:
        back = InlineKeyboardMarkup([[InlineKeyboardButton(t("plan.button_other_coin", user_id=user_id),
                                                           callback_data="tp")]])
        return unavailable_text(str(exc), symbol, user_id), back
    return render_plan(plan, alternatives, user_id, example), plan_keyboard(plan, alternatives, user_id)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def plan_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    _clear_inputs(context)
    try:
        if data == "tp":
            text, keyboard = picker(user_id)
            await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        elif data == "tp_other":
            context.user_data["awaiting_plan_coin"] = True
            back = InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="tp")]])
            await query.edit_message_text(_tr("plan.other_prompt", user_id), parse_mode=ParseMode.HTML,
                                          reply_markup=back)
        elif data.startswith("tp_watch:"):
            _prefix, symbol, level = data.split(":", 2)
            watch = _watch_store().add(user_id, symbol, resolve_coin_id(symbol), float(level))
            key = "plan.watch_set" if watch else "plan.watch_limit"
            await answer_once(query, t(key, user_id=user_id, symbol=symbol, price=format_price(float(level))),
                              show_alert=True)
        elif data.startswith("tp:"):
            await answer_once(query)
            text, keyboard = await _plan_message(context, user_id, data.split(":", 1)[1])
            await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    finally:
        await answer_once(query)


async def handle_plan_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The ticker typed after "✏️ Другая монета"."""
    user_id = update.effective_user.id
    context.user_data.pop("awaiting_plan_coin", None)
    symbol = (update.message.text or "").strip().split()[0].upper()[:20] if update.message.text else ""
    text, keyboard = await _plan_message(context, user_id, symbol)
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

async def run_entry_watches(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fire watches whose coin traded at or below the level."""
    store = _watch_store()
    watches = store.active()
    if not watches:
        return
    rows = await _market_rows(context, list({w.coin_id for w in watches}))
    for watch in watches:
        price = (rows.get(watch.coin_id) or {}).get("current_price")
        if not price or price > watch.level or not config.is_user_allowed(watch.user_id):
            continue
        if not store.mark_fired(watch.id):
            continue
        uid = watch.user_id
        head = _tr("plan.watch_fired", uid, symbol=watch.symbol, level=format_price(watch.level),
                   price=format_price(price))
        try:
            text, keyboard = await _plan_message(context, uid, watch.symbol)
            await context.bot.send_message(uid, f"{head}\n\n{text}", parse_mode=ParseMode.HTML,
                                           reply_markup=keyboard)
        except Exception:  # noqa: BLE001 — one user's failure must not stop the others
            logger.warning("Entry watch %s for %s failed to send", watch.id, watch.symbol, exc_info=True)


async def run_drawdown_notices(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tell a user when the drawdown protocol state of their portfolio changes."""
    store = _notice_store()
    now = time.time()
    for uid in get_portfolio_manager().get_all_users():
        if not config.is_user_allowed(uid):
            continue
        try:
            _p, _m, panel = await _panel(context, uid, style_of(uid))
        except Exception:  # noqa: BLE001
            logger.warning("Drawdown check failed for a user", exc_info=True)
            continue
        if panel.value <= 0:
            continue
        state = panel.state
        previous = store.get(uid)
        if previous is None:
            store.set(uid, state.value, now)          # first look: remember, do not announce
            continue
        prev_state = DrawdownState(previous[0])
        if state == prev_state:
            continue
        recent = now - previous[1] < NOTICE_REPEAT_SECONDS
        worse = _SEVERITY[state] > _SEVERITY[prev_state]
        text = None
        if worse and (not recent or _SEVERITY[state] - _SEVERITY[prev_state] >= 2):
            text = f"<b>{_tr('risk.notice_title', uid)}</b>\n{light_line(panel, uid)}"
        elif state == DrawdownState.NORMAL and not recent:
            text = _tr("risk.notice_recovered", uid)
        if text is None:
            store.set(uid, state.value, previous[1])  # track the state, keep the quiet period
            continue
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton(t("risk.button_panel", user_id=uid),
                                                               callback_data="risk")]])
        try:
            await context.bot.send_message(uid, text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
            store.set(uid, state.value, now)
        except Exception:  # noqa: BLE001
            logger.warning("Drawdown notice failed to send", exc_info=True)
