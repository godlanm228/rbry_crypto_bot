"""Portfolio buttons: add a coin (a button, then the amount), the help for a sale."""

import html

from core.formatting import as_html, format_price, plain
from core.portfolio_db import get_portfolio_manager
from core.utils import resolve_coin_id
from locales.translator import t
from modules.ui import _get_market_rows
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes


ADD_MENU_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "TON", "DOGE", "ADA", "LINK"]


async def add_coin_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Coin buttons first; typing the full command stays available."""
    query = update.callback_query
    # Coin buttons first; typing the full command stays available.
    user_id = query.from_user.id
    rows = [ADD_MENU_COINS[i:i + 3] for i in range(0, len(ADD_MENU_COINS), 3)]
    keyboard = [[InlineKeyboardButton(sym, callback_data=f"add_pick:{sym}") for sym in row] for row in rows]
    keyboard.append([InlineKeyboardButton(t("portfolio.add_pick_other", user_id=user_id), callback_data="add_other")])
    keyboard.append([InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="portfolio")])
    text = (
        f"<b>{plain(t('portfolio.add_pick_title', user_id=user_id))}</b>\n\n"
        f"{plain(t('portfolio.add_pick_hint', user_id=user_id))}"
    )
    await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))


async def add_pick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """A coin was picked (or 'another one'): wait for the amount."""
    query = update.callback_query
    user_id = query.from_user.id
    coin = query.data.split(":", 1)[1] if query.data.startswith("add_pick:") else None
    context.user_data["awaiting_add"] = {"coin": coin}
    if coin:
        rows, _ts = await _get_market_rows(context.application, [resolve_coin_id(coin)])
        row = rows.get(resolve_coin_id(coin)) or {}
        now = format_price(row["current_price"]) if row.get("current_price") else "n/a"
        prompt = t("portfolio.add_amount_prompt", user_id=user_id, coin=coin, price=now)
    else:
        prompt = t("portfolio.add_other_prompt", user_id=user_id)
    keyboard = [[InlineKeyboardButton(t("portfolio.add_cancel", user_id=user_id), callback_data="portfolio")]]
    await query.edit_message_text(as_html(prompt), parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))


async def sell_coin_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """How to record a sale, with the user's coins."""
    query = update.callback_query
    # Handler for "Sell" button - list user's coins
    user_id = query.from_user.id
    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)

    coin_list = ", ".join([coin.upper() for coin in portfolio.keys()]) if portfolio else t("common.not_available", user_id=user_id)

    help_text = (
        f"<b>{plain(t('portfolio.sell_coin_help_title', user_id=user_id))}</b>\n\n"
        f"<code>{t('portfolio.sell_coin_help_text', user_id=user_id)}</code>\n\n"
        f"<b>{plain(t('portfolio.sell_coin_help_coins', user_id=user_id, coins=coin_list))}</b>\n\n"
        f"<b>{plain(t('portfolio.sell_coin_help_example', user_id=user_id))}</b>"
    )
    keyboard = [[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
    await query.edit_message_text(help_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))


async def _handle_add_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, pending: dict):
    """Text after the add-coin menu: 'AMOUNT [PRICE]' or 'COIN AMOUNT [PRICE]'."""
    from core.utils import add_portfolio_entry

    user_id = update.effective_user.id
    parts = (update.message.text or "").split()
    coin = pending.get("coin")
    if coin is None and parts:
        coin, parts = parts[0].upper(), parts[1:]
    if not coin or len(parts) not in (1, 2):
        await update.message.reply_text(t("portfolio.add_format_hint", user_id=user_id))
        return  # keep waiting for a valid reply
    raw_amount = parts[0]
    raw_price = parts[1] if len(parts) == 2 else None
    if raw_price is None:
        coin_id = resolve_coin_id(coin)
        rows, _ts = await _get_market_rows(context.application, [coin_id])
        current = (rows.get(coin_id) or {}).get("current_price")
        if not current:
            await update.message.reply_text(t("portfolio.add_need_price", user_id=user_id, coin=html.escape(coin)))
            return
        raw_price = str(current)
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(t("portfolio.add_more", user_id=user_id), callback_data="add_coin_prompt"),
        InlineKeyboardButton(t("menu.portfolio", user_id=user_id), callback_data="portfolio"),
    ]])
    if await add_portfolio_entry(update, context, coin, raw_amount, raw_price, reply_markup=keyboard):
        context.user_data.pop("awaiting_add", None)
        from modules.risk_handlers import warn_concentration

        await warn_concentration(update, context, user_id, coin)
