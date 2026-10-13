import logging
import traceback
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from core.portfolio_db import get_portfolio_manager
from core.incidents import public_temporary_message, report_incident
from core.access.access_manager import get_access_manager
from core.access.presentation import format_access_decision
from core.analytics.user_analytics import get_user_analytics
from locales.translator import t
logger = logging.getLogger(__name__)

def _track_user_activity(user, action: str, details: dict | None = None):
    try:
        analytics = get_user_analytics()
        analytics.register_user(
            user_id=user.id,
            username=user.username,
            first_name=user.first_name,
            language_code=user.language_code,
        )
        analytics.log_activity(user.id, action, details)
    except Exception as e:
        logger.debug(f"User analytics error: {e}")

COIN_ID_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "USDT": "tether",
    "BNB": "binancecoin",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    # MATIC rebranded to POL (Sept 2024); keep both tickers resolving
    "MATIC": "polygon-ecosystem-token",
    "POL": "polygon-ecosystem-token",
    # CoinGecko renamed Toncoin to "Gram" (GRAM); users still type TON, which
    # CoinGecko now assigns to Tokamak Network.
    "TON": "the-open-network",
    "TONCOIN": "the-open-network",
    "LINK": "chainlink",
    "TRX": "tron",
    "SHIB": "shiba-inu",
    "ICP": "internet-computer",
    "NEAR": "near",
    "XLM": "stellar",
    "BCH": "bitcoin-cash",
    "UNI": "uniswap",
    "LTC": "litecoin",
    "ATOM": "cosmos",
}


def resolve_coin_id(symbol: str, registry=None) -> str:
    """Resolve a ticker through curated aliases, then the durable registry."""
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return ""
    mapped = COIN_ID_MAP.get(normalized)
    if mapped:
        return mapped
    try:
        if registry is None:
            from services.coin_registry_service import get_coin_registry_service

            registry = get_coin_registry_service()
        resolved = registry.resolve(normalized) if registry else None
        if resolved and resolved.get("coingecko_id"):
            return str(resolved["coingecko_id"])
    except Exception:
        logger.exception("Coin ID registry resolution failed symbol=%s", normalized)
    return normalized.lower()

def safe_run(task_name, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"[{task_name}] Ошибка: {e}")
        logger.debug(traceback.format_exc())
        return None

def get_avg_price_and_amount(entries):
    if not entries:
        return 0.0, 0.0
    total_cost = sum(e['amount'] * e['price'] for e in entries)
    total_amount = sum(e['amount'] for e in entries)
    if total_amount == 0:
        return 0.0, 0.0
    avg_price = total_cost / total_amount
    return avg_price, total_amount

# NOTE: Removed duplicate create_navigation_buttons() - using the one in handlers.py

async def add_entry_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add coin entry to user's portfolio using SQLite backend."""
    _track_user_activity(update.effective_user, "add_entry", {"args": context.args})
    if len(context.args) != 3:
        await update.message.reply_text(t("portfolio.add_usage", user_id=update.effective_user.id))
        return

    coin, amount, price = context.args
    coin = coin.upper()
    user_id = update.effective_user.id

    try:
        amount = float(amount)
        price = float(price)
    except ValueError:
        await update.message.reply_text(t("portfolio.invalid_numbers", user_id=user_id))
        return

    # Use new PortfolioManager
    pm = get_portfolio_manager()
    access_manager = get_access_manager()
    current_portfolio = pm.load_portfolio(user_id)
    existing_coin_ids = [
        resolve_coin_id(symbol)
        for symbol in current_portfolio
    ]
    coin_id = resolve_coin_id(coin)
    decision = access_manager.evaluate_portfolio_limit(
        user_id,
        existing_coin_ids,
        coin_id,
    )
    if not decision.allowed:
        await update.message.reply_text(format_access_decision(decision, user_id))
        return
    
    try:
        result = pm.add_entry(user_id, coin, amount, price)
        await update.message.reply_text(
            t(
                "portfolio.added",
                user_id=user_id,
                coin=coin,
                amount=amount,
                price=price,
                avg_price=result["avg_price"],
                total_amount=result["total_amount"],
            )
        )
    except Exception as e:
        logger.error(f"Error adding entry: {e}", exc_info=True)
        await report_incident(context, e, subsystem="portfolio_add", user_id=user_id)
        await update.message.reply_text(public_temporary_message(user_id))

async def sell_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Sell command with partial sell support and PnL tracking.
    Format: /sell COIN AMOUNT PRICE
    """
    _track_user_activity(update.effective_user, "sell", {"args": context.args})
    if len(context.args) != 3:
        await update.message.reply_text(t("portfolio.sell_usage", user_id=update.effective_user.id))
        return
    
    coin = context.args[0].upper()
    user_id = update.effective_user.id
    
    try:
        amount = float(context.args[1])
        sell_price = float(context.args[2])
    except ValueError:
        await update.message.reply_text(t("portfolio.invalid_numbers", user_id=user_id))
        return

    # Use new PortfolioManager
    pm = get_portfolio_manager()
    
    try:
        result = pm.sell(user_id, coin, amount, sell_price)
        pnl = result['pnl']
        remaining = result['remaining_amount']
        pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
        
        msg = t(
            "portfolio.sold",
            user_id=user_id,
            coin=coin,
            amount=amount,
            price=sell_price,
            pnl_emoji=pnl_emoji,
            pnl=pnl,
        )
        
        if result['was_full_sale']:
            msg += "\n" + t("portfolio.position_closed", user_id=user_id, coin=coin)
        else:
            msg += "\n" + t(
                "portfolio.position_remaining",
                user_id=user_id,
                amount=remaining,
                coin=coin,
            )
        
        await update.message.reply_text(msg)
        
    except ValueError:
        current = pm.load_portfolio(user_id)
        if coin not in current:
            error_msg = t("portfolio.sell_not_found", user_id=user_id, coin=coin)
        else:
            available = current[coin].get("amount", 0)
            error_msg = t(
                "portfolio.sell_insufficient",
                user_id=user_id,
                coin=coin,
                available=available,
            )
        keyboard = [[InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(error_msg, parse_mode="HTML", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error selling: {e}", exc_info=True)
        await report_incident(context, e, subsystem="portfolio_sell", user_id=user_id)
        await update.message.reply_text(public_temporary_message(user_id))

def _row_change_pct(row, horizon: str):
    """horizon: '1h' | '24h' | '7d'."""
    if not row:
        return None
    k_in_cur = f"price_change_percentage_{horizon}_in_currency"
    if row.get(k_in_cur) is not None:
        return row[k_in_cur]
    k_plain = f"price_change_percentage_{horizon}"
    return row.get(k_plain)
