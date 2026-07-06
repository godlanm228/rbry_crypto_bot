import logging
import traceback
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from core.portfolio_db import get_portfolio_manager
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
        await update.message.reply_text("Формат: /add_entry COIN AMOUNT PRICE")
        return

    coin, amount, price = context.args
    coin = coin.upper()
    user_id = update.effective_user.id

    try:
        amount = float(amount)
        price = float(price)
    except ValueError:
        await update.message.reply_text("Неверный формат чисел.")
        return

    # Use new PortfolioManager
    pm = get_portfolio_manager()
    
    try:
        result = pm.add_entry(user_id, coin, amount, price)
        await update.message.reply_text(
            f"✅ Добавлено: {coin} — {amount} @ ${price:.2f}\n"
            f"ℹ️ Новая средняя: ${result['avg_price']:.2f}, объём: {result['total_amount']:.6f}"
        )
    except Exception as e:
        logger.error(f"Error adding entry: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка при добавлении: {e}")

async def sell_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Sell command with partial sell support and PnL tracking.
    Format: /sell COIN AMOUNT PRICE
    """
    _track_user_activity(update.effective_user, "sell", {"args": context.args})
    if len(context.args) != 3:
        await update.message.reply_text("Формат: /sell COIN AMOUNT PRICE")
        return
    
    coin = context.args[0].upper()
    user_id = update.effective_user.id
    
    try:
        amount = float(context.args[1])
        sell_price = float(context.args[2])
    except ValueError:
        await update.message.reply_text("Неверный формат чисел.")
        return

    # Use new PortfolioManager
    pm = get_portfolio_manager()
    
    try:
        result = pm.sell(user_id, coin, amount, sell_price)
        pnl = result['pnl']
        remaining = result['remaining_amount']
        pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
        
        msg = (
            f"✅ Продано: {coin} — {amount} @ ${sell_price:.2f}\n"
            f"{pnl_emoji} PnL: ${pnl:+.2f}\n"
        )
        
        if result['was_full_sale']:
            msg += f"ℹ️ Позиция {coin} полностью закрыта"
        else:
            msg += f"ℹ️ Остаток: {remaining:.6f} {coin}"
        
        await update.message.reply_text(msg)
        
    except ValueError as e:
        # User-friendly error message with navigation button
        error_msg = f"❌ {str(e)}"
        keyboard = [[InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(error_msg, parse_mode="HTML", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error selling: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка при продаже: {e}")

def _row_change_pct(row, horizon: str):
    """horizon: '1h' | '24h' | '7d'."""
    if not row:
        return None
    k_in_cur = f"price_change_percentage_{horizon}_in_currency"
    if row.get(k_in_cur) is not None:
        return row[k_in_cur]
    k_plain = f"price_change_percentage_{horizon}"
    return row.get(k_plain)
