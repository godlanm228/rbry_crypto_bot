import atexit
import logging
import os
import threading
import asyncio
import time
import config

from core.api import coingecko_client
from datetime import datetime
from flask import Flask, jsonify
from telegram.ext import (
    ApplicationBuilder,
    Application,
    CommandHandler,
    CallbackQueryHandler,
    PreCheckoutQueryHandler,
    MessageHandler,
    filters,
    Defaults
)
from telegram.error import TelegramError

from modules.handlers import start_command, help_command, portfolio_command, menu_callback, handle_coin_input, stats_command, movers_command, analyze_command, unknown_command, settings_command
from core.utils import add_entry_command, sell_command
from modules.news import news_command
from modules.admin_commands import cmd_grant, cmd_revoke, cmd_user_stats, cmd_check_user, cmd_my_tier
from modules.payments.payment_manager import get_payment_manager
from modules.alerts import run_alerts_scan, AlertConfig
from services.market_snapshot import MarketSnapshotService
from services.history_snapshot import HistorySnapshotService
from services.coin_registry_service import get_coin_registry_service
from core.learning.auto_validator import validate_pending_predictions
from core.analytics.user_analytics import get_user_analytics

@atexit.register
def _shutdown_http_clients():
    try:
        # If the loop is already closed, create a temporary one to close the session.
        asyncio.run(coingecko_client.close_session())
    except RuntimeError:
        # If there is a running loop, schedule session close on it.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(coingecko_client.close_session())
        except Exception as e:
            logging.warning(f"Failed to close CoinGecko session cleanly: {e}", exc_info=True)

logger = logging.getLogger(__name__)

async def on_error(update, context):
    err = context.error
    # Use global logging instead of Application.logger (not present in PTB 20.x)
    import logging, traceback
    logging.exception("Unhandled error: %s", err)
    # Optional: notify owner quietly if configured
    try:
        owner_id = getattr(config, "OWNER_ID", None)
        if owner_id:
            tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))[-1000:]
            await context.bot.send_message(chat_id=owner_id, text=f"⚠️ Ошибка: {type(err).__name__}{str(err)[:200]}")
    except Exception:
        pass

from telegram import Update
from telegram.ext import ContextTypes

async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        analytics = get_user_analytics()
        analytics.register_user(
            user_id=update.effective_user.id,
            username=update.effective_user.username,
            first_name=update.effective_user.first_name,
            language_code=update.effective_user.language_code,
        )
        analytics.log_activity(update.effective_user.id, "health")
    except Exception as e:
        logger.debug(f"User analytics error: {e}")
    app = context.application
    ms = app.bot_data.get("market_snapshot")
    hs = app.bot_data.get("history_snapshot")
    ms_ts = ms.last_ts() if ms else None
    hs_ts = hs.last_ts() if hs else None

    def fmt(ts):
        import datetime
        return datetime.datetime.utcfromtimestamp(ts).strftime("%H:%M:%S") + " UTC" if ts else "n/a"

    text = (
        "⚙️ *Health*\n"
        f"Market snapshot: {fmt(ms_ts)}\n"
        f"History snapshot: {fmt(hs_ts)}\n"
        f"Intervals: market={config.SNAPSHOT_REFRESH_SECONDS}s, history={config.HISTORY_REFRESH_SECONDS}s\n"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def snapshot_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        analytics = get_user_analytics()
        analytics.register_user(
            user_id=update.effective_user.id,
            username=update.effective_user.username,
            first_name=update.effective_user.first_name,
            language_code=update.effective_user.language_code,
        )
        analytics.log_activity(update.effective_user.id, "debug")
    except Exception as e:
        logger.debug(f"User analytics error: {e}")
    app = context.application
    ms = app.bot_data.get("market_snapshot")
    if not ms:
        await update.message.reply_text("❌ MarketSnapshot не инициализирован")
        return

    snap = ms.get_latest()
    if not snap:
        await update.message.reply_text("❌ Нет данных в снапшоте")
        return

    lines = ["📊 **Debug Snapshot:**"]
    for coin_id, row in list(snap.markets.items())[:5]:
        price = row.get("current_price", "n/a")
        ch24 = row.get("price_change_percentage_24h", "n/a")
        lines.append(f"{coin_id}: ${price} ({ch24}%)")

    await update.message.reply_text("\n".join(lines), parse_mode='Markdown')


# Flask app for healthcheck
flask_app = Flask(__name__)

@flask_app.route("/")
def index():
    return jsonify(status="ok", ts=datetime.utcnow().isoformat())

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

async def on_startup(app):
    from modules.alerts import AlertConfig
    from modules.handlers import set_global_app_ref
    from core.portfolio_db import get_portfolio_manager
    from services.snapshot_id_manager import create_snapshot_id_manager
    import config
    from core.ai_analytics import ai_analyzer
    # Log AI API configuration presence (no health check, no secrets)
    if config.HUGGINGFACE_API_KEY:
        logger.info("HuggingFace API key detected")
    else:
        logger.warning("HuggingFace API key missing")
    if config.GROQ_API_KEY:
        logger.info("Groq API key detected")
    else:
        logger.warning("Groq API key missing")
    logger.info(
        "FinBERT config: ENABLE_FINBERT=%s REQUIRE_FINBERT=%s USE_LOCAL_FINBERT=%s ALLOW_MODEL_DOWNLOAD=%s",
        config.ENABLE_FINBERT,
        config.REQUIRE_FINBERT,
        config.USE_LOCAL_FINBERT,
        config.ALLOW_MODEL_DOWNLOAD,
    )
    try:
        fb = ai_analyzer.finbert_status
        logger.info(
            "FinBERT status: enabled=%s available=%s backend=%s reason=%s",
            fb.get("enabled"),
            fb.get("available"),
            fb.get("backend"),
            fb.get("reason"),
        )
    except Exception as e:
        logger.warning(f"FinBERT status unavailable: {e}")

    # Initialize portfolio database and optionally migrate old data
    pm = get_portfolio_manager()
    try:
        # Try to migrate old portfolio.json if it exists (for OWNER_ID)
        import os
        if os.path.exists(config.PORTFOLIO_FILE):
            logger.info(f"Found old portfolio.json, attempting migration for owner {config.OWNER_ID}")
            pm.migrate_from_json(config.PORTFOLIO_FILE, config.OWNER_ID)
    except Exception as e:
        logger.warning(f"Portfolio migration skipped or failed: {e}")

    # Initialize coin registry
    registry = get_coin_registry_service()
    await registry.refresh_if_needed(force=True)
    app.bot_data["coin_registry"] = registry

    # Create SnapshotIDManager for optimized coin tracking
    snapshot_manager = create_snapshot_id_manager(
        top20_coins=AlertConfig.TOP20_COINS,
    )

    # Get optimized list of coins for snapshot
    snapshot_ids = snapshot_manager.get_snapshot_ids()
    logger.info(f"Snapshot tracking {len(snapshot_ids)} unique coins")

    # MARKET snapshot — TOP-20 + eligible portfolio coins
    market_snapshot = MarketSnapshotService(snapshot_ids, ttl=config.SNAPSHOT_REFRESH_SECONDS)

    # HISTORY snapshot — only coins from eligible users' portfolios (minimum requests)
    from core.access.access_manager import get_access_manager, AccessTier
    access_manager = get_access_manager()
    eligible_tiers = {AccessTier.PREMIUM, AccessTier.BETA, AccessTier.OWNER}

    all_users = pm.get_all_users()
    eligible_users = [uid for uid in all_users if access_manager.get_user_tier(uid) in eligible_tiers]

    if eligible_users:
        history_coin_ids = pm.get_eligible_portfolio_coins(eligible_users)
        history_ids = []
        for c in history_coin_ids:
            resolved = registry.resolve(c)
            if not resolved:
                logger.warning(f"Unknown coin ticker: {c}")
                continue
            history_ids.append(resolved["coingecko_id"])
    else:
        # Fallback to TOP20 if no eligible users
        history_ids = AlertConfig.TOP20_COINS[:10]  # Limit to top 10 for efficiency

    logger.info(f"History tracking {len(history_ids)} coins from eligible users")

    history_snapshot = HistorySnapshotService(
        history_ids,
        days=config.HISTORY_DAYS,
        interval=config.HISTORY_REFRESH_SECONDS,
    )

    app.bot_data["market_snapshot"] = market_snapshot
    app.bot_data["history_snapshot"] = history_snapshot
    app.bot_data["snapshot_manager"] = snapshot_manager

    # Global app reference for handlers
    set_global_app_ref(app)

    # Initialize AI synthesis engine with data sources
    try:
        from core.ai_analytics import ai_analyzer
        ai_analyzer.set_data_sources(
            market_snapshot=market_snapshot,
            history_snapshot=history_snapshot
        )
        logger.info("AI synthesis engine data sources initialized")
    except Exception as e:
        logger.warning(f"Could not initialize AI synthesis data sources: {e}")

    # Run in background
    asyncio.create_task(market_snapshot.start())
    asyncio.create_task(history_snapshot.start())


async def learning_validation_job(context):
    history_snapshot = context.application.bot_data.get("history_snapshot")
    try:
        validate_pending_predictions(
            history_snapshot,
            min_age_hours=config.VALIDATION_MIN_AGE_HOURS,
            limit=50,
        )
    except Exception as e:
        logger.error(f"Learning validation job failed: {e}", exc_info=True)
async def scan_top20_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ручной запуск сканирования ТОП-20"""
    try:
        analytics = get_user_analytics()
        analytics.register_user(
            user_id=update.effective_user.id,
            username=update.effective_user.username,
            first_name=update.effective_user.first_name,
            language_code=update.effective_user.language_code,
        )
        analytics.log_activity(update.effective_user.id, "scan_top20")
    except Exception as e:
        logger.debug(f"User analytics error: {e}")
    await update.message.reply_text("🔍 Сканирую ТОП-20...")

    from modules.alerts import smart_alerts
    opportunities = await smart_alerts.check_top20_opportunities(context)

    if opportunities:
        await update.message.reply_text(f"Найдено возможностей: {len(opportunities)}")
    else:
        await update.message.reply_text("Интересных движений не обнаружено")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    application = (
        ApplicationBuilder()
        .token(config.TOKEN)
        .defaults(Defaults(block=False))
        .post_init(on_startup)
        .build()
    )

    # Flask in background
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("movers", movers_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("health", health_command))
    application.add_handler(CommandHandler("news", news_command))
    application.add_handler(CommandHandler("add_entry", add_entry_command))
    application.add_handler(CommandHandler("sell", sell_command))

    # Admin commands (owner-only)
    application.add_handler(CommandHandler("grant", cmd_grant))
    application.add_handler(CommandHandler("revoke", cmd_revoke))
    application.add_handler(CommandHandler("user_stats", cmd_user_stats))
    application.add_handler(CommandHandler("check_user", cmd_check_user))
    # User tier command (available to all users)
    application.add_handler(CommandHandler("my_tier", cmd_my_tier))

    # Payment handlers (Telegram Stars)
    payment_manager = get_payment_manager()
    application.add_handler(PreCheckoutQueryHandler(payment_manager.handle_pre_checkout_query))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, payment_manager.handle_successful_payment))

    application.add_handler(CallbackQueryHandler(menu_callback))
    
    # Message handler for custom coin input (must be after command handlers and callback handlers)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_coin_input))
    
    application.add_error_handler(on_error)
    application.add_handler(CommandHandler("debug", snapshot_debug))
    application.add_handler(CommandHandler("scan", scan_top20_command))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    # Alerts job
    application.job_queue.run_repeating(run_alerts_scan, interval=config.ALERTS_INTERVAL_SECONDS, first=60, name="smart_alerts_system")
    # Learning validation job
    application.job_queue.run_repeating(learning_validation_job, interval=config.VALIDATION_INTERVAL_SECONDS, first=300, name="learning_validation")

    logging.info("RBRY CryptoBot started")
    retry_delay = 5
    while True:
        try:
            application.run_polling(close_loop=False)
            logging.info("Polling stopped normally")
            break
        except Exception as e:
            logging.exception(f"Polling crashed: {e}")
            logging.info(f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)

if __name__ == "__main__":
    main()





