"""Analysis buttons: one coin, the portfolio, the top 10, and a coin typed by hand."""

import asyncio
import html
import logging
import re
import time

import config
from config import OWNER_ID as USER_ID
from core.access.access_manager import get_access_manager
from core.access.presentation import format_access_decision
from core.ai_analytics import ai_analyzer
from core.analyzer_full import analyze_coin_full
from core.formatting import as_html, format_price, plain
from core.incidents import public_temporary_message, report_incident
from core.market_signal import classify_price_momentum
from core.portfolio_db import get_portfolio_manager
from core.security.rate_limiter_decorator import get_rate_limiter
from core.utils import COIN_ID_MAP, _track_user_activity, resolve_coin_id
from locales.translator import t
from modules import paper_view
from modules.callback_utils import answer_once
from modules.news import get_significant_news
from modules.screens import show_analyze_prompt
from modules.ui import _get_market_rows, _public_incident, create_action_buttons, create_back_button, get_global_app_ref
from modules.user_prefs import get_show_explanations
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def build_news_context(user_id: int = None):
    """Собирает лёгкий news-context для AI-объяснений.

    Единственное место этой логики (раньше блок был скопирован в 3 хендлерах).
    RSS + sentiment - блокирующий I/O, поэтому уводим в отдельный поток.
    """
    # "No news analysis" is None, never "neutral" (missing evidence is not neutral evidence).
    fallback = {"count": 0, "sentiment": None, "sources": ["cryptonews", "cointelegraph", "decrypt"]}
    try:
        items = await asyncio.to_thread(get_significant_news, 3)
        sources = []
        combined_text = ""
        for it in items:
            n = it.get("news", {})
            if n.get("source"):
                sources.append(n.get("source"))
            combined_text += " " + (n.get("summary") or "") + " " + (n.get("title") or "")

        sentiment = None
        if combined_text and ai_analyzer.finbert and ai_analyzer.finbert.status.get("available"):
            try:
                label, _score, _backend = await ai_analyzer.finbert.analyze_async(combined_text[:512])
                sentiment = "positive" if label.startswith("pos") else "negative" if label.startswith("neg") else "neutral"
            except Exception:
                sentiment = None

        return {
            "count": len(items),
            "sentiment": sentiment,
            "sources": sources or fallback["sources"],
        }
    except Exception:
        return fallback


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The analysis menu."""
    query = update.callback_query
    user_id = query.from_user.id
    await show_analyze_prompt(user_id, context, query.message, prefer_edit=True)


async def analyze_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Choose a coin: popular ones as buttons, any other typed."""
    query = update.callback_query
    user_id = query.from_user.id
    # Set flag that we're waiting for coin input
    context.user_data["awaiting_coin_input"] = True

    # Показываем популярные монеты для выбора
    popular_coins = ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX"]
    keyboard = []
    for i in range(0, len(popular_coins), 2):
        row = []
        for j in range(2):
            if i+j < len(popular_coins):
                coin = popular_coins[i+j]
                row.append(InlineKeyboardButton(coin, callback_data=f"analyze_coin:{coin}"))
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="analyze")])

    # Updated message with hint for custom coin input
    message_text = f"{t('analysis.choose_coin', user_id=user_id)}\n\n{t('analysis.custom_hint', user_id=user_id)}"

    await query.edit_message_text(
        message_text,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def analyze_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Every coin of the portfolio, one card each."""
    query = update.callback_query
    # ACCESS CONTROL CHECK
    user_id = query.from_user.id
    access_manager = get_access_manager()

    # Check rate limit
    rate_limiter = get_rate_limiter()
    tier = access_manager.get_user_tier(user_id)
    allowed, reason = rate_limiter.check_rate_limit(user_id, tier.value)
    if not allowed:
        await answer_once(query, t("errors.system_busy", user_id=user_id), show_alert=True)
        return

    decision = access_manager.evaluate_access(user_id, 'basic_analysis')
    if not decision.allowed:
        keyboard = [[InlineKeyboardButton(t("tier.button_premium", user_id=user_id), callback_data="premium")],
                   [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
        await query.edit_message_text(
            as_html(format_access_decision(decision, user_id)),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    t_start = time.time()
    timestamp = time.strftime("%H:%M:%S")
    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)
    if not portfolio:
        await query.edit_message_text(as_html(t("portfolio.empty", user_id=user_id)), parse_mode=ParseMode.HTML, reply_markup=create_back_button(user_id))
        return

    await answer_once(query)
    await query.edit_message_text(as_html(t("analysis.analyzing_portfolio", user_id=user_id, time=timestamp)), parse_mode=ParseMode.HTML)

    coin_ids = [resolve_coin_id(c) for c in portfolio.keys()]
    unavailable = [
        coin_id
        for coin_id in coin_ids
        if not access_manager.check_coin_limit(user_id, coin_id)[0]
    ]
    if unavailable:
        limits = access_manager.get_limits(user_id)
        reason = t(
            "errors.portfolio_coins_not_in_tier",
            user_id=user_id,
            coins=", ".join(unavailable),
            allowed=", ".join(limits.get('allowed_coins') or ["all"]),
        )
        # `decision` here is the *allowed* basic-analysis decision, whose
        # text is empty; the refusal reason is the coin-tier message.
        await query.edit_message_text(
            as_html(reason),
            parse_mode=ParseMode.HTML,
            reply_markup=create_back_button(user_id),
        )
        return

    rate_limiter.record_request(user_id)
    access_manager.record_request(user_id)

    show_explanations = get_show_explanations(user_id)
    news_context = await build_news_context(user_id) if show_explanations and config.AI_EXPLANATIONS_ENABLED else None

    market_rows, market_ts = await _get_market_rows(context.application, coin_ids)
    data_map = {cid: {"market": market_rows.get(cid, {})} for cid in coin_ids}
    available_count = sum(1 for coin_id in coin_ids if market_rows.get(coin_id))
    logger.info(
        "Portfolio market coverage rows=%s requested=%s snapshot_ts=%s",
        available_count,
        len(coin_ids),
        market_ts,
    )

    t_analysis = time.time()
    parts = []
    for coin_sym, entry in portfolio.items():
        t_coin = time.time()
        cid = resolve_coin_id(coin_sym)
        pack = data_map.get(cid, {}) or {}
        m = pack.get("market", {}) or {}

        price = m.get("current_price")
        if not isinstance(price, (int, float)):
            parts.append(t("analysis.no_market_data", user_id=user_id, coin=coin_sym.upper()))
            continue

        # базовый анализ
        analysis_txt = await analyze_coin_full(
            cid, price, entry, m, context,
            user_id=user_id, show_explanations=show_explanations, news_context=news_context
        )

        parts.append(analysis_txt)
        logger.info(f"  - {coin_sym} total: {time.time() - t_coin:.3f}s")

    logger.info(f"✅ Total analysis time: {time.time() - t_analysis:.3f}s")
    logger.info(f"✅ TOTAL REQUEST TIME: {time.time() - t_start:.3f}s")

    blocks = parts if parts else [t("errors.analysis_error", user_id=user_id)]
    max_len = 4096
    current = ""
    first = True
    for block in blocks:
        if not current:
            current = block
        elif len(current) + 2 + len(block) > max_len:
            if first:
                await query.edit_message_text(as_html(current), parse_mode=ParseMode.HTML, reply_markup=create_back_button(user_id))
                first = False
            else:
                await query.message.reply_text(as_html(current), parse_mode=ParseMode.HTML)
            current = block
        else:
            current += "\n\n" + block
    if current:
        if first:
            await query.edit_message_text(as_html(current), parse_mode=ParseMode.HTML, reply_markup=create_back_button(user_id))
        else:
            await query.message.reply_text(as_html(current), parse_mode=ParseMode.HTML)


async def analyze_top10(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The ten largest coins without stablecoins and wrapped tokens."""
    query = update.callback_query
    # ACCESS CONTROL CHECK
    user_id = query.from_user.id
    access_manager = get_access_manager()

    # Check rate limit
    rate_limiter = get_rate_limiter()
    tier = access_manager.get_user_tier(user_id)
    allowed, reason = rate_limiter.check_rate_limit(user_id, tier.value)
    if not allowed:
        await answer_once(query, t("errors.system_busy", user_id=user_id), show_alert=True)
        return

    # TOP-10 is a paid batch feature in the central tier policy.
    decision = access_manager.evaluate_access(user_id, 'top10_analysis')
    if not decision.allowed:
        keyboard = [[InlineKeyboardButton(t("tier.button_premium", user_id=user_id), callback_data="premium")],
                   [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
        await query.edit_message_text(
            as_html(format_access_decision(decision, user_id)),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    # Record request
    rate_limiter.record_request(user_id)
    access_manager.record_request(user_id)

    await answer_once(query)
    await query.edit_message_text(as_html(t("analysis.analyzing_top10", user_id=user_id)), parse_mode=ParseMode.HTML)

    try:
        rows = await _top_markets_without_stables(context.application)
        if not rows:
            raise RuntimeError("top10 market coverage is empty")
    except Exception as e:
        error_text = await _public_incident(context, e, user_id, "top10_analysis")
        keyboard = [
            [InlineKeyboardButton(t("analysis.retry", user_id=user_id), callback_data="analyze_top10")],
            [InlineKeyboardButton(t("analysis.back_to_analysis", user_id=user_id), callback_data="analyze")],
            [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
        ]
        await query.edit_message_text(as_html(error_text), parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))
        return

    await query.edit_message_text(
        build_top10_text(rows, user_id), parse_mode=ParseMode.HTML, reply_markup=create_back_button(user_id)
    )


async def analyze_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """One coin's card."""
    query = update.callback_query
    # ACCESS CONTROL CHECK
    user_id = query.from_user.id
    access_manager = get_access_manager()
    rate_limiter = get_rate_limiter()
    tier = access_manager.get_user_tier(user_id)

    # Check rate limit
    allowed, reason = rate_limiter.check_rate_limit(user_id, tier.value)
    if not allowed:
        await answer_once(query, t("errors.system_busy", user_id=user_id), show_alert=True)
        return

    # 1) монета
    symbol = query.data.split(":", 1)[1].upper()
    coin_id = resolve_coin_id(symbol)

    # Check basic analysis access (FREE tier can use this)
    decision = access_manager.evaluate_access(user_id, 'basic_analysis')
    if not decision.allowed:
        keyboard = [
            [InlineKeyboardButton(t("premium.button", user_id=user_id), callback_data="premium")],
            [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]
        ]
        await query.edit_message_text(
            format_access_decision(decision, user_id),
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    # Check coin limit (FREE users can only analyze BTC/ETH)
    decision = access_manager.evaluate_coin_limit(user_id, coin_id)
    if not decision.allowed:
        keyboard = [
            [InlineKeyboardButton(t("tier.button_premium", user_id=user_id), callback_data="premium")],
            [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]
        ]
        await query.edit_message_text(
            format_access_decision(decision, user_id),
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    # Record request
    rate_limiter.record_request(user_id)
    access_manager.record_request(user_id)
    await answer_once(query)

    # 2) свежий маркет из снапшота (без сети)
    app = get_global_app_ref()
    markets, _snapshot_ts = await _get_market_rows(app, [coin_id])
    row = markets.get(coin_id) or {}

    current_price = row.get("current_price")  # аккуратно: None-guard уже есть утилитами в файле

    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)
    entry = portfolio.get(symbol, {}) or portfolio.get(coin_id, {})  # поддержим оба ключа

    try:
        show_explanations = get_show_explanations(user_id)
        news_context = await build_news_context(user_id) if show_explanations and config.AI_EXPLANATIONS_ENABLED else None

        text = await analyze_coin_full(
            coin_id=coin_id,
            current_price=float(current_price or 0.0),
            portfolio_entry=entry,
            indicators=row,
            context=context,
            user_id=user_id,
            show_explanations=show_explanations,
            news_context=news_context
        )
        if paper_view.enabled(user_id, USER_ID):     # the owner's test bot: the brain's verdict under it
            text += "\n\n" + paper_view.brain_block(symbol)
        await query.edit_message_text(as_html(text), parse_mode=ParseMode.HTML,
                                      reply_markup=create_action_buttons(user_id, plan_symbol=symbol))
    except Exception as e:
        error_text = await _public_incident(context, e, user_id, "coin_analysis")
        keyboard = [
            [InlineKeyboardButton(t("analysis.retry", user_id=user_id), callback_data=f"analyze_coin:{symbol}")],
            [InlineKeyboardButton(t("analysis.back_to_analysis", user_id=user_id), callback_data="analyze")],
            [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
        ]
        await query.edit_message_text(as_html(error_text), parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))


async def typed_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """A coin typed after "choose a coin": the same checks and the same card as the coin buttons."""
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "custom_coin_input", {"text": (update.message.text if update.message else "")[:20]})

    # Clear the flag
    context.user_data["awaiting_coin_input"] = False

    # Get the coin symbol from message (basic input hygiene)
    coin_symbol = update.message.text.strip().upper()[:20]
    if not re.fullmatch(r"[A-Z0-9\-]{2,20}", coin_symbol):
        await update.message.reply_text(t("analysis.coin_not_found", user_id=user_id))
        return

    # Check access control
    access_manager = get_access_manager()
    rate_limiter = get_rate_limiter()
    tier = access_manager.get_user_tier(user_id)

    # Check rate limit
    allowed, reason = rate_limiter.check_rate_limit(user_id, tier.value)
    if not allowed:
        await update.message.reply_text(t("errors.system_busy", user_id=user_id))
        return

    # Check basic analysis access
    decision = access_manager.evaluate_access(user_id, 'basic_analysis')
    if not decision.allowed:
        await update.message.reply_text(format_access_decision(decision, user_id))
        return

    registry = context.application.bot_data.get("coin_registry")
    coin_id = resolve_coin_id(coin_symbol, registry)

    # Coin limit must apply to typed input too, not only to buttons -
    # otherwise FREE users bypass the BTC/ETH restriction by typing
    decision = access_manager.evaluate_coin_limit(user_id, coin_id)
    if not decision.allowed:
        await update.message.reply_text(format_access_decision(decision, user_id))
        return

    # Count this request against the user's cooldown (was missing: the FREE
    # 2h cooldown never applied to the typed-input path)
    rate_limiter.record_request(user_id)
    access_manager.record_request(user_id)

    # Show analyzing message
    await update.message.reply_text(t("analysis.analyzing_coin", user_id=user_id, coin=coin_symbol))
    
    # Get market data
    try:
        rows, _snapshot_ts = await _get_market_rows(context.application, [coin_id])
        row = rows.get(coin_id)
        
        if not row:
            registry = context.application.bot_data.get("coin_registry")
            is_known = coin_symbol in COIN_ID_MAP or (
                registry is not None and registry.resolve(coin_symbol) is not None
            )
            key = "analysis.market_data_updating" if is_known else "analysis.coin_not_found"
            await update.message.reply_text(
                t(key, user_id=user_id, coin=coin_symbol)
            )
            return
        
        current_price = row.get("current_price")
        
        pm = get_portfolio_manager()
        portfolio = pm.load_portfolio(user_id)
        entry = portfolio.get(coin_symbol, {}) or portfolio.get(coin_id, {})

        show_explanations = get_show_explanations(user_id)
        news_context = await build_news_context(user_id) if show_explanations and config.AI_EXPLANATIONS_ENABLED else None

        text = await analyze_coin_full(
            coin_id=coin_id,
            current_price=float(current_price or 0.0),
            portfolio_entry=entry,
            indicators=row,
            context=context,
            user_id=user_id,
            show_explanations=show_explanations,
            news_context=news_context
        )
        
        # Send with back button
        keyboard = [[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="analyze")]]
        await update.message.reply_text(
            as_html(text), 
            parse_mode=ParseMode.HTML, 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    except Exception as e:
        await report_incident(
            context,
            e,
            subsystem="typed_coin_analysis",
            user_id=user_id,
        )
        await update.message.reply_text(public_temporary_message(user_id))


# Stablecoins, wrapped and staked tokens track another asset: they would
# fill a "top 10" without saying anything about the market.
_NOT_MARKET_SYMBOLS = {
    "usdt", "usdc", "usds", "dai", "fdusd", "usde", "tusd", "pyusd", "usd1", "usdd",
    "busd", "rlusd", "susde", "usdtb", "bsc-usd", "wbtc", "weth", "steth", "wsteth",
    "weeth", "wbeth", "cbbtc", "reth", "lbtc", "solvbtc", "meth", "figr_heloc",
}
_TOP_CACHE = {"ts": 0.0, "rows": []}
_TOP_CACHE_TTL = 300


def _is_market_coin(row: dict) -> bool:
    symbol = str(row.get("symbol") or "").lower()
    name = str(row.get("name") or "").lower()
    if symbol in _NOT_MARKET_SYMBOLS or name.startswith(("wrapped ", "staked ")):
        return False
    price = row.get("current_price")
    # Unlisted dollar stablecoins: a USD-named coin priced at ~$1.
    return not (symbol.startswith("usd") and isinstance(price, (int, float)) and 0.97 < price < 1.03)


async def _top_markets_without_stables(app, limit: int = 10) -> list:
    """Live top coins by market cap (cached 5 min); snapshot rows as fallback."""
    now = time.time()
    if now - _TOP_CACHE["ts"] < _TOP_CACHE_TTL and _TOP_CACHE["rows"]:
        return _TOP_CACHE["rows"][:limit]
    try:
        from core.api.coingecko_client import get_top_markets

        rows = [row for row in await get_top_markets(limit=40) if _is_market_coin(row)]
        if rows:
            _TOP_CACHE.update(ts=now, rows=rows)
            return rows[:limit]
    except Exception as error:
        logger.warning("Live top markets unavailable, using snapshot: %s", error)
    from modules.alerts import AlertConfig

    markets, _ts = await _get_market_rows(app, AlertConfig.TOP20_COINS)
    rows = sorted((r for r in markets.values() if r and _is_market_coin(r)),
                  key=lambda r: r.get("market_cap") or 0, reverse=True)
    return rows[:limit]


def build_top10_text(rows: list, user_id) -> str:
    """Two readable lines per coin plus a market summary on top."""
    def change(row, horizon):
        value = row.get(f"price_change_percentage_{horizon}_in_currency", row.get(f"price_change_percentage_{horizon}"))
        return value if isinstance(value, (int, float)) else None

    def fmt(value):
        return f"{value:+.1f}%" if value is not None else "n/a"

    ch24s = [change(r, "24h") for r in rows]
    ch7s = [change(r, "7d") for r in rows]
    parts = [
        f"<b>{plain(t('analyze.top10_title', user_id=user_id))}</b>",
        plain(t("analyze.top10_summary", user_id=user_id,
                up24=sum(1 for v in ch24s if v is not None and v > 0),
                up7=sum(1 for v in ch7s if v is not None and v > 0), n=len(rows))),
        "",
    ]
    for rank, (row, ch24, ch7) in enumerate(zip(rows, ch24s, ch7s), start=1):
        name = html.escape(str(row.get("name") or row.get("id") or "?"))
        symbol = html.escape(str(row.get("symbol") or "").upper())
        label = plain(t(classify_price_momentum(ch24, ch7).translation_key, user_id=user_id))
        parts.append(f"{rank}. <b>{name} ({symbol})</b> · {format_price(row.get('current_price'))}")
        parts.append(f"     {plain(t('analyze.top10_changes', user_id=user_id, ch24=fmt(ch24), ch7d=fmt(ch7)))} · {label}")
    parts += ["", f"<i>{plain(t('analyze.top10_footer', user_id=user_id))}</i>"]
    return "\n".join(parts)
