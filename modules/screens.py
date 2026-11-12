"""The bot's screens: help, settings, the analysis menu, the portfolio, its statistics and movers."""

import datetime
import html
import math

import config
from config import OWNER_ID as USER_ID
from core.formatting import as_html, format_price, plain
from core.portfolio_db import get_portfolio_manager
from core.portfolio_valuation import value_portfolio
from core.utils import resolve_coin_id
from locales.translator import t
from modules import paper_view
from modules.ui import _get_market_rows, fmt_asof_for_user, get_global_app_ref, respond_message
from modules.user_prefs import get_profile, get_show_explanations
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes


def build_help_text(user_id: int) -> str:
    parts = [
        t("help.title", user_id=user_id),
        "",
        t("help.commands_basic", user_id=user_id),
        t("help.cmd_start", user_id=user_id),
        t("help.cmd_help", user_id=user_id),
        t("help.cmd_portfolio", user_id=user_id),
        "",
        t("help.commands_portfolio", user_id=user_id),
        t("help.cmd_add_entry", user_id=user_id),
        t("help.cmd_sell", user_id=user_id),
        t("help.cmd_stats", user_id=user_id),
        t("help.cmd_movers", user_id=user_id),
        "",
        t("help.commands_analysis", user_id=user_id),
        t("help.cmd_analyze", user_id=user_id),
        t("help.cmd_news", user_id=user_id),
        "",
        t("help.commands_alerts", user_id=user_id),
        t("help.cmd_scan_now", user_id=user_id),
        "",
        t("help.example", user_id=user_id),
    ]
    if config.ALERTS_ENABLED:
        # Only claim monitoring that actually runs on this instance.
        parts += ["", t("help.alerts_active", user_id=user_id), t("help.alerts_desc", user_id=user_id)]
    return "\n".join(parts)


async def show_help(user_id: int, message, prefer_edit: bool):
    text = build_help_text(user_id)
    keyboard = [[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]
    await respond_message(
        message,
        as_html(text),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
        prefer_edit=prefer_edit,
    )


LANGUAGE_LABELS = {"ru": "Русский", "uk": "Українська", "en": "English", "de": "Deutsch"}


async def show_settings(user_id: int, message, prefer_edit: bool):
    prof = get_profile(user_id)
    show_expl = get_show_explanations(user_id)
    expl_status = t("settings.explanations_on", user_id=user_id) if show_expl else t("settings.explanations_off", user_id=user_id)
    lang_code = prof.get("lang") or "ru"
    language = LANGUAGE_LABELS.get(lang_code, lang_code)
    text = (
        f"{t('settings.title', user_id=user_id)}\n"
        f"{t('settings.user_id', user_id=user_id)}\n"
        f"{t('settings.timezone', user_id=user_id, tz=prof.get('tz') or t('settings.not_set', user_id=user_id))}\n"
        f"{t('settings.language', user_id=user_id, lang=language, language=language)}\n"
    )
    keyboard = [
        [InlineKeyboardButton(t("settings.button_timezone", user_id=user_id), callback_data="settings_tz")],
        [InlineKeyboardButton(t("settings.button_language", user_id=user_id), callback_data="settings_lang")],
    ]
    if config.AI_EXPLANATIONS_ENABLED:
        # The toggle only exists while there is something to toggle.
        text += f"{t('settings.explanations', user_id=user_id, status=expl_status)}\n"
        keyboard.append([InlineKeyboardButton(t("settings.button_explanations", user_id=user_id), callback_data="settings_explanations")])
    keyboard.append([InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")])
    await respond_message(
        message,
        as_html(text),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
        prefer_edit=prefer_edit,
    )


async def show_analyze_prompt(user_id: int, context: ContextTypes.DEFAULT_TYPE, message, prefer_edit: bool):
    keyboard = [
        [InlineKeyboardButton(t("analysis.my_portfolio", user_id=user_id), callback_data="analyze_portfolio")],
        [InlineKeyboardButton(t("analysis.top_10", user_id=user_id), callback_data="analyze_top10")],
        [InlineKeyboardButton(t("analysis.choose_coins", user_id=user_id), callback_data="analyze_custom")],
        [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]
    ]
    await respond_message(
        message,
        t("analysis.title", user_id=user_id),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
        prefer_edit=prefer_edit,
    )


async def show_paper(message, screen: str, prefer_edit: bool):
    """The test bot's paper account (modules/paper_view.py) for its owner."""
    text, keyboard = paper_view.all_screen() if screen == "all" else paper_view.account_screen()
    await respond_message(message, text, parse_mode=ParseMode.HTML, reply_markup=keyboard, prefer_edit=prefer_edit)


async def show_portfolio(user_id: int, context: ContextTypes.DEFAULT_TYPE, message, prefer_edit: bool):
    if paper_view.enabled(user_id, USER_ID):          # the owner's test bot: the paper account in its place
        await show_paper(message, "account", prefer_edit)
        return
    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)

    if not portfolio:
        text = t("portfolio.empty_hint", user_id=user_id)
        keyboard = [
            [InlineKeyboardButton(t("portfolio.add_coin_button", user_id=user_id), callback_data="add_coin_prompt")],
            [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
        ]
        await respond_message(message, text, reply_markup=InlineKeyboardMarkup(keyboard), prefer_edit=prefer_edit)
        return

    coin_ids = [resolve_coin_id(coin) for coin in portfolio]
    rows, snapshot_ts = await _get_market_rows(context.application, coin_ids)
    as_of = (
        datetime.datetime.fromtimestamp(snapshot_ts, datetime.timezone.utc).strftime("%H:%M:%S") + " UTC"
        if snapshot_ts
        else "n/a"
    )

    valuation = value_portfolio(
        portfolio,
        rows,
        coin_id_resolver=resolve_coin_id,
    )
    lines = [f"<b>{plain(t('portfolio.title', user_id=user_id))}</b>", "═" * 20, ""]

    for position in valuation.positions:
        coin = position.symbol
        amount = position.amount
        entry_price = position.entry_price
        current_price = position.current_price

        if current_price is not None:
            current_value = amount * current_price
            cost_basis = amount * entry_price
            pnl_usd = current_value - cost_basis
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            pnl_emoji = "📈" if pnl_usd > 0 else "📉" if pnl_usd < 0 else "➡️"

            entry_str = format_price(entry_price)
            current_str = format_price(current_price)
            value_str = format_price(current_value)

            lines.append(
                f"<b>{html.escape(coin.upper())}</b>\n"
                f"  {t('portfolio.amount_label', user_id=user_id)}: {amount:.6f}\n"
                f"  {t('portfolio.entry_price_label', user_id=user_id)}: {entry_str}\n"
                f"  {t('portfolio.current_price_label', user_id=user_id)}: {current_str}\n"
                f"  {t('portfolio.current_value_label', user_id=user_id)}: {value_str}\n"
                f"  {pnl_emoji} {t('portfolio.pnl_label', user_id=user_id)}: ${pnl_usd:+.2f} ({pnl_pct:+.1f}%)\n"
            )
        else:
            entry_str = format_price(entry_price)
            lines.append(
                f"<b>{html.escape(coin.upper())}</b>\n"
                f"  {t('portfolio.amount_label', user_id=user_id)}: {amount:.6f}\n"
                f"  {t('portfolio.entry_price_label', user_id=user_id)}: {entry_str}\n"
                f"  {t('portfolio.price_temporarily_unavailable', user_id=user_id)}\n"
            )

    total_cost_str = format_price(valuation.total_invested)
    if valuation.is_complete:
        total_value = valuation.total_current_value or 0.0
        total_pnl = valuation.total_pnl or 0.0
        total_pnl_pct = valuation.total_pnl_pct
        pnl_emoji = "📈" if total_pnl > 0 else "📉" if total_pnl < 0 else "➡️"
        total_value_str = format_price(total_value)
        pnl_pct_text = f" ({total_pnl_pct:+.1f}%)" if total_pnl_pct is not None else ""

        lines.append(
            f"\n{'═' * 20}\n"
            f"<b>{plain(t('portfolio.total_header', user_id=user_id))}</b>\n"
            f"  {t('portfolio.invested_label', user_id=user_id)}: {total_cost_str}\n"
            f"  {t('portfolio.current_value_label', user_id=user_id)}: {total_value_str}\n"
            f"  {pnl_emoji} {t('portfolio.pnl_label', user_id=user_id)}: ${total_pnl:+.2f}{pnl_pct_text}"
        )
    else:
        known_value_str = format_price(valuation.known_current_value)
        missing = ", ".join(valuation.missing_symbols)
        lines.append(
            f"\n{'═' * 20}\n"
            f"<b>{plain(t('portfolio.total_header', user_id=user_id))}</b>\n"
            f"  {t('portfolio.invested_label', user_id=user_id)}: {total_cost_str}\n"
            f"  {t('portfolio.known_value_label', user_id=user_id)}: {known_value_str}\n"
            f"  {t('portfolio.incomplete_totals', user_id=user_id, coins=missing)}"
        )

    lines.append(f"\n<i>{plain(t('portfolio.as_of', user_id=user_id, time=as_of))}</i>")

    keyboard = [
        [InlineKeyboardButton(t("portfolio.add_coin_button", user_id=user_id), callback_data="add_coin_prompt"),
         InlineKeyboardButton(t("portfolio.sell_coin_button", user_id=user_id), callback_data="sell_coin_prompt")],
        [InlineKeyboardButton(t("portfolio.analyze_button", user_id=user_id), callback_data="analyze"),
         InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
    ]
    await respond_message(
        message,
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
        prefer_edit=prefer_edit,
    )


async def show_stats(user_id: int, context: ContextTypes.DEFAULT_TYPE, message, prefer_edit: bool):
    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)

    if not portfolio:
        text = t("stats.empty_message", user_id=user_id)
        keyboard = [
            [InlineKeyboardButton(t("portfolio.add_coin_button", user_id=user_id), callback_data="add_coin_prompt")],
            [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
        ]
        await respond_message(message, text, reply_markup=InlineKeyboardMarkup(keyboard), prefer_edit=prefer_edit)
        return

    app = (context.application or get_global_app_ref())
    cids = [resolve_coin_id(c) for c in portfolio.keys()]
    rows, snapshot_ts = await _get_market_rows(app, cids)
    as_of = fmt_asof_for_user(user_id, snapshot_ts) if snapshot_ts else "n/a"
    if not rows:
        await respond_message(
            message,
            t("stats.no_data", user_id=user_id),
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
            prefer_edit=prefer_edit,
        )
        return

    valuation = value_portfolio(
        portfolio,
        rows or {},
        coin_id_resolver=resolve_coin_id,
    )
    msg = [t("stats.title", user_id=user_id), t("stats.divider", user_id=user_id), ""]

    def _f(x, default=None):
        try:
            value = float(x)
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    for coin, data in portfolio.items():
        cid = resolve_coin_id(coin)
        entry = _f(data.get("buy_price", data.get("price", 0.0)), 0.0)
        row = (rows or {}).get(cid, {}) or {}

        now = _f(row.get("current_price"), None)
        ch1 = row.get("price_change_percentage_1h_in_currency", row.get("price_change_percentage_1h"))
        ch24 = row.get("price_change_percentage_24h_in_currency", row.get("price_change_percentage_24h"))
        ch7 = row.get("price_change_percentage_7d_in_currency", row.get("price_change_percentage_7d"))

        if now is None:
            continue

        change = ((now - (entry or 0.0)) / (entry or 1.0) * 100.0) if entry else 0.0

        def fmt(x):
            return (
                f"{x:+.1f}%"
                if isinstance(x, (int, float))
                else t("common.not_available", user_id=user_id)
            )

        msg += [
            "━━━━━━━━━━━━━━━━━",
            t("stats.coin_header", user_id=user_id, coin=coin.upper()),
            t("stats.price", user_id=user_id, price=format_price(now)),
            t("stats.change_1h", user_id=user_id, change=fmt(ch1)),
            t("stats.change_24h", user_id=user_id, change=fmt(ch24)),
            t("stats.change_7d", user_id=user_id, change=fmt(ch7)),
            t("stats.profit_loss", user_id=user_id, pl=change),
        ]

    msg += ["", t("stats.total_invested", user_id=user_id, amount=valuation.total_invested)]
    if valuation.is_complete:
        msg.append(t("stats.current_value", user_id=user_id, amount=valuation.total_current_value or 0.0))
    else:
        msg += [
            t("portfolio.known_value", user_id=user_id, amount=valuation.known_current_value),
            t("portfolio.incomplete_totals", user_id=user_id, coins=", ".join(valuation.missing_symbols)),
        ]
    msg.append(t("portfolio.as_of", user_id=user_id, time=as_of))

    await respond_message(
        message,
        as_html("\n".join(msg)),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
        prefer_edit=prefer_edit,
    )


async def show_movers(user_id: int, context: ContextTypes.DEFAULT_TYPE, message, prefer_edit: bool):
    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)
    if not portfolio:
        text = t("movers.empty_message", user_id=user_id)
        keyboard = [
            [InlineKeyboardButton(t("portfolio.add_coin_button", user_id=user_id), callback_data="add_coin_prompt")],
            [InlineKeyboardButton(t("common.main_menu", user_id=user_id), callback_data="back_to_menu")]
        ]
        await respond_message(message, text, reply_markup=InlineKeyboardMarkup(keyboard), prefer_edit=prefer_edit)
        return

    app = (context.application or get_global_app_ref())
    cids = [resolve_coin_id(c) for c in portfolio.keys()]
    rows, snapshot_ts = await _get_market_rows(app, cids)
    as_of = fmt_asof_for_user(user_id, snapshot_ts) if snapshot_ts else "n/a"
    if not rows:
        await respond_message(
            message,
            as_html(t("movers.no_data", user_id=user_id)),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
            prefer_edit=prefer_edit,
        )
        return

    movers = []
    for coin, data in portfolio.items():
        cid = resolve_coin_id(coin)
        entry = float(data.get("buy_price", data.get("price", 0.0)) or 0.0)
        now = rows.get(cid, {}).get("current_price")
        if not isinstance(now, (int, float)) or entry <= 0:
            continue
        change_pct = (now - entry) / entry * 100.0
        movers.append((coin.upper(), change_pct, now, entry))

    movers.sort(key=lambda x: x[1], reverse=True)

    if not movers:
        await respond_message(
            message,
            as_html(t("movers.no_data", user_id=user_id)),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
            prefer_edit=prefer_edit,
        )
        return

    msg = [t("movers.title", user_id=user_id), t("movers.divider", user_id=user_id), ""]
    for coin, change, current, entry in movers:
        emoji = "🚀" if change > 10 else "📈" if change > 0 else "📉" if change > -10 else "💥"
        color = "🟢" if change > 0 else "🔴"
        change_str = f"{color}"
        msg += [
            t("movers.coin_entry", user_id=user_id, emoji=emoji, coin=coin, change_str=change_str),
            t("movers.change_line", user_id=user_id, change=change),
            t("movers.price_line", user_id=user_id, entry=format_price(entry), current=format_price(current)),
            ""
        ]
    msg.append(t("portfolio.as_of", user_id=user_id, time=as_of))

    await respond_message(
        message,
        as_html("\n".join(msg)),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]]),
        prefer_edit=prefer_edit,
    )
