import logging
import asyncio
import time
from typing import List, Dict, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from core.utils import COIN_ID_MAP, _row_change_pct
from core.api.coingecko_client import get_batch_historical_data
from core.api.binance_client import get_batch_ohlcv
from core.ai_analytics import ai_analyzer
from core.ai_explainer import generate_explanations
from core.history_cache import get as hist_get, set as hist_set
from core.analysis.technical_analysis import rsi, sma, macd
from core.analysis.signal_generator import (
    detect_patterns,
    detect_candlestick_patterns,
    detect_chart_formations,
)
from core.data_collectors.fear_greed_service import get_fear_greed_service
from locales.translator import t

logger = logging.getLogger(__name__)

_PATTERN_CACHE: Dict[tuple, list] = {}
_PATTERN_CACHE_TTL_BUCKET_SEC = 3600  # 1h buckets for 1h candles


def _pattern_cache_key(coin_id: str, window_hours: int) -> tuple:
    bucket = int(time.time() // _PATTERN_CACHE_TTL_BUCKET_SEC)
    return (coin_id, window_hours, bucket)


def _summarize_patterns(patterns: list, window_label: str) -> list:
    out = []
    for p in patterns:
        p_type = p.get("type") or p.get("pattern") or p.get("signal") or "pattern"
        strength = p.get("strength")
        if strength:
            out.append(f"{p_type} ({strength}, {window_label})")
        else:
            out.append(f"{p_type} ({window_label})")
    return out


async def get_historical_data(coin_id, days):
    """Fetch (timestamp, price, volume) for N days via batch API + local TTL cache."""
    cached = hist_get(coin_id, days)
    if cached is not None:
        return cached

    try:
        data_map = await get_batch_historical_data([coin_id], days=days)
        data = data_map.get(coin_id) or {}
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        import pandas as pd
        if not prices:
            return None

        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"]) if volumes else None

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        if vol_df is not None and not vol_df.empty:
            vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms")
            df = df.merge(vol_df, on="timestamp", how="left")

        # Alias for compatibility with pattern detectors.
        df["close"] = df["price"]

        hist_set(coin_id, days, df)
        return df
    except Exception as e:
        logger.error(f"Historical data error for {coin_id}: {e}")
        return None


async def analyze_coin_full(
    coin_id,
    current_price,
    portfolio_entry,
    indicators,
    context,
    user_id: int,
    show_explanations: bool,
    news_context: Optional[dict] = None,
):
    """Full analysis with AI growth assessment and optional explanations."""
    t_start = time.time()
    amount = portfolio_entry.get("amount", 0)
    avg_price = portfolio_entry.get("buy_price", portfolio_entry.get("price", 0))

    # Technical indicators
    tech_indicators = {}
    hist_snapshot = context.application.bot_data.get("history_snapshot") if context and context.application else None
    if hist_snapshot and hist_snapshot.get_latest():
        df = hist_snapshot.get_latest().by_coin.get(coin_id)
        if df is not None and len(df) >= 14:
            tech_indicators["rsi"] = rsi(df["price"], 14).iloc[-1]
            if len(df) >= 20:
                tech_indicators["sma_20"] = sma(df["price"], 20).iloc[-1]
            if len(df) >= 26:
                macd_line, _signal_line, _ = macd(df["price"])
                tech_indicators["macd"] = macd_line.iloc[-1] if not macd_line.empty else None

    growth_potential = await ai_analyzer.assess_growth_potential(
        coin_id, current_price, tech_indicators, indicators
    )

    profit_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price else 0
    rsi_val = tech_indicators.get("rsi")

    msg = []
    msg.append("━━━━━━━━━━━━━━━━━")
    msg.append(f"💰 **{coin_id.upper()}**")
    msg.append(f"Цена: ${current_price:.2f}")
    msg.append(f"Ваша позиция: {amount:.6f} @ ${avg_price:.2f}")
    msg.append(f"📊 P/L: {profit_pct:+.2f}%")
    msg.append("")

    msg.append("📈 **Индикаторы:**")
    if rsi_val is not None:
        rsi_signal = (
            "🔴 Перепродан" if rsi_val < 30 else
            "🟢 Перекуплен" if rsi_val > 70 else
            "⚪ Нейтрально"
        )
        msg.append(f"RSI(14): {rsi_val:.1f} {rsi_signal}")

    if "macd" in tech_indicators:
        macd_val = tech_indicators["macd"]
        if macd_val is not None:
            macd_signal = "📈 Бычий" if macd_val > 0 else "📉 Медвежий"
            msg.append(f"MACD: {macd_signal}")
    msg.append("")

    msg.append("🤖 **AI Прогноз:**")
    msg.append(
        f"Оценка потенциала: "
        f"{'🟢' * (growth_potential['score'] // 2)}"
        f"{'⚪' * (5 - growth_potential['score'] // 2)}"
    )
    msg.append(f"Рекомендация: **{growth_potential['recommendation']}**")
    msg.append(
        f"Целевая цена: ${growth_potential['target_price']:.2f} "
        f"({((growth_potential['target_price'] / current_price - 1) * 100):+.1f}%)"
    )
    msg.append(f"Уровень риска: {growth_potential['risk_level']}")

    if growth_potential["score"] >= 7:
        msg.append("💡 AI видит хороший потенциал роста")
    elif growth_potential["score"] <= 3:
        msg.append("⚠️ AI рекомендует осторожность")

    if show_explanations:
        hist_days = None
        hist_rows = None
        hist_ts = None
        df_hist = None
        if hist_snapshot and hist_snapshot.get_latest():
            hist_days = getattr(hist_snapshot, "days", None)
            df_hist = hist_snapshot.get_latest().by_coin.get(coin_id)
            if df_hist is not None and len(df_hist) > 0:
                hist_rows = len(df_hist)
                if "timestamp" in df_hist.columns:
                    hist_ts = df_hist["timestamp"].iloc[-1]

        volume_ratio = None
        if df_hist is not None and "volume" in df_hist.columns and len(df_hist) >= 20:
            avg_vol = df_hist["volume"].tail(20).mean()
            cur_vol = df_hist["volume"].iloc[-1]
            if avg_vol and avg_vol > 0:
                volume_ratio = cur_vol / avg_vol

        ch24 = _row_change_pct(indicators, "24h")
        ch7d = _row_change_pct(indicators, "7d")

        # Binance patterns (24h/7d/30d)
        patterns_summary = []
        patterns_source = "Binance"
        patterns_detail = "klines 1h, 24h/7d/30d"
        binance_rows = None
        binance_ts = None
        ohlcv_data = await get_batch_ohlcv([coin_id], interval="1h")
        df_binance = ohlcv_data.get(coin_id) if ohlcv_data else None
        if df_binance is None or len(df_binance) < 20:
            logger.info(f"PATTERNS_SKIP coin={coin_id} reason=binance_missing_or_short")
        else:
            df_binance = df_binance.tail(720)
            binance_rows = len(df_binance)
            if len(df_binance) > 0:
                binance_ts = df_binance.index[-1]
            windows = [(24, "24h"), (168, "7d"), (720, "30d")]
            for wh, label in windows:
                key = _pattern_cache_key(coin_id, wh)
                cached = _PATTERN_CACHE.get(key)
                if cached is not None:
                    patterns_summary += _summarize_patterns(cached, label)
                    logger.info(f"PATTERNS_CACHE coin={coin_id} window={label} count={len(cached)}")
                    continue
                df_win = df_binance.tail(wh)
                patterns = (
                    detect_patterns(df_win)
                    + detect_candlestick_patterns(df_win)
                    + detect_chart_formations(df_win)
                )
                _PATTERN_CACHE[key] = patterns
                patterns_summary += _summarize_patterns(patterns, label)
                logger.info(f"PATTERNS_CALC coin={coin_id} window={label} count={len(patterns)}")

        # Fear & Greed
        fear_greed_val = None
        try:
            fear_greed_val = await get_fear_greed_service().get_current_index()
        except Exception:
            fear_greed_val = None

        # Synthesis facts
        synth_signal = None
        synth_conf = None
        if growth_potential.get("_using_synthesis") and growth_potential.get("_synthesis_analysis"):
            synth_signal = growth_potential["_synthesis_analysis"].signal_type
            synth_conf = growth_potential["_synthesis_analysis"].confidence
        else:
            if growth_potential.get("recommendation") == "ПРОДАВАТЬ":
                synth_signal = "sell"
            elif growth_potential.get("recommendation") == "ПОКУПАТЬ":
                synth_signal = "buy"
            else:
                synth_signal = "hold"
            synth_conf = growth_potential.get("confidence")

        sections = {}
        sections["quotes"] = {
            "facts": (
                f"price=${current_price:.2f}, entry=${avg_price:.2f}, "
                f"amount={amount:.6f}, pnl={profit_pct:+.2f}%"
            )
        }

        facts = []
        if rsi_val is not None:
            facts.append(f"RSI={rsi_val:.1f}")
        if "macd" in tech_indicators:
            macd_val = tech_indicators.get("macd")
            if macd_val is not None:
                facts.append("MACD=bearish" if macd_val < 0 else "MACD=bullish")
        if ch24 is not None:
            facts.append(f"24h={ch24:+.2f}%")
        if ch7d is not None:
            facts.append(f"7d={ch7d:+.2f}%")
        if volume_ratio is not None:
            facts.append(f"volume_ratio={volume_ratio:.2f}x")
        sections["tech"] = {"facts": ", ".join(facts)} if facts else {"facts": "RSI=50.0, MACD=neutral"}

        pattern_facts = "none"
        if patterns_summary:
            pattern_facts = ", ".join(patterns_summary[:8])
        sections["patterns"] = {"facts": "patterns=" + pattern_facts}

        if fear_greed_val is not None:
            sections["sentiment"] = {
                "facts": f"fear_greed={fear_greed_val:.0f}/100 (0=extreme fear, 100=extreme greed)"
            }
        else:
            sections["sentiment"] = {"facts": "fear_greed=50/100 (0=extreme fear, 100=extreme greed)"}

        if news_context:
            news_sent = news_context.get("sentiment", "neutral")
            news_count = news_context.get("count", 0)
            sections["news"] = {"facts": f"news_sentiment={news_sent}, articles={news_count}"}
        else:
            sections["news"] = {"facts": "news_sentiment=neutral, articles=0"}

        if synth_signal:
            conf_text = f"{synth_conf:.2f}" if isinstance(synth_conf, (int, float)) else "0.40"
            sections["synthesis"] = {"facts": f"signal={synth_signal}, confidence={conf_text}"}

        lang_code = "ru"
        try:
            from modules.user_prefs import get_language
            lang_code = get_language(user_id) or "ru"
        except Exception:
            lang_code = "ru"

        explanations = await generate_explanations(sections, lang_code)
        if not explanations:
            logger.warning(f"AI explanations empty for {coin_id}")

        # Audit logs
        logger.info(
            f"ANALYSIS_SOURCE coin={coin_id} source=coingecko dataset=history_snapshot "
            f"days={hist_days} rows={hist_rows} last_ts={hist_ts}"
        )
        market_snapshot = context.application.bot_data.get("market_snapshot") if context and context.application else None
        market_ts = market_snapshot.get_latest().ts if market_snapshot and market_snapshot.get_latest() else None
        logger.info(
            f"ANALYSIS_SOURCE coin={coin_id} source=coingecko dataset=markets price_ts={market_ts}"
        )
        logger.info(
            f"ANALYSIS_SOURCE coin={coin_id} source=binance dataset=klines interval=1h rows={binance_rows} last_ts={binance_ts}"
        )
        logger.info(
            f"ANALYSIS_SOURCE coin=market source=alternative.me dataset=fear_greed value={fear_greed_val}"
        )
        if news_context:
            logger.info(
                f"ANALYSIS_SOURCE coin=market source=rss dataset=news "
                f"count={news_context.get('count')} sources={','.join(news_context.get('sources', []))}"
            )

        # Append explanation sections
        if explanations:
            msg.append("")
            msg.append(t("analysis.explain.title", user_id=user_id))

            def _append_section(key: str, source: str, detail: str):
                expl = explanations.get(key)
                if not expl:
                    return
                msg.append(t(f"analysis.explain.{key}_title", user_id=user_id))
                msg.append(expl)
                msg.append(t("analysis.explain.source", user_id=user_id, source=source, detail=detail))

            _append_section("quotes", "CoinGecko/Portfolio", "markets + portfolio")
            _append_section("tech", "CoinGecko", f"history {hist_days}d" if hist_days else "history")
            _append_section("patterns", patterns_source, patterns_detail)
            _append_section("sentiment", "Alternative.me", "Fear&Greed")
            _append_section("news", "RSS", "cryptonews/cointelegraph/decrypt")
            _append_section("synthesis", "AI", "multi-signal")

    logger.info(f"    - {coin_id} analyze_coin_full total: {time.time() - t_start:.3f}s")
    return "\n".join(msg)


async def get_indicators_batch(coin_ids: List[str], days: int = 30) -> Dict[str, dict]:
    """Batch indicators for a list of coins via CoinGecko markets."""
    result: Dict[str, dict] = {}
    try:
        from core.api.coingecko_client import get_markets_data
        logger.info(f"Request markets for {len(coin_ids)} coins: {coin_ids}")
        markets = await get_markets_data(coin_ids)
        logger.info(f"Markets received: {len(markets)}")

        for coin_id in coin_ids:
            row = markets.get(coin_id, {}) or {}

            def _chg(h: str):
                return row.get(
                    f"price_change_percentage_{h}_in_currency",
                    row.get(f"price_change_percentage_{h}"),
                )

            result[coin_id] = {
                "current_price": row.get("current_price"),
                "price_change_percentage_1h_in_currency": _chg("1h"),
                "price_change_percentage_24h_in_currency": _chg("24h"),
                "price_change_percentage_7d_in_currency": _chg("7d"),
                "total_volume": row.get("total_volume"),
                "market_cap": row.get("market_cap"),
            }
    except Exception as e:
        logger.error(f"Batch indicators error: {e}")

    return result


async def analyze_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    from core.portfolio_db import get_portfolio_manager
    from modules.user_prefs import get_show_explanations

    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)

    if not portfolio:
        await update.message.reply_text("Портфель пуст.")
        return

    coin_ids = [COIN_ID_MAP.get(coin.upper(), coin.lower()) for coin in portfolio.keys()]
    snap_service = context.application.bot_data.get("market_snapshot")
    indicators_batch = {}
    if snap_service:
        snap = snap_service.get_latest()
        if snap:
            for coin_id in coin_ids:
                indicators_batch[coin_id] = snap.markets.get(coin_id, {})

    show_explanations = get_show_explanations(user_id)

    # Build lightweight news context once per request
    news_context = None
    if show_explanations:
        try:
            from modules.news import get_significant_news

            items = get_significant_news(limit=3)
            sources = []
            combined_text = ""
            for it in items:
                n = it.get("news", {})
                if n.get("source"):
                    sources.append(n.get("source"))
                combined_text += " " + (n.get("summary") or "") + " " + (n.get("title") or "")

            sentiment = "neutral"
            if combined_text and ai_analyzer.finbert and ai_analyzer.finbert.status.get("enabled"):
                try:
                    label, _score, _backend = await ai_analyzer.finbert.analyze_async(combined_text[:512])
                    sentiment = (
                        "positive" if label.startswith("pos") else
                        "negative" if label.startswith("neg") else
                        "neutral"
                    )
                except Exception:
                    sentiment = "neutral"

            news_context = {
                "count": len(items),
                "sentiment": sentiment,
                "sources": sources or ["cryptonews", "cointelegraph", "decrypt"],
            }
        except Exception:
            news_context = {"count": 0, "sentiment": "neutral", "sources": ["cryptonews", "cointelegraph", "decrypt"]}

    results = []
    successful_analyses = 0
    errors_count = 0

    for coin, data in portfolio.items():
        coin_id = COIN_ID_MAP.get(coin.upper(), coin.lower())
        indicators = indicators_batch.get(coin_id, {})
        current_price = indicators.get("current_price")

        if current_price is None:
            results.append(f"❌ Не удалось получить цену для {coin.upper()}")
            errors_count += 1
            continue

        try:
            analysis = await analyze_coin_full(
                coin_id,
                current_price,
                data,
                indicators,
                context,
                user_id=user_id,
                show_explanations=show_explanations,
                news_context=news_context,
            )
            results.append(analysis)
            successful_analyses += 1
        except Exception as e:
            logger.error(f"Error analyzing {coin}: {e}")
            results.append(t("analysis.error_with_nav", user_id=user_id, error=f"{coin.upper()}: {str(e)}"))
            errors_count += 1

    # Send results with per-coin block integrity
    max_len = 4096
    current = ""
    for block in results:
        if not current:
            current = block
        elif len(current) + 2 + len(block) > max_len:
            await update.message.reply_text(current, parse_mode="Markdown")
            current = block
        else:
            current += "\n\n" + block
    if current:
        await update.message.reply_text(current, parse_mode="Markdown")

    summary = t(
        "analysis.portfolio_analysis_complete",
        user_id=user_id,
        success=successful_analyses,
        total=len(portfolio),
    )
    if errors_count > 0:
        summary += f"\n{t('analysis.portfolio_analysis_errors', user_id=user_id, count=errors_count)}"

    navigation_keyboard = [
        [
            InlineKeyboardButton("📊 Статистика", callback_data="stats"),
            InlineKeyboardButton("🚀 Движения", callback_data="movers"),
        ],
        [
            InlineKeyboardButton("📰 Новости", callback_data="news"),
            InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(navigation_keyboard)
    await update.message.reply_text(summary, reply_markup=reply_markup)
