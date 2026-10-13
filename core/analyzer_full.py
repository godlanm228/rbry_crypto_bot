import logging
import time
import config
from typing import List, Dict, Optional

from core.utils import _row_change_pct
from core.formatting import format_price
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
from core.access.access_manager import get_access_manager
from locales.translator import t

logger = logging.getLogger(__name__)

_PATTERN_CACHE: Dict[tuple, list] = {}
_PATTERN_CACHE_TTL_BUCKET_SEC = 3600  # 1h buckets for 1h candles


def _pattern_cache_key(coin_id: str, window_hours: int) -> tuple:
    bucket = int(time.time() // _PATTERN_CACHE_TTL_BUCKET_SEC)
    # Evict entries from previous buckets — the bucketed keys are never
    # requested again, so without this the cache grows forever.
    stale = [k for k in _PATTERN_CACHE if k[2] != bucket]
    for k in stale:
        _PATTERN_CACHE.pop(k, None)
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
    """Full analysis with an evidence-based, non-probabilistic assessment."""
    t_start = time.time()
    try:
        current_price = float(current_price)
    except (TypeError, ValueError):
        current_price = 0.0
    if current_price <= 0:
        return t("analysis.market_data_updating", user_id=user_id, coin=coin_id.upper())

    try:
        amount = float(portfolio_entry.get("amount", 0))
        avg_price = float(portfolio_entry.get("buy_price", portfolio_entry.get("price", 0)))
    except (TypeError, ValueError):
        amount = 0.0
        avg_price = 0.0
    has_position = amount > 0 and avg_price > 0

    # Technical indicators
    tech_indicators = {}
    hist_snapshot = context.application.bot_data.get("history_snapshot") if context and context.application else None
    history_max_age = max(7200, config.HISTORY_REFRESH_SECONDS * 3)
    if hist_snapshot and hasattr(hist_snapshot, "get_usable"):
        hist_set = hist_snapshot.get_usable(history_max_age)
    else:
        hist_set = hist_snapshot.get_latest() if hist_snapshot else None
        if hist_set and hist_set.ts and time.time() - hist_set.ts > history_max_age:
            hist_set = None
    if hist_snapshot and hist_snapshot.get_latest() and hist_set is None:
        logger.warning("History snapshot is stale for analysis coin=%s", coin_id)
    if hist_set:
        df = hist_set.by_coin.get(coin_id)
        if df is not None and len(df) >= 14:
            rsi_last = rsi(df["price"], 14).iloc[-1]
            if rsi_last == rsi_last:  # NaN during warm-up is "no RSI", not "nan"
                tech_indicators["rsi"] = float(rsi_last)
            if len(df) >= 20:
                tech_indicators["sma_20"] = sma(df["price"], 20).iloc[-1]
            if len(df) >= 26:
                macd_line, _signal_line, _ = macd(df["price"])
                tech_indicators["macd"] = macd_line.iloc[-1] if not macd_line.empty else None

    access_manager = get_access_manager()
    # Same engine for every tier: tiers limit coins and request frequency, not
    # the honesty of the assessment (FREE used to get the legacy rules).
    allow_synthesis = True
    allow_patterns = access_manager.has_feature(user_id, "patterns")
    growth_potential = await ai_analyzer.assess_growth_potential(
        coin_id,
        current_price,
        tech_indicators,
        indicators,
        allow_synthesis=allow_synthesis,
    )

    profit_pct = ((current_price - avg_price) / avg_price) * 100 if has_position else None
    rsi_val = tech_indicators.get("rsi")

    msg = []
    msg.append("━━━━━━━━━━━━━━━━━")
    msg.append(f"💰 **{coin_id.upper()}**")
    msg.append(t("analysis.price", user_id=user_id, price=format_price(current_price)))
    if has_position:
        msg.append(
            t(
                "analysis.position",
                user_id=user_id,
                amount=amount,
                price=format_price(avg_price),
            )
        )
        msg.append(t("analysis.position_pnl", user_id=user_id, pnl=profit_pct))
    else:
        msg.append(t("analysis.no_position", user_id=user_id))
    msg.append("")

    msg.append(t("analysis.indicators_title", user_id=user_id))
    if rsi_val is not None:
        rsi_signal_key = (
            "analysis.rsi_oversold" if rsi_val < 30 else
            "analysis.rsi_overbought" if rsi_val > 70 else
            "analysis.rsi_neutral"
        )
        msg.append(
            t(
                "analysis.rsi",
                user_id=user_id,
                value=rsi_val,
                signal=t(rsi_signal_key, user_id=user_id),
            )
        )

    if "macd" in tech_indicators:
        macd_val = tech_indicators["macd"]
        if macd_val is not None:
            macd_signal_key = (
                "analysis.macd_bullish" if macd_val > 0 else "analysis.macd_bearish"
            )
            msg.append(
                t(
                    "analysis.macd",
                    user_id=user_id,
                    signal=t(macd_signal_key, user_id=user_id),
                )
            )
    msg.append("")

    msg.append(t("analysis.market_assessment", user_id=user_id))
    method_key = (
        "analysis.method_synthesis"
        if growth_potential.method == "multi_factor_synthesis"
        else "analysis.method_rules"
    )
    msg.append(t("analysis.method", user_id=user_id, method=t(method_key, user_id=user_id)))
    if not growth_potential.is_available:
        msg.append(t("analysis.assessment_unavailable", user_id=user_id))
    else:
        msg.append(
            t(
                "analysis.scenario",
                user_id=user_id,
                scenario=t(f"analysis.scenario_{growth_potential.scenario}", user_id=user_id),
            )
        )
        msg.append(
            t(
                "analysis.risk_level",
                user_id=user_id,
                risk=t(f"analysis.risk_{growth_potential.risk_level}", user_id=user_id),
            )
        )
        if growth_potential.rule_score is not None:
            msg.append(
                t("analysis.rule_score", user_id=user_id, score=growth_potential.rule_score)
            )

    if show_explanations:
        hist_days = None
        hist_rows = None
        hist_ts = None
        df_hist = None
        if hist_set:
            hist_days = getattr(hist_snapshot, "days", None)
            df_hist = hist_set.by_coin.get(coin_id)
            if df_hist is not None and len(df_hist) > 0:
                hist_rows = len(df_hist)
                if "timestamp" in df_hist.columns:
                    hist_ts = df_hist["timestamp"].iloc[-1]

        # Facts computed by the shared synthesis pipeline are reused here —
        # the explanation layer must not refetch data or recompute patterns.
        synthesis_state = growth_potential.market_state

        volume_ratio = None
        if synthesis_state is not None and synthesis_state.technical.volume_ratio is not None:
            volume_ratio = synthesis_state.technical.volume_ratio
        elif df_hist is not None and "volume" in df_hist.columns and len(df_hist) >= 21:
            # Signal candle is excluded from its own baseline.
            avg_vol = df_hist["volume"].iloc[-21:-1].mean()
            cur_vol = df_hist["volume"].iloc[-1]
            if avg_vol and avg_vol > 0:
                volume_ratio = cur_vol / avg_vol

        ch24 = _row_change_pct(indicators, "24h")
        ch7d = _row_change_pct(indicators, "7d")

        # Patterns: reuse the shared pipeline's detections when present.
        patterns_summary = []
        patterns_source = "Binance"
        patterns_detail = "klines 1h, 24h/7d/30d"
        binance_rows = None
        binance_ts = None
        synthesis_patterns_source = (
            synthesis_state.quality.sources.get("patterns")
            if synthesis_state is not None else None
        )
        reuse_synthesis_patterns = bool(
            synthesis_patterns_source and synthesis_patterns_source.available
        )
        ohlcv_data = {}
        if allow_patterns and not reuse_synthesis_patterns:
            ohlcv_data = await get_batch_ohlcv([coin_id], interval="1h")
        df_binance = ohlcv_data.get(coin_id) if ohlcv_data else None
        if not allow_patterns:
            logger.info("PATTERNS_SKIP coin=%s reason=tier_policy", coin_id)
        elif reuse_synthesis_patterns:
            patterns_summary = _summarize_patterns(
                synthesis_state.patterns.detected_patterns, "1h/closed"
            )
            patterns_detail = "shared synthesis pipeline, closed 1h candles"
            logger.info(
                "PATTERNS_REUSE coin=%s count=%s",
                coin_id,
                len(synthesis_state.patterns.detected_patterns),
            )
        elif df_binance is None or len(df_binance) < 20:
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

        assessment_signal = growth_potential.scenario if growth_potential.is_available else None

        sections = {}
        quote_facts = [f"price={format_price(current_price)}"]
        if has_position:
            quote_facts.extend([
                f"entry={format_price(avg_price)}",
                f"amount={amount:.6f}",
                f"pnl={profit_pct:+.2f}%",
            ])
        sections["quotes"] = {"facts": ", ".join(quote_facts)}

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
        if facts:
            sections["tech"] = {"facts": ", ".join(facts)}

        if patterns_summary:
            sections["patterns"] = {"facts": "patterns=" + ", ".join(patterns_summary[:8])}

        if fear_greed_val is not None:
            sections["sentiment"] = {
                "facts": f"fear_greed={fear_greed_val:.0f}/100 (0=extreme fear, 100=extreme greed)"
            }
        if news_context:
            news_sent = news_context.get("sentiment")
            news_count = news_context.get("count", 0)
            if news_sent is not None or news_count:
                sections["news"] = {"facts": f"news_sentiment={news_sent}, articles={news_count}"}

        if assessment_signal:
            sections["synthesis"] = {
                "facts": f"scenario={assessment_signal}, method={growth_potential.method}"
            }

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
        market_latest = market_snapshot.get_latest() if market_snapshot else None
        market_ts = getattr(market_latest, "ts", None)
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
            _append_section("synthesis", "Multi-factor model", "scenario synthesis")

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

