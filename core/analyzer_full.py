import logging
import time
import config
from typing import List, Dict, Optional

from core.utils import _row_change_pct
from core.formatting import format_price
from core.api.binance_client import get_batch_ohlcv
from core.ai_analytics import ai_analyzer
from core.ai_explainer import generate_explanations
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
    # Evict entries from previous buckets - the bucketed keys are never
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

    msg = build_analysis_card(
        coin_id=coin_id,
        market_row=indicators or {},
        current_price=current_price,
        tech_indicators=tech_indicators,
        assessment=growth_potential,
        history_ready=hist_set is not None,
        amount=amount if has_position else None,
        avg_price=avg_price if has_position else None,
        user_id=user_id,
    )

    if show_explanations and config.AI_EXPLANATIONS_ENABLED:
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

        # Facts computed by the shared synthesis pipeline are reused here -
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


def _pct(value) -> str:
    return f"{value:+.1f}%" if isinstance(value, (int, float)) and value == value else "n/a"


def _evidence_reasons(assessment, user_id) -> str:
    """The one or two strongest weighted evidence components, in words."""
    evidence = getattr(assessment, "raw_analysis", None)
    components = getattr(evidence, "components", None) or ()
    scored = [
        (c.value * c.weight, c.name)
        for c in components
        if c.available and c.weight > 0 and abs(c.value) >= 0.1
    ]
    scored.sort(key=lambda item: abs(item[0]), reverse=True)
    phrases = [
        t(f"analysis.reason_{name}_{'up' if contribution > 0 else 'down'}", user_id=user_id)
        for contribution, name in scored[:2]
    ]
    return ", ".join(phrases) if phrases else t("analysis.card_reason_weak", user_id=user_id)


def build_analysis_card(*, coin_id, market_row, current_price, tech_indicators, assessment,
                        history_ready, amount, avg_price, user_id) -> list:
    """Readable /analyze card: what is measured, what it means, why the verdict."""
    name = market_row.get("name") or coin_id.replace("-", " ").title()
    symbol = (market_row.get("symbol") or "").upper()
    title = f"{name} ({symbol})" if symbol else name
    msg = [f"💰 **{title}** · {format_price(current_price)}"]
    msg.append(t("analysis.card_changes", user_id=user_id,
                 ch24=_pct(_row_change_pct(market_row, "24h")), ch7d=_pct(_row_change_pct(market_row, "7d"))))
    if amount and avg_price:
        pnl = (current_price - avg_price) / avg_price * 100
        msg.append(t("analysis.card_position", user_id=user_id, amount=f"{amount:g}",
                     price=format_price(avg_price), pnl=pnl))
    else:
        msg.append(t("analysis.no_position", user_id=user_id))
    msg.append("")

    state = getattr(assessment, "market_state", None)
    technical = getattr(state, "technical", None)
    msg.append(t("analysis.card_seen_title", user_id=user_id))
    rsi_val = tech_indicators.get("rsi")
    if rsi_val is not None:
        zone = "oversold" if rsi_val < 30 else "overbought" if rsi_val > 70 else "neutral"
        msg.append(t("analysis.card_rsi", user_id=user_id, value=rsi_val,
                     zone=t(f"analysis.rsi_zone_{zone}", user_id=user_id)))
    macd_val = tech_indicators.get("macd")
    if macd_val is not None and macd_val == macd_val:
        msg.append(t("analysis.card_macd_up" if macd_val > 0 else "analysis.card_macd_down", user_id=user_id))
    vol_pct = getattr(technical, "volatility_annual_pct", None)
    if vol_pct is not None:
        msg.append(t("analysis.card_volatility", user_id=user_id, pct=vol_pct,
                     level=t(f"analysis.risk_{assessment.risk_level}", user_id=user_id)))
    support = getattr(technical, "support_level", None)
    resistance = getattr(technical, "resistance_level", None)
    if support or resistance:
        none = t("analysis.card_level_none", user_id=user_id)
        sup = (f"{format_price(support)} ({_pct(-technical.support_distance_pct)})"
               if support and technical.support_distance_pct is not None else none)
        res = (f"{format_price(resistance)} ({_pct(technical.resistance_distance_pct)})"
               if resistance and technical.resistance_distance_pct is not None else none)
        msg.append(t("analysis.card_levels", user_id=user_id, support=sup, resistance=res))
    if not history_ready:
        msg.append(t("analysis.card_history_loading", user_id=user_id))
    elif rsi_val is None and vol_pct is None:
        msg.append(t("analysis.card_no_indicators", user_id=user_id))
    msg.append("")

    if not assessment.is_available:
        msg.append(t("analysis.assessment_unavailable", user_id=user_id))
    else:
        scenario = t(f"analysis.scenario_{assessment.scenario}", user_id=user_id)
        msg.append(t("analysis.card_verdict", user_id=user_id, scenario=scenario))
        reasons = _evidence_reasons(assessment, user_id)
        weak = t("analysis.card_reason_weak", user_id=user_id)
        if getattr(assessment, "direction", None) == "sideways" and reasons != weak:
            reasons = t("analysis.card_reason_weak_strongest", user_id=user_id, strongest=reasons)
        msg.append(t("analysis.card_reason", user_id=user_id, reasons=reasons))
    return msg
