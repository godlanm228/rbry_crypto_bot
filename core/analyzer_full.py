import logging
import asyncio
from telegram import Update , InlineKeyboardButton , InlineKeyboardMarkup
from telegram.ext import ContextTypes
from core.utils import COIN_ID_MAP
from core.api.coingecko_client import get_batch_historical_data
from core.ai_analytics import ai_analyzer
from core.history_cache import get as hist_get, set as hist_set
from typing import List, Dict
from core.analysis.technical_analysis import rsi, sma, macd

logger = logging.getLogger(__name__)

async def get_historical_data(coin_id, days):
    """Получает (timestamp, price, volume) за days. Возвращает DataFrame или None.
       Использует только батч-API и локальный TTL-кэш — БЕЗ snapshot-сервисов.
    """
    # 1) кэш
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

        # опционально добавим alias 'close' для совместимости с сигналами
        df["close"] = df["price"]

        hist_set(coin_id, days, df)
        return df
    except Exception as e:
        logger.error(f"Ошибка получения исторических данных для {coin_id}: {e}")
        return None
def _row_change_pct(row, horizon: str):
    if not row:
        return None
    k_in_cur = f"price_change_percentage_{horizon}_in_currency"
    if row.get(k_in_cur) is not None:
        return row[k_in_cur]
    k_plain = f"price_change_percentage_{horizon}"
    return row.get(k_plain)

async def analyze_coin_full(coin_id, current_price, portfolio_entry, indicators, context):
    """Полный анализ с AI оценкой потенциала роста"""
    import time
    t_start = time.time()
    amount = portfolio_entry.get("amount", 0)
    avg_price = portfolio_entry.get("buy_price", portfolio_entry.get("price", 0))

    # Получаем историю для технических индикаторов
    #t_hist = time.time()
    #hist_df = await get_historical_data(coin_id, 30)
    #logger.info(f"    - {coin_id} history: {time.time() - t_hist:.3f}s")

    # Расчет всех индикаторов
    tech_indicators = {}
    if context and context.application:
        hist_snapshot = context.application.bot_data.get("history_snapshot")
        if hist_snapshot and hist_snapshot.get_latest():
            df = hist_snapshot.get_latest().by_coin.get(coin_id)
            if df is not None and len(df) >= 14:
                tech_indicators['rsi'] = rsi(df['price'], 14).iloc[-1]
                if len(df) >= 20:
                    tech_indicators['sma_20'] = sma(df['price'], 20).iloc[-1]
                if len(df) >= 26:
                    macd_line, signal_line, _ = macd(df['price'])
                    tech_indicators['macd'] = macd_line.iloc[-1] if not macd_line.empty else None

    # AI оценка потенциала дальнейшего роста
    growth_potential = await ai_analyzer.assess_growth_potential(
        coin_id, current_price, tech_indicators, indicators
    )

    # Формируем сообщение
    profit_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price else 0
    rsi_val = tech_indicators.get('rsi')

    msg = []
    msg.append(f"━━━━━━━━━━━━━━━━━")
    msg.append(f"💰 **{coin_id.upper()}**")
    msg.append(f"Цена: ${current_price:.2f}")
    msg.append(f"Ваша позиция: {amount:.6f} @ ${avg_price:.2f}")
    msg.append(f"📊 P/L: {profit_pct:+.2f}%")
    msg.append("")

    # Технические индикаторы
    msg.append("📈 **Индикаторы:**")
    if rsi_val is not None:
        rsi_signal = "🔴 Перепродан" if rsi_val < 30 else "🟢 Перекуплен" if rsi_val > 70 else "⚪ Нейтрально"
        msg.append(f"RSI(14): {rsi_val:.1f} {rsi_signal}")

    if 'macd' in tech_indicators:
        macd_val = tech_indicators['macd']
        macd_signal = "📈 Бычий" if macd_val > 0 else "📉 Медвежий"
        msg.append(f"MACD: {macd_signal}")
    msg.append("")

    # AI прогноз с объяснением
    msg.append("🤖 **AI Прогноз:**")
    msg.append(f"Оценка потенциала: {'🟢' * (growth_potential['score']//2)}{'⚪' * (5 - growth_potential['score']//2)}")
    msg.append(f"Рекомендация: **{growth_potential['recommendation']}**")
    msg.append(f"Целевая цена: ${growth_potential['target_price']:.2f} ({((growth_potential['target_price']/current_price - 1)*100):+.1f}%)")
    msg.append(f"Уровень риска: {growth_potential['risk_level']}")

    if growth_potential['score'] >= 7:
        msg.append("💡 AI видит хороший потенциал роста")
    elif growth_potential['score'] <= 3:
        msg.append("⚠️ AI рекомендует осторожность")

    logger.info(f"    - {coin_id} analyze_coin_full total: {time.time() - t_start:.3f}s")
    return "\n".join(msg)


async def get_indicators_batch(coin_ids: List[str], days: int = 30) -> Dict[str, dict]:
    """Батч-получение индикаторов для списка монет за один запрос"""
    result: Dict[str, dict] = {}
    try:
        from core.api.coingecko_client import get_markets_data
        logger.info(f"Запрос markets для {len(coin_ids)} монет: {coin_ids}")
        markets = await get_markets_data(coin_ids)
        logger.info(f"Получено markets: {len(markets)} записей")

        for coin_id in coin_ids:
            row = markets.get(coin_id, {}) or {}

            # Нормализуем ключи изменений: берем *_in_currency если есть,
            # иначе — обычное поле без _in_currency (fallback).
            def _chg(h: str):
                return (
                    row.get(f"price_change_percentage_{h}_in_currency",
                            row.get(f"price_change_percentage_{h}"))
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
        logger.error(f"Ошибка батч-индикаторов: {e}")

    return result
async def analyze_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    from core.portfolio_db import get_portfolio_manager
    from locales.translator import t
    pm = get_portfolio_manager()
    portfolio = pm.load_portfolio(user_id)
    
    if not portfolio:
        await update.message.reply_text("Портфель пуст.")
        return

    coin_ids = [COIN_ID_MAP.get(coin.upper(), coin.lower()) for coin in portfolio.keys()]
    app = context.application
    snap_service = app.bot_data.get("market_snapshot")
    indicators_batch = {}
    if snap_service:
        snap = snap_service.get_latest()
        if snap:
            for coin_id in coin_ids:
                indicators_batch[coin_id] = snap.markets.get(coin_id, {})

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
            analysis = await analyze_coin_full(coin_id, current_price, data, indicators, context)
            results.append(analysis)
            successful_analyses += 1
        except Exception as e:
            logger.error(f"Error analyzing {coin}: {e}")
            results.append(t("analysis.error_with_nav", user_id=user_id, error=f"{coin.upper()}: {str(e)}"))
            errors_count += 1

    for result in results:
        await update.message.reply_text(result, parse_mode='Markdown')

    summary = t("analysis.portfolio_analysis_complete", user_id=user_id, success=successful_analyses, total=len(portfolio))
    if errors_count > 0:
        summary += f"\n{t('analysis.portfolio_analysis_errors', user_id=user_id, count=errors_count)}"

    navigation_keyboard = [
        [InlineKeyboardButton("📊 Статистика", callback_data="stats"),
         InlineKeyboardButton("🚀 Движения", callback_data="movers")],
        [InlineKeyboardButton("📰 Новости", callback_data="news"),
         InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(navigation_keyboard)
    await update.message.reply_text(summary, reply_markup=reply_markup)
