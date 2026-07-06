import feedparser
import logging
import asyncio
import html, re
from telegram import Update , InlineKeyboardMarkup, InlineKeyboardButton
from datetime import datetime, timedelta
from core.ai_analytics import ai_analyzer
from telegram.error import BadRequest
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from core.utils import _track_user_activity
from locales.translator import t

logger = logging.getLogger(__name__)
# Matching is done against lowercased text — keywords must be lowercase too,
# otherwise entries like "ETF"/"BlackRock" never match.
KEYWORDS = [
    "ban", "hacked", "exploit", "fraud", "lawsuit",
    "etf", "listing", "integration", "burn", "partnership",
    "investment", "blackrock", "regulation", "crackdown",
]

NEWS_KB = InlineKeyboardMarkup(
    [[InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]]
)

RSS_URLS = [
    "https://cryptonews.com/news/feed",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed"
]

def _categorize_news(title: str, summary: str, user_id: int = None):
    text = f"{(title or '').lower()} {(summary or '').lower()}"
    if any(k in text for k in ["hack", "exploit", "ban", "lawsuit", "fraud", "scam", "delist"]):
        return t("news.category_risk", user_id=user_id), "негатив"
    if any(k in text for k in ["etf", "approval", "inflow", "partnership", "adoption", "listing", "upgrade"]):
        return t("news.category_growth", user_id=user_id), "позитив"
    if any(k in text for k in ["tax", "regulation", "policy", "senate", "bill"]):
        return t("news.category_regulation", user_id=user_id), "нейтрально"
    return t("news.category_general", user_id=user_id), "нейтрально"
def _clean_summary(raw: str, limit: int = 200) -> str:
    if not raw:
        return ""
    # убираем html-теги
    text = re.sub(r"<[^>]+>", " ", raw)
    # убираем лишние пробелы
    text = re.sub(r"\s+", " ", text).strip()
    # обрезаем
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text

def _format_one(item: dict) -> str:
    n = item.get("news", {})
    title = (n.get("title") or "Без заголовка").strip()
    url   = n.get("url") or n.get("link") or ""
    summary = _clean_summary(n.get("summary", ""))
    analysis = item.get("analysis") or ""

    # экранируем для html
    title = html.escape(title)
    summary = html.escape(summary)
    analysis = html.escape(analysis)

    line = f"• <b>{title}</b>"
    if url:
        line = f'• <a href="{url}">{title}</a>'
    if summary:
        line += f"\n{summary}"
    if analysis:
        line += f"\n<b>{analysis}</b>"
    return line

def _format_news_block(items: list, user_id: int = None) -> str:
    lines = [_format_one(it) for it in items[:5] if it]
    return "\n\n".join(lines) if lines else t("news.no_news", user_id=user_id)

async def render_news(
    *,
    status_msg,
    items: list,
    reply_markup: InlineKeyboardMarkup,
    user_id: int = None,
    fallback_bot=None,
    fallback_chat_id: int | None = None
):
    """Единый вывод новостей: одинаковый для /news и для callback."""
    # Собираем тело
    body = _format_news_block(items, user_id).strip() if items else ""
    if not body:
        body = t("news.no_news", user_id=user_id)

    text = t("news.title", user_id=user_id) + "\n" + body

    # Печатаем и редактируем
    try:
        await status_msg.edit_text(
            text,
            parse_mode="HTML",
            disable_web_page_preview=True,
            reply_markup=reply_markup,
        )
    except BadRequest as e:
        # Страховка от пустого текста/«не изменено»
        s = str(e)
        if "Message is not modified" in s or "Message text is empty" in s:
            return
        if "Message can't be edited" in s:
            # Fallback to sending a new message
            if fallback_bot and fallback_chat_id:
                await fallback_bot.send_message(
                    chat_id=fallback_chat_id,
                    text=text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                    reply_markup=reply_markup,
                )
            else:
                await status_msg.reply_text(
                    text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                    reply_markup=reply_markup,
                )
            return
        raise

class NewsFetcher:
    """Fetches RSS news within a sliding 24h window.

    NOTE: intentionally stateless between calls. The old `last_check = now`
    approach made every second /news call return an empty list.
    """

    LOOKBACK_HOURS = 24

    def _normalize_sources(self):
        """Возвращает список источников RSS"""
        return RSS_URLS

    def fetch_news(self):
        """Тянем новости из источников RSS (за последние 24 часа)"""
        all_recent = []
        seen_links = set()
        cutoff = datetime.utcnow() - timedelta(hours=self.LOOKBACK_HOURS)
        sources = self._normalize_sources()

        for url in sources:
            try:
                feed = feedparser.parse(url)
                entries = getattr(feed, "entries", []) or []
                for entry in entries:
                    pp = getattr(entry, "published_parsed", None)
                    try:
                        published_dt = datetime(*pp[:6]) if pp else datetime.utcnow()
                    except Exception:
                        published_dt = datetime.utcnow()

                    link = getattr(entry, "link", "") or ""
                    if link and link in seen_links:
                        continue

                    if published_dt > cutoff:
                        if link:
                            seen_links.add(link)
                        all_recent.append({
                            "title": getattr(entry, "title", "") or "",
                            "summary": getattr(entry, "summary", "") or "",
                            "link": link,
                            "published": published_dt.isoformat(),
                        })
            except Exception as ex:
                logger.warning("fetch_news: источник упал %s: %s", url, ex)

        return all_recent

    def filter_news(self, news_list):
        """Фильтрует новости по ключевым словам"""
        filtered = []
        for news in news_list:
            title = (news.get("title") or "").lower()
            summary = (news.get("summary") or "").lower()
            if any(keyword in title or keyword in summary for keyword in KEYWORDS):
                filtered.append(news)
        return filtered

news_fetcher = NewsFetcher()

def fetch_news():
    """Глобальная функция для получения новостей"""
    return news_fetcher.fetch_news()

def score_news(news_item: dict, portfolio_symbols: list) -> float:
    """Оценивает важность новости для портфеля"""
    score = 0.0
    title = (news_item.get("title") or "").lower()
    summary = (news_item.get("summary") or "").lower()

    # Базовый скор за ключевые слова
    for keyword in KEYWORDS:
        if keyword in title:
            score += 0.3
        if keyword in summary:
            score += 0.2

    # Дополнительный скор за упоминание монет из портфеля
    for symbol in portfolio_symbols:
        if symbol.lower() in title or symbol.lower() in summary:
            score += 0.4

    return min(score, 1.0)  # Максимум 1.0

def get_significant_news(limit: int = 5, min_score: float = 0.3, portfolio_symbols: list = None):
    """Получает значимые новости с оценкой"""
    if portfolio_symbols is None:
        portfolio_symbols = []

    try:
        items = fetch_news()
    except Exception:
        items = []

    # If no news fetched, return empty list
    if not items:
        return []

    scored = []
    for item in items:
        try:
            # Создаем объект для анализа
            summary_text = _clean_summary(item.get("summary", ""), limit=500)
            analysis_result = ai_analyzer.analyze_news(summary_text)

            news_obj = {
                "news": item,
                "analysis": analysis_result
            }
            s = score_news(item, portfolio_symbols)
            scored.append((s, news_obj))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [item for s, item in scored if s >= min_score][:limit]

    if top:
        return top
    
    # If no news pass the threshold, return top scored items anyway (if any exist)
    return [item for _, item in scored][:limit]

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /news"""
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "news")
    await context.bot.send_chat_action(update.message.chat_id, ChatAction.TYPING)

    status_msg = await update.message.reply_text(t("common.loading", user_id=user_id))
    # RSS parsing + AI sentiment is blocking network I/O — keep it off the event loop
    items = await asyncio.to_thread(get_significant_news)
    news_kb = InlineKeyboardMarkup([[InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="back_to_menu")]])
    await render_news(
        status_msg=status_msg,
        items=items,
        reply_markup=news_kb,
        user_id=user_id,
        fallback_bot=context.bot,
        fallback_chat_id=update.message.chat_id,
    )
