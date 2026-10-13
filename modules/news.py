import feedparser
import httpx
import logging
import asyncio
import html, re
import threading
import time
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

# Keywords and tickers match whole words only: substring matching found
# "ban" inside "Bankruptcy"/"bank" and "op" inside "operations".
_KEYWORD_RE = re.compile(r"\b(" + "|".join(map(re.escape, KEYWORDS)) + r")\b")

FETCH_TIMEOUT_SECONDS = 10
CACHE_TTL_SECONDS = 600

RSS_URLS = [
    "https://cryptonews.com/news/feed",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed"
]

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

def _format_one(item: dict, user_id: int = None) -> str:
    n = item.get("news", {})
    title = html.escape((n.get("title") or t("news.untitled", user_id=user_id)).strip())
    url = (n.get("url") or n.get("link") or "").strip()
    summary = html.escape(_clean_summary(n.get("summary", "")))
    tone = item.get("analysis")  # "positive" / "negative" / "neutral" / None

    line = f"• <b>{title}</b>"
    if url.startswith(("https://", "http://")):
        line = f'• <a href="{html.escape(url, quote=True)}">{title}</a>'
    if summary:
        line += f"\n{summary}"
    if tone:
        # Shown only when a sentiment backend actually classified the text.
        line += f"\n<b>{html.escape(t(f'news.tone_{tone}', user_id=user_id))}</b>"
    return line

def _format_news_block(items: list, user_id: int = None) -> str:
    lines = [_format_one(it, user_id) for it in items[:5] if it]
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

    title = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(t("news.title", user_id=user_id)))
    text = title + "\n" + body

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

    def __init__(self):
        self._cache = (float("-inf"), [])
        self._lock = threading.Lock()

    def fetch_news(self):
        """RSS items from the last 24h, cached for CACHE_TTL_SECONDS."""
        with self._lock:
            fetched_at, items = self._cache
            if time.monotonic() - fetched_at < CACHE_TTL_SECONDS:
                return list(items)
            items = self._fetch_uncached()
            self._cache = (time.monotonic(), items)
            return list(items)

    def _fetch_uncached(self):
        all_recent = []
        seen_links = set()
        cutoff = datetime.utcnow() - timedelta(hours=self.LOOKBACK_HOURS)
        sources = self._normalize_sources()

        for url in sources:
            try:
                # feedparser has no timeout of its own; a hung feed used to pin
                # a worker thread forever.
                response = httpx.get(url, timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
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

news_fetcher = NewsFetcher()

def fetch_news():
    """Глобальная функция для получения новостей"""
    return news_fetcher.fetch_news()

def score_news(news_item: dict, portfolio_symbols: list) -> float:
    """Оценивает важность новости для портфеля"""
    score = 0.0
    title = (news_item.get("title") or "").lower()
    summary = (news_item.get("summary") or "").lower()

    # Базовый скор за ключевые слова (целые слова)
    score += 0.3 * len(set(_KEYWORD_RE.findall(title)))
    score += 0.2 * len(set(_KEYWORD_RE.findall(summary)))

    # Дополнительный скор за упоминание монет из портфеля
    for symbol in portfolio_symbols:
        sym_re = re.compile(r"(?<![\w$])\$?" + re.escape(symbol.lower()) + r"\b")
        if sym_re.search(title) or sym_re.search(summary):
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

    scored = sorted(
        ((score_news(item, portfolio_symbols), item) for item in items),
        key=lambda x: x[0],
        reverse=True,
    )
    top = [item for sc, item in scored if sc >= min_score][:limit]
    if not top:
        # If no news pass the threshold, return the best scored items anyway
        top = [item for _, item in scored][:limit]

    # Sentiment only for the items shown: one backend call per displayed
    # item instead of one per RSS entry (48 per /news before).
    result = []
    for item in top:
        summary_text = _clean_summary(item.get("summary", ""), limit=500)
        result.append({"news": item, "analysis": ai_analyzer.analyze_news(summary_text)})
    return result

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
