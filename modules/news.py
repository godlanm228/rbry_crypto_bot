import feedparser
import json
import urllib.request
import logging
import asyncio
import html, re
import threading
import time
from urllib.parse import urlparse

import config
from core.llm import groq_json
from telegram import Update , InlineKeyboardMarkup, InlineKeyboardButton
from datetime import datetime, timedelta
from core.ai_analytics import ai_analyzer
from telegram.error import BadRequest
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from core.utils import _track_user_activity
from locales.translator import t

logger = logging.getLogger(__name__)
# Matching is done against lowercased text - keywords must be lowercase too,
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
# Some feeds (cryptonews.com) answer 403 to clients without a browser-like agent.
FEED_HEADERS = {"User-Agent": "feedparser/6.0 +https://github.com/kurtmckee/feedparser/"}
# Tone and translations are per article and never change: cache by link.
ARTICLE_CACHE_TTL_SECONDS = 24 * 3600
_tone_cache: dict = {}         # link -> (ts, label | None)
_translation_cache: dict = {}  # (link, lang) -> (ts, title, summary)
_cache_lock = threading.Lock()

LANGUAGE_NAMES = {"ru": "Russian", "uk": "Ukrainian", "de": "German", "en": "English"}
SOURCE_NAMES = {"cointelegraph.com": "Cointelegraph", "decrypt.co": "Decrypt", "cryptonews.com": "Cryptonews"}

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

def _source_name(url: str) -> str:
    host = (urlparse(url).hostname or "").removeprefix("www.")
    return SOURCE_NAMES.get(host, host or "link")


def _format_one(item: dict, user_id: int = None) -> str:
    n = item.get("news", {})
    title = item.get("display_title") or (n.get("title") or t("news.untitled", user_id=user_id)).strip()
    summary = item.get("display_summary") or _clean_summary(n.get("summary", ""))
    url = (n.get("url") or n.get("link") or "").strip()
    tone = item.get("analysis")  # "positive" / "negative" / "neutral" / None

    # Only the source name is a link: whole blue headlines were hard to read.
    line = f"• <b>{html.escape(title)}</b>"
    if summary:
        line += f"\n{html.escape(summary)}"
    if url.startswith(("https://", "http://")):
        source = html.escape(_source_name(url))
        label = html.escape(t("news.source_label", user_id=user_id))
        line += f'\n{label}: <a href="{html.escape(url, quote=True)}">{source}</a>'
    if tone:
        # Shown only when a sentiment backend actually classified the text.
        line += f"\n{html.escape(t(f'news.tone_{tone}', user_id=user_id))}"
    return line


def localize_news(items: list, lang: str) -> list:
    """Translate headlines and write a short summary in the user's language.

    One LLM call per batch of uncached articles, cached per (link, language).
    Without the LLM the original headline and summary are shown; nothing is
    ever made up.
    """
    if not items or lang == "en" or lang not in LANGUAGE_NAMES:
        return items
    now = time.time()
    missing = []
    with _cache_lock:
        for idx, item in enumerate(items):
            link = item.get("news", {}).get("link") or ""
            cached = _translation_cache.get((link, lang))
            if cached and now - cached[0] < ARTICLE_CACHE_TTL_SECONDS:
                item["display_title"], item["display_summary"] = cached[1], cached[2]
            else:
                missing.append(idx)
    if missing and config.GROQ_API_KEY:
        language = LANGUAGE_NAMES[lang]
        payload = [
            {
                "id": idx,
                "title": items[idx]["news"].get("title", ""),
                "text": _clean_summary(items[idx]["news"].get("summary", ""), limit=600),
            }
            for idx in missing
        ]
        system = (
            f"You translate crypto news for a Telegram digest into {language}. "
            "Answer with a JSON object {\"items\": [{\"id\": int, \"title\": str, \"summary\": str}]}. "
            f"title: a faithful translation of the headline into {language}. "
            f"summary: one or two short sentences in {language} using ONLY facts from the given text; "
            "no new facts, no opinions, no investment advice. Keep tickers, numbers and the names of "
            "people, companies, products and projects untranslated (e.g. World, Coinbase, Base)."
        )
        result = groq_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=1600)
        translated = {}
        for entry in (result or {}).get("items", []):
            try:
                translated[int(entry["id"])] = (str(entry["title"]).strip(), str(entry.get("summary", "")).strip())
            except (KeyError, TypeError, ValueError):
                continue
        with _cache_lock:
            for idx, (title, summary) in translated.items():
                if idx not in missing or not title:
                    continue
                items[idx]["display_title"], items[idx]["display_summary"] = title, summary
                link = items[idx]["news"].get("link") or ""
                _translation_cache[(link, lang)] = (now, title, summary)
    return items


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
    from modules.user_prefs import get_language

    items = await asyncio.to_thread(localize_news, list(items or []), get_language(user_id) if user_id else "ru")
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
                # a worker thread forever. urllib (not httpx): cryptonews.com's
                # CDN answers 403 to httpx's TLS fingerprint.
                request = urllib.request.Request(url, headers=FEED_HEADERS)
                with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
                    feed = feedparser.parse(response.read())
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
                logger.warning("fetch_news: источник упал %s: %r", url, ex)

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
    now = time.time()
    for item in top:
        link = item.get("link") or ""
        with _cache_lock:
            cached = _tone_cache.get(link)
        if cached and now - cached[0] < ARTICLE_CACHE_TTL_SECONDS:
            tone = cached[1]
        else:
            summary_text = _clean_summary(item.get("summary", ""), limit=500)
            tone = ai_analyzer.analyze_news(summary_text)
            if tone is not None:  # do not cache failures: retry next time
                with _cache_lock:
                    _tone_cache[link] = (now, tone)
        result.append({"news": item, "analysis": tone})
    return result

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /news"""
    user_id = update.effective_user.id
    _track_user_activity(update.effective_user, "news")
    await context.bot.send_chat_action(update.message.chat_id, ChatAction.TYPING)

    status_msg = await update.message.reply_text(t("common.loading", user_id=user_id))
    # RSS parsing + AI sentiment is blocking network I/O - keep it off the event loop
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
