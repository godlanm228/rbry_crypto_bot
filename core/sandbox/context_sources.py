"""The market around the coins: macro, stablecoin supply, Fear & Greed, news.

All free and keyless, checked on 28.10.2026:
  macro        Yahoo Finance chart API (unofficial; a failure only leaves a gap)
  stablecoins  DefiLlama, total circulating USD since 2017
  fear_greed   alternative.me
  news         the bot's RSS feeds plus CoinDesk and The Block, and Binance's
               listing / delisting announcements (2289 / 446 back to 2017)

Timing, for features later: a US close (20:00-21:00 UTC) comes before the
crypto daily close at 00:00 UTC, so day D's macro value is known at the end
of crypto day D. Old headlines cannot be fetched again, so the news archive
only grows from the day collection started.
"""

from __future__ import annotations

import re
import time
import urllib.request
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
MACRO = {"spx": "^GSPC", "nasdaq": "^IXIC", "dxy": "DX-Y.NYB", "vix": "^VIX", "us10y": "^TNX", "gold": "GC=F"}
STABLECOINS_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
BINANCE_CMS = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
ANNOUNCEMENTS = {48: "listing", 161: "delisting"}
ANNOUNCEMENT_URL = "https://www.binance.com/en/support/announcement/{code}"
# modules/news.py RSS_URLS, plus two more sources for the archive
RSS_FEEDS = ["https://cryptonews.com/news/feed", "https://cointelegraph.com/rss", "https://decrypt.co/feed",
             "https://www.coindesk.com/arc/outboundfeeds/rss/", "https://www.theblock.co/rss.xml"]
FEED_HEADERS = {"User-Agent": "feedparser/6.0 +https://github.com/kurtmckee/feedparser/"}
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) rbry-research"}


def _day(ts_seconds: float) -> str:
    return datetime.fromtimestamp(ts_seconds, tz=timezone.utc).strftime("%Y-%m-%d")


def parse_yahoo(payload: dict) -> list[tuple[str, float]]:
    result = payload["chart"]["result"][0]
    closes = result["indicators"]["quote"][0].get("close") or []
    return [(_day(t), float(c)) for t, c in zip(result.get("timestamp") or [], closes) if c is not None]


def fetch_macro(client: httpx.Client, start: str = "2017-01-01") -> dict[str, list[tuple[str, float]]]:
    period1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    out = {}
    for name, ticker in MACRO.items():
        resp = client.get(YAHOO_CHART.format(ticker=ticker), headers=BROWSER_HEADERS,
                          params={"period1": period1, "period2": int(time.time()), "interval": "1d"})
        if resp.status_code == 200:
            out[name] = parse_yahoo(resp.json())
    return out


def parse_stablecoins(payload: list) -> list[tuple[str, float]]:
    return [(_day(int(r["date"])), float(r["totalCirculatingUSD"]["peggedUSD"]))
            for r in payload if r.get("totalCirculatingUSD", {}).get("peggedUSD") is not None]


def fetch_stablecoins(client: httpx.Client) -> list[tuple[str, float]]:
    resp = client.get(STABLECOINS_URL)
    resp.raise_for_status()
    return parse_stablecoins(resp.json())


def parse_fear_greed(payload: dict) -> list[tuple[str, float]]:
    return [(_day(int(d["timestamp"])), float(d["value"])) for d in payload.get("data", [])]


def fetch_fear_greed(client: httpx.Client) -> list[tuple[str, float]]:
    resp = client.get(FEAR_GREED_URL, params={"limit": 0, "format": "json"})
    resp.raise_for_status()
    return parse_fear_greed(resp.json())


# -- news -------------------------------------------------------------------------

def _clean(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()
    return text[:limit]


def parse_feed(raw: bytes, url: str) -> list[dict]:
    import feedparser
    source = (urlparse(url).hostname or "").removeprefix("www.")
    items = []
    for entry in getattr(feedparser.parse(raw), "entries", []) or []:
        when = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if not when or not getattr(entry, "link", ""):
            continue
        published = datetime(*when[:6], tzinfo=timezone.utc)
        items.append({"url": entry.link, "source": source, "kind": "article",
                      "published_ts": int(published.timestamp() * 1000),
                      "title": _clean(getattr(entry, "title", ""), 300),
                      "summary": _clean(getattr(entry, "summary", ""))})
    return items


def fetch_feeds(timeout: int = 15) -> tuple[list[dict], list[str]]:
    """(items, failed feeds). urllib, not httpx: cryptonews.com answers 403 to httpx (see modules/news.py)."""
    items, failed = [], []
    for url in RSS_FEEDS:
        try:
            request = urllib.request.Request(url, headers=FEED_HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                items += parse_feed(response.read(), url)
        except Exception:  # one broken feed must not stop the others
            failed.append(url)
    return items, failed


def parse_announcements(payload: dict, kind: str) -> tuple[list[dict], int]:
    catalogs = (payload.get("data") or {}).get("catalogs") or []
    if not catalogs:
        return [], 0
    articles = catalogs[0].get("articles") or []
    items = [{"url": ANNOUNCEMENT_URL.format(code=a["code"]), "source": "binance", "kind": kind,
              "published_ts": int(a["releaseDate"]), "title": _clean(a.get("title", ""), 300)}
             for a in articles if a.get("code") and a.get("releaseDate")]
    return items, int(catalogs[0].get("total") or 0)


def fetch_announcements(client: httpx.Client, catalog: int, page: int = 1, size: int = 50) -> tuple[list[dict], int]:
    resp = client.get(BINANCE_CMS, headers=BROWSER_HEADERS,
                      params={"type": 1, "catalogId": catalog, "pageNo": page, "pageSize": size})
    resp.raise_for_status()
    return parse_announcements(resp.json(), ANNOUNCEMENTS[catalog])


def all_announcements(client: httpx.Client, catalog: int, max_pages: Optional[int] = None) -> list[dict]:
    items, page = [], 1
    while True:
        batch, total = fetch_announcements(client, catalog, page)
        items += batch
        if not batch or len(items) >= total or (max_pages and page >= max_pages):
            return items
        page += 1
        time.sleep(1.0)
