"""The market around the coins: macro, stablecoin supply, Fear & Greed, news.

All free and keyless, checked on 28.10.2026:
  macro        Yahoo Finance chart API (unofficial; a failure only leaves a gap)
  stablecoins  DefiLlama, total circulating USD since 2017
  fear_greed   alternative.me
  dominance    CoinGecko /global: BTC, ETH, USDT, USDC shares and the total
               market cap; history is paid, so we keep a daily snapshot
  attention    Wikipedia page views of Bitcoin, Cryptocurrency and Ethereum,
               daily since 07.2015 (Wikimedia REST API): how many people look
  news         the bot's RSS feeds plus CoinDesk and The Block, and Binance's
               listing / delisting announcements (2289 / 446 back to 2017),
               its general news and maintenance notices (from 30.10.2026)

Timing, for features later: a US close (20:00-21:00 UTC) comes before the
crypto daily close at 00:00 UTC, so day D's macro value is known at the end
of crypto day D. Old headlines cannot be fetched again, so the news archive
only grows from the day collection started.
"""

from __future__ import annotations

import json
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
# BTC / ETH / stablecoin shares of the whole market: free only as today's value, so the
# archive grows from 29.10.2026 on, one snapshot right after each UTC close
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
WIKI_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
            "{article}/daily/{start}/{end}")
WIKI_PAGES = {"wiki_bitcoin": "Bitcoin", "wiki_crypto": "Cryptocurrency", "wiki_ethereum": "Ethereum"}
WIKI_HEADERS = {"User-Agent": "rbry-research/1.0 (market attention for a private test bot)"}
GLOBAL_SHARES = ("btc", "eth", "usdt", "usdc")
BINANCE_CMS = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
# 49 and 157 since 30.10.2026: Binance's own word on incidents (paused withdrawals and the like)
ANNOUNCEMENTS = {48: "listing", 161: "delisting", 49: "binance_news", 157: "maintenance"}
ANNOUNCEMENT_URL = "https://www.binance.com/en/support/announcement/{code}"
ARTICLE_DETAIL = "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query"
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


def parse_global(payload: dict) -> dict[str, float]:
    """Today's market shares (percent of all crypto) and the total market cap in USD."""
    data = payload.get("data") or {}
    shares = data.get("market_cap_percentage") or {}
    out = {f"{coin}_dominance": float(shares[coin]) for coin in GLOBAL_SHARES if shares.get(coin) is not None}
    total = (data.get("total_market_cap") or {}).get("usd")
    if total is not None:
        out["total_mcap_usd"] = float(total)
    return out


def fetch_global(client: httpx.Client) -> dict[str, float]:
    resp = client.get(COINGECKO_GLOBAL, headers=BROWSER_HEADERS)
    resp.raise_for_status()
    return parse_global(resp.json())


def parse_pageviews(payload: dict) -> list[tuple[str, float]]:
    return [(f"{i['timestamp'][:4]}-{i['timestamp'][4:6]}-{i['timestamp'][6:8]}", float(i["views"]))
            for i in payload.get("items", []) if i.get("views") is not None]


def fetch_pageviews(client: Optional[httpx.Client] = None, start: str = "20150701",
                    timeout: int = 60) -> dict[str, list[tuple[str, float]]]:
    """urllib, not httpx: Wikimedia answers 403 to httpx whatever the User-Agent (checked 29.10.2026)."""
    end = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = {}
    for name, article in WIKI_PAGES.items():
        request = urllib.request.Request(WIKI_URL.format(article=article, start=start, end=end), headers=WIKI_HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                out[name] = parse_pageviews(json.loads(response.read()))
        except (OSError, ValueError):
            continue
    return out


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


def _texts(node, out: list) -> None:
    if isinstance(node, dict):
        if node.get("node") == "text" and node.get("text"):
            out.append(node["text"])
        for value in node.values():
            _texts(value, out)
    elif isinstance(node, list):
        for value in node:
            _texts(value, out)


def parse_article(payload: dict) -> str:
    """The plain text of an announcement: its body is a JSON tree of nodes (or, now and then, HTML)."""
    body = (payload.get("data") or {}).get("body") or ""
    try:
        out: list = []
        _texts(json.loads(body), out)
        text = " ".join(out)
    except ValueError:
        text = body
    return _clean(text.replace("&nbsp;", " "), 3000)


def fetch_article_text(client: httpx.Client, url: str) -> str:
    code = url.rstrip("/").rsplit("/", 1)[-1]
    resp = client.get(ARTICLE_DETAIL, headers=BROWSER_HEADERS, params={"articleCode": code})
    resp.raise_for_status()
    return parse_article(resp.json())


def all_announcements(client: httpx.Client, catalog: int, max_pages: Optional[int] = None) -> list[dict]:
    items, page = [], 1
    while True:
        batch, total = fetch_announcements(client, catalog, page)
        items += batch
        if not batch or len(items) >= total or (max_pages and page >= max_pages):
            return items
        page += 1
        time.sleep(1.0)
