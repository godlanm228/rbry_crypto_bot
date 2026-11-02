"""Watch for urgent events and tell the owner at once (the rules are in core/sandbox/urgent.py).

    python scripts/urgent_watch.py            # a round every minute, forever (deploy/rbry-urgent.service)
    python scripts/urgent_watch.py --once     # one round, then exit
    python scripts/urgent_watch.py --test     # send the owner how the newest urgent story looked, marked as a test

Each round reads the five news feeds, Binance's own announcements and the
public Telegram channels of on-chain sleuths and Whale Alert (new items go into
market.db's news archive, which the 15-minute archive job fills too), asks the
bot's model whom each new danger headline is about (cached in the state; the
names of ENTITIES when it does not answer), groups the danger headlines of the
last FOLLOW_HOURS into stories, checks Binance's
6-hour tickers of BTC, ETH and the stablecoins, and looks for trouble at an
exchange before the news (core/sandbox/exchange_risk.py): money leaving it for
unknown wallets, its token falling on its own, its withdrawals stuck. What was sent is kept
in runtime/learning/urgent_state.json, so a restart never repeats a message;
on the very first start everything already in the archive counts as sent.
Messages go only to TELEGRAM_OWNER_ID of .env.test, through the test bot.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RBRY_ENV_FILE", str(ROOT / ".env.test"))   # the Groq key for translations

import httpx  # noqa: E402

from core.notify import notify_owner, owner_id  # noqa: E402
from core.sandbox import context_sources  # noqa: E402
from core.sandbox import exchange_risk as er  # noqa: E402
from core.sandbox import urgent as ur  # noqa: E402
from core.sandbox import wallets  # noqa: E402
from core.sandbox.market_store import DEFAULT_DB_PATH, MarketStore  # noqa: E402

logger = logging.getLogger("urgent")
STATE = ROOT / "runtime" / "learning" / "urgent_state.json"
PORTFOLIO = ROOT / "runtime" / "test" / "data" / "portfolio.db"
INTERVAL = 60
CATALOGS = (49, 157, 161)            # Binance's own news, maintenance notices, delistings
TICKER = "https://api.binance.com/api/v3/ticker"
PRICE = "https://api.binance.com/api/v3/ticker/price"
DOWN_MINUTES = 30                    # no source answered this long: tell the owner the watch is blind
SIGMA_REFRESH_S = 3600
NETWORKS = "https://www.binance.com/bapi/capital/v1/public/capital/getNetworkCoinAll"
NETWORKS_EVERY_S = 900               # 2.4 MB a call: every 15 minutes, at once when another signal fires
COINBASE = "https://api.exchange.coinbase.com/products/{product}/ticker"
GAP_SHOWN = 0.003                    # a price gap with Coinbase this large goes into the card


def owner_coins() -> set:
    """The owner's coins in the test bot's portfolio (read only)."""
    owner = owner_id()
    if owner is None or not PORTFOLIO.exists():
        return set()
    try:
        with sqlite3.connect(f"file:{PORTFOLIO}?mode=ro", uri=True, timeout=5) as conn:
            return {str(c).upper() for (c,) in conn.execute("SELECT coin FROM portfolio WHERE user_id = ?", (owner,))}
    except sqlite3.Error:
        return set()


def fetch(client: httpx.Client) -> tuple[list[dict], int]:
    """New headlines from every source and how many sources answered."""
    items, failed = context_sources.fetch_feeds(timeout=10)
    answered = len(context_sources.RSS_FEEDS) - len(failed)
    for catalog in CATALOGS:
        try:
            batch, _ = context_sources.fetch_announcements(client, catalog, size=10)
            items += batch
            answered += 1
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("binance catalog %s: %r", catalog, exc)
    for channel in context_sources.TELEGRAM:
        try:
            items += context_sources.fetch_telegram(client, channel)
            answered += 1
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("telegram %s: %r", channel, exc)
    return items, answered


def tickers(client: httpx.Client, pairs) -> dict:
    """{pair: (price 6 hours ago, price now)} from Binance's rolling-window ticker."""
    try:
        resp = client.get(TICKER, params={"symbols": json.dumps(list(pairs), separators=(",", ":")),
                                          "windowSize": f"{ur.SHOCK_HOURS}h"})
        resp.raise_for_status()
        return {t["symbol"]: (float(t["openPrice"]), float(t["lastPrice"])) for t in resp.json()}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("tickers %s: %r", pairs, exc)
        return {}


def hour_tickers(client: httpx.Client, pairs) -> dict:
    """{pair: (change over the last hour, its USDT volume)} from Binance's rolling-window ticker."""
    try:
        resp = client.get(TICKER, params={"symbols": json.dumps(list(pairs), separators=(",", ":")),
                                          "windowSize": "1h"})
        resp.raise_for_status()
        return {t["symbol"]: (float(t["lastPrice"]) / float(t["openPrice"]) - 1, float(t["quoteVolume"]))
                for t in resp.json() if float(t["openPrice"]) > 0}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("hour tickers %s: %r", pairs, exc)
        return {}


def usual_volume(store: MarketStore, pair: str, days: int = 30) -> float:
    """The median hourly USDT volume of the last ``days``."""
    since = int(time.time() * 1000) - days * 86_400_000
    with store.connect() as conn:
        vols = sorted(v for (v,) in conn.execute("SELECT quote_volume FROM klines WHERE symbol = ? AND tf = '1h' "
                                                 "AND ts >= ?", (pair, since)) if v is not None)
    return float(vols[len(vols) // 2]) if vols else 0.0


def price_gap(client: httpx.Client) -> str:
    """BTC on Binance against Coinbase, in dollars: a wide gap means money cannot move between them."""
    try:
        cb = {p: float(client.get(COINBASE.format(product=p)).json()["price"]) for p in ("BTC-USD", "USDT-USD")}
        bn = float(client.get(f"{TICKER}/price", params={"symbol": "BTCUSDT"}).json()["price"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("price gap: %r", exc)
        return ""
    gap = bn * cb["USDT-USD"] / cb["BTC-USD"] - 1
    if abs(gap) < GAP_SHOWN:
        return ""
    return f"BTC на Binance {'дешевле' if gap < 0 else 'дороже'} Coinbase на {ur._num(abs(gap) * 100)}%"


def sigmas(store: MarketStore) -> dict:
    since = int(time.time() * 1000) - (ur.SIGMA_DAYS + 1) * 86_400_000
    out = {}
    with store.connect() as conn:
        for pair in ur.WATCHED:
            closes = [c for (c,) in conn.execute(
                "SELECT close FROM klines WHERE symbol = ? AND tf = '1h' AND ts >= ? ORDER BY ts", (pair, since))]
            out[pair] = ur.sigma_of(closes)
    return out


SUBJECT = ("You read one crypto news headline. Answer with the party it reports trouble at: the exchange, wallet, "
           "protocol, bridge, blockchain, stablecoin, token or company that was hacked or drained, halted withdrawals "
           "or its network, lost its peg, went insolvent or shut down, or that authorities charged, banned, fined or "
           "raided. Never the attacker (a hacker group, North Korea), the authority (SEC, FBI, a state), an analyst, "
           "investor or the outlet. If no such party is named (\"an exchange\", \"crypto markets\") or the headline is "
           "not about such trouble, answer an empty string. Use the party's usual short name, e.g. \"Bybit\", \"Aave\", "
           "\"Solana\"; and the ticker of that party's own token if it has one (\"BNB\" for Binance, \"SOL\" for Solana, "
           "\"AAVE\" for Aave), else an empty string. Answer with a JSON object {\"subject\": str, \"ticker\": str}.")
ASK_PER_ROUND = 10                   # model calls a round: a fresh crisis is asked first, a backlog over a few rounds
ASK_TRIES = 3                        # unanswered this many times: the names of ENTITIES decide (ur.read)


def name_subject(title: str):
    """[subject, ticker] whom a danger headline is about, by the bot's model; None when it does not answer."""
    from core.llm import groq_json
    result = groq_json(SUBJECT, title[:400], max_tokens=300, timeout=10)
    if result is None or not isinstance(result.get("subject", ""), str):
        return None
    return [result.get("subject") or "", str(result.get("ticker") or "")]


def translate(title: str):
    from core.llm import groq_json
    result = groq_json("Translate this crypto news headline into Russian. Answer with a JSON object "
                       "{\"title\": str}. Faithful translation only; keep names, tickers and numbers as they are.",
                       title, max_tokens=200, timeout=10)
    return (result or {}).get("title")


def summarize(text: str):
    from core.llm import groq_json
    result = groq_json("Summarize this official crypto exchange notice in Russian in two or three short sentences, "
                       "using ONLY facts from the text: what happened, what is paused, what the exchange promises. "
                       "No new facts, no opinions, no advice; keep numbers and names as they are. "
                       "Answer with a JSON object {\"summary\": str}.", text[:3000], max_tokens=400, timeout=15)
    return (result or {}).get("summary")


def follow_headlines(store: MarketStore, now_ms: int) -> list:
    """What follow-ups read: the headlines, and Whale Alert's posts (its analyses tell of an exchange's troubles)."""
    frame = store.news(since_ts=now_ms - ur.FOLLOW_HOURS * ur.HOUR_MS)
    return [ur.Headline(r.url, r.source, r.kind, int(r.published_ts), r.title,
                        (r.summary or "") if r.source == "binance" else "")
            for r in frame.itertuples(index=False)
            if r.kind != "listing" and (r.kind != "onchain" or r.source == "whale_alert")]


def headlines(store: MarketStore, now_ms: int) -> list:
    frame = store.news(since_ts=now_ms - ur.FOLLOW_HOURS * ur.HOUR_MS)
    return [ur.Headline(r.url, r.source, r.kind, int(r.published_ts), r.title,
                        (r.summary or "") if r.source == "binance" else "")
            for r in frame.itertuples(index=False) if r.kind not in ("listing", "onchain")]


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def etherscan_key() -> str:
    """ETHERSCAN_API_KEY of .env.test (or RBRY_ENV_FILE); empty when there is none."""
    try:
        from dotenv import dotenv_values
        return dotenv_values(os.environ.get("RBRY_ENV_FILE", str(ROOT / ".env.test"))).get("ETHERSCAN_API_KEY") or ""
    except (ImportError, OSError):
        return ""


class Watch:
    def __init__(self, store: MarketStore, send=notify_owner, translate=translate, summarize=summarize,
                 name=name_subject):
        self.store, self.send, self.translate, self.summarize, self.name = store, send, translate, summarize, name
        self._tried: set = set()
        self._asked: dict = {}
        self.state = load_state()
        self.first = not self.state
        self.state.setdefault("stories", {})
        self.state.setdefault("alerts", {})
        self.state.setdefault("last_answer", time.time())
        self._sigmas, self._sigmas_at = {}, 0.0
        self.state.setdefault("situations", {})
        self._volumes, self._volumes_at, self._networks_at = {}, 0.0, 0.0
        key = etherscan_key()
        self.wallets = wallets.WalletWatch(key, self.state.setdefault("wallets", {})) if key else None

    def _known(self) -> dict:
        known = {}
        for key, rec in self.state["stories"].items():
            if rec["entity"] not in known or rec["last_ts"] > known[rec["entity"]][1]:
                known[rec["entity"]] = (key, rec["last_ts"])
        return known

    def _market(self, client, story, held) -> str:
        prices = tickers(client, list(ur.WATCHED))
        if story and story.token:                    # apart: a token Binance does not list fails its own call only
            prices.update(tickers(client, [f"{story.token}USDT"]))
        changes = {p: last / start - 1 for p, (start, last) in prices.items() if start > 0}
        return ur.market_line(changes, held)

    def subjects(self, hs: list) -> list:
        """The headlines with whom the model says each danger headline is about (asked once, newest first)."""
        cache = self.state.setdefault("subjects", {})
        asked = 0
        for h in sorted(hs, key=lambda x: -x.ts):
            if h.url in cache or not ur.worth_asking(h) or asked >= ASK_PER_ROUND:
                continue
            asked += 1
            found = self.name(h.title)
            if found is not None:
                cache[h.url] = found
                continue
            self._asked[h.url] = self._asked.get(h.url, 0) + 1
            if self._asked[h.url] >= ASK_TRIES:
                cache[h.url] = None
            break        # busy or down (Groq: 30 a minute, the test bot's too): the rest wait for the next round
        current = {h.url for h in hs}
        self.state["subjects"] = {u: v for u, v in cache.items() if u in current}
        return [replace(h, subject=c[0], ticker=c[1]) if (c := cache.get(h.url)) else h for h in hs]

    def _deliver(self, text: str) -> bool:
        ok = self.send(text)
        logger.info("sent=%s %s", ok, text.splitlines()[0])
        return ok

    def bodies(self, client, now_ms: int) -> None:
        """Binance's own notices: their titles say little ("Security Incident Update"), so read the text once."""
        since = now_ms - ur.FOLLOW_HOURS * ur.HOUR_MS
        notices = [r for kind in ("binance_news", "maintenance")
                   for r in self.store.news(since_ts=since, kind=kind).itertuples(index=False)]
        for r in notices:
            if ((r.summary or "").strip()
                    or r.url in self._tried or ur.routine(r.title)):
                continue
            self._tried.add(r.url)
            try:
                text = context_sources.fetch_article_text(client, r.url)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                logger.warning("announcement text %s: %r", r.url, exc)
                continue
            if text:
                self.store.set_summary(r.url, text)

    def _story(self, entity: str):
        """The newest story record about ``entity``, or None."""
        found = self._known().get(entity)
        return self.state["stories"][found[0]] if found else None

    def news_round(self, client, now_ms: int, held: set, hs: list, fhs: list) -> None:
        known = self._known()
        for story in ur.stories(hs, known):
            record = self.state["stories"].get(story.key)
            verdict = ur.due(story, record, now_ms, held)
            if verdict is None:
                if record is not None:
                    record["last_ts"] = max(record["last_ts"], story.last_ts)
                continue
            if verdict == "silent" or self.first:
                rec = ur.record_of(story, now_ms, record)
                ur.follow(rec, story.entity, fhs, held, now_ms, quiet=True)
                self.state["stories"][story.key] = rec
                continue
            market = self._market(client, story, held)
            text = (ur.text_new(story, self.translate, market) if verdict == "new"
                    else ur.text_update(story, record, self.translate, market))
            if self._deliver(text):
                rec = ur.record_of(story, now_ms, record)
                only = None if verdict == "new" else {ur.update_lead(story, record).url}
                ur.follow(rec, story.entity, fhs, held, now_ms, quiet=True, only=only)   # what it said is told
                self.state["stories"][story.key] = rec
        for entity, (key, _) in self._known().items():      # what is new in the stories already told
            rec = self.state["stories"][key]
            ur.follow(rec, entity, fhs, held, now_ms, quiet=self.first)
            if ur.follow_due(rec, now_ms) and self._deliver(
                    ur.text_follow(entity, rec, fhs, self.translate, self.summarize)):
                ur.follow_told(rec)
        for h in hs:
            coins = ur.delisted(h, held)
            key = f"delist:{h.url}"
            if coins and key not in self.state["alerts"]:
                if self.first or now_ms - h.ts > ur.MAX_LATE_HOURS * ur.HOUR_MS or self._deliver(ur.text_delisting(h, coins)):
                    self.state["alerts"][key] = now_ms

    def market_round(self, client, now_ms: int, held: set, hs: list) -> None:
        if time.time() - self._sigmas_at > SIGMA_REFRESH_S:
            self._sigmas, self._sigmas_at = sigmas(self.store), time.time()
        prices = tickers(client, list(ur.WATCHED) + list(ur.STABLE_PAIRS))
        changes = {p: last / start - 1 for p, (start, last) in prices.items() if start > 0}
        recent = [s for s in ur.stories(hs, self._known())
                  if s.key in self.state["stories"] and now_ms - s.last_ts <= ur.SHOCK_HOURS * ur.HOUR_MS]
        for pair in ur.WATCHED:
            if pair not in prices:
                continue
            start, last = prices[pair]
            z = ur.fall(start, last, self._sigmas.get(pair, float("nan")))
            key = f"fall:{pair}"
            if z is None or now_ms - self.state["alerts"].get(key, 0) < ur.REPEAT_HOURS * ur.HOUR_MS:
                continue
            text = ur.text_fall(pair, start, last, self._sigmas[pair], z,
                                {p: c for p, c in changes.items() if p in ur.WATCHED}, held,
                                recent[-1] if recent else None)
            if self._deliver(text):
                self.state["alerts"][key] = now_ms
        stable = {p: prices[p][1] for p in ur.STABLE_PAIRS if p in prices}
        off = [(p, v) for p, v in ur.depegs(stable)
               if now_ms - self.state["alerts"].get(f"depeg:{p}", 0) >= ur.REPEAT_HOURS * ur.HOUR_MS]
        if off and self._deliver(ur.text_depeg(off, stable)):
            for p, _ in off:
                self.state["alerts"][f"depeg:{p}"] = now_ms

    def risk_signals(self, client, now_ms: int) -> dict:
        """{exchange: {signal: (since ms, what is seen)}} this round."""
        found: dict[str, dict] = {}
        posts = self.store.news(since_ts=now_ms - 2 * ur.HOUR_MS, kind="onchain")
        transfers = [t for t in (er.parse_whale(r.summary or r.title, int(r.published_ts))
                                 for r in posts.itertuples(index=False)) if t]
        for exchange, (usd, n, first) in er.outflows(transfers, now_ms).items():
            if er.outflow_alarm(usd, n):
                found.setdefault(exchange, {})["outflow"] = (
                    first, f"с {exchange} ушло {ur.usd_text(usd, True)} на новые неизвестные кошельки за "
                           f"{er.OUTFLOW_MINUTES} мин ({n} переводов, Whale Alert)")
        if time.time() - self._volumes_at > SIGMA_REFRESH_S:
            self._volumes = {p: usual_volume(self.store, p) for p in er.CANARIES.values()}
            self._volumes_at = time.time()
        moves = hour_tickers(client, list(er.CANARIES.values()) + ["BTCUSDT"])
        for exchange, pair in er.CANARIES.items():
            if pair in moves and "BTCUSDT" in moves:
                own = er.canary(moves[pair][0], moves["BTCUSDT"][0], moves[pair][1], self._volumes.get(pair, 0.0))
                if own is not None:
                    found.setdefault(exchange, {})["canary"] = (
                        now_ms, f"{ur.base_of(pair)} {ur.pct(own)} за час сверх движения BTC, "
                                f"объём ×{moves[pair][1] / self._volumes[pair]:.0f} к обычному")
        if self.wallets is not None:                # our own look at the hot wallets (plan item 7)
            try:
                eth = float(client.get(PRICE, params={"symbol": "ETHUSDT"}).json()["price"])
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                logger.warning("eth price: %r", exc)
                eth = 0.0
            for exchange, seen in self.wallets.round(client, now_ms, eth).items():
                found.setdefault(exchange, {})["wallets"] = seen
        if found or time.time() - self._networks_at > NETWORKS_EVERY_S:
            self._networks_at = time.time()
            try:
                resp = client.get(NETWORKS)
                resp.raise_for_status()
                self.state["stuck"] = er.stuck(resp.json())
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("binance networks: %r", exc)
        coins = self.state.get("stuck", [])
        if len(coins) >= er.STUCK_COINS:
            found.setdefault("Binance", {})["withdrawals"] = (
                now_ms, f"на Binance закрыт вывод {len(coins)} из {len(er.MAJORS)} главных монет: {', '.join(coins)}")
        return found

    def risk_round(self, client, now_ms: int, hs: list) -> None:
        """Open, update and close the situations; the first round with this check only takes note."""
        found = self.risk_signals(client, now_ms)
        quiet = self.first or not self.state.get("risk_seeded")
        situations = self.state["situations"]
        for exchange in list(situations):
            rec = situations[exchange]
            if exchange in found:
                rec["seen"] = now_ms             # still going on: the same situation, not a new one every day
            if "withdrawals" in rec["signals"] and "withdrawals" not in found.get(exchange, {}):
                story = self._story(exchange)              # the news may have told it already
                told = story is not None and "access:open" in story.get("told", {})
                if quiet or told or self._deliver(er.text_reopened(exchange, ur.clock, now_ms)):
                    rec["reopened"] = now_ms
                    rec["signals"].pop("withdrawals")      # closing them again is news again
                    if story is not None:
                        ur.tell(story, "access:open")
            newest = max([s["at"] for s in rec["signals"].values()] + [rec.get("reopened", 0), rec.get("seen", 0)])
            if now_ms - newest > er.SITUATION_HOURS * ur.HOUR_MS:
                del situations[exchange]
        for exchange, signals in found.items():
            rec = situations.get(exchange)
            if rec is None:
                sit = er.Situation(exchange, now_ms)
                for name, (at, text) in signals.items():
                    sit.add(name, at, text)
                if quiet or self._deliver(er.text_situation(sit, ur.clock, price_gap(client),
                                                            self._news_line(exchange, hs))):
                    situations[exchange] = {"opened": now_ms, "signals": sit.signals}
                    self._closed_told(exchange, sit.signals)
                continue
            sit = er.Situation(exchange, rec["opened"], rec["signals"])
            for name, (at, text) in signals.items():
                if name in sit.signals:
                    continue
                sit.add(name, at, text)
                if quiet or self._deliver(er.text_more(sit, name, ur.clock)):
                    rec["signals"] = dict(sit.signals)
                    if name == "withdrawals":
                        rec.pop("reopened", None)
                    self._closed_told(exchange, {name: None})
                else:
                    sit.signals.pop(name)              # not delivered: try again next round
        self.state["risk_seeded"] = True

    def _closed_told(self, exchange: str, signals: dict) -> None:
        """Stuck withdrawals told by the situation are not news when the headlines catch up."""
        story = self._story(exchange)
        if "withdrawals" in signals and story is not None:
            ur.tell(story, "access:closed")

    def _news_line(self, exchange: str, hs: list) -> str:
        for s in ur.stories(hs, self._known()):
            if s.entity == exchange and s.urgent():
                kinds = ", ".join(ur.KIND_RU[k] for k in sorted(s.kinds & ur.MONEY)) or "тревожные новости"
                return f"новости: {kinds} ({ur._sources(s.sources)})"
        return ""

    def health(self, answered: int) -> None:
        now = time.time()
        if answered:
            if self.state.get("blind_told") and self._deliver("✅ Слежка за срочными новостями снова работает."):
                self.state["blind_told"] = False
            self.state["last_answer"] = now
        elif now - self.state["last_answer"] >= DOWN_MINUTES * 60 and not self.state.get("blind_told"):
            if self._deliver(f"⚠️ Слежка за срочными новостями не работает уже {DOWN_MINUTES} минут: "
                             "ни одна лента не отвечает. Сообщу, когда заработает."):
                self.state["blind_told"] = True

    def round(self) -> None:
        now_ms = int(time.time() * 1000)
        held = owner_coins()
        with httpx.Client(timeout=15, headers=context_sources.BROWSER_HEADERS) as client:
            items, answered = fetch(client)
            new = self.store.save_news(items)
            self.health(answered)
            self.bodies(client, now_ms)
            fhs = self.subjects(follow_headlines(self.store, now_ms))
            named = {h.url: h for h in fhs}
            hs = [named.get(h.url, h) for h in headlines(self.store, now_ms)]
            self.news_round(client, now_ms, held, hs, fhs)
            self.market_round(client, now_ms, held, hs)
            self.risk_round(client, now_ms, hs)
        horizon = now_ms - 2 * ur.FOLLOW_HOURS * ur.HOUR_MS
        self.state["stories"] = {k: v for k, v in self.state["stories"].items() if v["last_ts"] >= horizon}
        self.state["alerts"] = {k: v for k, v in self.state["alerts"].items() if v >= horizon - 30 * 86_400_000}
        if self.first:
            logger.info("first start: %d stories already in the archive count as sent", len(self.state["stories"]))
            self.first = False
        save_state(self.state)
        logger.debug("round: %d new items, %d sources answered", new, answered)


def test_message(store: MarketStore) -> int:
    """The newest urgent story of the archive, as its first message looked, marked as a test."""
    now_ms = int(time.time() * 1000)
    found = [s for s in ur.stories(headlines(store, now_ms)) if s.urgent()]
    if not found:
        print("no urgent story in the archive")
        return 1
    story = found[-1]
    first = ur.Story(story.key, story.entity, story.token)
    for item in story.items:                 # the headlines up to the one that made it urgent
        first.items.append(item)
        if first.urgent():
            break
    text = "ТЕСТ: так пришло бы первое сообщение\n\n" + ur.text_new(first, translate)
    print(text)
    return 0 if notify_owner(text) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    store = MarketStore(args.db)
    if args.test:
        return test_message(store)
    watch = Watch(store)
    while True:
        started = time.monotonic()
        try:
            watch.round()
        except Exception:                      # one bad round must not stop the watch
            logger.exception("round failed")
        if args.once:
            return 0
        time.sleep(max(5.0, INTERVAL - (time.monotonic() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
