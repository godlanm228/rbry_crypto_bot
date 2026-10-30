"""Urgent events for the owner as they happen: force majeure in the news, sharp falls, depegs.

Warnings, not trading signals: news did not predict prices (audit 14.10.2026),
but the owner has to know at once when money may be at risk. Vlad, 30.10.2026:
only the really important and urgent, almost instantly (the daily digest is
Redmond Hub's job).

News. A story is one exchange or stablecoin named in danger headlines; later
headlines about it join the story for STORY_HOURS, or FOLLOW_HOURS once it was
sent, so the next day's "what we know" is an update, not a new alarm. Sent
  - at the first headline about paused withdrawals, a depeg or insolvency, or
    a theft of MIN_THEFT_USD or more ("multi-billion" counts), marked "one
    source" until another outlet confirms it;
  - once two outlets report a theft without an amount, or authorities acting
    against it (charges, arrests, a ban, a shutdown).
A data leak is not a theft. Binance's own notices need no second source;
routine network upgrades are not incidents. A Binance delisting of a coin the
owner holds is urgent too. Updates: confirmation by another outlet at once, a
new kind of danger or relief (withdrawals resumed, losses covered) at most
every UPDATE_MINUTES.

Shadow run on the archive 26-30.10.2026 (291 articles): one story, Binance
29.10 at 21:12 UTC (CoinDesk; Cointelegraph confirmed 12 minutes later).
Bithumb's and Coinbase's data leaks, Polymarket's $2.65M exploit, the Keystone
breach and KuCoin leaving Russia stay quiet.

Market. BTC or ETH down more than SHOCK_SIGMAS typical 6-hour moves (the
standard deviation of 6-hour log returns over SIGMA_DAYS): 1.4 times a month
on 2017-2026 hourly closes, 13-19 a year in every year; the old fixed -12% in
an hour fired for BTC once in eight years. A stablecoin more than DEPEG away
from 1 USDT: USDC ten times since 2018 (Tether doubts 2019, Terra 05.2022, SVB
03.2023), FDUSD 04.2025, USDe 10.2025, every one a real event.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

import numpy as np

HOUR_MS = 3_600_000
STORY_HOURS = 12
FOLLOW_HOURS = 72
MAX_LATE_HOURS = 2             # a story that became urgent longer ago (the watcher was down) is not sent
MIN_THEFT_USD = 100e6
UPDATE_MINUTES = 15
SHOCK_SIGMAS = 3.5
SHOCK_HOURS = 6
SIGMA_DAYS = 90
DEPEG = 0.02
REPEAT_HOURS = 24              # one fall or depeg message per pair a day
WATCHED = ("BTCUSDT", "ETHUSDT")
STABLE_PAIRS = ("USDCUSDT", "FDUSDUSDT", "USDEUSDT")
PAST_DEPEGS = "USDC в 03.2023, FDUSD в 04.2025, USDe в 10.2025"

# headline names: (pattern, canonical name, its Binance token for the market's reaction)
ENTITIES = (
    (r"binance", "Binance", "BNB"), (r"coinbase", "Coinbase", None), (r"okx", "OKX", None),
    (r"bybit", "Bybit", None), (r"kraken", "Kraken", None), (r"bitget", "Bitget", None),
    (r"kucoin", "KuCoin", None), (r"gate\.io", "Gate", None), (r"htx|huobi", "HTX", None),
    (r"mexc", "MEXC", None), (r"upbit", "Upbit", None), (r"bithumb", "Bithumb", None),
    (r"bitfinex", "Bitfinex", None), (r"crypto\.com", "Crypto.com", None), (r"gemini", "Gemini", None),
    (r"robinhood", "Robinhood", None), (r"hyperliquid", "Hyperliquid", None),
    (r"tether|usdt", "Tether", None), (r"circle|usdc", "USDC", None), (r"ethena|usde", "Ethena", "ENA"),
    (r"fdusd|first digital", "FDUSD", None),
)
_ENTITIES = [(re.compile(rf"(?<![\w.]){p}(?!\w)", re.I), name, token) for p, name, token in ENTITIES]

KINDS = {
    "theft": r"\b(hack(s|ed|er|ers|ing)?|exploit(s|ed)?|drain(s|ed)?|stolen|steals?|heist|breach(es|ed)?|"
             r"compromised|security incident|unauthori[sz]ed)\b",
    "withdrawals": r"\b(suspen|paus|halt|freez|froze|stop|block|disabl|clos)\w*\b[^.;:]{0,40}\bwithdrawals?\b"
                   r"|\bwithdrawals?\b[^.;:]{0,30}\b(suspen|paus|halt|frozen|stopp|disabl|clos)\w*",
    "depeg": r"\bde-?peg\w*|\b(los|break)\w*\s+(its\s+)?(dollar\s+)?peg\b",
    "insolvency": r"\binsolven\w*|\bbankrupt\w*|\bchapter 11\b",
    "enforcement": r"\b(sue[sd]?|suing|lawsuit|charge[sd]|arrest\w*|detain\w*|indict\w*|raid(s|ed)?|ban(s|ned)?|"
                   r"seiz\w*|shut(s|ting)? down)\b",
    "relief": r"\b(resum\w*|reopen\w*|restor\w*|back online|reimburs\w*|recover(s|ed)?|"
              r"cover\w*\s+(all\s+)?(the\s+)?(user\s+)?loss\w*|safu)\b",
}
_KINDS = {k: re.compile(rx, re.I) for k, rx in KINDS.items()}
MONEY = {"theft", "withdrawals", "depeg", "insolvency"}
NEWS_KINDS = MONEY | {"enforcement", "relief"}
_DATA_LEAK = re.compile(r"\b(data|personal|customer (info|information|contact)|records|e-?mails?|leak\w*)\b", re.I)
_AFTERMATH = re.compile(r"\b(launder\w*|recover(s|ed|y)?|returned|refund\w*|lessons?|anniversary|months? after|"
                        r"years? after|post-?mortem|sentenced|convicted)\b", re.I)
_ROUTINE = re.compile(r"\b(upgrade|hard fork|maintenance|scheduled|network update)\b", re.I)
_AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|bn|b|million|mn|m|thousand|k)\b", re.I)
_MULTI = re.compile(r"\bmulti-?billion|\bbillions\b", re.I)
_DELIST = re.compile(r"Binance Will Delist (.+?) on \d{4}-\d{2}-\d{2}")

KIND_RU = {"theft": "взлом / кража", "withdrawals": "выводы остановлены", "depeg": "отвязка от доллара",
           "insolvency": "банкротство", "enforcement": "власти против", "relief": "хорошая новость"}
SOURCE_RU = {"coindesk.com": "CoinDesk", "cointelegraph.com": "Cointelegraph", "decrypt.co": "Decrypt",
             "theblock.co": "The Block", "cryptonews.com": "Cryptonews", "binance": "Binance (официально)"}
WARNING = "Это предупреждение, не совет купить или продать."


@dataclass(frozen=True)
class Headline:
    url: str
    source: str
    kind: str          # article | binance_news | maintenance | delisting
    ts: int            # published, ms
    title: str
    text: str = ""     # the body of Binance's own notices ("Security Incident Update" says little by itself)


@dataclass(frozen=True)
class Reading:
    entity: str
    token: Optional[str]
    kinds: frozenset
    usd: float         # the largest amount in the headline; MIN_THEFT_USD * 10 for "multi-billion"
    exact: bool        # the amount is a number, not "billions"
    aftermath: bool


def amount_usd(title: str) -> tuple[float, bool]:
    best, exact = 0.0, False
    for num, unit in _AMOUNT.findall(title):
        u = unit.lower()
        scale = 1e9 if u.startswith("b") else 1e6 if u.startswith("m") else 1e3
        value = float(num.replace(",", "")) * scale
        if value > best:
            best, exact = value, True
    if not best and _MULTI.search(title):
        return 1e9, False
    return best, exact


def routine(title: str) -> bool:
    return bool(_ROUTINE.search(title))


def read(h: Headline) -> Optional[Reading]:
    """What a headline says: about whom, which dangers, how much money. None when it names no one we watch."""
    official = h.source == "binance"
    if h.kind in ("listing", "delisting") or (official and _ROUTINE.search(h.title)):
        return None
    found = [(m.start(), name, token) for rx, name, token in _ENTITIES for m in [rx.search(h.title)] if m]
    if found:
        _, name, token = min(found, key=lambda x: x[0])
    elif official:
        name, token = "Binance", "BNB"
    else:
        return None
    words = f"{h.title} {h.text}" if official else h.title
    kinds = {k for k, rx in _KINDS.items() if rx.search(words)}
    usd, exact = amount_usd(words)
    if "theft" in kinds and _DATA_LEAK.search(h.title) and not exact:
        kinds.discard("theft")                  # "breach ... exposed customer data": a leak, not stolen money
    if not kinds:
        return None
    return Reading(name, token, frozenset(kinds), usd, exact, bool(_AFTERMATH.search(h.title)))


@dataclass
class Story:
    key: str
    entity: str
    token: Optional[str]
    items: list = field(default_factory=list)          # (Headline, Reading), oldest first

    @property
    def kinds(self) -> set:
        return set().union(*(r.kinds for _, r in self.items))

    @property
    def sources(self) -> list:
        out = []
        for h, _ in self.items:
            if h.source not in out:
                out.append(h.source)
        return out

    def sources_of(self, kind: str) -> list:
        return list(dict.fromkeys(h.source for h, r in self.items if kind in r.kinds))

    @property
    def first_ts(self) -> int:
        return self.items[0][0].ts

    @property
    def last_ts(self) -> int:
        return self.items[-1][0].ts

    @property
    def usd(self) -> tuple[float, bool]:
        exact = [r.usd for _, r in self.items if r.exact and "theft" in r.kinds]
        if exact:
            return max(exact), True
        return max((r.usd for _, r in self.items if "theft" in r.kinds), default=0.0), False

    def urgent(self) -> bool:
        kinds = self.kinds
        if kinds & (MONEY - {"theft"}):
            return True
        if "theft" in kinds and (self.usd[0] >= MIN_THEFT_USD or len(self.sources_of("theft")) >= 2
                                 or "binance" in self.sources_of("theft")):
            return True
        return "enforcement" in kinds and len(self.sources_of("enforcement")) >= 2


def stories(headlines: Iterable[Headline], known: Optional[dict] = None) -> list[Story]:
    """Danger headlines grouped by whom they are about.

    ``known``: {entity: (key, newest headline ms)} of stories already sent, so a
    story keeps its key when its first headlines leave the window.
    """
    known = known or {}
    open_: dict[str, Story] = {}
    out = []
    for h in sorted(headlines, key=lambda x: x.ts):
        r = read(h)
        if r is None:
            continue
        s = open_.get(r.entity)
        window = (FOLLOW_HOURS if s is not None and s.entity in known and s.key == known[s.entity][0]
                  else STORY_HOURS) * HOUR_MS
        if s is None or h.ts - s.last_ts > window:
            if r.aftermath:
                continue                        # "the hacker laundered ...": old news does not start a story
            prior = known.get(r.entity)
            key = prior[0] if prior and h.ts - prior[1] <= FOLLOW_HOURS * HOUR_MS else f"{r.entity}|{h.ts}"
            s = Story(key, r.entity, r.token)
            open_[r.entity] = s
            out.append(s)
        s.items.append((h, r))
    return out


def due(story: Story, record: Optional[dict], now_ms: int) -> Optional[str]:
    """'new', 'update', 'silent' (urgent, but too late to be news) or None."""
    if record is None:
        if not story.urgent():
            return None
        return "silent" if now_ms - story.last_ts > MAX_LATE_HOURS * HOUR_MS else "new"
    if len(story.sources) >= 2 > len(record["sources"]):
        return "update"                          # the confirmation goes out at once
    if "binance" in story.sources and "binance" not in record["sources"]:
        return "update"                          # so does Binance's own word
    fresh = (story.kinds - set(record["kinds"])) & NEWS_KINDS
    if fresh and now_ms - record["sent_at"] >= UPDATE_MINUTES * 60_000:
        return "update"
    return None


def record_of(story: Story, now_ms: int) -> dict:
    return {"entity": story.entity, "sent_at": now_ms, "sources": story.sources, "kinds": sorted(story.kinds),
            "last_ts": story.last_ts}


def delisted(h: Headline, coins: set) -> list[str]:
    """The owner's coins in a Binance spot delisting notice."""
    m = _DELIST.match(h.title) if h.kind == "delisting" else None
    if not m:
        return []
    bases = [b.strip() for b in re.split(r",\s*|\s+and\s+|\s*&\s*", m.group(1))]
    return [b for b in bases if b in coins]


# -- the market ---------------------------------------------------------------------------

def sigma_of(closes: Iterable[float], hours: int = SHOCK_HOURS) -> float:
    """Typical move: the standard deviation of ``hours``-hour log returns of hourly closes."""
    c = np.asarray(list(closes), dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if len(c) < hours * 10:
        return math.nan
    return float(np.std(np.log(c[hours:] / c[:-hours])))


def fall(open_price: float, last_price: float, sigma: float) -> Optional[float]:
    """How many typical moves the price fell, when it is a shock (SHOCK_SIGMAS or more)."""
    if not (open_price > 0 and last_price > 0 and sigma > 0):
        return None
    z = -math.log(last_price / open_price) / sigma
    return z if z >= SHOCK_SIGMAS else None


def depegs(prices: dict) -> list[tuple[str, float]]:
    return [(pair, p) for pair, p in prices.items() if p and abs(p - 1.0) > DEPEG]


# -- words --------------------------------------------------------------------------------

def _num(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def pct(x: float) -> str:
    return ("+" if x > 0 else "−" if x < 0 else "") + _num(abs(x) * 100) + "%"


def usd_text(value: float, exact: bool) -> str:
    if not exact:
        return "миллиарды долларов" if value >= 1e9 else ""
    if value >= 1e9:
        return f"${_num(value / 1e9)} млрд"
    return f"${_num(value / 1e6, 0)} млн"


def clock(ts_ms: int) -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.fromtimestamp(ts_ms / 1000, ZoneInfo("Europe/Berlin")).strftime("%H:%M")
    except Exception:   # no tz database: UTC, and say so
        return datetime.fromtimestamp(ts_ms / 1000, timezone.utc).strftime("%H:%M UTC")


def base_of(pair: str) -> str:
    return pair[:-4] if pair.endswith("USDT") else pair


def market_line(changes: dict, held: set) -> str:
    """'Рынок за 6 ч: BTC −1,2% (в портфеле), BNB −3,0%' from {pair: 6-hour change}."""
    parts = [f"{base_of(p)} {pct(c)}" + (" (в портфеле)" if base_of(p) in held else "")
             for p, c in changes.items() if c is not None and math.isfinite(c)]
    return f"Рынок за {SHOCK_HOURS} ч: " + ", ".join(parts) if parts else ""


def _quote(h: Headline, translate: Callable[[str], Optional[str]]) -> list[str]:
    ru = translate(h.title) if translate else None
    return [f"«{ru}»", h.title] if ru and ru.strip() and ru.strip() != h.title else [f"«{h.title}»"]


def _sources(names: Iterable[str]) -> str:
    return ", ".join(SOURCE_RU.get(s, s) for s in names)


def text_new(story: Story, translate=None, market: str = "") -> str:
    kinds = [KIND_RU[k] for k in ("withdrawals", "theft", "depeg", "insolvency", "enforcement") if k in story.kinds]
    lead = next(h for h, r in story.items if r.kinds & (MONEY | {"enforcement"}))
    n = len(story.sources)
    confirm = "пока 1 источник, жду подтверждения" if n == 1 else f"пишут {n} источника: {_sources(story.sources)}"
    lines = [f"🚨 СРОЧНО · {story.entity}: {', '.join(kinds)}",
             f"{SOURCE_RU.get(lead.source, lead.source)}, {clock(lead.ts)} — {confirm}"]
    lines += _quote(lead, translate)
    amount = usd_text(*story.usd)
    if amount:
        lines.append(f"Сумма в заголовках: {'до ' if story.usd[1] else ''}{amount}")
    if market:
        lines.append(market)
    lines += [lead.url, WARNING]
    return "\n".join(lines)


def text_update(story: Story, record: dict, translate=None, market: str = "", summarize=None) -> str:
    lines = []
    new_sources = [s for s in story.sources if s not in record["sources"]]
    if "binance" in new_sources:
        h = next(h for h, _ in reversed(story.items) if h.source == "binance")
        summary = (summarize(h.text) if summarize and h.text else None) or h.text[:400]
        lines += [f"🏛 {story.entity}: Binance официально, {clock(h.ts)}", f"«{h.title}»"]
        if summary:
            lines.append(summary)
        amount = usd_text(*story.usd)
        if amount and story.usd[1]:
            lines.append(f"Сумма: до {amount}")
        if market:
            lines.append(market)
        lines.append(h.url)
        return "\n".join(x for x in lines if x)
    if len(story.sources) >= 2 > len(record["sources"]):
        lines.append(f"✅ {story.entity}: подтверждают {len(story.sources)} источника — {_sources(story.sources)}")
    fresh = [k for k in ("withdrawals", "theft", "depeg", "insolvency", "enforcement", "relief")
             if k in story.kinds and k not in record["kinds"]]
    if fresh:
        sign = "🟢" if fresh == ["relief"] else "⚠️"
        lines.append(f"{sign} {story.entity} — новое: {', '.join(KIND_RU[k] for k in fresh)}")
        lead = next(h for h, r in reversed(story.items) if r.kinds & set(fresh))
    else:
        lead = next((h for h, _ in reversed(story.items) if h.source in new_sources), story.items[-1][0])
    lines += _quote(lead, translate)
    lines.append(f"{SOURCE_RU.get(lead.source, lead.source)}, {clock(lead.ts)}")
    amount = usd_text(*story.usd)
    if amount and story.usd[1]:
        lines.append(f"Сумма в заголовках: до {amount}")
    if market:
        lines.append(market)
    lines.append(lead.url)
    return "\n".join(lines)


def text_fall(pair: str, open_price: float, last_price: float, sigma: float, z: float,
              others: dict, held: set, news: Optional[Story] = None) -> str:
    coin = base_of(pair)
    lines = [f"📉 РЕЗКОЕ ПАДЕНИЕ · {coin} {pct(last_price / open_price - 1)} за {SHOCK_HOURS} часов"
             + (" (в портфеле)" if coin in held else ""),
             f"Это в {_num(z)} раза сильнее обычного движения за {SHOCK_HOURS} часов (обычно ±{_num(sigma * 100)}%). "
             f"Так бывает примерно 1–2 раза в месяц.",
             f"Цена: ${_num(last_price, 2 if last_price < 100 else 0)} "
             f"({SHOCK_HOURS} часов назад ${_num(open_price, 2 if open_price < 100 else 0)})"]
    rest = market_line({p: c for p, c in others.items() if p != pair}, held)
    if rest:
        lines.append(rest)
    if news is not None:
        lines.append(f"В новостях за эти часы: {news.entity} — "
                     f"{', '.join(KIND_RU[k] for k in sorted(news.kinds & NEWS_KINDS))} ({len(news.sources)} ист.)")
    return "\n".join(lines)


def text_depeg(off: list, all_prices: dict) -> str:
    """One message for the stablecoins that left 1 USDT; all of them up at once means USDT itself fell."""
    known = {p: v for p, v in all_prices.items() if v}
    if len(known) >= 2 and all(v > 1 + DEPEG for v in known.values()):
        head = "🚨 СРОЧНО · USDT дешевеет: другие стейблкоины стоят дороже 1 USDT"
    else:
        head = "🚨 СРОЧНО · Стейблкоин отвязался от доллара"
    lines = [head] + [f"{base_of(p)} = {_num(v, 3)} USDT ({pct(v - 1)})" for p, v in off]
    lines += [f"Такое уже бывало: {PAST_DEPEGS}.", WARNING]
    return "\n".join(lines)


def text_delisting(h: Headline, coins: list) -> str:
    return "\n".join([f"🚨 СРОЧНО · Binance убирает твою монету: {', '.join(coins)}",
                      f"«{h.title}»", f"Binance (официально), {clock(h.ts)}", h.url])
