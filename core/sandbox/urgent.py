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
owner holds is urgent too. The confirmation by another outlet goes out at once;
after that only a new fact about the money (Vlad, 31.10.2026: "не мусолить одну
и ту же новость", but what is new about it, yes): see "what is new in a known
story" below.

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
             "theblock.co": "The Block", "cryptonews.com": "Cryptonews", "binance": "Binance (официально)",
             "lookonchain": "Lookonchain", "zachxbt": "ZachXBT", "spotonchain": "Spot On Chain",
             "wublockchain": "Wu Blockchain", "whale_alert": "Whale Alert"}
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


def _whom(title: str, found: list) -> tuple:
    """Of the names in a headline, the one its danger words are about: the nearest name before one of them
    ("Coinbase leads crypto stock selloff following $6.3B Binance hack", "Binance May Have Been Hacked as
    Tether and Circle Freeze ..." are both Binance's), else the nearest after; the first named on a tie."""
    spots = [m.span() for rx in _KINDS.values() for m in rx.finditer(title)]
    before = [(ks - f[1], f[0], i) for i, f in enumerate(found) for ks, _ in spots if f[1] <= ks]
    after = [(max(0, f[0] - ke), f[0], i) for i, f in enumerate(found) for _, ke in spots]
    if before or after:
        return found[min(before or after)[2]]
    return min(found, key=lambda f: f[0])


def read(h: Headline) -> Optional[Reading]:
    """What a headline says: about whom, which dangers, how much money. None when it names no one we watch."""
    official = h.source == "binance"
    if h.kind in ("listing", "delisting") or (official and _ROUTINE.search(h.title)):
        return None
    found = [(m.start(), m.end(), name, token) for rx, name, token in _ENTITIES for m in [rx.search(h.title)] if m]
    if found:
        _, _, name, token = _whom(h.title, found)
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
    """'new', 'update' (the confirmation), 'silent' (urgent, but too late to be news) or None; what comes
    later is follow()'s."""
    if record is None:
        if not story.urgent():
            return None
        return "silent" if now_ms - story.last_ts > MAX_LATE_HOURS * HOUR_MS else "new"
    if len(story.sources) >= 2 > len(record["sources"]):
        return "update"                          # the confirmation goes out at once
    return None


def record_of(story: Story, now_ms: int, prior: Optional[dict] = None) -> dict:
    """What was told of the story; what follow() keeps of it carries over."""
    prior = prior or {}
    rec = {"entity": story.entity, "sent_at": now_ms, "sources": story.sources, "kinds": sorted(story.kinds),
           "last_ts": story.last_ts, "usd_told": max(prior.get("usd_told", 0.0), story.usd[0])}
    for k in ("told", "pending", "looked"):
        if k in prior:
            rec[k] = prior[k]
    return rec


def update_lead(story: Story, record: dict) -> Headline:
    """The headline the confirmation quotes: the newest from an outlet not heard before."""
    return next((h for h, _ in reversed(story.items) if h.source not in record["sources"]), story.items[-1][0])


# -- what is new in a known story (Vlad, 31.10.2026) -------------------------------------------------
# After the first message and its confirmation a story's later headlines go out only with a new fact about the
# money, each once: replayed on the 92 headlines of the Binance hack (29-31.10.2026), 9 messages in two days
# (the amount, "funds are safe", the restart dates, customer wallets hit, more breaches, the money set aside,
# Binance's own updates, the proof-of-reserves doubts, 90% replenished) and none of the hacker's transfers,
# prices, explainers, politics, AI forecasts or the same news from ten outlets.
GATHER_MINUTES = 15            # outlets report a fact within minutes of each other: one message
HOLD_HOURS = 3                 # the amount and the authorities wait this long at most for a message about the money
SCALE_X = 1.5                  # a larger amount is news once it is this many times the one told
_DAY = (r"(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|next week|"
        r"within \d+\s*(hours?|days?)|in \d+\s*(hours?|days?)|\d{1,2} (oct|nov|dec)\w*|(oct|nov|dec)\w* \d{1,2})")
FACTS = (
    # (type, what, stage, pattern); "safe" goes in stages (promised, funded, done): a later one is news again
    ("safe", "pledge", 1, r"funds? (are|were|is|remain\w*) (safe|secure|unaffected|not affected)|"
                          r"(customer|user)s?'? funds? (were |are )?(unaffected|not affected|safe)|"
                          r"\bwill cover (all )?(the )?(hack\w* )?loss|pledg\w* to cover|\bsafu\b|reimburs\w*|compensat\w*"),
    ("safe", "funded", 2, r"(set\w* aside|reserv\w*|arrang\w*|ready|holds?) [^.;]{0,40}\bto cover|"
                          r"cover [^.;]{0,30}loss\w* [^.;]{0,20}(company|own) funds"),
    ("safe", "done", 3, r"replenish\w*|fully (covered|restored|reimbursed)|reimbursed (all|in full)"),
    ("risk", "customers", 0, r"(customer|user|client)s?'? (assets|funds|deposits|balances|money)[^.;]{0,40}"
                             r"\b(drain\w*|hit|affected|lost|missing|exposed|stolen|at risk)\b|"
                             r"\b(drain\w*|hit|emptied)[^.;]{0,30}wallets backing customer|shortfall"),
    ("risk", "insolvency", 0, KINDS["insolvency"]),
    ("risk", "theft", 0, KINDS["theft"]),
    ("risk", "depeg", 0, KINDS["depeg"]),
    ("peg", "back", 0, r"\b(regain\w*|restor\w*|back to) (its )?(dollar )?(peg|\$1)\b|\bre-?peg\w*"),
    ("flag", "por", 0, r"proof[- ]of[- ]reserves?[^.;]{0,40}(delay\w*|postpon\w*|miss\w*|fail\w*|withh\w*)|"
                       r"(delay\w*|postpon\w*)[^.;]{0,40}proof[- ]of[- ]reserves?"),
    ("flag", "inside", 0, r"inside[- ]job|insider (attack|theft)"),
    ("flag", "wider", 0, r"(additional|wider|more|new|second) (points? of )?breach\w*|breach (is |was )?wider"),
    ("flag", "run", 0, r"\bbank run\b|\brun on\b"),
    ("access", "closed", 0, KINDS["withdrawals"]),
    ("access", "date", 0, rf"(resum|restart|reopen|restor)\w*[^.;]{{0,40}}{_DAY}|{_DAY}[^.;]{{0,30}}(resum|restart|reopen)\w*"
                          r"|target\w* [^.;]{0,20}(withdrawal )?restart"),
    ("access", "open", 0, r"withdrawals? (have |has )?(resumed|reopened|restored|back)|(resumes|reopens|restores) withdrawals"),
    ("law", "action", 0, r"\b(charged|indicted|sued|lawsuit|arrest\w*|raid\w*|licen[cs]e (revoked|suspended)|banned|"
                         r"seiz\w*|shut down)\b"),
)
_FACTS = [(t, what, stage, re.compile(rx, re.I)) for t, what, stage, rx in FACTS]
_DAY_RX = re.compile(_DAY, re.I)
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")                  # "$6.34B" does not end a sentence
_HACKER_SELLS = re.compile(r"\b(hacker|exploiter|attacker|thief)s?\b[^.;]{0,60}\b(dump\w*|sell\w*|sold|swap\w*|"
                           r"offload\w*)\b|selling pressure", re.I)
_TICKER = re.compile(r"[$#]([A-Z][A-Z0-9]{1,9})\b")
_BOUGHT = re.compile(r"\b(for|into|to)\s+[$#][A-Z][A-Z0-9]{1,9}\b")     # "swapping them for $ETH": bought, not sold
_TRIM = re.compile(r"^[^\w$#«\"']+|\s*\b(Read Analysis|Details)\s*$")   # Whale Alert's "📉 📉 📉 ... Read Analysis"
URGENT_FACTS = {"risk", "flag", "access", "safe", "peg", "official", "held"}
FACT_RU = {"risk": ("🔴", "деньги под угрозой"), "flag": ("🟠", "тревожный знак"), "access": ("🔑", "вывод"),
           "safe": ("🟢", "деньги клиентов"), "peg": ("🟢", "привязка к доллару"), "official": ("📢", "официально"),
           "held": ("🪙", "твоя монета"), "scale": ("📈", "сумма"), "law": ("⚖️", "власти")}
_ORDER = list(FACT_RU)


def whom_at(text: str, span: tuple) -> Optional[str]:
    """Whom the words at ``span`` are about: the nearest name before them, else the nearest after."""
    names = [(m.start(), m.end(), name) for rx, name, _ in _ENTITIES for m in rx.finditer(text)]
    before = [(span[0] - end, start, name) for start, end, name in names if end <= span[0]]
    after = [(max(0, start - span[1]), start, name) for start, end, name in names]
    return min(before or after)[2] if (before or after) else None


def facts_of(h: Headline, entity: str, held: Iterable[str] = ()) -> list[tuple[str, str, int]]:
    """[(type, what, stage)] the headline tells about ``entity``'s money."""
    official = h.source == "binance"
    if h.title.rstrip().endswith("?") or (official and (entity != "Binance" or read(h) is None)):
        return []                               # "Are funds SAFU?" asks; the exchange's routine notices tell nothing
    words = _DECIMAL.sub(",", f"{h.title} {h.text}" if official else h.title)
    out = []
    for t, what, stage, rx in _FACTS:
        m = rx.search(words)
        if not m or (not official and whom_at(words, m.span()) != entity):
            continue                            # about someone else the headline names
        if what == "date":
            day = _DAY_RX.search(m.group(0))
            what = f"date:{day.group(0).lower() if day else 'soon'}"
        out.append((t, what, stage))
    if official and h.kind == "binance_news":
        out.append(("official", h.url, 0))
    m = _HACKER_SELLS.search(words)
    if m and (official or whom_at(words, m.span()) == entity):
        out += [("held", c, 0) for c in sorted(set(_TICKER.findall(_BOUGHT.sub("", h.title))) & set(held))]
    return out


def _key(t: str, what: str) -> str:
    return "safe" if t == "safe" else "scale" if t == "scale" else f"{t}:{what}"


def tell(record: dict, key: str, stage: int = 1, usd: float = 0.0) -> None:
    """Mark a fact told; withdrawals open and closed undo each other, and what is pending of it goes."""
    told = record.setdefault("told", {})
    told[key] = max(stage, told.get(key, 0), 1)
    if key == "scale":
        record["usd_told"] = max(record.get("usd_told", 0.0), usd)
    if key == "access:open":
        for k in [k for k in told if k == "access:closed" or k.startswith("access:date:")]:
            del told[k]
    elif key == "access:closed":
        told.pop("access:open", None)
    record["pending"] = [p for p in record.get("pending", []) if p["key"] != key or (key == "safe" and p["stage"] > stage)]


def follow(record: dict, entity: str, headlines: Iterable[Headline], held: Iterable[str], now_ms: int,
           quiet: bool = False, only: Optional[set] = None) -> None:
    """Put the new facts of headlines not looked at yet into record['pending']; ``quiet`` marks them told instead
    (what the first message or a restart already covers), ``only`` limits that to these urls. A record seen for
    the first time takes everything so far as told."""
    hs = sorted(headlines, key=lambda h: h.ts)
    if "looked" not in record:
        quiet, only = True, None
    looked = set(record.get("looked", []))
    record.setdefault("told", {})
    pending = record.setdefault("pending", [])
    for h in hs:
        if h.url in looked or (only is not None and h.url not in only):
            continue
        looked.add(h.url)
        found = [(t, what, stage, 0.0) for t, what, stage in facts_of(h, entity, held)]
        r = read(h)
        if r is not None and r.entity == entity and r.exact and "theft" in r.kinds:
            bar = SCALE_X * max([record.get("usd_told", 0.0)] + [p["usd"] for p in pending if p["key"] == "scale"])
            if r.usd >= bar:
                found.append(("scale", "amount", 0, r.usd))
        silent = quiet or now_ms - h.ts > MAX_LATE_HOURS * HOUR_MS
        for t, what, stage, usd in found:
            key = _key(t, what)
            if key == "safe":
                if stage <= max([record["told"].get(key, 0)] + [p["stage"] for p in pending if p["key"] == key]):
                    continue
            elif key != "scale" and (key in record["told"] or any(p["key"] == key for p in pending)):
                continue
            if silent:
                tell(record, key, stage, usd)
                continue
            pending[:] = [p for p in pending if p["key"] != key or key not in ("safe", "scale")]
            pending.append({"key": key, "type": t, "what": what, "stage": stage, "usd": usd, "url": h.url, "at": now_ms})
    current = {h.url for h in hs}
    record["looked"] = sorted(u for u in looked if u in current)


def follow_due(record: dict, now_ms: int) -> bool:
    """Facts about the money go after GATHER_MINUTES; the amount and the authorities wait up to HOLD_HOURS for them."""
    pending = record.get("pending") or []
    if not pending:
        return False
    urgent = [p["at"] for p in pending if p["type"] in URGENT_FACTS]
    at = min(urgent) + GATHER_MINUTES * 60_000 if urgent else min(p["at"] for p in pending) + HOLD_HOURS * HOUR_MS
    return now_ms >= at


def follow_told(record: dict) -> None:
    for p in list(record.get("pending", [])):
        tell(record, p["key"], p["stage"], p["usd"])
    record["pending"] = []


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


def text_update(story: Story, record: dict, translate=None, market: str = "") -> str:
    """The confirmation by another outlet."""
    lines = [f"✅ {story.entity}: подтверждают {len(story.sources)} источника — {_sources(story.sources)}"]
    lead = update_lead(story, record)
    lines += _quote(lead, translate)
    lines.append(f"{SOURCE_RU.get(lead.source, lead.source)}, {clock(lead.ts)}")
    amount = usd_text(*story.usd)
    if amount and story.usd[1]:
        lines.append(f"Сумма в заголовках: до {amount}")
    if market:
        lines.append(market)
    lines.append(lead.url)
    return "\n".join(lines)


def text_follow(entity: str, record: dict, headlines: Iterable[Headline], translate=None, summarize=None) -> str:
    """'🔄 Binance — что нового': the pending facts, one line per headline, the most urgent first."""
    hs = list(headlines)
    by_url = {h.url: h for h in hs}
    support: dict = {}                       # fact -> the outlets that tell it, for "пока 1 источник"
    for h in hs:
        for t, what, _ in facts_of(h, entity):
            support.setdefault(f"{t}:{what}", set()).add(h.source)
    groups: dict = {}
    for p in record.get("pending", []):
        if p["url"] in by_url:
            groups.setdefault(p["url"], []).append(p)
    lines = [f"🔄 {entity} — что нового"]
    for url, ps in sorted(groups.items(), key=lambda kv: min(_ORDER.index(p["type"]) for p in kv[1])):
        h = by_url[url]
        ps = sorted(ps, key=lambda p: _ORDER.index(p["type"]))
        icons = "".join(dict.fromkeys(FACT_RU[p["type"]][0] for p in ps))
        if any(p["type"] == "official" for p in ps):
            summary = (summarize(h.text) if summarize and h.text else None) or h.text[:400]
            lines.append(f"{icons} {entity} официально, {clock(h.ts)}: «{h.title}»")
            lines += [summary, h.url] if summary else [h.url]
            continue
        coins = [p["what"] for p in ps if p["type"] == "held"]
        labels = []
        for p in ps:
            if p["type"] == "held":
                label = f"твоя монета {coins[0]}" if len(coins) == 1 else f"твои монеты {', '.join(coins)}"
            elif p["type"] == "scale":
                label = f"сумма уже {usd_text(p['usd'], True)}"
            else:
                label = FACT_RU[p["type"]][1]
            labels.append(label)
        label = ", ".join(dict.fromkeys(labels))
        checked = [len(support.get(f"{p['type']}:{p['what']}", ())) for p in ps if p["type"] not in ("held", "scale")]
        alone = bool(checked) and max(checked) <= 1
        title = _TRIM.sub("", h.title)
        title = title if len(title) <= 280 else title[:279] + "…"
        ru = translate(title) if translate else None
        source = f"{SOURCE_RU.get(h.source, h.source)}, {clock(h.ts)}" + (" · пока 1 источник" if alone else "")
        lines.append(f"{icons} {label[:1].upper() + label[1:]}: «{(ru or title).strip()}» ({source})")
    lines.append(WARNING)
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
