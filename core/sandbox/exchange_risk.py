"""Trouble at an exchange before the news: what the bot notices, what it may be, what to do.

The Binance hack of, minute by minute (UTC): from 20:06 Whale Alert's
public channel showed large transfers "from #Binance to unknown new wallet"
($2.8B in 36 minutes); at 20:32 BNB was 3% down in an hour on 200 times its
usual volume while BTC barely moved; the first headline came at 21:12, Binance
paused withdrawals at 02:23. Those who moved their money in the first hours
kept it; those who sold coins on the news sold near the bottom.

Signals, all from public data without keys:
  outflow    Whale Alert: OUTFLOW_USD or more leaving one exchange for new
             unknown wallets within OUTFLOW_MINUTES, in OUTFLOW_MIN_TRANSFERS
             transfers or more. On the channel's 15680 posts of 06.2025-10.2026
             it rang twice: Coinbase (most likely its own move to
             new wallets) and the hack at 20:30, 42 minutes before the news;
             counting any unknown wallet from $1B it rang 8 times, mostly
             exchanges moving their own money;
  canary     BNB at least CANARY_OWN below BTC's move over an hour on
             CANARY_VOLUME_X times its usual hourly volume: since 2022 about
             1.7 hours a year, Binance's own crises (12.2022 reserves panic, the
             SEC suit, CZ's plea) and the hack;
  withdrawals  Binance's public network list: STUCK_COINS or more of its major
             coins with no network open for withdrawals;
  news       the urgent news stories (core/sandbox/urgent.py) about the exchange.
A situation is one exchange with at least one signal; its card says what is
seen, what it may be and how sure the bot is, and what to do for each place
money can be kept. Updates follow as new signals arrive. No automatic
withdrawals: the owner is woken and decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

HOUR_MS = 3_600_000
OUTFLOW_MINUTES = 60
OUTFLOW_USD = 0.5e9
OUTFLOW_MIN_TRANSFERS = 3
CANARY_OWN = -0.04
CANARY_VOLUME_X = 5.0
MAJORS = ("BTC", "ETH", "USDT", "USDC", "BNB", "SOL", "XRP")
STUCK_COINS = 3
SITUATION_HOURS = 24           # signals of one exchange within this belong to one situation
UPDATE_MINUTES = 15
CANARIES = {"Binance": "BNBUSDT"}

_WHALE = re.compile(r"^\s*([\d,.]+)\s+[$#]?([A-Za-z0-9]+)\s+\(([\d,.]+)\s+USD\)\s+transferred from\s+(.+?)\s+to\s+(.+?)"
                    r"(?:\s+Details)?\s*$")
_ALIASES = {"okex": "OKX", "okx": "OKX", "huobi": "HTX", "htx": "HTX", "gate.io": "Gate", "gate": "Gate",
            "crypto.com": "Crypto.com"}


@dataclass(frozen=True)
class Transfer:
    ts: int
    usd: float
    token: str
    source: str        # the exchange it left ("Binance"), or "" when it did not come from one
    to_unknown: bool   # to a wallet nobody has labelled
    fresh: bool        # ... that is new


def exchange_name(label: str) -> str:
    """'#Binance' -> 'Binance'; anything without a hash tag is not an exchange."""
    label = label.strip()
    if not label.startswith("#"):
        return ""
    name = label[1:].strip()
    return _ALIASES.get(name.lower(), name)


def parse_whale(text: str, ts: int) -> Optional[Transfer]:
    """One Whale Alert post: '185,017,485 $XRP (235,816,974 USD) transferred from #Binance to unknown new wallet'."""
    m = _WHALE.match(text.replace(" ", " "))
    if not m:
        return None
    usd = float(m.group(3).replace(",", ""))
    dst = m.group(5).lower()
    return Transfer(ts, usd, m.group(2).upper(), exchange_name(m.group(4)),
                    dst.startswith("unknown"), "new wallet" in dst)


def outflows(transfers: Iterable[Transfer], now_ms: int, minutes: int = OUTFLOW_MINUTES) -> dict:
    """{exchange: (usd, transfers, first ms)} sent to new unknown wallets within the last ``minutes``."""
    out: dict[str, list] = {}
    for t in transfers:
        if t.source and t.fresh and now_ms - minutes * 60_000 <= t.ts <= now_ms:
            usd, n, first = out.get(t.source, (0.0, 0, t.ts))
            out[t.source] = [usd + t.usd, n + 1, min(first, t.ts)]
    return {k: tuple(v) for k, v in out.items()}


def outflow_alarm(usd: float, n: int) -> bool:
    return usd >= OUTFLOW_USD and n >= OUTFLOW_MIN_TRANSFERS


def canary(token_change: float, btc_change: float, volume: float, usual_volume: float) -> Optional[float]:
    """The token's own move over the hour (BTC's taken out) when it is a fall on heavy volume, else None."""
    if not usual_volume or usual_volume <= 0:
        return None
    own = token_change - btc_change
    return own if own <= CANARY_OWN and volume / usual_volume >= CANARY_VOLUME_X else None


def stuck(payload: dict, majors: Iterable[str] = MAJORS) -> list[str]:
    """Major coins with no network open for withdrawals, from Binance's public network list."""
    out = []
    for coin in (payload.get("data") or []):
        if coin.get("coin") in majors:
            nets = coin.get("networkList") or []
            if nets and not any(n.get("withdrawEnable") for n in nets):
                out.append(coin["coin"])
    return sorted(out)


# the situation

@dataclass
class Situation:
    exchange: str
    opened: int
    signals: dict = field(default_factory=dict)   # name -> {"at": ms, "text": what was seen}

    def add(self, name: str, at: int, text: str) -> bool:
        """True when the signal is new to this situation."""
        if name in self.signals:
            return False
        self.signals[name] = {"at": at, "text": text}
        return True


def _kinds(signals: Iterable[str]) -> set[str]:
    """Our own wallet watch (core/sandbox/wallets.py) is an outflow seen first-hand."""
    return {"outflow" if x == "wallets" else x for x in signals}


def level(signals: Iterable[str]) -> str:
    s = _kinds(signals)
    if "withdrawals" in s or ("news" in s and s & {"outflow", "canary"}) or {"outflow", "canary"} <= s:
        return "высокая"
    if s & {"outflow", "news"}:
        return "средняя"
    return "низкая"


def hypotheses(signals: Iterable[str]) -> list[str]:
    """What it may be, the likelier first, from which signals are seen."""
    s = _kinds(signals)
    if "withdrawals" in s:
        return ["биржа остановила выводы: взлом или нехватка денег",
                "реже: большая внеплановая починка кошельков"]
    if {"outflow", "canary"} <= s:
        return ["взлом горячих кошельков или массовое бегство клиентов",
                "реже: биржа сама переводит деньги, а токен падает на слухах"]
    if "outflow" in s:
        return ["взлом горячих кошельков",
                "биржа переносит деньги в холодное хранение",
                "крупный клиент или фонд выводит свои деньги"]
    if "canary" in s:
        return ["кто-то знает о проблеме биржи раньше новостей",
                "крупная ликвидация или распродажа токена",
                "слух"]
    return ["новости о проблеме биржи"]


EVACUATION = (
    ("монеты на споте этой биржи", "вывести на свой кошелёк сейчас, стейблкоины первыми; дешёвая быстрая сеть, "
     "адрес проверить по первым и последним символам. Если выводы уже стоят — не продавать в панике"),
    ("фьючерсы и плечо там же", "снизить плечо или закрыть, чтобы паника не ликвидировала; маржу — на спот и вывести"),
    ("Earn и стейкинг там же", "гибкие — выкупить и вывести; заблокированные быстро не вернуть, следить за биржей"),
    ("свой кошелёк", "взлом биржи его не касается; ничего «срочного» не подписывать: на панике ловят фишингом"),
    ("другие биржи", "паника перекидывается: держать там только нужное для торговли"),
)


def _line(sig: dict) -> str:
    return sig["text"]


def text_situation(sit: Situation, clock, gap: str = "", news: str = "") -> str:
    lines = [f"🆘 {sit.exchange}: похоже на беду · уверенность {level(sit.signals)}", "Что видно:"]
    for name, sig in sorted(sit.signals.items(), key=lambda kv: kv[1]["at"]):
        lines.append(f"• {clock(sig['at'])} {_line(sig)}")
    if gap:
        lines.append(f"• {gap}")
    lines.append(f"• {news}" if news else "• новостей пока нет")
    lines.append("Что это может быть: " + "; ".join(hypotheses(sit.signals)) + ".")
    lines.append(f"Если там твои деньги:")
    lines += [f"— {where}: {what}." for where, what in EVACUATION]
    lines.append("Это предупреждение о сохранности денег, не совет купить или продать.")
    return "\n".join(lines)


def text_more(sit: Situation, name: str, clock) -> str:
    sig = sit.signals[name]
    return (f"⚠️ {sit.exchange}: новое — {clock(sig['at'])} {_line(sig)}\n"
            f"Уверенность теперь: {level(sit.signals)}. Что это может быть: {hypotheses(sit.signals)[0]}.")


def text_reopened(exchange: str, clock, at: int) -> str:
    return f"🟢 {exchange}: выводы снова открыты ({clock(at)})."
