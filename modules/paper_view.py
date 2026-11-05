"""The paper account in the owner's test bot: the watcher's books on a screen.

The paper watcher (scripts/panic_watch.py, every hour on the test VM) writes
runtime/learning/watch/state.json and equity.jsonl; this screen only reads them.
Shown in place of the holdings portfolio, to the owner of the test bot only:
scripts/run_test_bot.py sets RBRY_PAPER_VIEW, the production bot never does.
"""

from __future__ import annotations

import html
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

WATCH_DIR = Path(__file__).resolve().parents[1] / "runtime" / "learning" / "watch"
PAPER_DIR = Path(__file__).resolve().parents[1] / "runtime" / "learning" / "paper"
WORDS = {"up": "рост", "flat": "боковик", "down": "падение", "calm": "спокойно", "nervous": "нервно",
         "wide": "широкий рынок", "narrow": "узкий рынок", "alts": "сезон альтов", "btc": "сезон BTC",
         "hot": "фандинг горячий", "neutral": "фандинг обычный", "cold": "фандинг холодный"}
EVENT_RU = {"SPIKE": "свеча +3% на объёме", "DUMP": "свеча −3% на объёме", "BREAKOUT": "пробой максимума недели",
            "BREAKDOWN": "пробой минимума недели", "ABSORB-BUY": "объём без движения (покупки)",
            "ABSORB-SELL": "объём без движения (продажи)", "TAKER-BUY": "70% рыночных покупок",
            "TAKER-SELL": "70% рыночных продаж", "FUND-HIGH": "фандинг ≥ +0.1%", "FUND-LOW": "фандинг ≤ −0.1%",
            "OI-UP-UP": "OI +10% и цена +3%", "OI-UP-DOWN": "OI +10% и цена −3%", "OI-FLUSH": "волна ликвидаций",
            "CROWD-LONG": "толпа в лонгах", "CROWD-SHORT": "толпа в шортах", "STRONG-VS-BTC": "сильнее BTC",
            "CAPITULATION-TURN": "разворот после −15%", "FELL-WITH-BTC": "упала сильнее BTC"}
SPARK = "▁▂▃▄▅▆▇█"
ACCOUNT = "account"
LIMIT = 3900                         # Telegram takes 4096 characters a message


def _fit(lines: list[str]) -> str:
    text = "\n".join(lines)
    while len(text) > LIMIT and len(lines) > 3:
        lines = lines[:-2] + ["…", lines[-1]]
        text = "\n".join(lines)
    return text


def enabled(user_id: int, owner_id: int) -> bool:
    return os.environ.get("RBRY_PAPER_VIEW") == "1" and user_id == owner_id


def _state() -> dict:
    """The watcher's state; unreadable (being replaced, damaged) counts as none rather than breaking the screen
    or a coin's analysis."""
    path = WATCH_DIR / "state.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def _equity(book: str, points: int = 40) -> list[float]:
    """A book's value at every hour since the account opened, thinned to ``points``."""
    path = WATCH_DIR / "equity.jsonl"
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            v = json.loads(line)["books"].get(book) if line.strip() else None
        except (ValueError, KeyError, AttributeError):  # a line being written right now
            continue
        if v is not None:
            values.append(float(v))
    step = max(1, math.ceil(len(values) / points))
    return values[::step] + ([values[-1]] if values and (len(values) - 1) % step else [])


def spark(values: list[float]) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return SPARK[3] * len(values)
    return "".join(SPARK[int((x - lo) / (hi - lo) * (len(SPARK) - 1))] for x in values)


def _price(x) -> str:
    if x is None:
        return "—"
    if x >= 100:
        return f"${x:,.2f}"
    if x >= 1:
        return f"${x:.3f}"
    return f"${x:.{max(4, 3 - math.floor(math.log10(x)))}f}" if x > 0 else "$0"


def _position(p: dict) -> str:
    side = "лонг" if p["side"] > 0 else "шорт"
    parts = [f"<b>{html.escape(p['base'])}</b> {side} ${p['size']:.2f}", f"{p['net_pct']:+.1f}% ({p['pnl_usd']:+.2f} $)",
             f"вход {_price(p['entry'])}"]
    if "stop" in p:
        parts.append(f"стоп {_price(p['stop'])}")
    if "half" in p:
        parts.append("половина закрыта" if p["half_sold"] else f"половина на {_price(p['half'])}")
    if "target" in p:
        parts.append(f"цель {_price(p['target'])}")
    parts.append(f"до {p['due']}")
    return "• " + " · ".join(parts) + f" · <i>{html.escape(p['source'])}</i>"


def _closed(p: dict) -> str:
    mark = "✅" if p["pnl_usd"] > 0 else "❌"
    side = "лонг" if p["side"] > 0 else "шорт"
    word = {"stop": "стоп", "breakeven": "без убытка", "time": "срок", "target": "цель", "keep": "трейлинг",
            "liquidated": "ликвидация"}.get(p["status"], p["status"])
    return (f"{mark} {html.escape(p['base'])} {side} {p['net_pct']:+.1f}% ({p['pnl_usd']:+.2f} $) · {word} · "
            f"{p['exit']} · <i>{html.escape(p['source'])}</i>")


def _keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Все счета", callback_data="paper:all"),
         InlineKeyboardButton("🔄 Обновить", callback_data="paper:account")],
        [InlineKeyboardButton("⬅️ Меню", callback_data="back_to_menu")]])


def account_screen() -> tuple[str, InlineKeyboardMarkup]:
    s = _state()
    acc = s.get("books", {}).get(ACCOUNT)
    if not acc:
        return ("📒 Бумажный счёт ещё не открыт или дозорный не отработал первый час.", _keyboard())
    opened = (s.get("account") or {}).get("opened", "?")
    regime = s.get("regime") or ""
    words = {"up": "рост", "flat": "боковик", "down": "падение", "calm": "спокойно", "nervous": "нервно"}
    regime_ru = " · ".join(words.get(x, x) for x in regime.split("|")) if regime else "неизвестен"
    value = acc["value"]
    lines = [f"📒 <b>Счёт Разгон</b> (бумага, деньги не двигаются)",
             f"Баланс <b>${value:.2f}</b> из $100 ({value - 100:+.2f} $, {value - 100:+.1f}%) · свободно ${acc['cash']:.2f}",
             f"С {opened}: {spark(_equity(ACCOUNT)) or '—'}",
             f"Режим рынка сейчас: {regime_ru}. Обновлено {s.get('updated', '?')}.",
             "Места в счёте: " + (", ".join(f"{html.escape(a['label'])} — до {a['places']}"
                                           for a in s.get("allocation", []) if a["places"]) or "—")
             + " (вместе не больше 5)", "",
             f"<b>Открыто ({len(acc['open'])} из 5):</b>"]
    lines += [_position(p) for p in acc["open"]] or ["— ничего: стратегии ждут своего момента"]
    lines += ["", "<b>Последние сделки:</b>"]
    lines += [_closed(p) for p in acc.get("last", [])[:8]] or ["— пока не было"]
    if acc["closed"]:
        lines += ["", f"Закрыто {acc['closed']}: в плюс {acc['up_of10']:.1f} из 10, в среднем "
                      f"{acc['usd_per_100'] * 0.2:+.2f} $ на сделку в $20."]
    return _fit(lines), _keyboard()


def _tail(path: Path, lines: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 400_000))
        chunk = fh.read().decode("utf-8", errors="ignore").splitlines()[-lines:]
    out = []
    for line in chunk:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def brain_block(symbol: str) -> str:
    """One verdict in plain words under a coin's analysis (plan item 11): the brain's place for it today, what
    the watcher saw on it in a day, the market now, and whether the paper account holds it."""
    base = symbol.upper().removesuffix("USDT")
    s = _state()
    lines = ["🧠 Что думает бот (бумага, тест)"]
    regime = s.get("regime") or ""
    market = s.get("market") or {}
    parts = [" · ".join(WORDS.get(x, x) for x in regime.split("|"))] if regime else []
    parts += [WORDS.get(market.get(k, ""), "") for k in ("breadth_state", "alt_state", "fund_state")]
    parts = [p for p in parts if p]
    if parts:
        lines.append("Рынок сейчас: " + ", ".join(parts) + ".")
    decisions = _tail(PAPER_DIR / "decisions.jsonl", 1)
    if decisions and decisions[-1].get("places"):
        places = sorted(decisions[-1]["places"], key=lambda p: -float(p[2]))
        rank = next((i for i, p in enumerate(places, 1) if p[1] == base), None)
        if rank:
            lines.append(f"Место среди {len(places)} монет по прогнозу мозга на 2 недели: {rank} "
                         f"({'в верхней трети' if rank <= len(places) / 3 else 'в нижней трети' if rank > 2 * len(places) / 3 else 'в середине'}).")
    seen = [h for h in _tail(WATCH_DIR / "seen.jsonl", 30) if any(c.get("base") == base for c in h.get("coins", []))]
    if seen:
        last = seen[-1]
        coin = next(c for c in last["coins"] if c.get("base") == base)
        when = datetime.fromtimestamp((last["hour"] + 3_600_000) / 1000, timezone.utc).strftime("%d.%m %H:%M UTC")
        lines.append("Дозорный видел: " + ", ".join(EVENT_RU.get(e, e) for e in coin.get("events", [])) + f" ({when}).")
    held = [p for p in (s.get("books", {}).get("account", {}) or {}).get("open", []) if p.get("base") == base]
    for p in held:
        lines.append(f"Счёт Разгон держит: {'лонг' if p['side'] > 0 else 'шорт'} ${p['size']:.0f}, {p['net_pct']:+.1f}%"
                     + (f", стоп {_price(p['stop'])}" if "stop" in p else "") + f", до {p['due']} · {p['source']}.")
    if len(lines) <= 2 and not held and not seen:
        lines.append("Сигналов по ней сейчас нет: ни одна стратегия её не держит и не ждёт.")
    return "\n".join(lines)


def all_screen() -> tuple[str, InlineKeyboardMarkup]:
    s = _state()
    books = s.get("books", {})
    lines = ["📊 <b>Все бумажные счета</b> (у каждой стратегии свои $100, общий счёт — сверху)", ""]
    for name, b in books.items():
        title = "Счёт Разгон (общий)" if name == ACCOUNT else b.get("label", name)
        stats = (f", в плюс {b['up_of10']:.1f} из 10" if b.get("closed") else "")
        lines.append(f"<b>{html.escape(title)}</b>: ${b['value']:.2f} · открыто {len(b['open'])} · закрыто "
                     f"{b.get('closed', 0)}{stats}\n{spark(_equity(name)) or '—'}")
        for p in b["open"][:5]:
            lines.append("  " + _position(p))
        lines.append("")
    if not books:
        lines.append("Дозорный ещё не записал ни одного часа.")
    lines.append(f"Обновлено {s.get('updated', '?')}.")
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📒 Счёт Разгон", callback_data="paper:account"),
                                InlineKeyboardButton("⬅️ Меню", callback_data="back_to_menu")]])
    return _fit(lines), kb
