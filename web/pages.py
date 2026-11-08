"""The owner's site in HTML: the paper accounts, every trade with what it saw and why, the coins, C1, the market
hours, the system and the cabinet. Server-rendered; the only script is static/app.js (passkeys, charts, numbers)
and the chart library next to it. Every value from a data file goes through e() or a number formatter."""

from __future__ import annotations

import hashlib
import html
import math
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

from web import charts, data

STATIC = Path(__file__).resolve().parent / "static"


def _version() -> str:
    """The site's own files' fingerprint: a deploy changes the addresses, so no browser keeps an old copy."""
    h = hashlib.sha256()
    for name in ("app.css", "app.js"):
        try:
            h.update((STATIC / name).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:10]


V = _version()

BERLIN = ZoneInfo("Europe/Berlin")
STATUS = {"open": "открыта", "stop": "стоп", "breakeven": "остаток по цене входа", "time": "срок вышел",
          "target": "цель", "keep": "трейлинг", "liquidated": "ликвидация"}
BOOKS = {"account": "Счёт Разгон", "nostop": "паника без стопа", "selector": "модель выбора"}
EVENTS = {"SPIKE": "свеча +3% на объёме", "DUMP": "свеча −3% на объёме", "BREAKOUT": "пробой максимума недели",
          "BREAKDOWN": "пробой минимума недели", "OI-FLUSH": "волна ликвидаций", "CAPITULATION-TURN": "разворот после −15%",
          "FELL-WITH-BTC": "упала сильнее BTC", "STRONG-VS-BTC": "сильнее BTC", "TAKER-BUY": "70% рыночных покупок",
          "TAKER-SELL": "70% рыночных продаж", "FUND-HIGH": "фандинг ≥ +0.1%", "FUND-LOW": "фандинг ≤ −0.1%",
          "OI-UP-UP": "OI +10% и цена +3%", "OI-UP-DOWN": "OI +10% и цена −3%", "CROWD-LONG": "толпа в лонгах",
          "CROWD-SHORT": "толпа в шортах", "ABSORB-BUY": "объём без движения (покупки)",
          "ABSORB-SELL": "объём без движения (продажи)", "LISTING-24H": "первые сутки листинга"}
STATES = {"wide": "широкий рынок", "narrow": "узкий рынок", "alts": "сезон альтов", "btc": "сезон BTC",
          "hot": "фандинг горячий", "neutral": "фандинг обычный", "cold": "фандинг холодный"}
EVENTS_RU = {"device added": "устройство добавлено", "binding code used": "введён код привязки",
             "app code confirmed": "подтверждён код приложения", "passkey added": "добавлен passkey",
             "login with passkey": "вход по passkey", "login with the app code": "вход по коду приложения",
             "device revoked": "устройство отвязано", "logout everywhere": "выход везде",
             "wrong binding code": "неверный код привязки", "wrong app code": "неверный код приложения",
             "wrong app code while binding": "неверный код приложения при привязке",
             "passkey login without its request or of another device": "чужой passkey или устаревший запрос",
             "passkey not accepted": "passkey не принят",
             "passkey dialog failed": "окно passkey не открылось на устройстве"}
e = html.escape

# icons: 24×24 strokes drawn here (no icon font, no request)
ICONS = {
    "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v9.5h13V10"/><path d="M10 19.5v-5h4v5"/>',
    "trades": '<path d="M4 18 9.5 12l4 3.5L20 8"/><path d="M15 8h5v5"/>',
    "coins": '<circle cx="9" cy="9" r="5.5"/><path d="M14.4 7.2A5.5 5.5 0 1 1 7.2 14.4"/>',
    "c1": '<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/>',
    "market": '<path d="M2 12h4l2.5-6 4 13 3-9 1.5 2H22"/>',
    "system": '<rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="13" width="18" height="7" rx="2"/>'
              '<path d="M7 7.5h.01M7 16.5h.01"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/>',
    "more": '<circle cx="5" cy="12" r="1.3"/><circle cx="12" cy="12" r="1.3"/><circle cx="19" cy="12" r="1.3"/>',
    "skip": '<circle cx="12" cy="12" r="8.5"/><path d="M6 6l12 12"/>',
    "up": '<path d="M7 17 17 7"/><path d="M8 7h9v9"/>',
    "down": '<path d="M7 7l10 10"/><path d="M17 8v9H8"/>',
    "key": '<circle cx="8" cy="15" r="4"/><path d="M11 12l9-9M17 6l3 3M15 8l2 2"/>',
    "shield": '<path d="M12 3 5 6v5c0 4.5 3 8 7 10 4-2 7-5.5 7-10V6z"/><path d="m9 12 2 2 4-4"/>',
    "back": '<path d="M15 5l-7 7 7 7"/>',
}


def icon(name: str, cls: str = "") -> str:
    return (f'<svg class="i {cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


# the mark: the Hub's hexagon, a berry inside, the leaf of RedBerry
LOGO = ('<svg class="logo" viewBox="0 0 64 64" aria-hidden="true"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#ff4a64"/><stop offset="1" stop-color="#b5112c"/></linearGradient></defs>'
        '<path d="M32 3 57 17.5v29L32 61 7 46.5v-29z" fill="#080b12" stroke="url(#lg)" stroke-width="3"/>'
        '<path d="M32 9.5 51.5 21v22L32 54.5 12.5 43V21z" fill="none" stroke="#2c9fd0" stroke-opacity=".45"/>'
        '<g fill="url(#lg)"><circle cx="27" cy="29" r="5"/><circle cx="37" cy="29" r="5"/><circle cx="32" cy="37" r="5"/>'
        '<circle cx="23.5" cy="37.5" r="4.2"/><circle cx="40.5" cy="37.5" r="4.2"/><circle cx="27.5" cy="45" r="4"/>'
        '<circle cx="36.5" cy="45" r="4"/></g>'
        '<path d="M32 24c-2-4-6-5-9-4 2 3 5 4.5 9 4zm0 0c2-4 6-5 9-4-2 3-5 4.5-9 4z" fill="#3dae5b"/></svg>')

NAV = (("/", "Обзор", "home"), ("/trades", "Сделки", "trades"), ("/coins", "Монеты", "coins"), ("/c1", "C1", "c1"),
       ("/hours", "Рынок", "market"), ("/health", "Система", "system"), ("/me", "Кабинет", "user"))
PHONE_TABS = ("/", "/trades", "/coins", "/c1")


# formatting

def event_ru(what: str) -> str:
    return EVENTS_RU.get(what) or next((v + what[len(k):] for k, v in EVENTS_RU.items() if what.startswith(k + ":")), what)


def when(ms: Optional[int], fmt: str = "%d.%m %H:%M") -> str:
    if not ms:
        return "—"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone(BERLIN).strftime(fmt)


def ago(seconds: Optional[float]) -> str:
    if seconds is None:
        return "нет файла"
    if seconds < 120:
        return f"{seconds:.0f} с назад"
    if seconds < 7200:
        return f"{seconds / 60:.0f} мин назад"
    return f"{seconds / 3600:.1f} ч назад"


def _num(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def pct(x, digits: int = 1) -> str:
    x = _num(x)
    return "—" if x is None else f"{x:+.{digits}f}%".replace("-", "−")


def share(x, digits: int = 1) -> str:
    x = _num(x)
    return "—" if x is None else pct(100 * x, digits)


def money(x) -> str:
    x = _num(x)
    if x is None:
        return "—"
    a = abs(x)
    body = f"{a:,.2f}".replace(",", " ") if a >= 1 else f"{a:.6g}"
    return f"{'−' if x < 0 else ''}${body}"


def signed_money(x) -> str:
    """A result in dollars: always cents, always a sign."""
    x = _num(x)
    if x is None:
        return "—"
    return f"{'+' if x > 0 else '−' if x < 0 else ''}${abs(x):,.2f}".replace(",", " ")


def tone(x) -> str:
    x = _num(x)
    return "pos" if x is not None and x > 0 else "neg" if x is not None and x < 0 else ""


def badge(base: str, big: bool = False) -> str:
    text = str(base)[:4]
    return f'<span class="badge{" lg" if big else ""}{" long" if len(text) > 3 else ""}">{e(text)}</span>'


def coin_link(base: str, big: bool = False) -> str:
    return f'<a class="coin" href="/coin/{quote(str(base))}">{badge(base, big)}<span>{e(str(base))}</span></a>'


def side_chip(side) -> str:
    return (f'<span class="chip long">{icon("up")}лонг</span>' if (_num(side) or 1) > 0
            else f'<span class="chip short">{icon("down")}шорт</span>')


def source_chip(source: str) -> str:
    return f'<span class="chip s-{charts.series_of(source)}"><i class="sdot"></i>{e(source)}</span>'


def status_chip(t: dict) -> str:
    st = t.get("status")
    if st == "open":
        return '<span class="chip open">открыта</span>'
    won = (_num(t.get("net_pct")) or 0) > 0
    return f'<span class="chip {"won" if won else "lost"}">{e(STATUS.get(st, str(st or "")))}</span>'


def card(title: str, body: str, cls: str = "c12", more: str = "") -> str:
    head = f'<div class="card-h"><h2>{e(title)}</h2>{more}</div>' if title or more else ""
    return f'<section class="card {cls}">{head}{body}</section>'


def fact(label: str, value: str, cls: str = "", extra: str = "") -> str:
    return f'<div class="fact"><span>{label}</span><b class="{cls}">{value}</b>{extra}</div>'


def lw(payload: dict, cls: str = "", label: str = "график") -> str:
    return (f'<div class="lw {cls}" role="img" aria-label="{e(label)}" data-chart="{charts.attr(payload)}">'
            f'<div class="readout"></div></div>')


def gauge(value: Optional[float]) -> str:
    return f'<div class="gauge"><i data-left="{value:.0f}"></i></div>' if value is not None else ""


def layout(title: str, body: str, active: str = "", device: str = "", sub: str = "", charts_on: bool = False) -> str:
    nav = tabs = health = ""
    if device:
        nav = "".join(f'<a href="{h}" class="{"on" if h == active else ""}">{icon(i)}<span>{n}</span></a>'
                      for h, n, i in NAV)
        on_more = active not in PHONE_TABS
        tabs = ('<nav class="tabs-m">'
                + "".join(f'<a href="{h}" class="{"on" if h == active else ""}">{icon(i)}{n}</a>'
                          for h, n, i in NAV if h in PHONE_TABS)
                + f'<details><summary class="{"on" if on_more else ""}">{icon("more")}Ещё</summary><div class="sheet">'
                + "".join(f'<a href="{h}">{icon(i)}{n}</a>' for h, n, i in NAV if h not in PHONE_TABS)
                + '</div></details></nav>')
        health = _health_dot()
    top = (f'<header class="top"><a class="brand" href="/">{LOGO}<span><b>RBRY</b><small>paper terminal</small></span></a>'
           f'<nav class="nav">{nav}</nav><div class="top-right">{health}'
           + (f'<span class="who">{e(device)}</span>' if device else "") + '</div></header>')
    lib = '<script src="/static/vendor/lightweight-charts.standalone.production.js" defer></script>' if charts_on else ""
    head = f'<div class="head"><h1>{e(title)}' + (f'<span class="sub">{sub}</span>' if sub else "") + '</h1></div>'
    return ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<meta name="referrer" content="same-origin"><meta name="theme-color" content="#04060b">'
            '<meta name="apple-mobile-web-app-capable" content="yes">'
            '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">'
            '<meta name="apple-mobile-web-app-title" content="RBRY">'
            '<link rel="icon" href="/static/img/icon-64.png"><link rel="apple-touch-icon" href="/static/img/icon-180.png">'
            f'<title>{e(title or "Вход")} · RBRY</title>'
            '<link rel="preload" href="/static/vendor/onest-cyrillic-wght-normal.woff2" as="font" type="font/woff2" crossorigin>'
            f'<link rel="stylesheet" href="/static/app.css?v={V}">{lib}<script src="/static/app.js?v={V}" defer></script></head>'
            f'<body>{top}<main>{head if title else ""}{body}</main>{tabs}</body></html>')


def _health_dot() -> str:
    try:
        units, stale = data.services(), data.stale()
    except Exception:  # noqa: BLE001 - the dot is a hint, never a reason for a page to fail
        return ""
    down = [u for u, s in units if s not in ("active", "?")]
    if not down and not stale:
        cls, text = "ok", "всё работает"
    else:
        cls, text = ("bad" if down else "warn"), f"проблем: {len(down) + len(stale)}"
    return f'<a class="status" href="/health"><i class="dot {cls}"></i><span>{text}</span></a>'


# login and binding

def _door(title: str, sub: str, body: str, step: int = 0) -> str:
    steps = ""
    if step:
        steps = '<div class="steps">' + "".join(
            f'<i class="{"done" if i < step else "now" if i == step else ""}"></i>' for i in (1, 2, 3)) + '</div>'
    return layout("", f'<div class="door"><section class="card">{LOGO}<h1>{e(title)}<span class="sub">{sub}</span></h1>'
                      f'{steps}{body}</section></div>')


def login(device: str, has_passkey: bool, message: str = "") -> str:
    note = f'<p class="msg">{e(message)}</p>' if message else ""
    if not has_passkey:
        return _door("Вход", "у этого устройства ещё нет ключа",
                     note + '<p>Привяжи устройство: <a href="/enroll">ввести код привязки</a>. Код приходит в Telegram '
                     'после команды на сервере.</p>')
    return _door("Вход", f'Устройство: <b>{e(device)}</b>',
                 note + '<p><button id="passkey-login" class="big">Войти по passkey</button></p><p id="msg" class="msg"></p>'
                 '<details class="more-h"><summary>Нет доступа к passkey</summary><form method="post" action="/login/totp">'
                 '<label>Код из приложения <input name="otp" inputmode="numeric" autocomplete="one-time-code" '
                 'maxlength="6" required></label><div class="actions"><button>Войти</button></div></form>'
                 '<p class="muted small">Вход по коду — на 2 часа, о нём сразу придёт сообщение в Telegram.</p></details>')


def enroll_code(message: str = "") -> str:
    note = f'<p class="msg">{e(message)}</p>' if message else ""
    return _door("Привязка устройства", "шаг 1 из 3 · код из Telegram",
                 note + '<form method="post" action="/enroll"><label>Код привязки <input name="code" autocomplete="off" '
                 'required placeholder="ABCD-EF01-2345"></label><div class="actions"><button class="big">Дальше</button>'
                 '</div></form><p class="muted small">Код живёт 15 минут и работает один раз, только на устройстве, '
                 'для которого его выпустили.</p>', 1)


def enroll_totp(qr_svg: str, secret: str, uri: str, first: bool, message: str = "") -> str:
    note = f'<p class="msg">{e(message)}</p>' if message else ""
    if first:
        how = (f'<p>Добавь в приложение для одноразовых кодов (на iPhone — «Пароли» → «Коды», или Google Authenticator, '
               f'2FAS) этот QR-код или ссылку:</p><div class="qr">{qr_svg}</div>'
               f'<p class="small"><a href="{e(uri)}">Добавить на этом устройстве</a> · ключ: <code>{e(secret)}</code></p>')
    else:
        how = '<p>Приложение для кодов уже привязано на другом устройстве: введи из него текущий код.</p>'
    return _door("Привязка: приложение для кодов", "шаг 2 из 3", note + how +
                 '<form method="post" action="/enroll/totp"><label>Код из приложения <input name="otp" '
                 'inputmode="numeric" autocomplete="one-time-code" maxlength="6" required></label>'
                 '<div class="actions"><button class="big">Проверить</button></div></form>', 2)


def enroll_passkey(device: str) -> str:
    return _door("Привязка: passkey", "шаг 3 из 3",
                 f'<p>Последний шаг: создать ключ входа для «{e(device)}». Телефон попросит Face ID, ноутбук — '
                 'Windows Hello. Сохраняй ключ на этом устройстве.</p>'
                 '<p><button id="passkey-register" class="big">Создать passkey</button></p><p id="msg" class="msg"></p>', 3)


# the site

def _trade_href(book: str, t: dict) -> Optional[str]:
    try:
        return f'/trade/{quote(book)}/{int(t["moment"])}/{quote(str(t["base"]))}'
    except (KeyError, TypeError, ValueError):
        return None


def _position(p: dict, book: str, index: dict) -> str:
    full = index.get((p.get("moment"), p.get("base")), {})
    due_ts, now = _num(full.get("due_ts")), _time.time() * 1000
    left = ""
    if due_ts and _num(p.get("moment")):
        span = max(due_ts - p["moment"], 1)
        done = min(max((now - p["moment"]) / span, 0), 1)
        hours = max((due_ts - now) / 3_600_000, 0)
        rest = ("ещё " + (f"{hours:.0f} ч" if hours >= 1 else "меньше часа")) if hours else "срок вышел"
        left = (f'<div class="time-left"><div class="bar"><i data-w="{done * 100:.0f}"></i></div>'
                f'<span><em>{rest}</em><em>до {e(str(p.get("due", "—")))}</em></span></div>')
    href = _trade_href(book, p) or "/trades"
    base = str(p.get("base", "?"))
    return (f'<a class="pos-card" href="{href}"><div class="pos-row"><span class="coin">{badge(base)}'
            f'<span>{e(base)}</span></span><span class="pos-res {tone(p.get("net_pct"))}">{pct(p.get("net_pct"))}</span></div>'
            f'<div class="pos-meta">{side_chip(p.get("side"))}{source_chip(str(p.get("source", "")))}'
            f'<span class="chip">{money(p.get("size"))} · вход {money(p.get("entry"))}</span>'
            f'<span class="chip {tone(p.get("pnl_usd"))}">{signed_money(p.get("pnl_usd"))}</span></div>{left}</a>')


def overview(device: str) -> str:
    s, j = data.state(), data.journal()
    books = s.get("books") or {}
    acc = books.get("account") or {}
    index = {(t["moment"], t["base"]): t for _, t in data.all_trades(j)}
    parts = []
    if acc:
        value = _num(acc.get("value"))
        value = 100.0 if value is None else value
        whole, cents = f"{value:,.2f}".replace(",", " ").split(".")
        closed = [t for t in ((j.get("books") or {}).get("account", {}).get("trades") or []) if t.get("status") != "open"]
        won = sum((_num(t.get("net_pct")) or 0) > 0 for t in closed)
        kpis = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in (
            ("свободно", money(acc.get("cash"))), ("открыто сделок", str(len(acc.get("open") or []))),
            ("закрыто в плюс", f"{won} из {len(closed)}" if closed else "—"),
            ("пропущено", e(str(acc.get("skipped", 0))))))
        eq = data.equity("account")
        chart = lw(charts.balance(eq), "", "баланс Счёта Разгон") if len(eq) > 1 else '<p class="empty">Пока мало точек.</p>'
        parts.append(
            f'<section class="card hero c8"><div class="card-h"><h2>Счёт Разгон · бумага</h2>'
            f'<div class="ranges"><button data-days="1">24ч</button><button data-days="7">7д</button>'
            f'<button data-days="0" class="on">всё</button></div></div>'
            f'<div class="hero-top"><div class="big"><span>${whole}</span><span class="cents">.{cents}</span></div>'
            f'<span class="delta {tone(value - 100)}">{icon("up" if value >= 100 else "down")}'
            f'{signed_money(value - 100)} · {pct(value - 100, 2)}</span>'
            f'<span class="muted small">из $100 · обновлено {e(str(s.get("updated", "?")))}</span></div>'
            f'<div class="kpis">{kpis}</div>{chart}</section>')
    caps = {a.get("label"): a.get("places") for a in s.get("allocation") or [] if isinstance(a, dict)}
    opened = [p for p in acc.get("open") or [] if isinstance(p, dict)]
    slots = "".join(f'<div class="slot full s-{charts.series_of(str(p.get("source", "")))}">{e(str(p.get("base", "")))}</div>'
                    for p in opened[:5])
    slots += '<div class="slot">+</div>' * max(0, 5 - len(opened))
    cap_text = "".join(f'<span class="chip s-{charts.series_of(str(k))}"><i class="sdot"></i>{e(str(k))} — до {e(str(v))}'
                       f'</span>' for k, v in caps.items())
    side = (card("Места счёта", f'<div class="slots">{slots}</div><div class="pos-meta">{cap_text}</div>'
                 '<p class="chart-note">Пять мест по ~$20. Модель выбора берёт не больше трёх: два всегда остаются '
                 'панике.</p>', "")
            + card("Паника сегодня", pulse_today(), "", '<a class="more" href="/hours">рынок →</a>'))
    parts.append(f'<div class="c4 stack">{side}</div>')
    parts.append(card("Открытые позиции", f'<div class="positions">{"".join(_position(p, "account", index) for p in opened)}'
                      '</div>' if opened else '<p class="empty">Открытых сделок нет — счёт ждёт сигнала.</p>', "c12",
                      '<a class="more" href="/trades">все сделки →</a>'))
    eqs = data.equity_books()
    for key in ("nostop", "selector"):
        b = books.get(key)
        if not b:
            continue
        v = _num(b.get("value")) or 100.0
        line = [x for _, x in eqs.get(key, [])][-72:]
        parts.append(f'<a class="card strat c4 s-{key}" href="/trades?book={key}"><div class="strat-h"><i class="sdot"></i>'
                     f'{e(str(b.get("label") or BOOKS.get(key, key)))}</div><div class="strat-v"><div><b>{money(v)}</b>'
                     f'<div class="small {tone(v - 100)}">{pct(v - 100, 2)}</div><div class="small muted">открыто '
                     f'{len(b.get("open") or [])} · закрыто {e(str(b.get("closed", 0)))}</div></div>'
                     f'{charts.spark(line, key)}</div></a>')
    c1 = data.c1_state()
    if c1:
        parts.append(f'<a class="card strat c4 s-c1" href="/c1"><div class="strat-h"><i class="sdot"></i>C1 · условные $10 000'
                     f'</div><div class="strat-v"><div><b>{money(c1.get("value_usd"))}</b><div class="small '
                     f'{tone(c1.get("return_pct"))}">{pct(c1.get("return_pct"), 2)}</div><div class="small muted">с '
                     f'{e(str(c1.get("from", "—")))}</div></div><span class="chip">'
                     f'{"держит BTC" if c1.get("btc_in") else "BTC-половина в USDT"}</span></div></a>')
    main3 = {k: eqs[k] for k in ("account", "nostop", "selector") if len(eqs.get(k, [])) > 1}
    if len(main3) > 1:
        payload = charts.compare(main3, BOOKS)
        legend = "".join(f'<span class="s-{x["key"]}"><i></i>{e(x["label"])}<b class="{tone(x["data"][-1]["value"])}">'
                         f'{pct(x["data"][-1]["value"], 2)}</b></span>' for x in payload["series"])
        parts.append(card("Стратегии рядом: доходность с начала", f'<div class="legend">{legend}</div>'
                          f'{lw(payload, "", "доходность стратегий")}<p class="chart-note">Все три книги с одного старта: '
                          f'Счёт Разгон — общий счёт, паника и модель выбора — каждая на своих $100.</p>', "c12"))
    return layout("Обзор", f'<div class="grid">{"".join(parts)}</div>', "/", device,
                  sub="бумажные счета: ни одной настоящей сделки", charts_on=True)


def pulse_today() -> str:
    day0 = int(datetime.now(BERLIN).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    today = [h for h in data.hours(48) if (h.get("hour") or 0) >= day0]
    if not today:
        return '<p class="empty">Часов за сегодня ещё нет.</p>'
    top = max(today, key=lambda h: len(h.get("panic") or []))
    n = len(top.get("panic") or [])
    nervous = sum(bool(h.get("nervous")) for h in today)
    most = (f'Больше всего — {n} из нужных 4 в {when(top["hour"], "%H:%M")}: {e(", ".join((top.get("panic") or [])[:12]))}.'
            if n else "Ни одной монеты в панике.")
    return (f'{charts.panic_hours(today)}<p class="small">Рынок нервный {nervous} ч из {len(today)}. {most}</p>'
            '<p class="chart-note">Число — монет в панике за час. Паника входит, когда их 4 и больше в нервном рынке '
            '(ячейка светится). Жёлтая черта — нервный час.</p>')


def _trade_row(book: str, t: dict) -> str:
    href = _trade_href(book, t)
    if href is None:
        return ""
    held = t.get("held") or {}
    base = str(t["base"])
    return (f'<a class="row trade-row" href="{href}"><span class="when">{when(t.get("moment"))}</span>'
            f'<span class="coin-col"><span class="coin">{badge(base)}<span>{e(base)}</span></span></span>'
            f'<span class="side-col">{side_chip(t.get("side"))}</span>'
            f'<span class="strategy-col">{source_chip(str(t.get("source", "")))}</span>'
            f'<span class="status-col">{status_chip(t)}</span>'
            f'<span class="meta-col"><span class="when">{when(t.get("moment"))}</span>{side_chip(t.get("side"))}'
            f'{status_chip(t)}</span>'
            f'<span class="res {tone(t.get("net_pct"))}">{pct(t.get("net_pct"))}</span>'
            f'<span class="usd {tone(t.get("pnl_usd"))}" title="лучший момент {share(held.get("best"))} · худший '
            f'{share(held.get("worst"))}">{signed_money(t.get("pnl_usd"))}</span></a>')


def _trade_list(rows: list[tuple[str, dict]]) -> str:
    if not rows:
        return '<p class="empty">Сделок нет.</p>'
    head = ('<div class="row trade-row row-h"><span>вход</span><span>монета</span><span class="side-col">сторона</span>'
            '<span class="strategy-col">стратегия</span><span class="status-col">статус</span><span class="res">итог</span>'
            '<span class="usd">$</span></div>')
    return f'<div class="rows">{head}{"".join(_trade_row(b, t) for b, t in rows)}</div>'


def _summary(items: list[dict]) -> str:
    closed = [t for t in items if t.get("status") != "open"]
    won = sum((_num(t.get("net_pct")) or 0) > 0 for t in closed)
    nets = [_num(t.get("net_pct")) for t in closed if _num(t.get("net_pct")) is not None]
    usd = sum(_num(t.get("pnl_usd")) or 0 for t in items)
    mean = sum(nets) / len(nets) if nets else None
    return ('<div class="stats">'
            f'<div class="stat"><span>сделок</span><b>{len(items)}</b><em>открыто {len(items) - len(closed)}</em></div>'
            f'<div class="stat"><span>закрыто в плюс</span><b>{won} из {len(closed)}</b>'
            f'<em>{f"{won / len(closed) * 100:.0f}% сделок" if closed else "пока не закрывали"}</em></div>'
            f'<div class="stat"><span>средний итог закрытых</span><b class="{tone(mean)}">{pct(mean)}</b>'
            f'<em>лучший {pct(max(nets)) if nets else "—"}</em></div>'
            f'<div class="stat"><span>итог в $</span><b class="{tone(usd)}">{signed_money(usd)}</b>'
            f'<em>вместе с открытыми</em></div></div>')


def _book_tabs(books: dict, active: str, skips_n: int) -> str:
    tabs = "".join(f'<a href="/trades?book={quote(k)}" class="s-{k}{" on" if k == active else ""}"><i class="sdot"></i>'
                   f'{e(BOOKS.get(k, str(b.get("label", k))))}<span class="count">{len(b.get("trades") or [])}</span></a>'
                   for k, b in books.items())
    return (f'<nav class="filters">{tabs}<a href="/skips" class="{"on" if active == "skips" else ""}">{icon("skip")}'
            f'пропуски<span class="count">{skips_n}</span></a></nav>')


def _skips_of(j: dict) -> list[dict]:
    out = [s for s in ((j.get("books") or {}).get("account", {}).get("skips") or [])
           if isinstance(s, dict) and s.get("reason") != "no_price"]
    return sorted(out, key=lambda s: -(s.get("moment") or 0))


def trades(device: str, book: str) -> str:
    j = data.journal()
    books = j.get("books") or {}
    book = book if book in books else "account"
    items = [t for t in books.get(book, {}).get("trades") or [] if isinstance(t, dict) and "moment" in t]
    rows = sorted(((book, t) for t in items), key=lambda bt: -bt[1]["moment"])
    return layout("Сделки", _book_tabs(books, book, len(_skips_of(j))) + '<div class="grid">'
                  + card("", _summary(items)) + card("", _trade_list(rows)) + '</div>', "/trades", device,
                  sub=e(BOOKS.get(book, book)) + " · нажми на сделку: что модель видела и почему вошла")


def _factor_rows(by: dict, values: dict, words: dict, n: int = 6) -> str:
    ranked = sorted(((k, v) for k, v in by.items() if _num(v) is not None), key=lambda kv: kv[1])
    up = [kv for kv in reversed(ranked) if kv[1] > 0][:n]
    down = [kv for kv in ranked if kv[1] < 0][:n]
    biggest = max([abs(v) for _, v in up + down] + [1e-9])

    def row(f, added):
        value = values.get(f)
        shown = f"{value:.4g}" if isinstance(value, float) else str(value) if value is not None else "—"
        kind = "plus" if added > 0 else "minus"
        return (f'<div class="why-row"><span>{e(str(words.get(f, f)))}<span class="v">тогда: {e(shown)}</span></span>'
                f'<div class="why-bar"><i class="{kind}" data-w="{abs(added) / biggest * 50:.1f}"></i></div>'
                f'<b class="{kind}">{added * 100:+.2f}</b></div>')
    return '<div class="why">' + "".join(row(f, a) for f, a in up + down) + '</div>'


def _market_facts(m: dict, regimes: dict, with_regime: bool = True) -> str:
    btc = m.get("btc") or {}
    states = m.get("states") or {}
    regime = regimes.get(m.get("regime"), m.get("regime") or "—")
    fg = _num((m.get("fear_greed") or {}).get("value"))
    out = "".join(fact(k, v, c) for k, v, c in (
        *((("режим", e(str(regime)), ""),) if with_regime else ()), ("BTC", money(btc.get("close")), ""),
        ("BTC за час", share(btc.get("r1")), tone(btc.get("r1"))), ("за 4 часа", share(btc.get("r4")), tone(btc.get("r4"))),
        ("за сутки", share(btc.get("r24")), tone(btc.get("r24"))), ("за неделю", share(btc.get("r168")), tone(btc.get("r168"))),
        ("BTC к 50-дневной", share(btc.get("vs_sma50")), tone(btc.get("vs_sma50"))),
        ("нервность недели", share(btc.get("nervousness"), 0) + " к обычной", "")))
    out += fact("страх и жадность", e(str(int(fg))) if fg is not None else "—", "", gauge(fg))
    out += fact("рынок", e(" · ".join(STATES.get(states.get(k), "—") for k in ("breadth_state", "alt_state", "fund_state")))
                ).replace('class="fact"', 'class="fact wide"', 1)
    return out


def _chips(items: list, cls: str = "") -> str:
    return "".join(f'<span class="chip {cls}">{e(str(x))}</span>' for x in items) or '<span class="muted small">нет</span>'


def trade(device: str, book: str, moment: int, base: str) -> str:
    j = data.journal()
    t = data.trade(book, moment, base)
    if t is None:
        return layout("Сделка не найдена", card("", '<p class="empty">Такой сделки в журнале нет.</p>'
                                                f'<p><a href="/trades">{icon("back")} к сделкам</a></p>'), "/trades", device)
    words, actions, regimes = j.get("words") or {}, j.get("actions") or {}, j.get("regimes") or {}
    d, pick = t.get("decision") or {}, t.get("pick") or {}
    m, coin_then, why = d.get("market") or {}, pick.get("coin") or {}, pick.get("why") or {}
    held = t.get("held") or {}
    end = t.get("exit_ts") or ((j.get("newest_hour") or moment) + data.HOUR_MS)
    rows = data.prices(t.get("symbol") or f"{base}USDT", moment - 36 * data.HOUR_MS, end + 24 * data.HOUR_MS)
    side_word = "лонг (спот)" if (_num(t.get("side")) or 1) > 0 else "шорт (фьючерс 1x)"
    rule = actions.get(pick.get("action"), t.get("rule_words") or t.get("rule", ""))
    hero = (f'<div class="trade-hero">{coin_link(base, True)}'
            f'<div class="big {tone(t.get("net_pct"))}">{pct(t.get("net_pct"), 2)}</div>'
            f'<span class="delta {tone(t.get("pnl_usd"))}">{signed_money(t.get("pnl_usd"))}</span>'
            f'{side_chip(t.get("side"))}{source_chip(str(t.get("source", "")))}{status_chip(t)}</div>')
    plan = '<div class="plan">' + "".join(fact(k, v, c) for k, v, c in (
        ("вход", money(t.get("entry")), ""), ("сумма", money(t.get("size")), ""),
        ("стоп", money(t.get("stop")) if t.get("stop") else "без стопа", "neg" if t.get("stop") else "muted"),
        ("цель", money(t.get("target")) if t.get("target") else "—", "pos" if t.get("target") else ""),
        ("решение", when(d.get("decided_ms") or moment), ""), ("закрыть до", e(str(t.get("due", "—"))), ""),
        ("выход", e(str(t.get("exit") or "ещё держим")), ""),
        ("лучший / худший момент", f'{share(held.get("best"))} / {share(held.get("worst"))}', ""),
        ("BTC за то же время", share(held.get("btc")), tone(held.get("btc"))))) + '</div>'
    chart = (lw(charts.candles(rows, [t], book), "tall", f"свечи {base}") if rows
             else '<p class="empty">Свечей этой монеты в базе нет.</p>')
    parts = [card("", hero + chart + '<p class="chart-note">Часовые свечи по Берлину. Стрелка — вход, кружок — выход, '
                  'пунктир — уровни сделки. Тяни и щипай график, чтобы смотреть ближе.</p>'),
             card("План и как прошла", f'<p class="small">{e(str(t.get("source", "")))}: <b>{side_word}</b> {e(base)}'
                  + (f' · {e(str(rule))}' if rule else "") + f'</p>{plan}')]
    if m:
        panic = m.get("panic") or []
        urgent = [u.get("story", "") for u in m.get("urgent") or [] if isinstance(u, dict)]
        body = (f'<div class="facts">{_market_facts(m, regimes)}'
                f'<div class="fact wide"><span>монет в панике: {len(panic)}</span><div class="pos-meta">{_chips(panic[:15])}</div></div>'
                f'<div class="fact wide"><span>срочные истории</span><div class="pos-meta">{_chips(urgent, "ev")}</div></div></div>')
        parts.append(card("Рынок в момент решения", body, "c7"))
    else:
        parts.append(card("Рынок в момент решения", '<p class="empty">Снимка рынка нет: сделка решена до 07.11.2026, '
                          'когда решения начали записывать, что видели.</p>', "c7"))
    if coin_then:
        events = [EVENTS.get(x, x) for x in coin_then.get("events", [])]
        facts = "".join(fact(k, v, c) for k, v, c in (
            ("за час", share(coin_then.get("r1")), tone(coin_then.get("r1"))),
            ("за 4 часа", share(coin_then.get("r4")), tone(coin_then.get("r4"))),
            ("за сутки", share(coin_then.get("r24")), tone(coin_then.get("r24"))),
            ("обычный ход за день", share(coin_then.get("sigma_day")), ""),
            ("оборот в день", money(coin_then.get("turnover")), "")))
        parts.append(card("Монета тогда", f'<div class="pos-meta">{_chips(events, "ev")}</div><div class="facts">{facts}</div>',
                          "c5"))
    if why.get("by"):
        chosen = pick.get("action")
        fc = sorted(((a, v) for a, v in (why.get("forecasts") or {}).items() if _num(v) is not None), key=lambda kv: -kv[1])
        forecasts = "".join(f'<div class="fc-row{" chosen" if a == chosen else ""}"><span>{e(str(actions.get(a, a)))}'
                            f'{" · выбран" if a == chosen else ""}</span><b class="{tone(v)}">{share(v, 2)}</b></div>'
                            for a, v in fc)
        parts.append(card("Почему модель так решила",
                          f'<p class="small">Прогноз выбранного варианта <b class="mono">{share(pick.get("forecast"), 2)}</b> '
                          f'(любой прогноз начинается с <span class="mono">{share(why.get("base"), 2)}</span>). '
                          f'Синее добавило к прогнозу, оранжевое убавило — в пунктах результата сделки.</p>'
                          f'{_factor_rows(why["by"], why.get("features") or {}, words)}'
                          f'<h3>Прогноз по каждому способу сделки</h3><div class="forecast">{forecasts}</div>', "c7"))
    if d.get("candidates"):
        cand = "".join(f'<tr><td>{coin_link(str(c.get("base", "")))}</td>'
                       f'<td>{e(str(actions.get(c.get("action"), c.get("action", ""))))}</td>'
                       f'<td class="n {tone(c.get("forecast"))}">{share(c.get("forecast"), 2)}</td>'
                       f'<td><span class="chip {"won" if c.get("over_bar") else ""}">'
                       f'{"выше планки" if c.get("over_bar") else "ниже планки"}</span></td></tr>'
                       for c in d["candidates"] if isinstance(c, dict))
        parts.append(card("Кого ещё рассматривала модель в этот час", f'<div class="table-wrap"><table><tr><th>монета</th>'
                          f'<th>как</th><th class="n">прогноз</th><th></th></tr>{cand}</table></div>', "c5"))
    if d.get("rule"):
        parts.append(card("Правило стратегии", f'<p>{e(str(d["rule"]))}</p>'))
    return layout(f"{base} · {BOOKS.get(book, book)}", f'<div class="grid">{"".join(parts)}</div>', "/trades", device,
                  sub=f'<a href="/trades?book={quote(book)}">{icon("back")} все сделки</a>', charts_on=True)


def skips(device: str) -> str:
    j = data.journal()
    out = _skips_of(j)
    rows = "".join(f'<tr><td class="mono small">{when(s.get("moment"))}</td><td>{coin_link(str(s.get("base", "")))}</td>'
                   f'<td>{source_chip(str(s.get("source", "")))}</td><td>{e(str(s.get("why", s.get("reason", ""))))}</td>'
                   f'<td class="small muted">{e(", ".join(map(str, s.get("holding") or [])))}</td></tr>' for s in out)
    table = (f'<div class="table-wrap"><table><tr><th>когда</th><th>монета</th><th>стратегия</th><th>почему не взята</th>'
             f'<th>что держали</th></tr>{rows}</table></div>' if out else '<p class="empty">Пропусков нет.</p>')
    return layout("Пропуски", _book_tabs(j.get("books") or {}, "skips", len(out)) + f'<div class="grid">{card("", table)}</div>',
                  "/trades", device, sub="сигналы, которые Счёт Разгон не взял, и почему")


def coins(device: str) -> str:
    items = data.coins()
    rows = "".join(
        f'<a class="row coin-row" href="/coin/{quote(str(c["base"]))}" data-name="{e(str(c["base"]).lower())}">'
        f'<span class="coin">{badge(c["base"])}<span>{e(str(c["base"]))}</span></span>'
        f'<span class="small muted">{c["trades"]} сдел. · {c["won"]} из {c["closed"]} в плюс'
        + (f' · {c["skips"]} пропуск.' if c["skips"] else "") + '</span>'
        + ('<span class="chip open">в позиции</span>' if c["open"] else '<span></span>')
        + f'<span class="res {tone(c["pnl"])}">{signed_money(c["pnl"]) if c["pnl"] else ""}</span>'
          f'<span class="when">{when(c["last"])}</span></a>' for c in items)
    body = ('<input type="search" id="coin-filter" placeholder="Найти монету…" autocomplete="off" aria-label="поиск монеты">'
            f'<div class="rows coin-list">{rows}</div>' if items else '<p class="empty">Монет пока нет.</p>')
    return layout("Монеты", f'<div class="grid">{card("", body)}</div>', "/coins", device,
                  sub="всё, что стратегии брали или пропускали; $ — итог в Счёте Разгон")


def coin(device: str, base: str) -> str:
    j = data.journal()
    mine = [(b, t) for b, t in data.all_trades(j) if t["base"] == base]
    skipped = [s for s in _skips_of(j) if s.get("base") == base]
    if not mine and not skipped:
        return layout(base, card("", '<p class="empty">По этой монете сделок нет.</p>'
                                 f'<p><a href="/coins">{icon("back")} все монеты</a></p>'), "/coins", device)
    newest = j.get("newest_hour") or int(_time.time() * 1000)
    start = min([t["moment"] for _, t in mine] + [s.get("moment") or newest for s in skipped]) - 48 * data.HOUR_MS
    start = max(start, newest - 21 * 24 * data.HOUR_MS)
    symbol = next((t.get("symbol") for _, t in mine if t.get("symbol")), f"{base}USDT")
    rows = data.prices(symbol, start, newest + data.HOUR_MS)
    uniq = {(t["moment"], t.get("side"), charts.series_of(str(t.get("source", "")))): t for _, t in mine}
    chart = (lw(charts.candles(rows, list(uniq.values())), "tall", f"свечи {base}") if rows
             else '<p class="empty">Свечей этой монеты в базе нет.</p>')
    legend = "".join(f'<span class="s-{k}"><i></i>{n}</span>' for k, n in (("nostop", "паника"), ("selector", "модель выбора")))
    closed = [t for _, t in mine if t.get("status") != "open"]
    won = sum((_num(t.get("net_pct")) or 0) > 0 for t in closed)
    pnl = sum(_num(t.get("pnl_usd")) or 0 for b, t in mine if b == "account")
    parts = [card("", f'<div class="trade-hero">{badge(base, True)}'
                  f'<span class="chip">{len(mine)} сдел. · {won} из {len(closed)} в плюс</span>'
                  f'<span class="chip">{len(skipped)} пропуск.</span>'
                  f'<span class="delta {tone(pnl)}">{signed_money(pnl)} в Счёте Разгон</span></div>'
                  f'<div class="legend">{legend}</div>{chart}'
                  '<p class="chart-note">Стрелки — входы стратегий (цвет — чья), кружки — выходы.</p>'),
             card("Сделки по монете", _trade_list(mine))]
    if skipped:
        rows_s = "".join(f'<tr><td class="mono small">{when(s.get("moment"))}</td><td>{source_chip(str(s.get("source", "")))}</td>'
                         f'<td>{e(str(s.get("why", s.get("reason", ""))))}</td></tr>' for s in skipped)
        parts.append(card("Пропуски", f'<div class="table-wrap"><table><tr><th>когда</th><th>стратегия</th>'
                          f'<th>почему</th></tr>{rows_s}</table></div>'))
    return layout(base, f'<div class="grid">{"".join(parts)}</div>', "/coins", device,
                  sub=f'<a href="/coins">{icon("back")} все монеты</a>', charts_on=True)


def c1(device: str) -> str:
    st = data.c1_state()
    decisions = data.c1_decisions(7)
    parts = [card("", f'<div class="hero-top"><div class="big">{money(st.get("value_usd"))}</div>'
                  f'<span class="delta {tone(st.get("return_pct"))}">{pct(st.get("return_pct"), 2)}</span>'
                  f'<span class="chip s-c1"><i class="sdot"></i>{"держит BTC" if st.get("btc_in") else "BTC-половина в USDT"}'
                  f'</span></div><p class="small muted">С {e(str(st.get("from", "—")))} по {e(str(st.get("through", "—")))} · '
                  f'условные $10 000. Половина в BTC, если вчера он закрылся выше 50-дневной средней; половина — 3 лучших '
                  f'монеты дня в лонг и 3 худших в шорт по прогнозу мозга на 14 дней.</p>', "hero c12")]
    opened = [p for p in st.get("open") or [] if isinstance(p, dict)]
    if opened:
        cards_ = "".join(
            f'<div class="pos-card"><div class="pos-row"><span class="coin">{badge(str(p.get("history", "?")))}'
            f'<span>{e(str(p.get("history", "?")))}</span></span><span class="pos-res {tone(p.get("ret"))}">'
            f'{share(p.get("ret"))}</span></div><div class="pos-meta">{side_chip(p.get("side"))}'
            f'<span class="chip">с {e(str(p.get("day", ""))[:10])}</span>'
            + (f'<span class="chip">{money(p.get("size"))}</span>' if p.get("size") else "") + '</div></div>'
            for p in opened)
        parts.append(card("Открыто", f'<div class="positions">{cards_}</div>'))
    for d in reversed(decisions):
        orders = "".join(
            f'<tr><td>{coin_link(str(o.get("base", "")))}</td><td>{side_chip(o.get("side"))}</td>'
            f'<td class="n mono">{(_num(o.get("place")) or 0):.2f}</td><td>{_factors_inline(o.get("why") or {})}</td></tr>'
            for o in d.get("orders", []) if isinstance(o, dict))
        parts.append(card(f"Решение за {d.get('day', '')}", f'<p class="small muted">Принято {e(str(d.get("decided_at", "")))}'
                          f' · монет в оценке {e(str(d.get("coins", "—")))} · модель {e(str(d.get("champion", "")))}</p>'
                          f'<div class="table-wrap"><table><tr><th>монета</th><th>сторона</th><th class="n">место</th>'
                          f'<th>что подняло / опустило</th></tr>{orders}</table></div>', "c6"))
    return layout("C1", f'<div class="grid">{"".join(parts)}</div>', "/c1", device,
                  sub="дневная стратегия на бумаге · место 0 — худшая монета дня, 1 — лучшая")


def _factors_inline(why: dict) -> str:
    by = why.get("by") or {}
    if not by:
        return '<span class="muted small">не записано</span>'
    ranked = sorted(((k, v) for k, v in by.items() if _num(v) is not None), key=lambda kv: kv[1])
    up = [str(f) for f, v in reversed(ranked) if v > 0][:3]
    down = [str(f) for f, v in ranked if v < 0][:3]
    return (f'<span class="small"><span class="pos">↑</span> {e(", ".join(up) or "—")}<br>'
            f'<span class="neg">↓</span> {e(", ".join(down) or "—")}</span>')


def hours_page(device: str) -> str:
    j = data.journal()
    regimes = j.get("regimes") or {}
    hrs = [h for h in data.hours(72) if isinstance(h, dict) and "hour" in h]
    parts = []
    if hrs:
        now = hrs[-1]
        mood = regimes.get(now.get("regime"), now.get("regime") or "—")
        nervous = bool(now.get("nervous"))
        parts.append(card(f"Сейчас · {when(now.get('hour'), '%H:%M')}",
                          f'<div class="regime"><i class="dot {"warn" if nervous else "ok"}"></i>'
                          f'<span class="mood">{e(str(mood))}</span><span class="chip">'
                          f'{"нервный рынок" if nervous else "спокойно"}</span></div>'
                          f'<div class="facts">{_market_facts(now, regimes, False)}'
                          + fact("монет в панике", str(len(now.get("panic") or []))) + '</div>', "c5"))
        btc_line = [(h["hour"], _num((h.get("btc") or {}).get("close"))) for h in hrs]
        parts.append(card("BTC за 3 дня", lw(charts.line_payload([p for p in btc_line if p[1]], "BTC", charts.INK),
                                              "", "цена BTC"), "c7"))
        bars = [(h["hour"], len(h.get("panic") or [])) for h in hrs]
        parts.append(card("Монет в панике по часам", lw(charts.bars_payload(bars, "в панике", "#d9732d", 4), "tiny",
                                                         "монет в панике по часам")
                          + '<p class="chart-note">Пунктир — 4 монеты: с этого паника входит, если рынок нервный.</p>'))
    body_rows = []
    for h in reversed(hrs):
        btc = h.get("btc") or {}
        taken = "; ".join(f"{k}: {', '.join(map(str, v))}" for k, v in (h.get("taken") or {}).items() if v)
        body_rows.append(f'<tr><td class="mono small">{when(h.get("hour"))}</td>'
                         f'<td>{"<span class=chip>нервный</span>" if h.get("nervous") else ""}</td>'
                         f'<td>{e(str(regimes.get(h.get("regime"), h.get("regime") or "")))}</td>'
                         f'<td class="n mono {tone(btc.get("r1"))}">{share(btc.get("r1"))}</td>'
                         f'<td class="n mono {tone(btc.get("r24"))}">{share(btc.get("r24"))}</td>'
                         f'<td class="n mono">{len(h.get("panic") or [])}</td><td class="small">{e(taken)}</td></tr>')
    parts.append(card("Час за часом", '<div class="table-wrap"><table><tr><th>час</th><th></th><th>режим</th>'
                      '<th class="n">BTC за час</th><th class="n">за сутки</th><th class="n">в панике</th><th>взято</th></tr>'
                      + "".join(body_rows) + '</table></div>'))
    return layout("Рынок", f'<div class="grid">{"".join(parts)}</div>', "/hours", device,
                  sub="что видел дозорный каждый час за последние 3 дня", charts_on=True)


def health(device: str) -> str:
    limit = data.limits()
    files = "".join(
        f'<div class="item"><span class="t"><i class="dot {"bad" if s is None else "ok" if s < limit.get(n, 3 * 3600) else "warn"}">'
        f'</i><b>{e(n)}</b></span><span class="mono small muted">{ago(s)}</span></div>' for n, s in data.freshness())
    units = "".join(
        f'<div class="item"><span class="t"><i class="dot {"ok" if s == "active" else "bad" if s == "failed" else ""}"></i>'
        f'<b class="mono small">{e(u)}</b></span><span class="small {"pos" if s == "active" else "neg" if s == "failed" else "muted"}">'
        f'{e(s)}</span></div>' for u, s in data.services())
    return layout("Система", '<div class="grid">' + card("Службы", f'<div class="list">{units}</div>', "c6")
                  + card("Данные", f'<div class="list">{files}</div>', "c6") + '</div>'
                  '<p class="foot">Графики: <a href="https://github.com/tradingview/lightweight-charts" rel="noreferrer">'
                  'TradingView Lightweight Charts™</a> (Apache 2.0) · шрифты Unbounded, Onest, JetBrains Mono (OFL)</p>',
                  "/health", device, sub="что работает на сервере и насколько свежие данные")


def me(device: str, devices: list[dict], sessions: list[dict], events: list[dict]) -> str:
    dev = "".join(
        f'<div class="item"><span class="t">{icon("shield")}<span><b>{e(d["name"])}</b><br><span class="small muted mono">'
        f'{e(d["ip"])} · passkey: {d["passkeys"]} · вход: {when(int(d["last_used"] * 1000)) if d.get("last_used") else "—"}'
        f'</span></span></span>'
        + ('<span class="chip lost">отвязано</span>' if d.get("revoked") else
           f'<form method="post" action="/me/revoke" class="inline"><input type="hidden" name="name" value="{e(d["name"])}">'
           f'<button class="danger">Отвязать</button></form>') + '</div>' for d in devices)
    ses = "".join(f'<div class="item"><span class="t">{icon("key")}<span><b>{e(s["name"])}</b> · {e(s["method"])}<br>'
                  f'<span class="small muted mono">с {when(int(s["created"] * 1000))} · активна '
                  f'{when(int(s["last_seen"] * 1000))} · до {when(int(s["expires"] * 1000))}</span></span></span></div>'
                  for s in sessions)
    log = "".join(f'<tr><td class="mono small">{when(int(x["ts"] * 1000))}</td><td>{e(x.get("device") or "")}</td>'
                  f'<td class="mono small muted">{e(x.get("ip") or "")}</td>'
                  f'<td class="{"ok" if x["ok"] else "neg"}">{e(event_ru(x["what"]))}</td></tr>' for x in events)
    return layout("Кабинет", '<div class="grid">'
                  + card("Устройства", f'<div class="list">{dev}</div><p class="small muted">Новое устройство привязывается '
                         'командой на сервере: <code>scripts/web.py enroll --device имя --ip 100.x.y.z</code> — код придёт '
                         'в Telegram.</p>', "c6")
                  + card("Сессии", f'<div class="list">{ses or "<p class=empty>Сессий нет.</p>"}</div><div class="actions">'
                         '<form method="post" action="/me/logout-all" class="inline"><button class="danger">Выйти везде</button>'
                         '</form><form method="post" action="/logout" class="inline"><button>Выйти здесь</button></form></div>',
                         "c6")
                  + card("Журнал входов", f'<div class="table-wrap log"><table>{log}</table></div>')
                  + '</div>', "/me", device, sub="устройства, сессии и каждый вход")


def plain(title: str, text: str) -> str:
    return layout(title, card("", f"<p>{e(text)}</p>"))
