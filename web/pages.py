"""The owner's site in HTML, as a HUD of the Redberry family: the account first, then what is held, what happened,
the market, the strategies; every trade with the market it was decided in (logged, or rebuilt and marked so) and
why. Server-rendered; the only script is static/app.js (passkeys, charts, motion) and the chart library next to it.
Every value from a data file goes through e() or a number formatter."""

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
HOUR_MS = 3_600_000
STATUS = {"open": "открыта", "stop": "стоп", "breakeven": "по цене входа", "time": "срок вышел",
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

ICONS = {
    "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v9.5h13V10"/><path d="M10 19.5v-5h4v5"/>',
    "trades": '<path d="M4 18 9.5 12l4 3.5L20 8"/><path d="M15 8h5v5"/>',
    "coins": '<path d="M12 2.8 20 7.4v9.2L12 21.2 4 16.6V7.4z"/><path d="M12 7.5v9M9.5 9.5h4a1.5 1.5 0 0 1 0 3h-3a1.5 1.5 0 0 0 0 3h4"/>',
    "c1": '<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/>',
    "market": '<path d="M2 12h4l2.5-6 4 13 3-9 1.5 2H22"/>',
    "system": '<rect x="3" y="4" width="18" height="7" rx="1"/><rect x="3" y="13" width="18" height="7" rx="1"/>'
              '<path d="M7 7.5h.01M7 16.5h.01"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/>',
    "more": '<circle cx="5" cy="12" r="1.3"/><circle cx="12" cy="12" r="1.3"/><circle cx="19" cy="12" r="1.3"/>',
    "skip": '<circle cx="12" cy="12" r="8.5"/><path d="M6 6l12 12"/>',
    "up": '<path d="M7 17 17 7"/><path d="M8 7h9v9"/>',
    "down": '<path d="M7 7l10 10"/><path d="M17 8v9H8"/>',
    "in": '<path d="M3 12h13"/><path d="m11 6 6 6-6 6"/><path d="M21 4v16"/>',
    "out": '<path d="M9 12h12"/><path d="m16 6 6 6-6 6"/><path d="M3 4v16"/>',
    "key": '<circle cx="8" cy="15" r="4"/><path d="M11 12l9-9M17 6l3 3M15 8l2 2"/>',
    "shield": '<path d="M12 3 5 6v5c0 4.5 3 8 7 10 4-2 7-5.5 7-10V6z"/><path d="m9 12 2 2 4-4"/>',
    "back": '<path d="M15 5l-7 7 7 7"/>',
}


def icon(name: str) -> str:
    return (f'<svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


LOGO = ('<svg class="logo" viewBox="0 0 64 64" aria-hidden="true"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#ff4a64"/><stop offset="1" stop-color="#b5112c"/></linearGradient></defs>'
        '<path d="M32 3 57 17.5v29L32 61 7 46.5v-29z" fill="#060912" stroke="url(#lg)" stroke-width="3"/>'
        '<path d="M32 9.5 51.5 21v22L32 54.5 12.5 43V21z" fill="none" stroke="#2c9fd0" stroke-opacity=".5"/>'
        '<g fill="url(#lg)"><circle cx="27" cy="29" r="5"/><circle cx="37" cy="29" r="5"/><circle cx="32" cy="37" r="5"/>'
        '<circle cx="23.5" cy="37.5" r="4.2"/><circle cx="40.5" cy="37.5" r="4.2"/><circle cx="27.5" cy="45" r="4"/>'
        '<circle cx="36.5" cy="45" r="4"/></g>'
        '<path d="M32 24c-2-4-6-5-9-4 2 3 5 4.5 9 4zm0 0c2-4 6-5 9-4-2 3-5 4.5-9 4z" fill="#3dae5b"/></svg>')

NAV = (("/", "Обзор", "home"), ("/trades", "Сделки", "trades"), ("/coins", "Монеты", "coins"), ("/c1", "C1", "c1"),
       ("/hours", "Рынок", "market"), ("/health", "Система", "system"), ("/me", "Кабинет", "user"))
PHONE_TABS = ("/", "/trades", "/coins", "/c1")


# ── formatting ─────────────────────────────────────────────────────────────

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


def pct(x, digits: int = 2) -> str:
    x = _num(x)
    if x is None:
        return "—"
    if round(x, digits) == 0:
        return f"{0:.{digits}f}%"                      # no "−0.00%"
    return f"{x:+.{digits}f}%".replace("-", "−")


def share(x, digits: int = 2) -> str:
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


def series(source: str) -> str:
    return charts.series_of(source)


# ── pieces ────────────────────────────────────────────────────────────────

def hexb(base: str, big: bool = False, key: str = "", live: bool = False) -> str:
    text = str(base)[:4]
    cls = ("lg " if big else "") + ("long4" if len(text) > 3 else "")
    return (f'<span class="hexwrap{" live" if live else ""}{f" s-{key}" if key else ""}">'
            f'<span class="hexb {cls}">{e(text)}</span></span>')


def coin_link(base: str, big: bool = False, key: str = "", live: bool = False) -> str:
    return f'<a class="coin" href="/coin/{quote(str(base))}">{hexb(base, big, key, live)}<span>{e(str(base))}</span></a>'


def side_tag(side) -> str:
    return (f'<span class="side long">{icon("up")}лонг</span>' if (_num(side) or 1) > 0
            else f'<span class="side short">{icon("down")}шорт</span>')


def strat_tag(source: str) -> str:
    return f'<span class="strat s-{series(source)}"><i class="tick-c"></i>{e(source)}</span>'


def status_tag(t: dict) -> str:
    st = t.get("status")
    if st == "open":
        return '<span class="st open">открыта</span>'
    won = (_num(t.get("net_pct")) or 0) > 0
    return f'<span class="st {"won" if won else "lost"}">{e(STATUS.get(st, str(st or "")))}</span>'


def sec(title: str, body: str, cls: str = "w12", meta: str = "") -> str:
    return (f'<section class="sec {cls}"><header class="sec-h"><i class="hx"></i><h2>{e(title)}</h2>'
            f'<span class="meta">{meta}</span></header>{body}</section>')


def lw(payload: dict, cls: str = "", label: str = "график") -> str:
    return (f'<div class="lw {cls}" role="img" aria-label="{e(label)}" data-chart="{charts.attr(payload)}">'
            f'<div class="readout"></div></div>')


def gauge(value: Optional[float]) -> str:
    return f'<div class="gauge"><i data-left="{value:.0f}"></i></div>' if value is not None else ""


def kvs(items, cls: str = "grid-kv") -> str:
    """(label, value html, tone class[, wide]) pairs on hairlines."""
    out = []
    for it in items:
        label, value, c = it[0], it[1], it[2] if len(it) > 2 else ""
        wide = len(it) > 3 and it[3]
        out.append(f'<div{" class=\"wide\"" if wide else ""}><span>{label}</span><b class="{c}">{value}</b></div>')
    return f'<div class="{cls}">{"".join(out)}</div>'


def figs(items) -> str:
    return '<div class="figs">' + "".join(
        f'<div class="fig"><span>{k}</span><b class="{c}">{v}</b>' + (f'<em>{x}</em>' if x else "") + '</div>'
        for k, v, c, x in items) + '</div>'


def count(value: Optional[float], text: str, key: str = "", fmt: str = "usd") -> str:
    """A number that counts up on load and flashes when it differs from the last visit (app.js)."""
    if value is None:
        return text
    return f'<span data-count="{value:.6g}" data-fmt="{fmt}"' + (f' data-key="{e(key)}"' if key else "") + f'>{text}</span>'


# ── frame ─────────────────────────────────────────────────────────────────

def _ticker() -> str:
    try:
        s = data.state()
        hrs = data.hours(2)
    except Exception:  # noqa: BLE001 - the ticker is decoration, never a reason for a page to fail
        return ""
    items = []
    acc = (s.get("books") or {}).get("account") or {}
    if _num(acc.get("value")) is not None:
        items.append(f'<span class="tk s-account"><i class="hx"></i>Счёт Разгон <b>{money(acc["value"])}</b>'
                     f'<span class="{tone(acc["value"] - 100)}">{pct(acc["value"] - 100)}</span></span>')
    for p in acc.get("open") or []:
        if isinstance(p, dict):
            items.append(f'<span class="tk s-{series(str(p.get("source", "")))}"><i class="hx"></i><b>{e(str(p.get("base", "")))}</b>'
                         f'{"лонг" if (_num(p.get("side")) or 1) > 0 else "шорт"}'
                         f'<span class="{tone(p.get("net_pct"))}">{pct(p.get("net_pct"))}</span></span>')
    if hrs:
        btc = hrs[-1].get("btc") or {}
        if _num(btc.get("close")):
            items.append(f'<span class="tk"><i class="hx"></i>BTC <b>{money(btc["close"])}</b>'
                         f'<span class="{tone(btc.get("r24"))}">{share(btc.get("r24"))} за сутки</span></span>')
        fg = _num((hrs[-1].get("fear_greed") or {}).get("value"))
        if fg is not None:
            items.append(f'<span class="tk"><i class="hx"></i>страх и жадность <b>{fg:.0f}</b></span>')
        items.append(f'<span class="tk"><i class="hx"></i>паника <b>{len(hrs[-1].get("panic") or [])}</b> монет · '
                     f'{"рынок нервный" if hrs[-1].get("nervous") else "рынок спокоен"}</span>')
    if not items:
        return ""
    run = "".join(items)
    return f'<div class="ticker" aria-hidden="true"><div class="ticker-track">{run}{run}</div></div>'


def _health_dot() -> str:
    try:
        units, stale = data.services(), data.stale()
    except Exception:  # noqa: BLE001
        return ""
    down = [u for u, s in units if s not in ("active", "?")]
    if not down and not stale:
        cls, text = "ok", "всё работает"
    else:
        cls, text = ("bad" if down else "warn"), f"проблем: {len(down) + len(stale)}"
    return f'<a class="status" href="/health"><i class="dot {cls}"></i><span>{text}</span></a>'


def layout(title: str, body: str, active: str = "", device: str = "", crumb: str = "", charts_on: bool = False,
           head: bool = True, right: str = "") -> str:
    nav = tabs = health = ticker = ""
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
        ticker = _ticker()
    top = (f'<header class="top"><a class="brand" href="/">{LOGO}<span><b>RBRY</b><small>paper · hud</small></span></a>'
           f'<nav class="nav">{nav}</nav><div class="top-right">{health}'
           + (f'<span class="who">{e(device)}</span>' if device else "") + '</div></header>')
    lib = '<script src="/static/vendor/lightweight-charts.standalone.production.js" defer></script>' if charts_on else ""
    page_h = (f'<div class="page-h"><div>{crumb}<h1>{e(title)}</h1></div>{right}</div>' if head and title else "")
    return ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<meta name="referrer" content="same-origin"><meta name="theme-color" content="#03050a">'
            '<meta name="apple-mobile-web-app-capable" content="yes">'
            '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">'
            '<meta name="apple-mobile-web-app-title" content="RBRY">'
            '<link rel="icon" href="/static/img/icon-64.png"><link rel="apple-touch-icon" href="/static/img/icon-180.png">'
            f'<title>{e(title or "Вход")} · RBRY</title>'
            '<link rel="preload" href="/static/vendor/unbounded-cyrillic-wght-normal.woff2" as="font" type="font/woff2" crossorigin>'
            f'<link rel="stylesheet" href="/static/app.css?v={V}">{lib}<script src="/static/app.js?v={V}" defer></script></head>'
            f'<body>{top}{ticker}<main>{page_h}{body}</main>{tabs}</body></html>')


def _crumb(href: str, text: str) -> str:
    return f'<a class="crumb" href="{href}">{icon("back")}{e(text)}</a>'


# ── login and binding ────────────────────────────────────────────────────

def _door(title: str, sub: str, body: str, step: int = 0) -> str:
    steps = ""
    if step:
        steps = '<div class="steps">' + "".join(
            f'<i class="{"done" if i < step else "now" if i == step else ""}"></i>' for i in (1, 2, 3)) + '</div>'
    return layout("", f'<div class="door"><section class="hud">{LOGO}<h1>{e(title)}</h1><p class="who-l">{sub}</p>'
                      f'{steps}{body}</section></div>', head=False)


def login(device: str, has_passkey: bool, message: str = "") -> str:
    note = f'<p class="msg">{e(message)}</p>' if message else ""
    if not has_passkey:
        return _door("Вход", "у этого устройства ещё нет ключа",
                     note + '<p class="small">Привяжи устройство: <a href="/enroll">ввести код привязки</a> — код приходит '
                     'в Telegram после команды на сервере.</p>')
    return _door("Вход", f'Устройство: <b>{e(device)}</b>',
                 note + '<button id="passkey-login" class="big">Войти по passkey</button><p id="msg" class="msg"></p>'
                 '<details class="more-h"><summary>Нет доступа к passkey</summary><form method="post" action="/login/totp">'
                 '<label>Код из приложения <input name="otp" inputmode="numeric" autocomplete="one-time-code" '
                 'maxlength="6" required></label><div class="actions"><button>Войти на 2 часа</button></div></form></details>')


def enroll_code(message: str = "") -> str:
    note = f'<p class="msg">{e(message)}</p>' if message else ""
    return _door("Привязка", "шаг 1 из 3 · код из Telegram",
                 note + '<form method="post" action="/enroll"><label>Код привязки <input name="code" autocomplete="off" '
                 'required placeholder="ABCD-EF01-2345"></label><div class="actions"><button class="big">Дальше</button>'
                 '</div></form><p class="muted small">15 минут, один раз, только на этом устройстве.</p>', 1)


def enroll_totp(qr_svg: str, secret: str, uri: str, first: bool, message: str = "") -> str:
    note = f'<p class="msg">{e(message)}</p>' if message else ""
    if first:
        how = (f'<p class="small">Добавь в приложение кодов («Пароли» → «Коды», Google Authenticator, 2FAS):</p>'
               f'<div class="qr">{qr_svg}</div>'
               f'<p class="small"><a href="{e(uri)}">Добавить на этом устройстве</a> · ключ: <code>{e(secret)}</code></p>')
    else:
        how = '<p class="small">Приложение кодов уже привязано: введи из него текущий код.</p>'
    return _door("Привязка", "шаг 2 из 3 · приложение кодов", note + how +
                 '<form method="post" action="/enroll/totp"><label>Код из приложения <input name="otp" '
                 'inputmode="numeric" autocomplete="one-time-code" maxlength="6" required></label>'
                 '<div class="actions"><button class="big">Проверить</button></div></form>', 2)


def enroll_passkey(device: str) -> str:
    return _door("Привязка", f"шаг 3 из 3 · passkey для «{e(device)}»",
                 '<p class="small">Телефон попросит Face ID, ноутбук — Windows Hello.</p>'
                 '<button id="passkey-register" class="big">Создать passkey</button><p id="msg" class="msg"></p>', 3)


# ── the site ─────────────────────────────────────────────────────────────

def _trade_href(book: str, t: dict) -> Optional[str]:
    try:
        return f'/trade/{quote(book)}/{int(t["moment"])}/{quote(str(t["base"]))}'
    except (KeyError, TypeError, ValueError):
        return None


def _since_midnight(points: list[tuple[int, float]]) -> Optional[tuple[float, float]]:
    """(value at the first hour of today in Berlin, now)."""
    if not points:
        return None
    day0 = int(datetime.now(BERLIN).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    before = [v for ms, v in points if ms <= day0]
    start = before[-1] if before else points[0][1]
    return start, points[-1][1]


def _position_row(p: dict, index: dict) -> str:
    full = index.get((p.get("moment"), p.get("base")), {})
    due_ts, now, moment = _num(full.get("due_ts")), _time.time() * 1000, _num(p.get("moment"))
    beam = left = ""
    if due_ts and moment:
        done = min(max((now - moment) / max(due_ts - moment, 1), 0), 1)
        hours = max((due_ts - now) / HOUR_MS, 0)
        left = (f"ещё {hours:.0f} ч" if hours >= 1 else "меньше часа") if hours else "срок вышел"
        beam = f'<span class="beam"><i data-w="{done * 100:.0f}"></i></span>'
    base, src = str(p.get("base", "?")), str(p.get("source", ""))
    return (f'<a class="ln pos-row" href="{_trade_href("account", p) or "/trades"}">'
            f'<span class="c-coin"><span class="coin">{hexb(base, key=series(src), live=True)}<span>{e(base)}</span></span></span>'
            f'<span class="c-side">{side_tag(p.get("side"))}</span><span class="c-strat">{strat_tag(src)}</span>'
            f'<span class="c-meta">{side_tag(p.get("side"))}{strat_tag(src)}<span class="when">{e(left)}</span></span>'
            f'<span class="res {tone(p.get("net_pct"))}">{count(_num(p.get("net_pct")), pct(p.get("net_pct")), "", "pct")}</span>'
            f'<span class="usd {tone(p.get("pnl_usd"))}">{signed_money(p.get("pnl_usd"))}</span>'
            f'<span class="left">{e(left)}<br>до {e(str(p.get("due", "—")))}</span>{beam}</a>')


def _feed(j: dict, n: int = 8) -> str:
    """What happened to the account, newest first: entries, exits with their result, skipped signals."""
    events = []
    books = j.get("books") or {}
    for t in (books.get("account") or {}).get("trades") or []:
        if not isinstance(t, dict) or "moment" not in t:
            continue
        href = _trade_href("account", t) or "/trades"
        events.append((t["moment"], href, "in", "in",
                       f'вход <b>{e(str(t["base"]))}</b> {"лонг" if (_num(t.get("side")) or 1) > 0 else "шорт"} · '
                       f'{e(str(t.get("source", "")))}', ""))
        if t.get("status") != "open" and t.get("exit_ts"):
            won = (_num(t.get("net_pct")) or 0) > 0
            events.append((t["exit_ts"], href, "out", "out-w" if won else "out-l",
                           f'выход <b>{e(str(t["base"]))}</b> · {e(STATUS.get(t.get("status"), str(t.get("status"))))}',
                           f'<span class="{tone(t.get("net_pct"))} mono">{pct(t.get("net_pct"))}</span>'))
    for s in (books.get("account") or {}).get("skips") or []:
        if isinstance(s, dict) and s.get("moment") and s.get("reason") != "no_price":
            events.append((s["moment"], f'/coin/{quote(str(s.get("base", "")))}', "skip", "skip",
                           f'пропуск <b>{e(str(s.get("base", "")))}</b> · {e(str(s.get("why", s.get("reason", ""))))}', ""))
    events.sort(key=lambda x: -x[0])
    if not events:
        return '<p class="empty">Событий пока нет.</p>'
    return '<div class="rows feed">' + "".join(
        f'<a class="ln" href="{href}"><span class="when">{when(ms, "%d.%m %H:%M")}</span>'
        f'<span class="ev-i {cls}">{icon(ic)}</span><span class="what">{what}</span><span>{extra}</span></a>'
        for ms, href, ic, cls, what, extra in events[:n]) + '</div>'


def _market_now() -> str:
    hrs = [h for h in data.hours(48) if isinstance(h, dict) and "hour" in h]
    if not hrs:
        return '<p class="empty">Часового журнала нет.</p>'
    now = hrs[-1]
    regimes = data.journal().get("regimes") or {}
    btc = now.get("btc") or {}
    fg = _num((now.get("fear_greed") or {}).get("value"))
    head = kvs([("режим", e(str(regimes.get(now.get("regime"), now.get("regime") or "—"))), ""),
                ("BTC", money(btc.get("close")), ""), ("за сутки", share(btc.get("r24")), tone(btc.get("r24"))),
                ("страх и жадность", (f"{fg:.0f}" if fg is not None else "—") + gauge(fg), "")])
    day0 = int(datetime.now(BERLIN).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    today = [h for h in hrs if h["hour"] >= day0]
    nervous = sum(bool(h.get("nervous")) for h in today)
    return (f'{head}<h3>Паника по часам сегодня <span class="muted small">· нервных часов {nervous} из {len(today)}</span></h3>'
            f'{charts.panic_hours(today)}')


def overview(device: str) -> str:
    s, j = data.state(), data.journal()
    books = s.get("books") or {}
    acc = books.get("account") or {}
    index = {(t["moment"], t["base"]): t for _, t in data.all_trades(j)}
    eqs = data.equity_books()
    parts = []
    if acc:
        value = _num(acc.get("value"))
        value = 100.0 if value is None else value
        whole, cents = f"{value:,.2f}".replace(",", " ").split(".")
        today = _since_midnight(eqs.get("account", []))
        day = (today[1] - today[0]) if today else None
        closed = [t for t in ((j.get("books") or {}).get("account", {}).get("trades") or []) if t.get("status") != "open"]
        won = sum((_num(t.get("net_pct")) or 0) > 0 for t in closed)
        opened = [p for p in acc.get("open") or [] if isinstance(p, dict)]
        caps = {a.get("label"): a.get("places") for a in s.get("allocation") or [] if isinstance(a, dict)}
        places = "".join(f'<span class="place full s-{series(str(p.get("source", "")))}" title="{e(str(p.get("base", "")))}">'
                         f'{e(str(p.get("base", ""))[:3])}</span>' for p in opened[:5])
        places += '<span class="place free">+</span>' * max(0, 5 - len(opened))
        eq = eqs.get("account", [])
        chart = (f'<div class="chart-top"><span class="src">баланс по часам · Берлин</span><div class="ranges">'
                 f'<button data-days="1">24ч</button><button data-days="7">7д</button><button data-days="0" class="on">всё</button>'
                 f'</div></div>{lw(charts.balance(eq), "", "баланс Счёта Разгон")}' if len(eq) > 1 else "")
        parts.append(
            f'<section class="w12 hud hero"><div class="hero-l">'
            f'<div class="label">Счёт Разгон<span class="tag">бумага</span></div>'
            f'<div class="big">{count(value, f"${whole}", "acct", "usd-int")}<span class="cents">.{cents}</span></div>'
            f'<div class="deltas"><div class="delta"><span>сегодня</span><b class="{tone(day)}">'
            f'{signed_money(day) if day is not None else "—"} · {pct(day) if day is not None else "—"}</b></div>'
            f'<div class="delta"><span>с начала</span><b class="{tone(value - 100)}">{signed_money(value - 100)} · '
            f'{pct(value - 100)}</b></div></div>'
            + kvs([("места", f'<span class="places">{places}</span>', ""), ("свободно", money(acc.get("cash")), ""),
                   ("закрыто", f'{len(closed)} · в плюс {won}', "")], "kv")
            + f'<div class="src">обновлено {e(str(s.get("updated", "?")))} · модель выбора до {e(str(caps.get("модель выбора", "—")))}'
              f' мест, паника до {e(str(caps.get("паника", "—")))}</div></div><div>{chart}</div></section>')
    opened = [p for p in acc.get("open") or [] if isinstance(p, dict)]
    parts.append(sec("В сделках", ('<div class="rows">' + "".join(_position_row(p, index) for p in opened) + '</div>')
                     if opened else '<p class="empty">Ничего не держим — счёт ждёт сигнала.</p>', "w12",
                     f'{len(opened)} из 5 · <a href="/trades">все сделки →</a>'))
    parts.append(sec("Лента", _feed(j), "w7", '<a href="/trades">сделки →</a>'))
    parts.append(sec("Рынок сейчас", _market_now(), "w5", '<a href="/hours">подробно →</a>'))
    strat = []
    for key in ("nostop", "selector"):
        b = books.get(key)
        if not b:
            continue
        v = _num(b.get("value")) or 100.0
        line = [x for _, x in eqs.get(key, [])][-96:]
        strat.append(f'<a class="strat-c s-{key}" href="/trades?book={key}"><span class="nm"><i class="tick-c"></i>'
                     f'{e(str(b.get("label") or BOOKS.get(key, key)))}</span><span class="v">{money(v)}</span>'
                     f'<span class="p {tone(v - 100)}">{pct(v - 100)}</span><span class="sub">открыто {len(b.get("open") or [])}'
                     f' · закрыто {e(str(b.get("closed", 0)))}</span>{charts.spark(line, key)}</a>')
    c1 = data.c1_state()
    if c1:
        strat.append(f'<a class="strat-c s-c1" href="/c1"><span class="nm"><i class="tick-c"></i>C1 · $10 000</span>'
                     f'<span class="v">{money(c1.get("value_usd"))}</span><span class="p {tone(c1.get("return_pct"))}">'
                     f'{pct(c1.get("return_pct"))}</span><span class="sub">с {e(str(c1.get("from", "—")))} · '
                     f'{"держит BTC" if c1.get("btc_in") else "BTC-половина в USDT"}</span></a>')
    main3 = {k: eqs[k] for k in ("account", "nostop", "selector") if len(eqs.get(k, [])) > 1}
    compare = ""
    if len(main3) > 1:
        payload = charts.compare(main3, BOOKS)
        legend = "".join(f'<span class="s-{x["key"]}"><i></i>{e(x["label"])}<b class="{tone(x["data"][-1]["value"])}">'
                         f'{pct(x["data"][-1]["value"])}</b></span>' for x in payload["series"])
        compare = f'<h3>Доходность с одного старта</h3><div class="legend">{legend}</div>{lw(payload, "short", "доходность")}'
    parts.append(sec("Стратегии", f'<div class="strats">{"".join(strat)}</div>{compare}', "w12"))
    return layout("", f'<div class="cols">{"".join(parts)}</div>', "/", device, charts_on=True, head=False)


def _trade_row(book: str, t: dict) -> str:
    href = _trade_href(book, t)
    if href is None:
        return ""
    base, src = str(t["base"]), str(t.get("source", ""))
    return (f'<a class="ln trade-row" href="{href}"><span class="when">{when(t.get("moment"))}</span>'
            f'<span class="c-coin"><span class="coin">{hexb(base, key=series(src), live=t.get("status") == "open")}'
            f'<span>{e(base)}</span></span></span>'
            f'<span class="c-side">{side_tag(t.get("side"))}</span><span class="c-strat">{strat_tag(src)}</span>'
            f'<span class="c-st">{status_tag(t)}</span>'
            f'<span class="c-meta"><span class="when">{when(t.get("moment"))}</span>{side_tag(t.get("side"))}{status_tag(t)}</span>'
            f'<span class="res {tone(t.get("net_pct"))}">{pct(t.get("net_pct"))}</span>'
            f'<span class="usd {tone(t.get("pnl_usd"))}">{signed_money(t.get("pnl_usd"))}</span></a>')


def _trade_list(rows: list[tuple[str, dict]]) -> str:
    if not rows:
        return '<p class="empty">Сделок нет.</p>'
    head = ('<div class="ln trade-row ln-h"><span>вход</span><span>монета</span><span class="c-side">сторона</span>'
            '<span class="c-strat">стратегия</span><span class="c-st">статус</span><span class="res">итог</span>'
            '<span class="usd">$</span></div>')
    return f'<div class="rows">{head}{"".join(_trade_row(b, t) for b, t in rows)}</div>'


def _summary(items: list[dict]) -> str:
    closed = [t for t in items if t.get("status") != "open"]
    won = sum((_num(t.get("net_pct")) or 0) > 0 for t in closed)
    nets = [_num(t.get("net_pct")) for t in closed if _num(t.get("net_pct")) is not None]
    usd = sum(_num(t.get("pnl_usd")) or 0 for t in items)
    mean = sum(nets) / len(nets) if nets else None
    return figs([("сделок", str(len(items)), "", f"открыто {len(items) - len(closed)}"),
                 ("в плюс", f"{won} из {len(closed)}", "", f"{won / len(closed) * 100:.0f}%" if closed else ""),
                 ("средний итог", pct(mean), tone(mean), f"лучший {pct(max(nets))}" if nets else ""),
                 ("итог в $", signed_money(usd), tone(usd), "с открытыми")])


def _book_tabs(books: dict, active: str, skips_n: int) -> str:
    tabs = "".join(f'<a href="/trades?book={quote(k)}" class="s-{k}{" on" if k == active else ""}">'
                   f'{e(BOOKS.get(k, str(b.get("label", k))))}<span class="n">{len(b.get("trades") or [])}</span></a>'
                   for k, b in books.items())
    return (f'<nav class="tabs">{tabs}<a href="/skips" class="{"on" if active == "skips" else ""}">пропуски'
            f'<span class="n">{skips_n}</span></a></nav>')


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
    return layout("Сделки", _book_tabs(books, book, len(_skips_of(j))) + '<div class="cols">'
                  + sec(BOOKS.get(book, book), _summary(items), "w12") + sec("Все сделки", _trade_list(rows), "w12")
                  + '</div>', "/trades", device)


def _factor_rows(by: dict, values: dict, words: dict, n: int = 6) -> str:
    ranked = sorted(((k, v) for k, v in by.items() if _num(v) is not None), key=lambda kv: kv[1])
    up = [kv for kv in reversed(ranked) if kv[1] > 0][:n]
    down = [kv for kv in ranked if kv[1] < 0][:n]
    biggest = max([abs(v) for _, v in up + down] + [1e-9])

    def row(f, added):
        value = values.get(f)
        shown = f"{value:.4g}" if isinstance(value, float) else str(value) if value is not None else "—"
        kind = "plus" if added > 0 else "minus"
        return (f'<div class="why-row"><span>{e(str(words.get(f, f)))}<small>тогда: {e(shown)}</small></span>'
                f'<div class="why-bar"><i class="{kind}" data-w="{abs(added) / biggest * 50:.1f}"></i></div>'
                f'<b class="{kind}">{added * 100:+.2f}</b></div>')
    return '<div class="why">' + "".join(row(f, a) for f, a in up + down) + '</div>'


def _market_kv(m: dict, regimes: dict, with_regime: bool = True) -> list:
    btc = m.get("btc") or {}
    states = m.get("states") or {}
    fg = _num((m.get("fear_greed") or {}).get("value"))
    out = []
    if with_regime:
        out.append(("режим", e(str(regimes.get(m.get("regime"), m.get("regime") or "—"))), ""))
    out += [("рынок", "нервный" if m.get("nervous") else "спокойный", "warn" if m.get("nervous") else ""),
            ("BTC", money(btc.get("close")), ""), ("за час", share(btc.get("r1")), tone(btc.get("r1"))),
            ("за 4 часа", share(btc.get("r4")), tone(btc.get("r4"))), ("за сутки", share(btc.get("r24")), tone(btc.get("r24"))),
            ("за неделю", share(btc.get("r168")), tone(btc.get("r168"))),
            ("к 50-дневной", share(btc.get("vs_sma50")), tone(btc.get("vs_sma50"))),
            ("нервность недели", share(btc.get("nervousness")), ""),
            ("страх и жадность", (f"{fg:.0f}" if fg is not None else "—") + gauge(fg), ""),
            ("ширина · альты · фандинг",
             e(" · ".join(STATES.get(states.get(k), "—") for k in ("breadth_state", "alt_state", "fund_state"))), "", True)]
    return out


def _tags(items: list, key: str = "") -> str:
    return ('<div class="tags">' + "".join(f'<span class="tag-c{f" s-{key}" if key else ""}">{e(str(x))}</span>' for x in items)
            + '</div>') if items else '<span class="none-t">нет</span>'


def trade(device: str, book: str, moment: int, base: str) -> str:
    j = data.journal()
    t = data.trade(book, moment, base)
    if t is None:
        return layout("Сделка не найдена", '<p class="empty">Такой сделки в журнале нет.</p>', "/trades", device,
                      crumb=_crumb("/trades", "все сделки"))
    words, actions, regimes = j.get("words") or {}, j.get("actions") or {}, j.get("regimes") or {}
    d, pick = t.get("decision") or {}, t.get("pick") or {}
    m, coin_then, why = d.get("market") or {}, pick.get("coin") or {}, pick.get("why") or {}
    source_note = '<span class="src">записано в момент решения</span>'
    logged = None
    if not m:
        r = data.rebuilt().get(f"{moment}:{base}") or {}
        m, coin_then, logged = r.get("market") or {}, coin_then or r.get("coin") or {}, r.get("logged")
        if m:
            source_note = ('<span class="src rebuilt">восстановлено по базе свечей кодом дозорного · тогда не '
                           'записывалось</span>')
    held = t.get("held") or {}
    end = t.get("exit_ts") or ((j.get("newest_hour") or moment) + HOUR_MS)
    rows = data.prices(t.get("symbol") or f"{base}USDT", moment - 36 * HOUR_MS, end + 24 * HOUR_MS)
    src = str(t.get("source", ""))
    rule = actions.get(pick.get("action"), t.get("rule_words") or t.get("rule", ""))
    head = (f'<div class="trade-head"><div class="trade-id">{hexb(base, True, series(src), t.get("status") == "open")}'
            f'<div><h1>{e(base)}</h1><div class="meta">{side_tag(t.get("side"))}{strat_tag(src)}{status_tag(t)}'
            + (f'<span class="muted small">{e(str(rule))}</span>' if rule else "") + '</div></div></div>'
            f'<div class="trade-res"><div class="big {tone(t.get("net_pct"))}">{count(_num(t.get("net_pct")), pct(t.get("net_pct")), "", "pct")}</div>'
            f'<div class="usd {tone(t.get("pnl_usd"))}">{signed_money(t.get("pnl_usd"))} на {money(t.get("size"))}</div></div></div>')
    levels = '<div class="levels">' + "".join(
        f'<div class="s-{c}"><span>{k}</span><b class="{tc}">{v}</b></div>' for k, v, c, tc in (
            ("вход", money(t.get("entry")), "account", ""),
            ("стоп", money(t.get("stop")) if t.get("stop") else "нет", "lvl-stop", "neg" if t.get("stop") else "muted"),
            ("цель", money(t.get("target")) if t.get("target") else "нет", "lvl-target", "pos" if t.get("target") else "muted"),
            ("лучший момент", share(held.get("best")), "lvl-best", tone(held.get("best"))),
            ("худший", share(held.get("worst")), "lvl-worst", tone(held.get("worst"))),
            ("BTC за то же время", share(held.get("btc")), "lvl-btc", tone(held.get("btc"))))) + '</div>'
    timeline = ('<div class="timeline">'
                f'<div class="tl"><span>решение</span>{when(d.get("decided_ms") or moment)}</div>'
                f'<div class="tl"><span>вход</span>{when(moment)}</div>'
                + (f'<div class="tl"><span>половина</span>{when(t.get("half_ts"))}</div>' if t.get("half_ts") else "")
                + (f'<div class="tl"><span>выход</span>{e(str(t.get("exit")))}</div>' if t.get("exit")
                   else f'<div class="tl"><span>закрыть до</span>{e(str(t.get("due", "—")))}</div>')
                + '</div>')
    chart = (lw(charts.candles(rows, [t], book), "tall", f"свечи {base}") if rows
             else '<p class="empty">Свечей этой монеты в базе нет.</p>')
    parts = [f'<section class="w12 hud">{chart}{timeline}</section>']
    if m:
        panic = m.get("panic") or []
        urgent = [u.get("story", "") for u in m.get("urgent") or [] if isinstance(u, dict)]
        check = ""
        if logged:
            was, now_ = set(map(str, logged.get("panic") or [])), set(map(str, panic))
            notes = []
            if bool(logged.get("nervous")) != bool(m.get("nervous")):
                notes.append(f'журнал считал рынок {"нервным" if logged.get("nervous") else "спокойным"}')
            shares = set(map(str, logged.get("stocks") or []))
            if was - now_ - shares:
                notes.append("в журнале ещё " + ", ".join(sorted(was - now_ - shares)))
            if now_ - was:
                notes.append("в журнале не было " + ", ".join(sorted(now_ - was)))
            if was & shares:
                notes.append("журнал считал монетами токенизированные акции " + ", ".join(sorted(was & shares))
                             + " (исправлено 05.11)")
            check = ('<p class="src">сверка с журналом того часа: '
                     + ("совпадает ✓" if not notes else e("; ".join(notes))) + '</p>')
        body = (kvs(_market_kv(m, regimes)) + f'<h3>Монеты в панике · {len(panic)}</h3>{_tags(panic[:20], "nostop")}'
                + ('<h3>Срочные истории</h3>' + _tags(urgent) if "urgent" in m else "") + check)
        parts.append(sec("Рынок в момент решения", body, "w7", source_note))
    else:
        parts.append(sec("Рынок в момент решения", '<p class="empty">Снимка нет: не записан и не восстановлен.</p>', "w7"))
    if coin_then:
        events = [EVENTS.get(x, x) for x in coin_then.get("events", [])]
        parts.append(sec(f"{base} в тот час", kvs([
            ("за час", share(coin_then.get("r1")), tone(coin_then.get("r1"))),
            ("за 4 часа", share(coin_then.get("r4")), tone(coin_then.get("r4"))),
            ("за сутки", share(coin_then.get("r24")), tone(coin_then.get("r24"))),
            ("обычный ход за день", share(coin_then.get("sigma_day")), ""),
            ("оборот в день", money(coin_then.get("turnover")), "")]) + f'<h3>События</h3>{_tags(events, series(src))}', "w5"))
    if why.get("by"):
        chosen = pick.get("action")
        fc = sorted(((a, v) for a, v in (why.get("forecasts") or {}).items() if _num(v) is not None), key=lambda kv: -kv[1])
        forecasts = "".join(f'<div class="fc{" chosen" if a == chosen else ""}"><span>{e(str(actions.get(a, a)))}</span>'
                            f'<b class="{tone(v)}">{share(v)}</b></div>' for a, v in fc)
        parts.append(sec("Почему модель вошла", f'<p class="small muted">прогноз {share(pick.get("forecast"))} · '
                         f'база {share(why.get("base"))} · синее добавило, оранжевое убавило</p>'
                         f'{_factor_rows(why["by"], why.get("features") or {}, words)}', "w7"))
        parts.append(sec("Прогноз по способам", f'<div>{forecasts}</div>', "w5"))
    elif src == "модель выбора":
        parts.append(sec("Почему модель вошла", '<p class="empty">Причины модели начали записывать 07.11.2026; модель '
                         'того дня не сохранилась, восстановить её доводы честно нельзя.</p>', "w12"))
    if d.get("candidates"):
        cand = "".join(f'<div class="ln coin-row"><span>{coin_link(str(c.get("base", "")))}</span>'
                       f'<span class="small muted">{e(str(actions.get(c.get("action"), c.get("action", ""))))}</span>'
                       f'<span class="st {"won" if c.get("over_bar") else ""}">{"выше планки" if c.get("over_bar") else "ниже планки"}</span>'
                       f'<span class="res {tone(c.get("forecast"))}">{share(c.get("forecast"))}</span><span></span></div>'
                       for c in d["candidates"] if isinstance(c, dict))
        parts.append(sec("Кого ещё рассматривала модель", f'<div class="rows">{cand}</div>', "w12"))
    if d.get("rule"):
        parts.append(sec("Правило стратегии", f'<p class="small">{e(str(d["rule"]))}</p>', "w12"))
    body = head + levels + f'<div class="cols">{"".join(parts)}</div>'
    return layout(f"{base} · {BOOKS.get(book, book)}", body, "/trades", device, charts_on=True, head=False,
                  crumb="").replace("<main>", f'<main>{_crumb("/trades?book=" + quote(book), BOOKS.get(book, book))}', 1)


def skips(device: str) -> str:
    j = data.journal()
    out = _skips_of(j)
    rows = "".join(f'<a class="ln coin-row" href="/coin/{quote(str(s.get("base", "")))}">'
                   f'<span class="coin">{hexb(str(s.get("base", "")), key=series(str(s.get("source", ""))))}'
                   f'<span>{e(str(s.get("base", "")))}</span></span>'
                   f'<span class="small">{e(str(s.get("why", s.get("reason", ""))))}'
                   + (f' <span class="muted">· держали {e(", ".join(map(str, s.get("holding") or [])))}</span>' if s.get("holding") else "")
                   + f'</span><span>{strat_tag(str(s.get("source", "")))}</span><span></span>'
                     f'<span class="when">{when(s.get("moment"))}</span></a>' for s in out)
    body = f'<div class="rows">{rows}</div>' if out else '<p class="empty">Пропусков нет.</p>'
    return layout("Сделки", _book_tabs(j.get("books") or {}, "skips", len(out))
                  + f'<div class="cols">{sec("Не взятые сигналы", body)}</div>', "/trades", device)


def coins(device: str) -> str:
    items = data.coins()
    rows = "".join(
        f'<a class="ln coin-row" href="/coin/{quote(str(c["base"]))}" data-name="{e(str(c["base"]).lower())}">'
        f'<span class="coin">{hexb(c["base"], live=bool(c["open"]))}<span>{e(str(c["base"]))}</span></span>'
        f'<span class="small muted">{c["trades"]} сдел. · {c["won"]} из {c["closed"]} в плюс'
        + (f' · {c["skips"]} пропуск.' if c["skips"] else "") + '</span>'
        + f'<span class="c-open">{"<span class=\"st open\">в позиции</span>" if c["open"] else ""}</span>'
        + f'<span class="res {tone(c["pnl"])}">{signed_money(c["pnl"]) if c["pnl"] else ""}</span>'
          f'<span class="when">{when(c["last"])}</span></a>' for c in items)
    body = ('<input type="search" id="coin-filter" placeholder="Найти монету" autocomplete="off" aria-label="поиск монеты">'
            f'<div class="rows">{rows}</div>' if items else '<p class="empty">Монет пока нет.</p>')
    return layout("Монеты", f'<div class="cols">{sec(f"Всего {len(items)}", body, "w12", "$ — итог в Счёте Разгон")}</div>',
                  "/coins", device)


def coin(device: str, base: str) -> str:
    j = data.journal()
    mine = [(b, t) for b, t in data.all_trades(j) if t["base"] == base]
    skipped = [s for s in _skips_of(j) if s.get("base") == base]
    crumb = _crumb("/coins", "монеты")
    if not mine and not skipped:
        return layout(base, '<p class="empty">По этой монете сделок нет.</p>', "/coins", device, crumb=crumb)
    newest = j.get("newest_hour") or int(_time.time() * 1000)
    start = min([t["moment"] for _, t in mine] + [s.get("moment") or newest for s in skipped]) - 48 * HOUR_MS
    start = max(start, newest - 21 * 24 * HOUR_MS)
    symbol = next((t.get("symbol") for _, t in mine if t.get("symbol")), f"{base}USDT")
    rows = data.prices(symbol, start, newest + HOUR_MS)
    uniq = {(t["moment"], t.get("side"), series(str(t.get("source", "")))): t for _, t in mine}
    chart = (lw(charts.candles(rows, list(uniq.values())), "tall", f"свечи {base}") if rows
             else '<p class="empty">Свечей этой монеты в базе нет.</p>')
    closed = [t for _, t in mine if t.get("status") != "open"]
    won = sum((_num(t.get("net_pct")) or 0) > 0 for t in closed)
    pnl = sum(_num(t.get("pnl_usd")) or 0 for b, t in mine if b == "account")
    legend = '<div class="legend"><span class="s-nostop"><i></i>паника</span><span class="s-selector"><i></i>модель выбора</span></div>'
    parts = [sec("Итог", figs([("сделок", str(len(mine)), "", f"в плюс {won} из {len(closed)}"),
                               ("пропусков", str(len(skipped)), "", ""),
                               ("в Счёте Разгон", signed_money(pnl), tone(pnl), "")]), "w12"),
             f'<section class="w12 hud">{legend}{chart}</section>',
             sec("Сделки", _trade_list(mine), "w12")]
    if skipped:
        rows_s = "".join(f'<div class="ln coin-row"><span class="when">{when(s.get("moment"))}</span>'
                         f'<span class="small">{e(str(s.get("why", s.get("reason", ""))))}</span>'
                         f'<span>{strat_tag(str(s.get("source", "")))}</span><span></span><span></span></div>' for s in skipped)
        parts.append(sec("Пропуски", f'<div class="rows">{rows_s}</div>', "w12"))
    return layout(base, f'<div class="cols">{"".join(parts)}</div>', "/coins", device, crumb=crumb, charts_on=True,
                  right=hexb(base, True))


def c1(device: str) -> str:
    st = data.c1_state()
    decisions = data.c1_decisions(7)
    v = _num(st.get("value_usd"))
    whole = f"{v:,.2f}".replace(",", " ").split(".") if v is not None else ["—", "—"]
    parts = [f'<section class="w12 hud s-c1"><div class="label">C1 · условные $10 000<span class="tag">бумага</span></div>'
             f'<div class="big">{count(v, "$" + whole[0], "c1", "usd-int")}<span class="cents">.{whole[1]}</span></div>'
             + kvs([("с начала", pct(st.get("return_pct")), tone(st.get("return_pct"))),
                    ("период", f'{e(str(st.get("from", "—")))} → {e(str(st.get("through", "—")))}', ""),
                    ("BTC-половина", "держит BTC" if st.get("btc_in") else "в USDT", ""),
                    ("решений", e(str(st.get("decisions", len(decisions)))), "")], "kv") + '</section>']
    opened = [p for p in st.get("open") or [] if isinstance(p, dict)]
    if opened:
        rows = "".join(
            f'<div class="ln coin-row"><span class="coin">{hexb(str(p.get("history", "?")), key="c1", live=True)}'
            f'<span>{e(str(p.get("history", "?")))}</span></span><span>{side_tag(p.get("side"))}</span>'
            f'<span class="when">с {e(str(p.get("day", ""))[:10])}</span><span class="res {tone(p.get("ret"))}">{share(p.get("ret"))}</span>'
            f'<span class="usd">{money(p.get("size"))}</span></div>' for p in opened)
        parts.append(sec("Открыто", f'<div class="rows">{rows}</div>', "w12", f"{len(opened)} позиций"))
    for d in reversed(decisions):
        orders = "".join(
            f'<div class="ln coin-row"><span>{coin_link(str(o.get("base", "")))}</span><span>{side_tag(o.get("side"))}</span>'
            f'<span class="small">{_factors_inline(o.get("why") or {})}</span>'
            f'<span class="res mono">{(_num(o.get("place")) or 0):.2f}</span><span></span></div>'
            for o in d.get("orders", []) if isinstance(o, dict))
        parts.append(sec(f"Решение за {d.get('day', '')}", f'<div class="rows">{orders}</div>', "w6",
                         f'{e(str(d.get("decided_at", "")))} · {e(str(d.get("coins", "—")))} монет'))
    return layout("C1", f'<div class="cols">{"".join(parts)}</div>', "/c1", device)


def _factors_inline(why: dict) -> str:
    by = why.get("by") or {}
    if not by:
        return '<span class="none-t">не записано</span>'
    ranked = sorted(((k, v) for k, v in by.items() if _num(v) is not None), key=lambda kv: kv[1])
    up = [str(f) for f, v in reversed(ranked) if v > 0][:3]
    down = [str(f) for f, v in ranked if v < 0][:3]
    return f'<span class="pos">↑</span> {e(", ".join(up) or "—")}<br><span class="neg">↓</span> {e(", ".join(down) or "—")}'


def hours_page(device: str) -> str:
    j = data.journal()
    regimes = j.get("regimes") or {}
    hrs = [h for h in data.hours(72) if isinstance(h, dict) and "hour" in h]
    parts = []
    if hrs:
        now = hrs[-1]
        mood = regimes.get(now.get("regime"), now.get("regime") or "—")
        parts.append(f'<section class="w12 hud"><div class="label">Сейчас · {when(now.get("hour"), "%H:%M")}</div>'
                     f'<div class="big">{e(str(mood))}</div>{kvs(_market_kv(now, regimes, False))}</section>')
        btc_line = [(h["hour"], _num((h.get("btc") or {}).get("close"))) for h in hrs]
        parts.append(sec("BTC · 3 дня", lw(charts.line_payload([p for p in btc_line if p[1]], "BTC", charts.INK), "short",
                                           "цена BTC"), "w7"))
        bars = [(h["hour"], len(h.get("panic") or [])) for h in hrs]
        parts.append(sec("Монет в панике", lw(charts.bars_payload(bars, "в панике", "#d9732d", 4), "short",
                                              "монет в панике по часам"), "w5", "пунктир — 4: порог входа"))
    rows = []
    for h in reversed(hrs):
        btc = h.get("btc") or {}
        taken = "; ".join(f"{k}: {', '.join(map(str, v))}" for k, v in (h.get("taken") or {}).items() if v)
        rows.append(f'<tr><td class="mono small">{when(h.get("hour"))}</td>'
                    f'<td>{"<span class=warn>нервный</span>" if h.get("nervous") else ""}</td>'
                    f'<td>{e(str(regimes.get(h.get("regime"), h.get("regime") or "")))}</td>'
                    f'<td class="n mono {tone(btc.get("r1"))}">{share(btc.get("r1"))}</td>'
                    f'<td class="n mono {tone(btc.get("r24"))}">{share(btc.get("r24"))}</td>'
                    f'<td class="n mono">{len(h.get("panic") or [])}</td><td class="small">{e(taken)}</td></tr>')
    parts.append(sec("Час за часом", '<div class="table-wrap"><table><tr><th>час</th><th></th><th>режим</th>'
                     '<th class="n">BTC 1ч</th><th class="n">24ч</th><th class="n">паника</th><th>взято</th></tr>'
                     + "".join(rows) + '</table></div>', "w12", "журнал дозорного"))
    return layout("Рынок", f'<div class="cols">{"".join(parts)}</div>', "/hours", device, charts_on=True)


def health(device: str) -> str:
    limit = data.limits()
    files = "".join(
        f'<div class="ln"><span class="t"><i class="dot {"bad" if s is None else "ok" if s < limit.get(n, 3 * 3600) else "warn"}">'
        f'</i>{e(n)}</span><span class="mono small muted">{ago(s)}</span></div>' for n, s in data.freshness())
    units = "".join(
        f'<div class="ln"><span class="t"><i class="dot {"ok" if s == "active" else "bad" if s == "failed" else ""}"></i>'
        f'<span class="mono small">{e(u)}</span></span><span class="small {"pos" if s == "active" else "neg" if s == "failed" else "muted"}">'
        f'{e(s)}</span></div>' for u, s in data.services())
    return layout("Система", '<div class="cols">' + sec("Службы", f'<div class="rows list">{units}</div>', "w6")
                  + sec("Данные", f'<div class="rows list">{files}</div>', "w6") + '</div>'
                  '<p class="foot">графики: <a href="https://github.com/tradingview/lightweight-charts" rel="noreferrer">'
                  'TradingView Lightweight Charts™</a> (Apache 2.0) · шрифты Unbounded, Onest, JetBrains Mono (OFL)</p>',
                  "/health", device)


def me(device: str, devices: list[dict], sessions: list[dict], events: list[dict]) -> str:
    dev = "".join(
        f'<div class="ln"><span class="t">{icon("shield")}<span>{e(d["name"])}<small>'
        f'{e(d["ip"])} · passkey {d["passkeys"]} · вход {when(int(d["last_used"] * 1000)) if d.get("last_used") else "—"}'
        f'</small></span></span>'
        + ('<span class="st lost">отвязано</span>' if d.get("revoked") else
           f'<form method="post" action="/me/revoke" class="inline"><input type="hidden" name="name" value="{e(d["name"])}">'
           f'<button class="danger">Отвязать</button></form>') + '</div>' for d in devices)
    ses = "".join(f'<div class="ln"><span class="t">{icon("key")}<span>{e(s["name"])} · {e(s["method"])}<small>'
                  f'с {when(int(s["created"] * 1000))} · активна {when(int(s["last_seen"] * 1000))} · до '
                  f'{when(int(s["expires"] * 1000))}</small></span></span></div>' for s in sessions)
    log = "".join(f'<tr><td class="mono small">{when(int(x["ts"] * 1000))}</td><td>{e(x.get("device") or "")}</td>'
                  f'<td class="mono small muted">{e(x.get("ip") or "")}</td>'
                  f'<td class="{"" if x["ok"] else "neg"}">{e(event_ru(x["what"]))}</td></tr>' for x in events)
    return layout("Кабинет", '<div class="cols">'
                  + sec("Устройства", f'<div class="rows list">{dev}</div><p class="src">новое: '
                        '<code>scripts/web.py enroll --device имя --ip 100.x.y.z</code></p>', "w6")
                  + sec("Сессии", f'<div class="rows list">{ses or "<p class=empty>Сессий нет.</p>"}</div><div class="actions">'
                        '<form method="post" action="/me/logout-all" class="inline"><button class="danger">Выйти везде</button>'
                        '</form><form method="post" action="/logout" class="inline"><button>Выйти здесь</button></form></div>', "w6")
                  + sec("Журнал входов", f'<div class="table-wrap"><table>{log}</table></div>', "w12")
                  + '</div>', "/me", device)


def plain(title: str, text: str) -> str:
    return layout(title, f'<p class="empty">{e(text)}</p>')
