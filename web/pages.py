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

from web import charts, data, words

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
REACTION_MS = 10 * 60_000     # the paper entry: the 5-minute candle 10 minutes after the hour (core/brain/events)
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
    "gear": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 '
            '1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1'
            'a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1'
            'a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 '
            '1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1'
            'a1.7 1.7 0 0 0-1.5 1z"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.9 1.9 0 0 0 3.4 0"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2'
           'M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "logout": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="m16 17 5-5-5-5"/><path d="M21 12H9"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
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
       ("/hours", "Рынок", "market"))
SETTINGS = (("/settings#theme", "Тема", "sun"), ("/settings#notify", "Уведомления", "bell"),
            ("/settings#server", "Состояние сервера", "system"))
CABINET = (("/me#devices", "Устройства", "shield"), ("/me#sessions", "Сессии", "key"),
           ("/me#logins", "Журнал входов", "list"), ("/me#events", "Журнал событий", "list"))


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
    if a < 0.005 and a < 1e-6:
        return "$0.00"                                  # 3.55e-15 of free cash is no cash (09.11)
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

def _problems() -> int:
    """Services down and data older than it should be: a dot on the settings icon, nothing when all is fine."""
    try:
        return len([u for u, st in data.services() if st not in ("active", "?")]) + len(data.stale())
    except Exception:  # noqa: BLE001
        return 0


def _menu(name: str, title: str, icn: str, items, extra: str = "", dot: bool = False) -> str:
    links = "".join(f'<a href="{h}">{icon(i)}{n}</a>' for h, n, i in items)
    return (f'<details class="menu"><summary aria-label="{e(title)}" title="{e(title)}">{icon(icn)}'
            + ('<i class="pdot"></i>' if dot else "") + f'</summary><div class="menu-p"><b>{e(title)}</b>{links}{extra}'
            f'</div></details>')


def layout(title: str, body: str, active: str = "", device: str = "", crumb: str = "", charts_on: bool = False,
           head: bool = True, right: str = "") -> str:
    nav = tabs = menus = ""
    if device:
        nav = "".join(f'<a href="{h}" class="{"on" if h == active else ""}">{icon(i)}<span>{n}</span></a>'
                      for h, n, i in NAV)
        tabs = ('<nav class="tabs-m">' + "".join(f'<a href="{h}" class="{"on" if h == active else ""}">{icon(i)}{n}</a>'
                                                 for h, n, i in NAV) + '</nav>')
        problems = _problems()
        theme = ('<div class="seg theme-pick"><button data-theme-set="dark">Тёмная</button>'
                 '<button data-theme-set="light">Светлая</button><button data-theme-set="system">Как в системе</button></div>')
        menus = (_menu("settings", "Настройки", "gear", SETTINGS, theme + '<span class="soon">Язык — скоро</span>', problems > 0)
                 + _menu("cabinet", f"Кабинет · {device}", "user", CABINET,
                         '<form method="post" action="/logout"><button class="link">' + icon("logout") + 'Выйти</button></form>'))
    top = (f'<header class="top"><a class="brand" href="/">{LOGO}<span><b>RBRY</b><small>paper · hud</small></span></a>'
           f'<nav class="nav">{nav}</nav><div class="top-right">{menus}</div></header>')
    lib = '<script src="/static/vendor/lightweight-charts.standalone.production.js" defer></script>' if charts_on else ""
    page_h = (f'<div class="page-h"><div>{crumb}<h1>{e(title)}</h1></div>{right}</div>' if head and title else "")
    return ('<!doctype html><html lang="ru" data-theme="dark"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<meta name="referrer" content="same-origin"><meta name="theme-color" content="#03050a">'
            '<meta name="apple-mobile-web-app-capable" content="yes">'
            '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">'
            '<meta name="apple-mobile-web-app-title" content="RBRY">'
            f'<script src="/static/theme.js?v={V}"></script>'
            '<link rel="icon" href="/static/img/icon-64.png"><link rel="apple-touch-icon" href="/static/img/icon-180.png">'
            f'<title>{e(title or "Вход")} · RBRY</title>'
            '<link rel="preload" href="/static/vendor/unbounded-cyrillic-wght-normal.woff2" as="font" type="font/woff2" crossorigin>'
            f'<link rel="stylesheet" href="/static/app.css?v={V}">{lib}<script src="/static/app.js?v={V}" defer></script></head>'
            f'<body{" data-feed=\"1\"" if device else ""}>{top}<main>{page_h}{body}</main>{tabs}'
            '<div class="toasts" aria-live="polite"></div></body></html>')


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
    c1_line = _c1_points(c1)
    if len(c1_line) > 1:
        main3["c1"] = c1_line
    compare = ""
    if len(main3) > 1:
        payload = charts.compare(main3, {**BOOKS, "c1": "C1"})
        legend = "".join(f'<span class="s-{x["key"]}"><i></i>{e(x["label"])}<b class="{tone(x["data"][-1]["value"])}">'
                         f'{pct(x["data"][-1]["value"])}</b></span>' for x in payload["series"])
        compare = (f'<h3>Доходность с одного старта · с 30.10</h3><div class="legend">{legend}</div>'
                   f'{lw(payload, "short", "доходность")}<p class="src">Счёт Разгон, паника и модель выбора — по часам, '
                   f'каждая на своих $100; C1 — по дням на условных $10 000.</p>')
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
            ("ширина рынка", e(words.breadth(states.get("breadth"))), "", True),
            ("альты против BTC", e(words.alts(states.get("alts"))), "", True),
            ("фандинг", e(words.funding(states.get("funding"))), "", True)]
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
    rows = data.prices(t.get("symbol") or f"{base}USDT", moment - 30 * 24 * HOUR_MS, end + 24 * HOUR_MS)
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
                f'<div class="tl"><span>час закрылся</span>{when(moment)}</div>'
                f'<div class="tl"><span>решение</span>{when(d.get("decided_ms") or moment)}</div>'
                f'<div class="tl"><span>вход по цене</span>{when(moment + REACTION_MS)}</div>'
                + (f'<div class="tl"><span>половина</span>{when(t.get("half_ts"))}</div>' if t.get("half_ts") else "")
                + (f'<div class="tl"><span>выход</span>{e(str(t.get("exit")))}</div>' if t.get("exit")
                   else f'<div class="tl"><span>закрыть до</span>{e(str(t.get("due", "—")))}</div>')
                + '</div>')
    chart = '<p class="empty">Свечей этой монеты в базе нет.</p>'
    if rows:
        payload = charts.candles(rows, [t], book)
        fut = None
        if t.get("status") == "open":
            fut = charts.future(rows, int(_time.time() * 1000), int(_num(t.get("due_ts")) or 0), _num(t.get("entry")),
                                int(_num(t.get("side")) or 1), _num(pick.get("forecast")))
            if fut:
                payload["future"] = fut
        last = fut["whitespace"][-1] if fut else charts.local(min(end + 24 * HOUR_MS, rows[-1][0]))
        payload["visible"] = {"from": charts.local(moment - 36 * HOUR_MS), "to": last}
        legend = ('<div class="legend"><span class="s-account"><i></i>уровни сделки</span>'
                  + ('<span class="band1"><i></i>коридор обычного хода · 2 из 3</span><span class="band2"><i></i>19 из 20</span>'
                     + ('<span class="s-selector"><i></i>ожидание модели</span>' if fut.get("expect") else "") if fut else "")
                  + '</div>')
        chart = (f'<div class="chart-top">{legend}<div class="ranges"><button data-span="trade" class="on">сделка</button>'
                 f'<button data-days="3">3д</button><button data-days="7">7д</button><button data-days="30">30д</button></div></div>'
                 + lw(payload, "tall", f"свечи {base}")
                 + ('<p class="src">Справа — до срока сделки: коридор, куда цена этой монеты обычно доходит за оставшиеся часы '
                    '(по её ходу за 30 дней), и ожидание модели на конец срока. Это не путь цены, а разброс.</p>' if fut else ""))
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
    rows, by_reason = [], {}
    for s_ in out:
        o = data.skip_outcome(s_, j)
        reason = str(s_.get("why", s_.get("reason", "")))
        r = by_reason.setdefault(reason, {"n": 0, "closed": 0, "won": 0, "sum": 0.0, "usd": 0.0, "open": 0})
        r["n"] += 1
        if o is not None:
            if o.get("status") == "open":
                r["open"] += 1
            else:
                r["closed"] += 1
                r["won"] += (_num(o.get("net_pct")) or 0) > 0
                r["sum"] += _num(o.get("net_pct")) or 0
                r["usd"] += _num(o.get("pnl_usd")) or 0
        could = ("—" if o is None else
                 f'<span class="{tone(o.get("net_pct"))}">{pct(o.get("net_pct"))}</span>'
                 + (' <span class="st open">ещё идёт</span>' if o.get("status") == "open" else ""))
        rows.append(f'<a class="ln skip-row" href="{f"/trade/{o["book"]}/{int(o["moment"])}/{quote(str(o["base"]))}" if o else "/coins"}">'
                    f'<span class="when">{when(s_.get("moment"))}</span>'
                    f'<span class="coin">{hexb(str(s_.get("base", "")), key=series(str(s_.get("source", ""))))}<span>{e(str(s_.get("base", "")))}</span></span>'
                    f'<span class="small">{e(reason)}' + (f'<br><span class="muted">держали {e(", ".join(map(str, s_.get("holding") or [])))}</span>'
                                                       if s_.get("holding") else "")
                    + f'</span><span>{strat_tag(str(s_.get("source", "")))}</span><span class="res">{could}</span></a>')
    summary = "".join(
        f'<div class="ln reason-row"><span>{e(k)}</span><span class="small muted">{v["n"]} пропуск.'
        + (f' · закрылись {v["closed"]}, в плюс {v["won"]}' if v["closed"] else "") + (f' · ещё идут {v["open"]}' if v["open"] else "")
        + f'</span><span class="res {tone(v["sum"] / v["closed"] if v["closed"] else None)}">'
        f'{pct(v["sum"] / v["closed"]) if v["closed"] else "—"}</span><span class="usd {tone(v["usd"])}">'
        f'{signed_money(v["usd"]) if v["closed"] else ""}</span></div>' for k, v in by_reason.items())
    body = (f'<div class="rows">{"".join(rows)}</div>' if out else '<p class="empty">Пропусков нет.</p>')
    return layout("Сделки", _book_tabs(j.get("books") or {}, "skips", len(out)) + '<div class="cols">'
                  + sec("Что дали бы пропуски, по причинам", f'<div class="rows">{summary}</div><p class="src">Итог — как этот же '
                        'сигнал прошёл в книге своей стратегии (те же правила и цены), в среднем на сделку; $ — на ставку той книги.</p>',
                        "w12")
                  + sec("Не взятые сигналы", body, "w12", "«что было бы» справа") + '</div>', "/trades", device)


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


def _c1_href(p: dict) -> str:
    return f'/c1/pos/{quote(str(p.get("day", "")))}/{quote(str(p.get("history", "")))}'


def _c1_row(p: dict) -> str:
    base = str((p.get("order") or {}).get("base") or p.get("history", "?"))
    ret = _num(p.get("ret"))
    status = (f'<span class="st open">открыта</span>' if p["status"] == "open"
              else f'<span class="st {"won" if (ret or 0) > 0 else "lost"}">{e(words.exit_word(str(p.get("reason", ""))))}</span>')
    return (f'<a class="ln c1-row" href="{_c1_href(p)}"><span class="coin">{hexb(base, key="c1", live=p["status"] == "open")}'
            f'<span>{e(base)}</span></span><span>{side_tag(p.get("side"))}</span><span>{status}</span>'
            f'<span class="when">{e(_day(p.get("day")))}' + (f' → {e(_day(p.get("closed")))}' if p.get("closed") else "")
            + f'</span><span class="res {tone(ret)}">{share(ret)}</span><span class="usd">{money(p.get("size"))}</span></a>')


def _day(day) -> str:
    try:
        return datetime.fromisoformat(str(day)[:10]).strftime("%d.%m")
    except ValueError:
        return str(day or "—")


def c1(device: str) -> str:
    st = data.c1_state()
    decisions = data.c1_decisions(60)
    v = _num(st.get("value_usd"))
    whole = f"{v:,.2f}".replace(",", " ").split(".") if v is not None else ["—", "—"]
    positions = data.c1_positions(st, decisions)
    opened = [p for p in positions if p["status"] == "open"]
    closed = [p for p in positions if p["status"] == "closed"]
    won = sum((_num(p.get("ret")) or 0) > 0 for p in closed)
    mean = sum(_num(p.get("ret")) or 0 for p in closed) / len(closed) if closed else None
    parts = [f'<section class="w12 hud s-c1"><div class="label">C1 · условные $10 000<span class="tag">бумага</span></div>'
             f'<div class="big">{count(v, "$" + whole[0], "c1", "usd-int")}<span class="cents">.{whole[1]}</span></div>'
             + kvs([("с начала", pct(st.get("return_pct")), tone(st.get("return_pct"))),
                    ("период", f'{_day(st.get("from"))} → {_day(st.get("through"))}', ""),
                    ("половина в BTC", "держит BTC" if st.get("btc_in") else "в USDT: BTC ниже 50-дневной", ""),
                    ("закрыто", f"{len(closed)} · в плюс {won}" + (f" · в среднем {share(mean)}" if mean is not None else ""), "")],
                   "kv")
             + '<p class="src">Каждую ночь «мозг» ставит монеты дня по прогнозу на 14 дней: 3 лучшие берёт в лонг, '
               '3 худшие в шорт, по $1 000 на 14 дней. Вторая половина держит BTC, пока он выше своей 50-дневной средней.</p>'
               '</section>']
    if opened:
        parts.append(sec("Открыто", '<div class="rows">' + "".join(_c1_row(p) for p in opened) + '</div>', "w12",
                         f"{len(opened)} позиций · нажми, чтобы увидеть почему"))
    days: dict[str, dict] = {}
    for d in decisions:
        if isinstance(d, dict) and d.get("day"):
            days.setdefault(str(d["day"])[:10], {"opened": [], "closed": [], "decision": d})["decision"] = d
    for p in positions:
        days.setdefault(p["day"], {"opened": [], "closed": [], "decision": None})["opened"].append(p)
        if p.get("closed"):
            days.setdefault(str(p["closed"])[:10], {"opened": [], "closed": [], "decision": None})["closed"].append(p)
    hist = []
    for day in sorted(days, reverse=True):
        x = days[day]
        rows = "".join(f'<a class="ln hist-row" href="{_c1_href(p)}"><span class="ev-i in">{icon("in")}</span>'
                       f'<span class="what">открыта <b>{e(str((p.get("order") or {}).get("base") or p.get("history")))}</b> '
                       f'{"лонг" if (_num(p.get("side")) or 1) > 0 else "шорт"} · место '
                       f'{(_num((p.get("order") or {}).get("place")) or 0):.2f}<br><span class="small muted">'
                       f'{_factors_inline((p.get("order") or {}).get("why") or {}, 2)}</span></span>'
                       f'<span class="res {tone(p.get("ret"))}">{share(p.get("ret"))}</span></a>' for p in x["opened"])
        rows += "".join(f'<a class="ln hist-row" href="{_c1_href(p)}"><span class="ev-i {"out-w" if (_num(p.get("ret")) or 0) > 0 else "out-l"}">'
                        f'{icon("out")}</span><span class="what">закрыта <b>{e(str((p.get("order") or {}).get("base") or p.get("history")))}</b> '
                        f'{"лонг" if (_num(p.get("side")) or 1) > 0 else "шорт"} · {e(words.exit_word(str(p.get("reason", ""))))} · '
                        f'{money(p.get("pnl"))}</span><span class="res {tone(p.get("ret"))}">{share(p.get("ret"))}</span></a>'
                        for p in x["closed"])
        d = x["decision"]
        note = (f'решение {e(str(d.get("decided_at", "")))} · монет в оценке {e(str(d.get("coins", "—")))} · половина в '
                f'{"BTC" if d.get("btc_up") else "USDT"}' if d else "")
        if rows:
            hist.append(f'<div class="hist-day"><div class="hist-h"><b>{_day(day)}</b><span class="src">{note}</span></div>'
                        f'<div class="rows">{rows}</div></div>')
    parts.append(sec("История сделок по дням", "".join(hist) or '<p class="empty">Истории пока нет.</p>', "w12"))
    return layout("C1", f'<div class="cols">{"".join(parts)}</div>', "/c1", device)


def c1_position(device: str, day: str, history: str) -> str:
    st = data.c1_state()
    pos = next((p for p in data.c1_positions(st, data.c1_decisions(60)) if p["day"] == day and str(p.get("history")) == history), None)
    crumb = _crumb("/c1", "C1")
    if pos is None:
        return layout("Позиция не найдена", '<p class="empty">Такой позиции C1 нет.</p>', "/c1", device, crumb=crumb)
    order = pos.get("order") or {}
    base = str(order.get("base") or history)
    ret = _num(pos.get("ret"))
    try:
        entry_ms = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000
    except ValueError:
        entry_ms = 0
    days_held = int((order.get("exit") or {}).get("days") or 14)
    due_ms = entry_ms + days_held * 86_400_000
    end_ms = (int(datetime.fromisoformat(str(pos["closed"])[:10]).replace(tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000
              if pos.get("closed") else int(_time.time() * 1000))
    rows = data.prices(f"{base}USDT", entry_ms - 10 * 86_400_000, end_ms + 2 * 86_400_000) if entry_ms else []
    marker = {"moment": entry_ms, "base": base, "side": pos.get("side"), "source": "C1", "status": "open"}
    chart = (lw(charts.candles(rows, [marker], "c1"), "tall", f"свечи {base}") if rows else
             '<p class="empty">Свечей этой монеты в базе нет (C1 берёт и монеты без пары к USDT на споте).</p>')
    head = (f'<div class="trade-head"><div class="trade-id">{hexb(base, True, "c1", pos["status"] == "open")}'
            f'<div><h1>{e(base)}</h1><div class="meta">{side_tag(pos.get("side"))}<span class="strat s-c1"><i class="tick-c"></i>C1</span>'
            + ('<span class="st open">открыта</span>' if pos["status"] == "open" else
               f'<span class="st {"won" if (ret or 0) > 0 else "lost"}">{e(words.exit_word(str(pos.get("reason", ""))))}</span>')
            + f'</div></div></div><div class="trade-res"><div class="big {tone(ret)}">{share(ret)}</div>'
            f'<div class="usd {tone(ret)}">{signed_money((ret or 0) * (_num(pos.get("size")) or 0))} на {money(pos.get("size"))}</div></div></div>')
    levels = '<div class="levels">' + "".join(f'<div class="s-{c}"><span>{k}</span><b>{v}</b></div>' for k, v, c in (
        ("решение за день", _day(day), "account"), ("вход", when(entry_ms), "account"),
        ("закрыть до", when(due_ms), "lvl-btc"),
        ("выход", _day(pos.get("closed")) if pos.get("closed") else "ещё держим", "lvl-target"),
        ("место у мозга", f'{(_num(order.get("place")) or 0):.2f}', "c1"),
        ("худший момент", share(pos.get("worst")), "lvl-worst"))) + '</div>'
    why = order.get("why") or {}
    feats = why.get("features") or {}
    by = why.get("by") or {}
    if by:
        ranked = sorted(((k, v) for k, v in by.items() if _num(v) is not None), key=lambda kv: kv[1])
        top = [kv for kv in reversed(ranked) if kv[1] > 0][:6] + [kv for kv in ranked if kv[1] < 0][:6]
        biggest = max([abs(v) for _, v in top] + [1e-9])
        why_html = ('<div class="why">' + "".join(
            f'<div class="why-row"><span>{e(words.c1_factor(f))}<small>тогда: {e(words.c1_value(f, feats.get(f)))}</small></span>'
            f'<div class="why-bar"><i class="{"plus" if a > 0 else "minus"}" data-w="{abs(a) / biggest * 50:.1f}"></i></div>'
            f'<b class="{"plus" if a > 0 else "minus"}">{"↑" if a > 0 else "↓"}</b></div>' for f, a in top) + '</div>'
            '<p class="src">Синее подняло монету в рейтинге дня, оранжевое опустило. Место 1 — лучшая монета дня, 0 — худшая.</p>')
    else:
        why_html = '<p class="empty">Причины C1 начали записывать 06.11; у этой позиции их нет.</p>'
    plan = (f'<p class="small">{"Лонг" if (_num(pos.get("side")) or 1) > 0 else "Шорт"} на {days_held} дней без стопа и цели: '
            f'C1 держит позицию весь срок и выходит по таймеру (или раньше, если монету снимают с торгов).</p>')
    body = (head + levels + '<div class="cols">' + f'<section class="w12 hud">{chart}</section>'
            + sec("Почему мозг выбрал эту монету", why_html, "w7") + sec("План", plan, "w5") + '</div>')
    return layout(f"{base} · C1", body, "/c1", device, charts_on=True, head=False).replace("<main>", f"<main>{crumb}", 1)


def _c1_points(st: dict) -> list[tuple[int, float]]:
    """C1's daily values as (ms at the day's close, value on a $100 scale) for the comparison."""
    out = []
    for v in st.get("values") or []:
        try:
            ms = int(datetime.fromisoformat(v["day"]).replace(tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000
            out.append((ms, float(v["value"]) / 100.0))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _factors_inline(why: dict, n: int = 3) -> str:
    by = why.get("by") or {}
    if not by:
        return '<span class="none-t">причины не записаны</span>'
    ranked = sorted(((k, v) for k, v in by.items() if _num(v) is not None), key=lambda kv: kv[1])
    up = [words.c1_factor(str(f)) for f, v in reversed(ranked) if v > 0][:n]
    down = [words.c1_factor(str(f)) for f, v in ranked if v < 0][:n]
    return (f'<span class="pos">↑</span> {e(", ".join(up) or "—")}<br>'
            f'<span class="neg">↓</span> {e(", ".join(down) or "—")}')


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


def _server() -> str:
    limit = data.limits()
    files = "".join(
        f'<div class="ln"><span class="t"><i class="dot {"bad" if s_ is None else "ok" if s_ < limit.get(n, 3 * 3600) else "warn"}">'
        f'</i>{e(n)}</span><span class="mono small muted">{ago(s_)}</span></div>' for n, s_ in data.freshness())
    units = "".join(
        f'<div class="ln"><span class="t"><i class="dot {"ok" if s_ == "active" else "bad" if s_ == "failed" else ""}"></i>'
        f'<span class="mono small">{e(u)}</span></span><span class="small {"pos" if s_ == "active" else "neg" if s_ == "failed" else "muted"}">'
        f'{e(s_)}</span></div>' for u, s_ in data.services())
    return (f'<div class="cols"><div class="w6"><h3>Службы</h3><div class="rows list">{units}</div></div>'
            f'<div class="w6"><h3>Данные</h3><div class="rows list">{files}</div></div></div>')


NOTIFY = (("entry", "Вход в сделку на Счёте Разгон"), ("exit", "Выход из сделки с итогом"),
          ("skip", "Пропущенный сигнал"), ("panic", "Массовая паника (4+ монет в нервный час)"),
          ("urgent", "Срочные новости (биржи, взломы)"), ("c1", "Решение C1 за день"))


def settings(device: str) -> str:
    theme = ('<div class="seg theme-pick big-seg"><button data-theme-set="dark">Тёмная</button>'
             '<button data-theme-set="light">Светлая</button><button data-theme-set="system">Как в системе</button></div>')
    notify = ('<div class="rows">' + "".join(
        f'<label class="ln switch-row"><span>{e(text)}</span><input type="checkbox" data-notify="{kind}" checked>'
        f'<i class="switch"></i></label>' for kind, text in NOTIFY)
        + '<label class="ln switch-row"><span>Звук</span><input type="checkbox" data-notify="sound"><i class="switch"></i></label>'
        '</div><p class="src">Пока сайт открыт, он раз в минуту проверяет новое и показывает уведомление в углу. '
        'Настройки хранятся на этом устройстве. В Telegram всё приходит как раньше.</p>'
        '<div class="actions"><button type="button" data-notify-test>Показать пример</button></div>')
    body = ('<div class="cols">' + f'<section class="sec w12" id="theme"><header class="sec-h"><i class="hx"></i><h2>Тема</h2></header>{theme}</section>'
            + f'<section class="sec w12" id="notify"><header class="sec-h"><i class="hx"></i><h2>Уведомления на сайте</h2></header>{notify}</section>'
            + f'<section class="sec w12" id="server"><header class="sec-h"><i class="hx"></i><h2>Состояние сервера</h2></header>{_server()}</section>'
            + '<section class="sec w12" id="lang"><header class="sec-h"><i class="hx"></i><h2>Язык</h2></header>'
              '<p class="small">Русский. Другие языки — позже.</p></section>'
            + '<p class="foot w12">графики: <a href="https://github.com/tradingview/lightweight-charts" rel="noreferrer">'
              'TradingView Lightweight Charts™</a> (Apache 2.0) · шрифты Unbounded, Onest, JetBrains Mono (OFL)</p></div>')
    return layout("Настройки", body, "", device)


LOGIN_EVENTS = ("login with passkey", "login with the app code", "wrong app code", "passkey not accepted",
                "passkey login without its request or of another device")


def me(device: str, devices: list[dict], sessions: list[dict], events: list[dict]) -> str:
    dev = "".join(
        f'<div class="ln"><span class="t">{icon("shield")}<span>{e(d["name"])}<small>'
        f'{e(d["ip"])} · passkey {d["passkeys"]} · вход {when(int(d["last_used"] * 1000)) if d.get("last_used") else "—"}'
        f'</small></span></span>'
        + ('<span class="st lost">отвязано</span>' if d.get("revoked") else
           f'<form method="post" action="/me/revoke" class="inline"><input type="hidden" name="name" value="{e(d["name"])}">'
           f'<button class="danger">Отвязать</button></form>') + '</div>' for d in devices)
    ses = "".join(f'<div class="ln"><span class="t">{icon("key")}<span>{e(s_["name"])} · {e(s_["method"])}<small>'
                  f'с {when(int(s_["created"] * 1000))} · активна {when(int(s_["last_seen"] * 1000))} · до '
                  f'{when(int(s_["expires"] * 1000))}</small></span></span></div>' for s_ in sessions)

    def table(rows):
        return ('<div class="table-wrap"><table>' + "".join(
            f'<tr><td class="mono small">{when(int(x["ts"] * 1000))}</td><td>{e(x.get("device") or "")}</td>'
            f'<td class="mono small muted">{e(x.get("ip") or "")}</td>'
            f'<td class="{"" if x["ok"] else "neg"}">{e(event_ru(x["what"]))}</td></tr>' for x in rows) + '</table></div>'
            if rows else '<p class="empty">Пусто.</p>')
    logins = [x for x in events if any(str(x["what"]).startswith(k) for k in LOGIN_EVENTS)]
    admin = [x for x in events if x not in logins]
    return layout("Кабинет", '<div class="cols">'
                  + f'<section class="sec w6" id="devices"><header class="sec-h"><i class="hx"></i><h2>Устройства</h2></header>'
                    f'<div class="rows list">{dev}</div><p class="src">новое: <code>scripts/web.py enroll --device имя --ip 100.x.y.z</code></p></section>'
                  + f'<section class="sec w6" id="sessions"><header class="sec-h"><i class="hx"></i><h2>Сессии</h2></header>'
                    f'<div class="rows list">{ses or "<p class=empty>Сессий нет.</p>"}</div><div class="actions">'
                    '<form method="post" action="/me/logout-all" class="inline"><button class="danger">Выйти везде</button>'
                    '</form><form method="post" action="/logout" class="inline"><button>Выйти здесь</button></form></div></section>'
                  + f'<section class="sec w6" id="logins"><header class="sec-h"><i class="hx"></i><h2>Журнал входов</h2></header>{table(logins)}</section>'
                  + f'<section class="sec w6" id="events"><header class="sec-h"><i class="hx"></i><h2>Журнал событий</h2>'
                    f'<span class="meta">привязки, отвязки, ошибки устройств</span></header>{table(admin)}</section>'
                  + '</div>', "/me", device)


def plain(title: str, text: str) -> str:
    return layout(title, f'<p class="empty">{e(text)}</p>')
