"""The owner's site in HTML: the paper accounts, every trade with what it saw and why, C1, the hours, the system,
and the cabinet (devices, sessions, logins). Plain server-side HTML, no script but the passkey dialog."""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

from web import charts, data

BERLIN = ZoneInfo("Europe/Berlin")
NAV = (("/", "Обзор"), ("/trades", "Сделки"), ("/skips", "Пропуски"), ("/c1", "C1"), ("/hours", "Часы"),
       ("/health", "Система"), ("/me", "Кабинет"))
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
             "passkey not accepted": "passkey не принят"}
e = html.escape


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


def pct(x, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.{digits}f}%"


def share(x, digits: int = 1) -> str:
    return pct(100 * x, digits) if isinstance(x, (int, float)) else "—"


def money(x) -> str:
    if not isinstance(x, (int, float)):
        return "—"
    return f"${x:,.2f}".replace(",", " ") if abs(x) >= 1 else f"${x:.6g}"


def tone(x) -> str:
    return "pos" if isinstance(x, (int, float)) and x > 0 else "neg" if isinstance(x, (int, float)) and x < 0 else ""


def layout(title: str, body: str, active: str = "", device: str = "") -> str:
    nav = "".join(f'<a href="{href}" class="{"on" if href == active else ""}">{name}</a>' for href, name in NAV) \
        if device else ""
    who = f'<span class="who">{e(device)}</span>' if device else ""
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<meta name="referrer" content="no-referrer"><title>{e(title)} · RBRY</title>'
            f'<link rel="stylesheet" href="/static/app.css"><script src="/static/app.js" defer></script></head>'
            f'<body><header><b>RBRY</b> <span class="muted">бумага</span>{who}<nav>{nav}</nav></header>'
            f'<main><h1>{e(title)}</h1>{body}</main></body></html>')


# login and binding

def login(device: str, has_passkey: bool, message: str = "") -> str:
    note = f'<p class="warn">{e(message)}</p>' if message else ""
    if not has_passkey:
        return layout("Вход", note + '<p>У этого устройства ещё нет ключа. Привяжи его: <a href="/enroll">ввести код '
                      'привязки</a> (код приходит в Telegram после команды на сервере).</p>')
    return layout("Вход", note + f'<p>Устройство: <b>{e(device)}</b></p>'
                  '<p><button id="passkey-login" class="big">Войти по passkey</button></p><p id="msg" class="warn"></p>'
                  '<details><summary>Нет доступа к passkey</summary><form method="post" action="/login/totp">'
                  '<label>Код из приложения <input name="otp" inputmode="numeric" autocomplete="one-time-code" '
                  'maxlength="6" required></label> <button>Войти</button></form>'
                  '<p class="muted">Вход по коду — на 2 часа, о нём сразу придёт сообщение в Telegram.</p></details>')


def enroll_code(message: str = "") -> str:
    note = f'<p class="warn">{e(message)}</p>' if message else ""
    return layout("Привязка устройства", note + '<form method="post" action="/enroll"><label>Код привязки из Telegram '
                  '<input name="code" autocomplete="off" required placeholder="ABCD-EF01-2345"></label> '
                  '<button>Дальше</button></form><p class="muted">Код живёт 15 минут и работает один раз, только на '
                  'устройстве, для которого его выпустили.</p>')


def enroll_totp(qr_svg: str, secret: str, uri: str, first: bool, message: str = "") -> str:
    note = f'<p class="warn">{e(message)}</p>' if message else ""
    if first:
        how = (f'<p>Добавь в приложение для одноразовых кодов (на iPhone — «Пароли» → «Коды», или Google Authenticator, '
               f'2FAS) этот QR-код или ссылку:</p><div class="qr">{qr_svg}</div>'
               f'<p><a href="{e(uri)}">Добавить на этом устройстве</a> · ключ: <code>{e(secret)}</code></p>')
    else:
        how = '<p>Приложение для кодов уже привязано на другом устройстве: введи из него текущий код.</p>'
    return layout("Привязка: приложение для кодов", note + how +
                  '<form method="post" action="/enroll/totp"><label>Код из приложения <input name="otp" '
                  'inputmode="numeric" autocomplete="one-time-code" maxlength="6" required></label> '
                  '<button>Проверить</button></form>')


def enroll_passkey(device: str) -> str:
    return layout("Привязка: passkey", f'<p>Последний шаг: создать ключ входа для «{e(device)}». Телефон попросит '
                  'Face ID, ноутбук — Windows Hello. Сохраняй ключ на этом устройстве.</p>'
                  '<p><button id="passkey-register" class="big">Создать passkey</button></p><p id="msg" class="warn"></p>')


# the site

def overview(device: str) -> str:
    s = data.state()
    books = s.get("books", {})
    acc = books.get("account", {})
    parts = []
    if acc:
        value = acc.get("value", 100.0)
        parts.append(f'<section class="card"><h2>Счёт Разгон</h2><p class="num {tone(value - 100)}">{money(value)} '
                     f'<span class="muted">из $100 ({pct(value - 100, 2)})</span></p>'
                     f'{charts.line(data.equity("account"), "баланс Счёта Разгон", 100.0)}'
                     f'<p class="muted">Свободно {money(acc.get("cash"))} · обновлено {e(s.get("updated", "?"))} · '
                     f'места: {e(", ".join(f"{a["label"]} — до {a["places"]}" for a in s.get("allocation", [])))}</p>'
                     f'{positions_table(acc.get("open", []), "account")}</section>')
    rows = "".join(f'<tr><td><a href="/trades?book={quote(k)}">{e(b.get("label") or BOOKS.get(k, k))}</a></td>'
                   f'<td class="{tone(b.get("value", 100) - 100)}">{money(b.get("value"))}</td>'
                   f'<td>{len(b.get("open", []))}</td><td>{b.get("closed", 0)}</td></tr>'
                   for k, b in books.items() if k != "account")
    c1 = data.c1_state()
    if c1:
        rows += (f'<tr><td><a href="/c1">C1 (условные $10 000)</a></td><td class="{tone(c1.get("return_pct"))}">'
                 f'{money(c1.get("value_usd"))}</td><td>{len(c1.get("open", []))}</td><td>{c1.get("closed", 0)}</td></tr>')
    parts.append(f'<section class="card"><h2>Счета стратегий</h2><table><tr><th>счёт</th><th>баланс</th>'
                 f'<th>открыто</th><th>закрыто</th></tr>{rows}</table></section>')
    parts.append(f'<section class="card"><h2>Паника сегодня</h2>{pulse_today()}</section>')
    return layout("Обзор", "".join(parts), "/", device)


def pulse_today() -> str:
    day0 = int(datetime.now(BERLIN).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    today = [h for h in data.hours(48) if h["hour"] >= day0]
    if not today:
        return '<p class="muted">Часов за сегодня ещё нет.</p>'
    top = max(today, key=lambda h: len(h.get("panic") or []))
    n = len(top.get("panic") or [])
    nervous = sum(bool(h.get("nervous")) for h in today)
    most = (f'больше всего монет в панике за час — {n} из нужных 4 ({when(top["hour"], "%H:%M")}: '
            f'{e(", ".join((top.get("panic") or [])[:12]))})' if n else "ни одной монеты в панике")
    return f'<p>Рынок нервный {nervous} ч из {len(today)}; {most}.</p>'


def positions_table(rows: list[dict], book: str) -> str:
    if not rows:
        return '<p class="muted">Открытых сделок нет.</p>'
    body = "".join(
        f'<tr><td><a href="/trade/{quote(book)}/{p["moment"]}/{quote(p["base"])}">{e(p["base"])}</a></td>'
        f'<td>{"лонг" if p["side"] > 0 else "шорт"}</td><td>{money(p.get("size"))}</td>'
        f'<td class="{tone(p.get("net_pct"))}">{pct(p.get("net_pct"))}</td><td>{e(p.get("source", ""))}</td>'
        f'<td>{e(p.get("due", ""))}</td></tr>' for p in rows)
    return f'<table><tr><th>монета</th><th>сторона</th><th>сумма</th><th>итог</th><th>стратегия</th><th>до</th></tr>{body}</table>'


def trades(device: str, book: str) -> str:
    j = data.journal()
    books = j.get("books", {})
    tabs = " · ".join(f'<a href="/trades?book={quote(k)}" class="{"on" if k == book else ""}">'
                      f'{e(BOOKS.get(k, b.get("label", k)))}</a>' for k, b in books.items())
    items = sorted(books.get(book, {}).get("trades", []), key=lambda t: -t["moment"])
    body = "".join(
        f'<tr><td>{when(t["moment"])}</td><td><a href="/trade/{quote(book)}/{t["moment"]}/{quote(t["base"])}">'
        f'{e(t["base"])}</a></td><td>{"лонг" if t.get("side", 1) > 0 else "шорт"}</td><td>{e(t.get("source", ""))}</td>'
        f'<td>{e(STATUS.get(t.get("status"), t.get("status", "")))}</td>'
        f'<td class="{tone(t.get("net_pct"))}">{pct(t.get("net_pct"))}</td>'
        f'<td class="{tone(t.get("pnl_usd"))}">{money(t.get("pnl_usd"))}</td>'
        f'<td>{share((t.get("held") or {}).get("best"))} / {share((t.get("held") or {}).get("worst"))}</td></tr>'
        for t in items)
    table = (f'<table><tr><th>вход</th><th>монета</th><th>сторона</th><th>стратегия</th><th>статус</th><th>итог</th>'
             f'<th>$</th><th>лучший / худший момент</th></tr>{body}</table>' if items else '<p class="muted">Сделок нет.</p>')
    return layout("Сделки", f'<p class="tabs">{tabs}</p>{table}', "/trades", device)


def _factor_rows(by: dict, values: dict, words: dict, n: int = 5) -> str:
    ranked = sorted(by.items(), key=lambda kv: kv[1])
    up = [kv for kv in reversed(ranked) if kv[1] > 0][:n]
    down = [kv for kv in ranked if kv[1] < 0][:n]

    def row(f, added):
        value = values.get(f)
        shown = f"{value:.4g}" if isinstance(value, float) else str(value) if value is not None else "—"
        return (f'<tr><td>{e(words.get(f, f))}</td><td>{e(shown)}</td>'
                f'<td class="{tone(added)}">{added * 100:+.2f}</td></tr>')
    return ('<table><tr><th>что</th><th>значение тогда</th><th>вклад, пункты</th></tr>'
            + "".join(row(f, a) for f, a in up + down) + '</table>')


def trade(device: str, book: str, moment: int, base: str) -> str:
    j = data.journal()
    t = data.trade(book, moment, base)
    if t is None:
        return layout("Сделка не найдена", '<p>Такой сделки в журнале нет.</p>', "/trades", device)
    words, actions = j.get("words", {}), j.get("actions", {})
    d, pick = t.get("decision") or {}, t.get("pick") or {}
    m, coin, why = d.get("market") or {}, pick.get("coin") or {}, pick.get("why") or {}
    held = t.get("held") or {}
    levels = {k: t[k] for k in ("stop", "half", "target") if t.get(k)}
    end = t.get("exit_ts") or (j.get("newest_hour", moment) + data.HOUR_MS)
    rows = data.prices(t.get("symbol") or f"{base}USDT", moment - 24 * data.HOUR_MS, end + 24 * data.HOUR_MS)
    plan = (f'<p>{e(t.get("source", ""))}: <b>{"лонг (спот)" if t.get("side", 1) > 0 else "шорт (фьючерс 1x)"}</b> '
            f'{e(base)} на {money(t.get("size"))} по {money(t.get("entry"))}, '
            f'{e(actions.get(pick.get("action"), t.get("rule_words") or t.get("rule", "")))}.</p>'
            f'<p>Решение {e(d.get("decided_at", when(moment)))}; закрыть до {e(t.get("due", "—"))}. '
            f'Статус: <b>{e(STATUS.get(t.get("status"), t.get("status", "")))}</b>'
            f'{", выход " + e(t["exit"]) if t.get("exit") else ""}; итог <b class="{tone(t.get("net_pct"))}">'
            f'{pct(t.get("net_pct"))} ({money(t.get("pnl_usd"))})</b>.</p>'
            f'<p>Пока держали: лучший момент {share(held.get("best"))}, худший {share(held.get("worst"))}, '
            f'BTC за то же время {share(held.get("btc"))}.</p>')
    parts = [f'<section class="card">{plan}{charts.trade(rows, moment, t.get("entry") or 0, t.get("exit_ts"), levels, t.get("side", 1))}</section>']
    if m:
        btc = m.get("btc") or {}
        states = m.get("states") or {}
        regime = j.get("regimes", {}).get(m.get("regime"), m.get("regime") or "—")
        urgent = ", ".join(u.get("story", "") for u in m.get("urgent") or []) or "нет"
        parts.append(f'<section class="card"><h2>Рынок в момент решения</h2><table>'
                     f'<tr><td>режим</td><td>{e(regime)}</td></tr>'
                     f'<tr><td>BTC</td><td>{money(btc.get("close"))}: за час {share(btc.get("r1"))}, за 4 ч '
                     f'{share(btc.get("r4"))}, за сутки {share(btc.get("r24"))}, за неделю {share(btc.get("r168"))}</td></tr>'
                     f'<tr><td>BTC против 50-дневной</td><td>{share(btc.get("vs_sma50"))}</td></tr>'
                     f'<tr><td>нервность недели</td><td>{share(btc.get("nervousness"), 0)} к обычной</td></tr>'
                     f'<tr><td>рынок</td><td>{e(", ".join(STATES.get(states.get(k), "—") for k in ("breadth_state", "alt_state", "fund_state")))}</td></tr>'
                     f'<tr><td>страх и жадность</td><td>{e(str((m.get("fear_greed") or {}).get("value", "—")))}</td></tr>'
                     f'<tr><td>монет в панике</td><td>{len(m.get("panic") or [])}: {e(", ".join((m.get("panic") or [])[:15]))}</td></tr>'
                     f'<tr><td>срочные истории</td><td>{e(urgent)}</td></tr></table></section>')
    else:
        parts.append('<section class="card"><p class="muted">Снимка рынка нет: сделка решена до 07.11.2026, когда '
                     'решения начали записывать, что видели.</p></section>')
    if coin:
        events = ", ".join(EVENTS.get(x, x) for x in coin.get("events", []))
        parts.append(f'<section class="card"><h2>Монета тогда</h2><p>События: {e(events or "—")}. Движение: за час '
                     f'{share(coin.get("r1"))}, за 4 ч {share(coin.get("r4"))}, за сутки {share(coin.get("r24"))}; '
                     f'обычный ход за день {share(coin.get("sigma_day"))}; оборот {money(coin.get("turnover"))} в день.</p></section>')
    if why.get("by"):
        forecasts = "".join(f'<tr><td>{e(actions.get(a, a))}</td><td class="{tone(v)}">{share(v, 2)}</td></tr>'
                            for a, v in sorted((why.get("forecasts") or {}).items(), key=lambda kv: -kv[1]))
        parts.append(f'<section class="card"><h2>Почему модель так решила</h2><p>Прогноз выбранного варианта '
                     f'{share(pick.get("forecast"), 2)} (база у всех прогнозов {share(why.get("base"), 2)}); ниже — что '
                     f'сильнее всего добавило и убавило (в пунктах результата сделки).</p>'
                     f'{_factor_rows(why["by"], why.get("features") or {}, words)}'
                     f'<h3>Прогноз по каждому способу сделки</h3><table>{forecasts}</table></section>')
    if d.get("candidates"):
        cand = "".join(f'<tr><td>{e(c["base"])}</td><td>{e(actions.get(c["action"], c["action"]))}</td>'
                       f'<td class="{tone(c["forecast"])}">{share(c["forecast"], 2)}</td>'
                       f'<td>{"выше планки" if c.get("over_bar") else "ниже планки"}</td></tr>' for c in d["candidates"])
        parts.append(f'<section class="card"><h2>Кого ещё рассматривала модель в этот час</h2><table>{cand}</table></section>')
    if d.get("rule"):
        parts.append(f'<section class="card"><h2>Правило стратегии</h2><p>{e(d["rule"])}</p></section>')
    return layout(f"{base} · {BOOKS.get(book, book)}", "".join(parts), "/trades", device)


def skips(device: str) -> str:
    j = data.journal()
    out = sorted(j.get("books", {}).get("account", {}).get("skips", []), key=lambda s: -s["moment"])
    out = [s for s in out if s["reason"] != "no_price"]
    body = "".join(f'<tr><td>{when(s["moment"])}</td><td>{e(s["base"])}</td><td>{e(s["source"])}</td>'
                   f'<td>{e(s.get("why", s["reason"]))}</td><td>{e(", ".join(s.get("holding") or []))}</td></tr>' for s in out)
    table = (f'<table><tr><th>когда</th><th>монета</th><th>стратегия</th><th>почему не взята</th><th>что держали</th></tr>'
             f'{body}</table>' if out else '<p class="muted">Пропусков нет.</p>')
    return layout("Пропуски Счёта Разгон", table, "/skips", device)


def c1(device: str) -> str:
    st = data.c1_state()
    decisions = data.c1_decisions(7)
    parts = [f'<section class="card"><p class="num {tone(st.get("return_pct"))}">{money(st.get("value_usd"))} '
             f'<span class="muted">{pct(st.get("return_pct"), 2)} с {e(st.get("from", "—"))} по {e(st.get("through", "—"))}</span></p>'
             f'<p class="muted">Половина в BTC, если вчера он закрылся выше 50-дневной средней (сейчас: '
             f'{"держит BTC" if st.get("btc_in") else "в USDT"}); половина — 3 лучших монеты дня в лонг, 3 худших в '
             f'шорт по прогнозу мозга на 14 дней.</p></section>']
    if st.get("open"):
        body = "".join(f'<tr><td>{e(p["day"][:10])}</td><td>{e(p["history"])}</td><td>{"лонг" if p["side"] > 0 else "шорт"}</td>'
                       f'<td class="{tone(p.get("ret"))}">{share(p.get("ret"))}</td></tr>' for p in st["open"])
        parts.append(f'<section class="card"><h2>Открыто</h2><table><tr><th>с</th><th>монета</th><th>сторона</th>'
                     f'<th>итог</th></tr>{body}</table></section>')
    for d in reversed(decisions):
        orders = "".join(
            f'<tr><td>{e(o["base"])}</td><td>{"лонг" if o["side"] > 0 else "шорт"}</td><td>{o.get("place", 0):.2f}</td>'
            f'<td>{_factors_inline(o.get("why") or {})}</td></tr>' for o in d.get("orders", []))
        parts.append(f'<section class="card"><h2>Решение за {e(d.get("day", ""))}</h2><p class="muted">Принято '
                     f'{e(d.get("decided_at", ""))}, монет в оценке {d.get("coins", "—")}, модель {e(str(d.get("champion", "")))}.'
                     f'</p><table><tr><th>монета</th><th>сторона</th><th>место (0 — худшая, 1 — лучшая)</th>'
                     f'<th>что подняло / опустило в рейтинге</th></tr>{orders}</table></section>')
    return layout("C1", "".join(parts), "/c1", device)


def _factors_inline(why: dict) -> str:
    by = why.get("by") or {}
    if not by:
        return '<span class="muted">не записано</span>'
    ranked = sorted(by.items(), key=lambda kv: kv[1])
    up = [f for f, v in reversed(ranked) if v > 0][:3]
    down = [f for f, v in ranked if v < 0][:3]
    return f'<span class="pos">↑ {e(", ".join(up))}</span><br><span class="neg">↓ {e(", ".join(down))}</span>'


def hours_page(device: str) -> str:
    j = data.journal()
    regimes = j.get("regimes", {})
    rows = "".join(
        f'<tr><td>{when(h["hour"])}</td><td>{"да" if h.get("nervous") else ""}</td>'
        f'<td>{e(regimes.get(h.get("regime"), h.get("regime") or ""))}</td>'
        f'<td>{share((h.get("btc") or {}).get("r1"))}</td><td>{share((h.get("btc") or {}).get("r24"))}</td>'
        f'<td>{len(h.get("panic") or [])}</td>'
        f'<td>{e("; ".join(f"{k}: {", ".join(v)}" for k, v in (h.get("taken") or {}).items() if v))}</td></tr>'
        for h in reversed(data.hours(72)))
    return layout("Часы рынка", f'<table><tr><th>час</th><th>нервный</th><th>режим</th><th>BTC за час</th>'
                  f'<th>BTC за сутки</th><th>в панике</th><th>взято</th></tr>{rows}</table>', "/hours", device)


def health(device: str) -> str:
    files = "".join(f'<tr><td>{e(n)}</td><td>{ago(s)}</td></tr>' for n, s in data.freshness())
    units = "".join(f'<tr><td>{e(u)}</td><td class="{"pos" if s == "active" else "neg" if s == "failed" else ""}">'
                    f'{e(s)}</td></tr>' for u, s in data.services())
    return layout("Система", f'<section class="card"><h2>Данные</h2><table>{files}</table></section>'
                  f'<section class="card"><h2>Службы</h2><table>{units}</table></section>', "/health", device)


def me(device: str, devices: list[dict], sessions: list[dict], events: list[dict]) -> str:
    dev = "".join(f'<tr><td>{e(d["name"])}</td><td>{e(d["ip"])}</td><td>{d["passkeys"]}</td>'
                  f'<td>{when(int(d["last_used"] * 1000)) if d.get("last_used") else "—"}</td>'
                  f'<td>{"отвязано" if d.get("revoked") else ""}</td><td>'
                  + ("" if d.get("revoked") else f'<form method="post" action="/me/revoke" class="inline">'
                     f'<input type="hidden" name="name" value="{e(d["name"])}"><button class="danger">Отвязать</button></form>')
                  + '</td></tr>' for d in devices)
    ses = "".join(f'<tr><td>{e(s["name"])}</td><td>{e(s["method"])}</td><td>{when(int(s["created"] * 1000))}</td>'
                  f'<td>{when(int(s["last_seen"] * 1000))}</td><td>{when(int(s["expires"] * 1000))}</td></tr>' for s in sessions)
    log = "".join(f'<tr><td>{when(int(x["ts"] * 1000))}</td><td>{e(x.get("device") or "")}</td><td>{e(x.get("ip") or "")}</td>'
                  f'<td class="{"" if x["ok"] else "neg"}">{e(event_ru(x["what"]))}</td></tr>' for x in events)
    return layout("Кабинет", f'<section class="card"><h2>Устройства</h2><table><tr><th>имя</th><th>адрес в tailnet</th>'
                  f'<th>passkey</th><th>последний вход</th><th></th><th></th></tr>{dev}</table>'
                  f'<p class="muted">Новое устройство привязывается командой на сервере: '
                  f'<code>scripts/web.py enroll --device имя --ip 100.x.y.z</code> — код придёт в Telegram.</p></section>'
                  f'<section class="card"><h2>Сессии</h2><table><tr><th>устройство</th><th>как</th><th>начата</th>'
                  f'<th>активна</th><th>до</th></tr>{ses}</table><form method="post" action="/me/logout-all">'
                  f'<button class="danger">Выйти везде</button></form> <form method="post" action="/logout" class="inline">'
                  f'<button>Выйти здесь</button></form></section>'
                  f'<section class="card"><h2>Журнал входов</h2><table>{log}</table></section>', "/me", device)


def plain(title: str, text: str) -> str:
    return layout(title, f"<p>{e(text)}</p>")
