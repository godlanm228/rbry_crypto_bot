"""The site over HTTP: only the owner's devices through the local proxy, the binding from code to passkey, every
page on sample data, the strict headers."""

import json
import sqlite3

import pyotp
import pytest
from aiohttp.test_utils import TestClient, TestServer
from softkey import ORIGIN, RP, SoftKey

from web import app as site
from web import data, pages
from web.auth import Auth

HOST = RP
PHONE = "100.64.0.2"
T0 = 1_794_000_000_000
H = 3_600_000


def _sample(tmp_path, monkeypatch):
    watch, paper = tmp_path / "watch", tmp_path / "paper"
    watch.mkdir()
    paper.mkdir()
    trade = {"moment": T0, "base": "TRUMP", "side": -1, "source": "модель выбора", "status": "open", "net_pct": 4.1,
             "pnl_usd": 0.82, "entry": 1.276, "size": 20.0, "rule": "S72S2T2", "due": "07.11 19:00", "symbol": "TRUMPUSDT",
             "stop": 1.45, "target": 0.91, "held": {"best": 0.12, "worst": -0.04, "btc": -0.05},
             "decision": {"decided_at": "2026-11-04 18:01:03 UTC",
                          "market": {"regime": "flat|nervous", "btc": {"close": 77000.0, "r24": -0.02},
                                     "states": {"breadth_state": "narrow"}, "fear_greed": {"value": 30.0},
                                     "panic": ["TRUMP"], "urgent": [{"story": "Binance"}]},
                          "candidates": [{"base": "TRUMP", "action": "S72S2T2", "forecast": 0.0168, "over_bar": True},
                                         {"base": "WIF", "action": "S24", "forecast": 0.004, "over_bar": False}]},
             "pick": {"action": "S72S2T2", "forecast": 0.0168, "coin": {"events": ["DUMP"], "r1": -0.06},
                      "why": {"forecasts": {"S72S2T2": 0.0168, "L24": -0.01}, "base": -0.003,
                              "features": {"mood": "nervous", "fell24": -0.1},
                              "by": {"mood": 0.006, "fell24": 0.004, "hour": -0.002}}}}
    journal = {"newest_hour": T0 + 5 * H, "words": {"mood": "настроение рынка", "fell24": "движение монеты за сутки"},
               "actions": {"S72S2T2": "шорт, 3 дня, стоп 2 хода, цель 2 стопа"}, "regimes": {"flat|nervous": "боковик · нервно"},
               "books": {"account": {"label": "Счёт Разгон", "trades": [trade], "skips": [
                   {"moment": T0, "base": "XPL", "source": "модель выбора", "reason": "cap", "holding": ["TIA (модель выбора)"],
                    "why": "места этой стратегии заняты"}]}}}
    (watch / "journal.json").write_text(json.dumps(journal, ensure_ascii=False), encoding="utf-8")
    (watch / "state.json").write_text(json.dumps({"updated": "07.11 10:14", "allocation": [
        {"label": "модель выбора", "places": 3}, {"label": "паника", "places": 5}],
        "books": {"account": {"label": "Счёт Разгон", "value": 101.98, "cash": 20.2,
                              "btc": {"usd": 40.12, "holding": True, "share": 0.5, "sma": 50},
                              "btc_words": "₿ В BTC $40.12 (правило v2)",
                              "open": [{**trade, "due": "07.11 19:00"}]},
                  "selector": {"label": "модель выбора", "value": 106.04, "open": [], "closed": 3}}}, ensure_ascii=False),
        encoding="utf-8")
    (watch / "equity.jsonl").write_text("".join(json.dumps({"hour": T0 + i * H, "books": {"account": 100 + i / 10}}) + "\n"
                                                for i in range(10)), encoding="utf-8")
    (watch / "hours.jsonl").write_text(json.dumps({"hour": T0, "nervous": True, "regime": "flat|nervous",
                                                   "btc": {"r1": -0.01}, "panic": ["TRUMP"], "taken": {"selector": ["TRUMP"]}},
                                                  ensure_ascii=False) + "\n", encoding="utf-8")
    (paper / "state.json").write_text(json.dumps({"value_usd": 10276.7, "return_pct": 2.77, "from": "2026-10-30",
                                                  "through": "2026-11-05", "btc_in": False, "closed": 0,
                                                  "open": [{"day": "2026-10-30 00:00:00", "history": "NIL", "side": -1,
                                                            "ret": 0.17}]}), encoding="utf-8")
    (paper / "decisions.jsonl").write_text(json.dumps({"day": "2026-11-05", "decided_at": "2026-11-06 01:25 UTC", "coins": 80,
                                                       "champion": "m1", "orders": [{"base": "NIL", "side": -1, "place": 0.01,
                                                       "why": {"by": {"ret_7d": -0.2, "rsi14": 0.05}}}]}) + "\n",
                                           encoding="utf-8")
    db = tmp_path / "market.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE klines (symbol TEXT, tf TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL)")
        conn.executemany("INSERT INTO klines VALUES (?, '1h', ?, ?, ?, ?, ?)",
                         [("TRUMPUSDT", T0 + i * H, 1.3 - i / 100, 1.31 - i / 100, 1.25 - i / 100, 1.28 - i / 100)
                          for i in range(-24, 30)])
    monkeypatch.setattr(data, "WATCH", watch)
    monkeypatch.setattr(data, "PAPER", paper)
    monkeypatch.setattr(data, "MARKET_DB", db)


@pytest.fixture
def made(tmp_path, monkeypatch):
    _sample(tmp_path, monkeypatch)
    told = []
    auth = Auth(RP, ORIGIN, tmp_path / "auth.db", notify=told.append)
    cfg = site.Config(host=HOST, origin=ORIGIN, rp_id=RP, secure_cookie=False)   # the test client speaks http
    return site.make_app(cfg, auth), auth, told


def _h(ip=PHONE, origin=True):
    out = {"Host": HOST, "X-Forwarded-For": ip}
    if origin:
        out["Origin"] = ORIGIN
    return out


@pytest.mark.asyncio
async def test_strangers_are_refused_before_anything(made):
    app, auth, _ = made
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/", headers={"Host": HOST})                       # no proxy header: not via Serve
        assert r.status == 403
        r = await client.get("/", headers=_h("100.64.0.99"))                    # a tailnet node not bound
        assert r.status == 403 and "не привязано" in await r.text()
        r = await client.get("/", headers={**_h(), "Host": "evil.example"})
        assert r.status == 421
        auth.add_device("iphone", PHONE)
        r = await client.get("/", headers=_h(), allow_redirects=False)
        assert r.status == 302 and r.headers["Location"] == "/enroll"
        r = await client.post("/enroll", data={"code": "x"}, headers=_h(origin=False))
        assert r.status == 403                                                  # a post must come from the site
        # the device tells why its passkey dialog failed: logged, not counted against it
        r = await client.post("/webauthn/client-error", json={"where": "register", "name": "NotAllowedError",
                                                              "message": "timed out"},
                              headers={**_h(), "User-Agent": "iPhone Safari"})
        assert r.status == 200
        what = auth.events(5)[0]
        assert "NotAllowedError" in what["what"] and "iPhone Safari" in what["what"] and not what["ok"]
        with auth._db() as db:
            assert db.execute("SELECT count(*) FROM fails").fetchone()[0] == 0
        r = await client.get("/enroll", headers=_h())
        assert r.headers["Content-Security-Policy"].startswith("default-src 'none'")
        assert r.headers["X-Frame-Options"] == "DENY" and "Strict-Transport-Security" in r.headers
        # a browser sends "Origin: null" on a form post under no-referrer: the policy must let the real origin go
        assert r.headers["Referrer-Policy"] == "same-origin" and 'content="same-origin"' in await r.text()


@pytest.mark.asyncio
async def test_binding_from_the_code_to_the_passkey_then_every_page(made):
    app, auth, told = made
    code = auth.add_device("iphone", PHONE)
    key = SoftKey()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/enroll", data={"code": code}, headers=_h(), allow_redirects=False)
        assert r.status == 302
        page = await (await client.get("/enroll", headers=_h())).text()
        assert "<svg" in page and "otpauth://" in page
        secret, _ = auth.totp_secret()
        r = await client.post("/enroll/totp", data={"otp": pyotp.TOTP(secret).now()}, headers=_h(), allow_redirects=False)
        assert r.status == 302
        assert "Создать passkey" in await (await client.get("/enroll", headers=_h())).text()
        options = await (await client.post("/webauthn/register/options", headers=_h())).text()
        r = await client.post("/webauthn/register/verify", json=key.create(options), headers=_h())
        assert r.status == 200 and any("привязано" in t for t in told)
        home = await (await client.get("/", headers=_h())).text()
        assert "в BTC" in home and "$40.12" in home and "₿ В BTC $40.12 (правило v2)" in home
        pages = {"/": "Счёт Разгон", "/trades": "TRUMP", "/skips": "места этой стратегии заняты", "/c1": "NIL",
                 "/hours": "боковик · нервно", "/health": "Службы", "/me": "iphone", "/razgon": "Лестница версий",
                 f"/trade/account/{T0}/TRUMP": "Почему модель вошла", "/coins": "TRUMP",
                 "/coin/TRUMP": "в Счёте Разгон", "/coin/NOPE": "сделок нет"}
        for path, text in pages.items():
            r = await client.get(path, headers=_h())
            body = await r.text()
            assert r.status == 200 and text in body, path
        card = await (await client.get(f"/trade/account/{T0}/TRUMP", headers=_h())).text()
        assert "настроение рынка" in card and "Кого ещё рассматривала модель" in card and "<svg" in card
        assert "боковик · нервно" in card and "Binance" in card
        r = await client.get("/trade/account/1/NOPE", headers=_h())
        assert "не найдена" in await r.text()
        r = await client.post("/me/logout-all", headers=_h(), allow_redirects=False)
        assert r.status == 302
        r = await client.get("/", headers=_h(), allow_redirects=False)
        assert r.headers["Location"] == "/login"
        options = await (await client.post("/webauthn/login/options", headers=_h())).text()
        r = await client.post("/webauthn/login/verify", json=key.get(options), headers=_h())
        assert r.status == 200
        assert (await client.get("/", headers=_h())).status == 200


def test_without_the_owner_login_the_site_does_not_start(tmp_path, monkeypatch):
    # 08.11 audit: a web.env without RBRY_WEB_TS_LOGIN used to skip the Tailscale login check silently
    env = tmp_path / "web.env"
    env.write_text("RBRY_WEB_HOST=box.example.ts.net\n", encoding="utf-8")
    monkeypatch.setenv("RBRY_WEB_ENV", str(env))
    for k in ("RBRY_WEB_TS_LOGIN", "RBRY_WEB_DEV", "RBRY_WEB_HOST"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="TS_LOGIN"):
        site.Config.load()
    env.write_text("RBRY_WEB_HOST=box.example.ts.net\nRBRY_WEB_TS_LOGIN=owner@github\n", encoding="utf-8")
    assert site.Config.load().ts_login == "owner@github"


@pytest.mark.asyncio
async def test_a_broken_data_file_is_a_plain_page_with_the_same_headers(made, monkeypatch):
    app, auth, _ = made
    code = auth.add_device("iphone", PHONE)
    key = SoftKey()
    async with TestClient(TestServer(app)) as client:
        await client.post("/enroll", data={"code": code}, headers=_h(), allow_redirects=False)
        secret, _ = auth.totp_secret()
        await client.post("/enroll/totp", data={"otp": pyotp.TOTP(secret).now()}, headers=_h(), allow_redirects=False)
        options = await (await client.post("/webauthn/register/options", headers=_h())).text()
        assert (await client.post("/webauthn/register/verify", json=key.create(options), headers=_h())).status == 200

        def broken():
            raise KeyError("label")
        monkeypatch.setattr(data, "state", broken)
        r = await client.get("/", headers=_h())
        body = await r.text()
        assert r.status == 500 and "Страница не открылась" in body and "KeyError" not in body
        assert r.headers["Content-Security-Policy"].startswith("default-src 'none'")
        assert r.headers["Cross-Origin-Opener-Policy"] == "same-origin"
        assert r.headers["Cross-Origin-Resource-Policy"] == "same-origin"


@pytest.mark.asyncio
async def test_the_site_serves_only_its_own_files_and_pages_name_them_by_content(made):
    # 08.11 design: fonts and the chart library live on the site (CSP 'self'); a page names app.css/app.js with a
    # fingerprint, so a deploy is never hidden behind a day-old cached copy
    app, auth, _ = made
    auth.add_device("iphone", PHONE)
    async with TestClient(TestServer(app)) as client:
        for path, kind in (("/static/vendor/onest-cyrillic-wght-normal.woff2", "font/woff2"),
                           ("/static/vendor/lightweight-charts.standalone.production.js", "application/javascript"),
                           ("/static/img/icon-180.png", "image/png"), ("/static/app.css", "text/css")):
            r = await client.get(path, headers=_h())
            assert r.status == 200 and r.content_type == kind, path
            assert r.headers["Cache-Control"] == "private, max-age=86400"
        for path in ("/static/../web/app.py", "/static/vendor/LICENSE-onest.txt", "/static/web.env"):
            r = await client.get(path, headers=_h(), allow_redirects=False)
            assert r.status in (302, 404) and "import" not in await r.text(), path
        page = await (await client.get("/enroll", headers=_h())).text()
        assert f"/static/app.css?v={pages.V}" in page and f"/static/app.js?v={pages.V}" in page
        r = await client.get("/enroll", headers=_h())
        assert r.headers["Cache-Control"] == "no-store" and "font-src 'self'" in r.headers["Content-Security-Policy"]


@pytest.mark.asyncio
async def test_a_trade_without_its_own_snapshot_shows_the_rebuilt_one_marked_and_checked(made, tmp_path):
    # Vlad, 08.11: trades decided before 07.11 had no market; scripts/rebuild_snapshots.py replays their hour with
    # the watcher's code, the page marks it and compares it with what the hourly log wrote at the time
    app, auth, _ = made
    watch_dir = data.WATCH
    j = json.loads((watch_dir / "journal.json").read_text(encoding="utf-8"))
    t = j["books"]["account"]["trades"][0]
    t["decision"] = {"decided_at": "old"}
    t["pick"] = {"action": "S72S2T2"}
    (watch_dir / "journal.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    (watch_dir / "rebuilt.json").write_text(json.dumps({"trades": {f"{T0}:TRUMP": {
        "hour": T0 - H, "market": {"regime": "flat|nervous", "nervous": True, "panic": ["TRUMP", "WIF"],
                                   "btc": {"close": 76000.0, "r24": -0.031}, "fear_greed": {"value": 22.0}},
        "coin": {"events": ["DUMP"], "r1": -0.05, "r24": -0.12}, "logged": {"nervous": True, "panic": ["WIF", "TRUMP"]}}}},
        ensure_ascii=False), encoding="utf-8")
    code = auth.add_device("iphone", PHONE)
    key = SoftKey()
    async with TestClient(TestServer(app)) as client:
        await client.post("/enroll", data={"code": code}, headers=_h(), allow_redirects=False)
        secret, _ = auth.totp_secret()
        await client.post("/enroll/totp", data={"otp": pyotp.TOTP(secret).now()}, headers=_h(), allow_redirects=False)
        options = await (await client.post("/webauthn/register/options", headers=_h())).text()
        assert (await client.post("/webauthn/register/verify", json=key.create(options), headers=_h())).status == 200
        card = await (await client.get(f"/trade/account/{T0}/TRUMP", headers=_h())).text()
    assert "восстановлено" in card and "совпадает" in card and "боковик · нервно" in card
    assert "−3.10%" in card and "свеча −3% на объёме" in card and "WIF" in card


@pytest.mark.asyncio
async def test_v3_settings_feed_c1_position_skips_and_cabinet(made):
    # Vlad, 09.11: settings and the cabinet top right, notifications while the site is open, C1's positions open
    # to their reasons, a skipped signal with what it would have made
    app, auth, _ = made
    j = json.loads((data.WATCH / "journal.json").read_text(encoding="utf-8"))
    xpl = {"moment": T0, "base": "XPL", "side": -1, "source": "модель выбора", "status": "time", "net_pct": 13.66,
           "pnl_usd": 2.73, "entry": 0.072, "size": 20.0}
    j["books"]["selector"] = {"label": "модель выбора", "trades": [xpl], "skips": []}
    (data.WATCH / "journal.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    code = auth.add_device("iphone", PHONE)
    key = SoftKey()
    async with TestClient(TestServer(app)) as client:
        await client.post("/enroll", data={"code": code}, headers=_h(), allow_redirects=False)
        secret, _ = auth.totp_secret()
        await client.post("/enroll/totp", data={"otp": pyotp.TOTP(secret).now()}, headers=_h(), allow_redirects=False)
        options = await (await client.post("/webauthn/register/options", headers=_h())).text()
        assert (await client.post("/webauthn/register/verify", json=key.create(options), headers=_h())).status == 200
        page = await (await client.get("/settings", headers=_h())).text()
        assert "Тема" in page and "Уведомления на сайте" in page and "Состояние сервера" in page and "Службы" in page
        r = await client.get("/health", headers=_h(), allow_redirects=False)
        assert r.status == 302 and r.headers["Location"] == "/settings#server"
        feed = await (await client.get("/api/feed?since=0", headers=_h())).json()
        kinds = {x["kind"] for x in feed["events"]}
        assert {"entry", "skip"} <= kinds and feed["now"] > 0
        later = await (await client.get(f"/api/feed?since={feed['now']}", headers=_h())).json()
        assert later["events"] == []
        skips = await (await client.get("/skips", headers=_h())).text()
        assert "+13.66%" in skips and "Что дали бы пропуски" in skips
        pos = await (await client.get("/c1/pos/2026-10-30/NIL", headers=_h())).text()
        assert "NIL" in pos and "План" in pos
        me = await (await client.get("/me", headers=_h())).text()
        assert "Журнал входов" in me and "Журнал событий" in me and "добавлен passkey" in me
        assert (await client.get("/static/theme.js", headers=_h())).status == 200
        home = await (await client.get("/", headers=_h())).text()
        assert "ticker" not in home and "всё работает" not in home and 'class="menu"' in home


def test_the_future_zone_is_a_corridor_and_a_point_not_a_path():
    from web import charts
    rows = [(T0 + i * H, 1.0, 1.01, 0.99, 1.0 + 0.01 * ((-1) ** i)) for i in range(200)]
    fut = charts.future(rows, rows[-1][0], rows[-1][0] + 10 * H, entry=1.0, side=-1, forecast=0.02)
    assert len(fut["whitespace"]) == 10 and [b["label"] for b in fut["bands"]] == ["2 из 3", "19 из 20"]
    up1, up2 = fut["bands"][0]["up"][-1]["value"], fut["bands"][1]["up"][-1]["value"]
    assert up2 > up1 > rows[-1][4] > fut["bands"][0]["down"][-1]["value"]
    assert fut["expect"]["to"]["value"] == pytest.approx(0.98) and "+2.00%" in fut["expect"]["text"]
    assert charts.future(rows, rows[-1][0], rows[-1][0] - H, 1.0, 1, None) is None      # past its deadline: nothing ahead


def test_the_hour_repeated_when_summer_time_ends_is_drawn_once():
    # 09.11: 30 days of candles crossed 26.10.2026, Berlin's 02:00-03:00 came twice, the chart library threw
    from web import charts
    t = 1792886400000                                   # 26.10.2026 00:00 UTC = 02:00 summer time in Berlin
    rows = [(t + i * H, 1.0, 1.0, 1.0, 1.0) for i in range(4)]
    times = [x["time"] for x in charts.candles(rows, [])["data"]]
    assert times == sorted(set(times)) and len(times) == 3


def test_the_timeline_says_the_hour_closed_then_the_decision_then_the_entry(made, monkeypatch):
    # Vlad, 09.11: "вход 23:00" stood before "решение 23:01"; 23:00 was the hour's close, the paper entry is the
    # 5-minute candle 10 minutes later
    from web import pages
    t = json.loads((data.WATCH / "journal.json").read_text(encoding="utf-8"))["books"]["account"]["trades"][0]
    t["decision"]["decided_ms"] = T0 + 60_000
    monkeypatch.setattr(data, "trade", lambda book, moment, base: t)
    page = pages.trade("iphone", "account", T0, "TRUMP")
    a, b, c = page.index("час закрылся"), page.index("решение</span>"), page.index("вход по цене")
    assert a < b < c and pages.when(T0 + 10 * 60_000) in page[c:c + 120]


def _ladder():
    year = lambda g, dd: {"growth_pct": g, "drawdown_pct": dd, "n": 100}
    exam = lambda yearly, dd, y25, y26, liq: {
        "words": "массовая паника: 3 самых ликвидных; спот, без плеча", "learn_yearly_pct": yearly,
        "learn_worst_drawdown_pct": dd, "trades": 2067, "up_of10": 8.0, "liquidations": liq, "busy_pct": 14.0,
        **{str(y): year(yearly, dd) for y in range(2018, 2025)}, "2025": year(y25, -12.0), "2026": year(y26, -7.0)}
    return {"current": "v1", "updated": "2026-11-13 04:39",
            "versions": [{"name": "v1", "title": "паника как на бумаге", "since": "2026-11-13",
                          "exam": exam(29.6, -37.7, 14.9, 15.7, 0)}],
            "now": {"name": "v1", "exam": exam(29.6, -37.7, 14.9, 15.7, 0)},
            "steps": [{"key": "lev2", "title": "плечо 2x на покупках паники", "why": "перевес мал, плечо удваивает <его>",
                       "on": "v1", "decided": "2026-11-13", "accepted": False, "exam": exam(21.0, -77.0, 7.0, 34.0, 14),
                       "reasons": ["2018–2024: +21% в год против +30% — не быстрее",
                                   "просадка до -77% (2020, 2021) — глубже допустимых -50%"]}]}


def test_the_ladder_of_versions_shows_the_current_one_and_every_step_with_its_reasons(made, monkeypatch):
    # Vlad, 13.11: "I have not once seen a model develop" - the ladder is where a step shows, taken or not
    from web import pages
    assert "ещё не пришла с ноутбука" in pages.razgon("iphone")              # before the laptop's first exam
    (data.WATCH / "razgon-versions.json").write_text(json.dumps(_ladder(), ensure_ascii=False), encoding="utf-8")
    page = pages.razgon("iphone")
    assert "Текущая версия" in page and ">v1<" in page and "+30% в год" in page and "$115" in page
    assert "Деньги в сделках в среднем 14% времени" in page
    assert "плечо 2x на покупках паники" in page and "не принят" in page and "глубже допустимых -50%" in page
    assert "было +30% в год" in page and ">14<" in page                      # the step beside the version it was judged on
    assert "удваивает &lt;его&gt;" in page and "<его>" not in page         # the ladder's text is escaped like any other
    assert 'href="/razgon" class="on"' in page and "Развитие" in page
    assert "шагов 1 · принято 0" in page
    assert "Путь версий" in page and "версий 1" in page and "паника как на бумаге" in page
    assert "остальное время счёт стоит в деньгах" in page
    grown = _ladder()                                                        # 14.11: the second engine became v2
    v2 = {**grown["versions"][0]["exam"], "learn_yearly_pct": 52.0, "btc_pct": 24.0}
    grown["versions"].append({"name": "v2", "title": "половина свободных денег — в BTC", "since": "2026-11-14", "exam": v2})
    grown.update(current="v2", now={"name": "v2", "exam": v2})
    (data.WATCH / "razgon-versions.json").write_text(json.dumps(grown, ensure_ascii=False), encoding="utf-8")
    page = pages.razgon("iphone")
    assert ">v2<" in page and "версий 2" in page and "половина свободных денег — в BTC" in page
    assert page.index("<b>v1</b>") < page.index("<b>v2</b>")                # the path reads from the first version on
    assert "ещё 24% счёта в среднем стоит в BTC, остальное — в долларах" in page
    for bad in ({"current": "v9", "versions": []}, {"versions": "x"}, []):   # a damaged file is an empty ladder, not a crash
        (data.WATCH / "razgon-versions.json").write_text(json.dumps(bad), encoding="utf-8")
        assert "ещё не пришла с ноутбука" in pages.razgon("iphone")
