"""The site over HTTP: only the owner's devices through the local proxy, the binding from code to passkey, every
page on sample data, the strict headers."""

import json
import sqlite3

import pyotp
import pytest
from aiohttp.test_utils import TestClient, TestServer
from softkey import ORIGIN, RP, SoftKey

from web import app as site
from web import data
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
        pages = {"/": "Счёт Разгон", "/trades": "TRUMP", "/skips": "места этой стратегии заняты", "/c1": "NIL",
                 "/hours": "боковик · нервно", "/health": "Службы", "/me": "iphone",
                 f"/trade/account/{T0}/TRUMP": "Почему модель так решила"}
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
