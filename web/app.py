"""The owner's site (aiohttp, on 127.0.0.1 behind Tailscale Serve): every page behind the device check and a
session (web/auth.py), every response with a strict policy (no script or style from anywhere but the site).

How a request is known: Tailscale Serve proxies to 127.0.0.1 and sets X-Forwarded-For to the client's tailnet
address (it overwrites any copy the client sent). A request not from the local proxy, or without that header,
is refused; RBRY_WEB_DEV=1 (a test on the laptop's localhost) takes the local address itself as the device.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aiohttp import web

from web import data, pages
from web.auth import FULL, SESSION_HOURS, Auth, Locked

log = logging.getLogger("rbry.web")
ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
COOKIE = "rbry_s"
# what the site serves from static/: name -> content type (nothing else, whatever the path says)
ASSETS = {"app.css": "text/css", "app.js": "application/javascript", "theme.js": "application/javascript",
          "vendor/lightweight-charts.standalone.production.js": "application/javascript",
          "img/icon-64.png": "image/png", "img/icon-180.png": "image/png"}
ASSETS.update({f"vendor/{f}-{s}-wght-normal.woff2": "font/woff2" for f in ("unbounded", "onest", "jetbrains-mono")
               for s in ("cyrillic", "latin")})
PUBLIC = ("/login", "/login/totp", "/enroll", "/enroll/totp", "/webauthn/register/options", "/webauthn/register/verify",
          "/webauthn/login/options", "/webauthn/login/verify", "/webauthn/client-error",
          *(f"/static/{name}" for name in ASSETS))
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
       "connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")


@dataclass(frozen=True)
class Config:
    host: str                 # the Host a request must name (the node's ts.net name); "" in the dev mode
    origin: str               # https://<host>, what passkeys are bound to
    rp_id: str
    dev: bool = False
    ts_login: str = ""        # when set, Serve's Tailscale-User-Login must be this login
    secure_cookie: bool = True   # the session cookie only over https (Serve's); off in the dev mode and the tests

    @staticmethod
    def load() -> "Config":
        env_file = Path(os.environ.get("RBRY_WEB_ENV", ROOT / "runtime" / "web" / "web.env"))
        values = {}
        if env_file.exists():
            from dotenv import dotenv_values
            values = dotenv_values(env_file)
        get = lambda k, d="": os.environ.get(k, values.get(k, d)) or d    # noqa: E731
        dev = get("RBRY_WEB_DEV") == "1"
        host = get("RBRY_WEB_HOST", "localhost:8000" if dev else "")
        if not host:
            raise SystemExit("RBRY_WEB_HOST is not set (runtime/web/web.env)")
        if not dev and not get("RBRY_WEB_TS_LOGIN"):           # fail closed: no owner login, no site
            raise SystemExit("RBRY_WEB_TS_LOGIN is not set (runtime/web/web.env)")
        return Config(host=host, origin=get("RBRY_WEB_ORIGIN", f"{'http' if dev else 'https'}://{host}"),
                      rp_id=get("RBRY_WEB_RP_ID", host.split(":")[0]), dev=dev, ts_login=get("RBRY_WEB_TS_LOGIN"),
                      secure_cookie=not dev)


CFG = web.AppKey("cfg", Config)
AUTH = web.AppKey("auth", Auth)
IP = web.RequestKey("ip", str)
DEVICE = web.RequestKey("device", object)
SESSION = web.RequestKey("session", object)


def client_ip(request: web.Request, cfg: Config) -> Optional[str]:
    peer = request.remote or ""
    if peer not in ("127.0.0.1", "::1"):
        return None                                   # only the local proxy (Serve) may talk to the site
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return "127.0.0.1" if cfg.dev else None


def _page(text: str, status: int = 200) -> web.Response:
    return web.Response(text=text, status=status, content_type="text/html", charset="utf-8")


def _refuse(text: str, status: int = 403) -> web.Response:
    return _page(pages.plain("Нет доступа", text), status)


def _harden(response: web.StreamResponse, cfg: Config) -> None:
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    # same-origin, not no-referrer: with no-referrer a browser sends "Origin: null" on a form post, and the
    # gate's Origin check refused every binding code typed on the iPhone and the laptop (07.11)
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"      # no other page holds a handle to ours
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"    # nor embeds what we serve
    # the pages: never kept; the site's own files: kept a day (they change only with a deploy)
    static = isinstance(response, web.Response) and response.content_type in set(ASSETS.values())
    response.headers["Cache-Control"] = "private, max-age=86400" if static else "no-store"
    if not cfg.dev:
        response.headers["Strict-Transport-Security"] = "max-age=31536000"


@web.middleware
async def secure(request: web.Request, handler):
    try:
        response = await handler(request)
    except web.HTTPException as exc:          # a redirect or an error: the same headers, raised on
        _harden(exc, request.app[CFG])
        raise
    except Exception:                         # noqa: BLE001 - a broken data file: a plain page, the same headers
        log.exception("page failed: %s", request.path)
        response = _page(pages.plain("Ошибка", "Страница не открылась: данные не прочитались. Подробности — в журнале "
                                     "службы rbry-web."), 500)
    _harden(response, request.app[CFG])
    return response


@web.middleware
async def gate(request: web.Request, handler):
    cfg, auth = request.app[CFG], request.app[AUTH]
    if request.host != cfg.host:
        return _refuse("Неверный адрес сайта.", 421)
    if cfg.ts_login and request.headers.get("Tailscale-User-Login", "") != cfg.ts_login:
        return _refuse("Чужой аккаунт Tailscale.")
    ip = client_ip(request, cfg)
    device = auth.device_by_ip(ip) if ip else None
    if device is None:
        return _refuse("Это устройство не привязано к сайту.")
    if request.method == "POST" and request.headers.get("Origin") != cfg.origin:
        return _refuse("Запрос не с этого сайта.")
    request[IP], request[DEVICE] = ip, device
    session = auth.session(request.cookies.get(COOKIE), ip)
    request[SESSION] = session
    if request.path not in PUBLIC and (session is None or session.method not in FULL):
        if request.method == "GET":
            raise web.HTTPFound("/login" if device.passkeys else "/enroll")
        return web.json_response({"error": "нужен вход"}, status=401)
    return await handler(request)


def _set_session(response: web.StreamResponse, token: str, cfg: Config, hours: float = SESSION_HOURS) -> None:
    response.set_cookie(COOKIE, token, max_age=int(hours * 3600), httponly=True, secure=cfg.secure_cookie, samesite="Strict",
                        path="/")


async def _form(request: web.Request) -> dict:
    return dict(await request.post())


async def _json(request: web.Request) -> dict:
    try:
        return await request.json()
    except (ValueError, json.JSONDecodeError):
        return {}


def _locked() -> web.Response:
    return _page(pages.plain("Подожди", "Слишком много неудачных попыток: адрес заблокирован на 15 минут."), 429)


# pages

async def overview(request):
    return _page(pages.overview(request[DEVICE].name))


async def trades(request):
    return _page(pages.trades(request[DEVICE].name, request.query.get("book", "account")))


async def trade(request):
    try:
        moment = int(request.match_info["moment"])
    except ValueError:
        raise web.HTTPNotFound()
    return _page(pages.trade(request[DEVICE].name, request.match_info["book"], moment, request.match_info["base"]))


async def coins(request):
    return _page(pages.coins(request[DEVICE].name))


async def coin(request):
    return _page(pages.coin(request[DEVICE].name, request.match_info["base"][:40]))


async def skips(request):
    return _page(pages.skips(request[DEVICE].name))


async def c1(request):
    return _page(pages.c1(request[DEVICE].name))


async def hours(request):
    return _page(pages.hours_page(request[DEVICE].name))


async def health(request):
    raise web.HTTPFound("/settings#server")             # 09.11: the server's state lives in the settings now


async def settings(request):
    return _page(pages.settings(request[DEVICE].name))


async def c1_position(request):
    return _page(pages.c1_position(request[DEVICE].name, request.match_info["day"][:10], request.match_info["history"][:40]))


async def feed(request):
    """What happened since ``since`` (ms): the site's notifications while it is open."""
    try:
        since = int(request.query.get("since", "0"))
    except ValueError:
        since = 0
    return web.json_response({"now": int(time.time() * 1000), "events": data.feed(since)})


async def me(request):
    auth = request.app[AUTH]
    return _page(pages.me(request[DEVICE].name, auth.devices(), auth.sessions(), auth.events(60)))


async def logout(request):
    request.app[AUTH].logout(request.cookies.get(COOKIE), ip=request[IP])
    response = web.HTTPFound("/login")
    response.del_cookie(COOKIE, path="/")
    raise response


async def logout_all(request):
    request.app[AUTH].logout(None, everywhere=True, ip=request[IP])
    response = web.HTTPFound("/login")
    response.del_cookie(COOKIE, path="/")
    raise response


async def revoke(request):
    name = (await _form(request)).get("name", "")
    request.app[AUTH].revoke(name, ip=request[IP])
    raise web.HTTPFound("/me")


# login

async def login_page(request):
    d = request[DEVICE]
    return _page(pages.login(d.name, bool(d.passkeys)))


async def login_totp(request):
    auth, cfg = request.app[AUTH], request.app[CFG]
    try:
        token = auth.login_totp((await _form(request)).get("otp", ""), request[IP])
    except Locked:
        return _locked()
    except PermissionError:
        return _page(pages.login(request[DEVICE].name, True, "Код не подошёл."), 401)
    response = web.HTTPFound("/")
    _set_session(response, token, cfg, 2)
    raise response


async def login_options(request):
    try:
        return web.Response(text=request.app[AUTH].login_options(request[IP]), content_type="application/json")
    except Locked:
        return web.json_response({"error": "адрес заблокирован на 15 минут"}, status=429)
    except PermissionError as exc:
        return web.json_response({"error": str(exc)}, status=403)


async def login_verify(request):
    auth, cfg = request.app[AUTH], request.app[CFG]
    try:
        token = auth.login(await _json(request), request[IP])
    except Locked:
        return web.json_response({"error": "адрес заблокирован на 15 минут"}, status=429)
    except PermissionError:
        return web.json_response({"error": "ключ не подошёл"}, status=403)
    response = web.json_response({"ok": True})
    _set_session(response, token, cfg)
    return response


# binding

async def enroll_page(request):
    s, auth = request[SESSION], request.app[AUTH]
    if s is None or s.method != "enroll":
        return _page(pages.enroll_code())
    if not s.totp_ok:
        import segno
        secret, confirmed = auth.totp_secret()
        uri = auth.totp_uri()
        qr = segno.make(uri, error="m").svg_inline(scale=4, border=2, dark="#000", light="#fff")
        return _page(pages.enroll_totp(qr, secret, uri, first=not confirmed))
    return _page(pages.enroll_passkey(s.device.name))


async def enroll_code(request):
    auth, cfg = request.app[AUTH], request.app[CFG]
    try:
        token = auth.redeem((await _form(request)).get("code", ""), request[IP])
    except Locked:
        return _locked()
    except PermissionError:
        return _page(pages.enroll_code("Код не подошёл или устарел."), 401)
    response = web.HTTPFound("/enroll")
    _set_session(response, token, cfg, 0.25)
    raise response


async def enroll_totp(request):
    auth = request.app[AUTH]
    try:
        ok = auth.confirm_totp(request.cookies.get(COOKIE, ""), (await _form(request)).get("otp", ""), request[IP])
    except Locked:
        return _locked()
    except PermissionError:
        raise web.HTTPFound("/enroll")
    if not ok:
        import segno
        secret, confirmed = auth.totp_secret()
        uri = auth.totp_uri()
        qr = segno.make(uri, error="m").svg_inline(scale=4, border=2, dark="#000", light="#fff")
        return _page(pages.enroll_totp(qr, secret, uri, not confirmed, "Код не подошёл, попробуй следующий."), 401)
    raise web.HTTPFound("/enroll")


async def register_options(request):
    try:
        return web.Response(text=request.app[AUTH].register_options(request.cookies.get(COOKIE, ""), request[IP]),
                            content_type="application/json")
    except PermissionError as exc:
        return web.json_response({"error": str(exc)}, status=403)


async def register_verify(request):
    auth, cfg = request.app[AUTH], request.app[CFG]
    try:
        token = auth.register(request.cookies.get(COOKIE, ""), await _json(request), request[IP])
    except PermissionError as exc:
        return web.json_response({"error": str(exc)}, status=403)
    response = web.json_response({"ok": True})
    _set_session(response, token, cfg)
    return response


async def client_error(request):
    """Why a passkey dialog failed on the device (its own words and browser), into the login log."""
    body = await _json(request)
    text = " ".join(str(body.get(k, ""))[:300] for k in ("where", "name", "message"))
    agent = request.headers.get("User-Agent", "")[:200]
    request.app[AUTH].note(request[IP], request[DEVICE].name, f"passkey dialog failed: {text} | {agent}")
    return web.json_response({"ok": True})


async def static(request):
    name = request.match_info["name"]
    kind = ASSETS.get(name)
    path = STATIC / name
    if kind is None or not path.is_file():
        raise web.HTTPNotFound()
    text = kind.startswith("text/") or kind.endswith("javascript")
    return web.Response(body=path.read_bytes(), content_type=kind, charset="utf-8" if text else None)


def make_app(cfg: Config, auth: Optional[Auth] = None) -> web.Application:
    if auth is None:
        from core.notify import notify_owner
        auth = Auth(cfg.rp_id, cfg.origin, notify=notify_owner)
    app = web.Application(middlewares=[secure, gate], client_max_size=64 * 1024)
    app[CFG], app[AUTH] = cfg, auth
    app.router.add_get("/", overview)
    app.router.add_get("/trades", trades)
    app.router.add_get("/trade/{book}/{moment}/{base}", trade)
    app.router.add_get("/skips", skips)
    app.router.add_get("/coins", coins)
    app.router.add_get("/coin/{base}", coin)
    app.router.add_get("/c1", c1)
    app.router.add_get("/hours", hours)
    app.router.add_get("/health", health)
    app.router.add_get("/settings", settings)
    app.router.add_get("/c1/pos/{day}/{history}", c1_position)
    app.router.add_get("/api/feed", feed)
    app.router.add_get("/me", me)
    app.router.add_post("/me/revoke", revoke)
    app.router.add_post("/me/logout-all", logout_all)
    app.router.add_post("/logout", logout)
    app.router.add_get("/login", login_page)
    app.router.add_post("/login/totp", login_totp)
    app.router.add_post("/webauthn/login/options", login_options)
    app.router.add_post("/webauthn/login/verify", login_verify)
    app.router.add_get("/enroll", enroll_page)
    app.router.add_post("/enroll", enroll_code)
    app.router.add_post("/enroll/totp", enroll_totp)
    app.router.add_post("/webauthn/register/options", register_options)
    app.router.add_post("/webauthn/register/verify", register_verify)
    app.router.add_post("/webauthn/client-error", client_error)
    app.router.add_get("/static/{name:.+}", static)
    return app
