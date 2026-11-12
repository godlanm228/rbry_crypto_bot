"""The owner's site (web/): run it, bind a device, revoke one.

    python scripts/web.py serve [--port 8000]                    # deploy/rbry-web.service, 127.0.0.1 only
    python scripts/web.py enroll --device iphone --ip 100.x.y.z  # bind a device: its code goes to Telegram
    python scripts/web.py code --device iphone                   # a new code for a device not bound yet
    python scripts/web.py devices | revoke --device iphone | events

The settings are in runtime/web/web.env (RBRY_WEB_HOST = the node's ts.net name, RBRY_WEB_TS_LOGIN = the
owner's Tailscale login); RBRY_WEB_DEV=1 runs it on the laptop's localhost for a test.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RBRY_ENV_FILE", str(ROOT / ".env.test"))

from core.console import utf8_output  # noqa: E402

from core.notify import notify_owner  # noqa: E402
from web.app import Config, make_app  # noqa: E402
from web.auth import CODE_MINUTES, Auth  # noqa: E402


def _auth(cfg: Config) -> Auth:
    return Auth(cfg.rp_id, cfg.origin, notify=notify_owner)


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    for name in ("enroll", "code", "revoke"):
        p = sub.add_parser(name)
        p.add_argument("--device", required=True)
        if name == "enroll":
            p.add_argument("--ip", required=True, help="the device's tailnet address (100.x.y.z)")
    sub.add_parser("devices")
    sub.add_parser("events")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load()
    if args.cmd == "serve":
        from aiohttp import web
        web.run_app(make_app(cfg), host="127.0.0.1", port=args.port, access_log=None, print=None)
        return 0
    auth = _auth(cfg)
    if args.cmd in ("enroll", "code"):
        code = auth.add_device(args.device, args.ip) if args.cmd == "enroll" else auth.new_code(args.device)
        sent = notify_owner(f"🔐 Код привязки устройства «{args.device}» к сайту: {code}\n"
                            f"Действует {CODE_MINUTES} минут, один раз и только на этом устройстве. "
                            f"Открой {cfg.origin}/enroll на нём — в Safari/Chrome, не внутри Telegram "
                            f"(во встроенном браузере Telegram passkey не создаётся).")
        print(f"code for {args.device}: sent to Telegram" if sent else f"code for {args.device}: {code}")
    elif args.cmd == "devices":
        for d in auth.devices():
            print(d)
    elif args.cmd == "revoke":
        print("revoked" if auth.revoke(args.device, ip="cli") else "no such device")
    elif args.cmd == "events":
        for e in reversed(auth.events(100)):
            print(time.strftime("%d.%m %H:%M:%S", time.localtime(e["ts"])), e["device"], e["ip"], e["what"],
                  "ok" if e["ok"] else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
