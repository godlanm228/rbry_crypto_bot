"""Who may open the owner's site, and how (runtime/web/auth.db).

The site is reachable only inside the owner's tailnet (Tailscale Serve on the VM, nothing open to the
internet). On top of that, the site itself:
- knows the owner's devices by their tailnet address (100.x, stable while the node lives): a request from
  any other address is refused before anything else;
- binds a device once, with a one-time code the owner gets in Telegram (scripts/web.py enroll), and a
  one-time password app (TOTP) shared by the devices;
- logs in with the device's passkey (Face ID, Touch ID, Windows Hello); the TOTP code alone is the fallback,
  told to the owner at once;
- keeps a session per device (a stolen cookie does not work from another device), locks an address after
  repeated failures, and tells the owner of every login, binding and revocation.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("RBRY_WEB_AUTH_DB", ROOT / "runtime" / "web" / "auth.db"))
CODE_MINUTES = 15            # a binding code lives this long
ENROLL_MINUTES = 15          # and the binding that follows it
SESSION_HOURS = 12           # a passkey login
FALLBACK_HOURS = 2           # a login with the one-time password app alone
CHALLENGE_SECONDS = 300          # the page fetches it on load: Safari wants the dialog right on the tap
MAX_FAILS = 5                # failures from one address in FAIL_MINUTES lock it for FAIL_MINUTES
FAIL_MINUTES = 15
NOTES_PER_HOUR = 20          # what a device may report of its passkey dialogs: the login log does not flood
TOTP_ISSUER = "RBRY"

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, ip TEXT NOT NULL,
    created REAL NOT NULL, revoked REAL);
CREATE TABLE IF NOT EXISTS passkeys (cred_id TEXT PRIMARY KEY, device_id INTEGER NOT NULL, public_key BLOB NOT NULL,
    sign_count INTEGER NOT NULL, created REAL NOT NULL, last_used REAL);
CREATE TABLE IF NOT EXISTS totp (id INTEGER PRIMARY KEY CHECK (id = 1), secret TEXT NOT NULL, confirmed REAL,
    last_step INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS codes (code_hash TEXT PRIMARY KEY, device_id INTEGER NOT NULL, expires REAL NOT NULL,
    used REAL);
CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, device_id INTEGER NOT NULL, method TEXT NOT NULL,
    created REAL NOT NULL, expires REAL NOT NULL, last_seen REAL NOT NULL, totp_ok INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS challenges (device_id INTEGER NOT NULL, purpose TEXT NOT NULL, challenge BLOB NOT NULL,
    expires REAL NOT NULL, PRIMARY KEY (device_id, purpose));
CREATE TABLE IF NOT EXISTS events (ts REAL NOT NULL, ip TEXT, device TEXT, what TEXT NOT NULL, ok INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS fails (ip TEXT PRIMARY KEY, count INTEGER NOT NULL, since REAL NOT NULL);
"""
FULL = ("passkey", "totp")   # session methods that open the site; "enroll" opens only the binding


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def unb64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass
class Device:
    id: int
    name: str
    ip: str
    passkeys: int


@dataclass
class Session:
    device: Device
    method: str
    expires: float
    totp_ok: bool


class Locked(Exception):
    """Too many failures from this address."""


class Auth:
    def __init__(self, rp_id: str, origin: str, path: Path = DB_PATH, notify: Optional[Callable[[str], object]] = None,
                 clock: Callable[[], float] = time.time, rp_name: str = "RBRY"):
        self.rp_id, self.origin, self.rp_name = rp_id, origin, rp_name
        self.path, self.notify, self.clock = Path(path), notify, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)
        try:
            os.chmod(self.path, 0o600)      # the app's secret and the passkeys: the owner of the file alone
        except OSError:
            pass

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def _tell(self, text: str) -> None:
        if self.notify:
            try:
                self.notify(f"🔐 Сайт: {text}")
            except Exception:  # noqa: BLE001 - a message that does not go out never blocks a login
                pass

    def _event(self, db, ip: str, device: Optional[str], what: str, ok: bool) -> None:
        db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)", (self.clock(), ip, device, what, int(ok)))

    # devices

    def add_device(self, name: str, ip: str) -> str:
        """A device the owner binds (its tailnet address), and its one-time binding code."""
        with self._db() as db:
            if db.execute("SELECT 1 FROM devices WHERE ip = ? AND revoked IS NULL AND name != ?", (ip, name)).fetchone():
                raise ValueError(f"{ip} belongs to another device")
            row = db.execute("SELECT id FROM devices WHERE name = ?", (name,)).fetchone()
            if row:
                db.execute("UPDATE devices SET ip = ?, revoked = NULL WHERE id = ?", (ip, row["id"]))
            else:
                db.execute("INSERT INTO devices (name, ip, created) VALUES (?, ?, ?)", (name, ip, self.clock()))
            self._event(db, ip, name, "device added", True)
        return self.new_code(name)

    def new_code(self, name: str) -> str:
        code = "-".join(secrets.token_hex(2).upper() for _ in range(3))      # 6 hex pairs: ABCD-EF01-2345
        with self._db() as db:
            row = db.execute("SELECT id FROM devices WHERE name = ? AND revoked IS NULL", (name,)).fetchone()
            if not row:
                raise ValueError(f"no device {name}")
            db.execute("DELETE FROM codes WHERE device_id = ?", (row["id"],))
            db.execute("INSERT INTO codes VALUES (?, ?, ?, NULL)",
                       (_hash(code.upper()), row["id"], self.clock() + CODE_MINUTES * 60))
        return code

    def device_by_ip(self, ip: str) -> Optional[Device]:
        with self._db() as db:
            row = db.execute("SELECT d.id, d.name, d.ip, COUNT(p.cred_id) AS n FROM devices d LEFT JOIN passkeys p "
                             "ON p.device_id = d.id WHERE d.ip = ? AND d.revoked IS NULL GROUP BY d.id", (ip,)).fetchone()
        return Device(row["id"], row["name"], row["ip"], row["n"]) if row else None

    def devices(self) -> list[dict]:
        with self._db() as db:
            return [dict(r) for r in db.execute(
                "SELECT d.name, d.ip, d.created, d.revoked, COUNT(p.cred_id) AS passkeys, MAX(p.last_used) AS last_used "
                "FROM devices d LEFT JOIN passkeys p ON p.device_id = d.id GROUP BY d.id ORDER BY d.created")]

    def revoke(self, name: str, ip: str = "") -> bool:
        """A device's passkeys and sessions go, and the device with them (bind it again to come back)."""
        with self._db() as db:
            row = db.execute("SELECT id FROM devices WHERE name = ? AND revoked IS NULL", (name,)).fetchone()
            if not row:
                return False
            db.execute("DELETE FROM passkeys WHERE device_id = ?", (row["id"],))
            db.execute("DELETE FROM sessions WHERE device_id = ?", (row["id"],))
            db.execute("DELETE FROM codes WHERE device_id = ?", (row["id"],))
            db.execute("UPDATE devices SET revoked = ? WHERE id = ?", (self.clock(), row["id"]))
            self._event(db, ip, name, "device revoked", True)
        self._tell(f"устройство «{name}» отвязано, его ключи и сессии удалены.")
        return True

    # failures

    def check_lock(self, ip: str) -> None:
        with self._db() as db:
            row = db.execute("SELECT count, since FROM fails WHERE ip = ?", (ip,)).fetchone()
        if row and row["count"] >= MAX_FAILS and self.clock() - row["since"] < FAIL_MINUTES * 60:
            raise Locked(ip)

    def _fail(self, db, ip: str, device: Optional[str], what: str) -> None:
        now = self.clock()
        row = db.execute("SELECT count, since FROM fails WHERE ip = ?", (ip,)).fetchone()
        count = 1 if not row or now - row["since"] >= FAIL_MINUTES * 60 else row["count"] + 1
        since = now if count == 1 else row["since"]
        db.execute("INSERT OR REPLACE INTO fails VALUES (?, ?, ?)", (ip, count, since))
        self._event(db, ip, device, what, False)
        if count in (3, MAX_FAILS):
            self._tell(f"{count} неудачные попытки входа с {device or ip} ({what})"
                       + (f" — адрес заблокирован на {FAIL_MINUTES} минут." if count >= MAX_FAILS else "."))

    def _clear_fails(self, db, ip: str) -> None:
        db.execute("DELETE FROM fails WHERE ip = ?", (ip,))

    # sessions

    def _session(self, db, device: Device, method: str, hours: float, totp_ok: bool = False) -> str:
        token, now = secrets.token_urlsafe(32), self.clock()
        db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (_hash(token), device.id, method, now, now + hours * 3600, now, int(totp_ok)))
        return token

    def session(self, token: Optional[str], ip: str) -> Optional[Session]:
        """The live session of this token on this very device, or None."""
        if not token:
            return None
        device = self.device_by_ip(ip)
        if device is None:
            return None
        with self._db() as db:
            row = db.execute("SELECT method, expires, totp_ok FROM sessions WHERE token_hash = ? AND device_id = ?",
                             (_hash(token), device.id)).fetchone()
            if not row or row["expires"] < self.clock():
                return None
            db.execute("UPDATE sessions SET last_seen = ? WHERE token_hash = ?", (self.clock(), _hash(token)))
        return Session(device, row["method"], row["expires"], bool(row["totp_ok"]))

    def logout(self, token: Optional[str], everywhere: bool = False, ip: str = "") -> None:
        with self._db() as db:
            if everywhere:
                db.execute("DELETE FROM sessions")
                self._event(db, ip, None, "logout everywhere", True)
            elif token:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash(token),))

    def sessions(self) -> list[dict]:
        with self._db() as db:
            return [dict(r) for r in db.execute(
                "SELECT d.name, s.method, s.created, s.expires, s.last_seen FROM sessions s JOIN devices d "
                "ON d.id = s.device_id WHERE s.expires > ? ORDER BY s.last_seen DESC", (self.clock(),))]

    def note(self, ip: str, device: str, what: str) -> None:
        """A thing the device reported (a passkey dialog that failed there): logged, not counted as a failure,
        at most NOTES_PER_HOUR an hour from one address (the endpoint needs no session)."""
        with self._db() as db:
            recent = db.execute("SELECT COUNT(*) FROM events WHERE ip = ? AND ts > ? AND what LIKE 'passkey dialog failed%'",
                                (ip, self.clock() - 3600)).fetchone()[0]
            if recent < NOTES_PER_HOUR:
                self._event(db, ip, device, what[:600], False)

    def events(self, limit: int = 50) -> list[dict]:
        with self._db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM events ORDER BY ts DESC, rowid DESC LIMIT ?", (limit,))]

    # binding: the code, then the one-time password app, then the passkey

    def redeem(self, code: str, ip: str) -> str:
        """The binding code typed on the device it was made for: a short session that can only bind."""
        self.check_lock(ip)
        device = self.device_by_ip(ip)
        token = None
        with self._db() as db:
            row = db.execute("SELECT code_hash FROM codes WHERE code_hash = ? AND device_id = ? AND used IS NULL "
                             "AND expires > ?", (_hash(code.strip().upper()), device.id if device else -1,
                                                 self.clock())).fetchone()
            if device is None or not row:
                self._fail(db, ip, device.name if device else None, "wrong binding code")
            else:
                db.execute("UPDATE codes SET used = ? WHERE code_hash = ?", (self.clock(), row["code_hash"]))
                self._clear_fails(db, ip)
                self._event(db, ip, device.name, "binding code used", True)
                token = self._session(db, device, "enroll", ENROLL_MINUTES / 60)
        if token is None:                       # raised after the commit: the failure stays counted
            raise PermissionError("wrong or expired code")
        return token

    def totp_secret(self) -> tuple[str, bool]:
        """(the app's secret, whether it is confirmed); made on first use."""
        import pyotp
        with self._db() as db:
            row = db.execute("SELECT secret, confirmed FROM totp WHERE id = 1").fetchone()
            if row:
                return row["secret"], row["confirmed"] is not None
            secret = pyotp.random_base32()
            db.execute("INSERT INTO totp (id, secret) VALUES (1, ?)", (secret,))
            return secret, False

    def totp_uri(self) -> str:
        import pyotp
        secret, _ = self.totp_secret()
        return pyotp.TOTP(secret).provisioning_uri(name="owner", issuer_name=TOTP_ISSUER)

    def _totp_ok(self, db, otp: str) -> bool:
        """A code of the app, each 30-second step used once (a code seen over a shoulder cannot be replayed)."""
        import pyotp
        row = db.execute("SELECT secret, last_step FROM totp WHERE id = 1").fetchone()
        if not row or not otp or not otp.strip().isdigit():
            return False
        totp, now = pyotp.TOTP(row["secret"]), self.clock()
        for drift in (0, -1, 1):
            step = int(now // 30) + drift
            if step > row["last_step"] and secrets.compare_digest(totp.at(step * 30), otp.strip()):
                db.execute("UPDATE totp SET last_step = ? WHERE id = 1", (step,))
                return True
        return False

    def confirm_totp(self, token: str, otp: str, ip: str) -> bool:
        """The binding session proves the app: the first device confirms the new secret, a later one shows it
        holds the same app."""
        self.check_lock(ip)
        s = self.session(token, ip)
        if s is None or s.method != "enroll":
            raise PermissionError("no binding session")
        with self._db() as db:
            if not self._totp_ok(db, otp):
                self._fail(db, ip, s.device.name, "wrong app code while binding")
                return False
            db.execute("UPDATE totp SET confirmed = COALESCE(confirmed, ?) WHERE id = 1", (self.clock(),))
            db.execute("UPDATE sessions SET totp_ok = 1 WHERE token_hash = ?", (_hash(token),))
            self._clear_fails(db, ip)
            self._event(db, ip, s.device.name, "app code confirmed", True)
        return True

    def login_totp(self, otp: str, ip: str) -> str:
        """The fallback: a known device and the app's code, a short session, the owner told at once."""
        self.check_lock(ip)
        device = self.device_by_ip(ip)
        token = None
        with self._db() as db:
            confirmed = db.execute("SELECT confirmed FROM totp WHERE id = 1").fetchone()
            if device is None or not device.passkeys or not confirmed or confirmed["confirmed"] is None \
                    or not self._totp_ok(db, otp):
                self._fail(db, ip, device.name if device else None, "wrong app code")
            else:
                self._clear_fails(db, ip)
                self._event(db, ip, device.name, "login with the app code", True)
                token = self._session(db, device, "totp", FALLBACK_HOURS, totp_ok=True)
        if token is None:
            raise PermissionError("wrong code")
        self._tell(f"вход с «{device.name}» по коду из приложения (без passkey). Если это не ты — открой "
                   f"сайт с другого устройства и нажми «Выйти везде», затем «Отвязать».")
        return token

    # passkeys

    def _challenge(self, db, device: Device, purpose: str, challenge: bytes) -> None:
        db.execute("INSERT OR REPLACE INTO challenges VALUES (?, ?, ?, ?)",
                   (device.id, purpose, challenge, self.clock() + CHALLENGE_SECONDS))

    def _take_challenge(self, db, device: Device, purpose: str) -> Optional[bytes]:
        row = db.execute("SELECT challenge, expires FROM challenges WHERE device_id = ? AND purpose = ?",
                         (device.id, purpose)).fetchone()
        db.execute("DELETE FROM challenges WHERE device_id = ? AND purpose = ?", (device.id, purpose))
        return row["challenge"] if row and row["expires"] >= self.clock() else None

    def register_options(self, token: str, ip: str) -> str:
        from webauthn import generate_registration_options, options_to_json
        from webauthn.helpers.structs import (AuthenticatorSelectionCriteria, PublicKeyCredentialDescriptor,
                                              ResidentKeyRequirement, UserVerificationRequirement)
        s = self.session(token, ip)
        if s is None or s.method != "enroll" or not s.totp_ok:
            raise PermissionError("bind the app first")
        # each device its own user: a passkey kept in iCloud Keychain (the laptop's, made with the phone) must not
        # stand for, or block, the phone's own (07.11: the iPhone could not make its passkey)
        with self._db() as db:
            own = [r["cred_id"] for r in db.execute("SELECT cred_id FROM passkeys WHERE device_id = ?", (s.device.id,))]
            options = generate_registration_options(
                rp_id=self.rp_id, rp_name=self.rp_name, user_name=f"rbry-{s.device.name}",
                user_id=f"rbry-{s.device.name}".encode("utf-8"), user_display_name=f"RBRY · {s.device.name}",
                authenticator_selection=AuthenticatorSelectionCriteria(
                    resident_key=ResidentKeyRequirement.PREFERRED,
                    user_verification=UserVerificationRequirement.REQUIRED),
                exclude_credentials=[PublicKeyCredentialDescriptor(id=unb64url(c)) for c in own])
            self._challenge(db, s.device, "register", options.challenge)
        return options_to_json(options)

    def register(self, token: str, credential: dict, ip: str) -> str:
        """The device's new passkey; the binding session becomes a full one."""
        from webauthn import verify_registration_response
        s = self.session(token, ip)
        if s is None or s.method != "enroll" or not s.totp_ok:
            raise PermissionError("bind the app first")
        full, failure = None, "the request expired"
        with self._db() as db:
            challenge = self._take_challenge(db, s.device, "register")
            if challenge is not None:
                try:
                    done = verify_registration_response(credential=credential, expected_challenge=challenge,
                                                        expected_rp_id=self.rp_id, expected_origin=self.origin,
                                                        require_user_verification=True)
                except Exception as exc:  # noqa: BLE001 - the library's many errors are all "not this passkey"
                    self._fail(db, ip, s.device.name, f"passkey not accepted: {type(exc).__name__}")
                    failure = "passkey not accepted"
                else:
                    db.execute("INSERT INTO passkeys VALUES (?, ?, ?, ?, ?, NULL)",
                               (b64url(done.credential_id), s.device.id, done.credential_public_key, done.sign_count,
                                self.clock()))
                    db.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash(token),))
                    self._event(db, ip, s.device.name, "passkey added", True)
                    full = self._session(db, s.device, "passkey", SESSION_HOURS, totp_ok=True)
        if full is None:
            raise PermissionError(failure)
        self._tell(f"устройство «{s.device.name}» привязано: passkey добавлен.")
        return full

    def login_options(self, ip: str) -> str:
        from webauthn import generate_authentication_options, options_to_json
        from webauthn.helpers.structs import PublicKeyCredentialDescriptor, UserVerificationRequirement
        self.check_lock(ip)
        device = self.device_by_ip(ip)
        if device is None or not device.passkeys:
            raise PermissionError("this device has no passkey")
        with self._db() as db:
            own = [r["cred_id"] for r in db.execute("SELECT cred_id FROM passkeys WHERE device_id = ?", (device.id,))]
            options = generate_authentication_options(
                rp_id=self.rp_id, allow_credentials=[PublicKeyCredentialDescriptor(id=unb64url(c)) for c in own],
                user_verification=UserVerificationRequirement.REQUIRED)
            self._challenge(db, device, "login", options.challenge)
        return options_to_json(options)

    def login(self, credential: dict, ip: str) -> str:
        """A passkey of this very device: a passkey synced to another device does not open the site from it."""
        from webauthn import verify_authentication_response
        self.check_lock(ip)
        device = self.device_by_ip(ip)
        if device is None:
            raise PermissionError("unknown device")
        token, failure = None, "not this device's passkey"
        with self._db() as db:
            challenge = self._take_challenge(db, device, "login")
            cred_id = str(credential.get("id", "")) if isinstance(credential, dict) else ""
            row = db.execute("SELECT public_key, sign_count FROM passkeys WHERE cred_id = ? AND device_id = ?",
                             (cred_id, device.id)).fetchone()
            if challenge is None or row is None:
                self._fail(db, ip, device.name, "passkey login without its request or of another device")
            else:
                try:
                    done = verify_authentication_response(
                        credential=credential, expected_challenge=challenge, expected_rp_id=self.rp_id,
                        expected_origin=self.origin, credential_public_key=row["public_key"],
                        credential_current_sign_count=row["sign_count"], require_user_verification=True)
                except Exception as exc:  # noqa: BLE001
                    self._fail(db, ip, device.name, f"passkey not accepted: {type(exc).__name__}")
                    failure = "passkey not accepted"
                else:
                    db.execute("UPDATE passkeys SET sign_count = ?, last_used = ? WHERE cred_id = ?",
                               (done.new_sign_count, self.clock(), cred_id))
                    self._clear_fails(db, ip)
                    self._event(db, ip, device.name, "login with passkey", True)
                    token = self._session(db, device, "passkey", SESSION_HOURS, totp_ok=True)
        if token is None:
            raise PermissionError(failure)
        self._tell(f"вход с «{device.name}» (passkey).")
        return token
