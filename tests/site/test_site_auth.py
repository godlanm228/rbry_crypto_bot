"""The site's door: a binding code, the app's codes once each, a passkey of this very device, locks, revocation."""

import pyotp
import pytest
from softkey import ORIGIN, RP, SoftKey

from web.auth import Auth, Locked, b64url, unb64url
PHONE, LAPTOP = "100.64.0.2", "100.64.0.3"


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def door(tmp_path):
    told = []
    clock = Clock()
    auth = Auth(RP, ORIGIN, tmp_path / "auth.db", notify=told.append, clock=clock)
    return auth, told, clock


def _otp(auth, clock):
    secret, _ = auth.totp_secret()
    return pyotp.TOTP(secret).at(clock())


def _bind(auth, clock, name, ip, key):
    code = auth.add_device(name, ip)
    token = auth.redeem(code, ip)
    assert auth.confirm_totp(token, _otp(auth, clock), ip)
    clock.t += 31                                    # the next app code
    return auth.register(token, key.create(auth.register_options(token, ip)), ip)


def test_a_device_is_bound_by_its_code_the_app_and_its_passkey_then_logs_in(door):
    auth, told, clock = door
    phone = SoftKey()
    full = _bind(auth, clock, "iphone", PHONE, phone)
    s = auth.session(full, PHONE)
    assert s.method == "passkey" and s.device.name == "iphone" and s.device.passkeys == 1
    assert any("привязано" in t for t in told)
    token = auth.login(phone.get(auth.login_options(PHONE)), PHONE)
    assert auth.session(token, PHONE).method == "passkey" and any("вход с «iphone» (passkey)" in t for t in told)
    assert auth.session(token, LAPTOP) is None                       # a cookie does not work from another device


def test_a_code_works_once_on_its_own_device_and_not_after_its_time(door):
    auth, _, clock = door
    code = auth.add_device("iphone", PHONE)
    auth.add_device("laptop", LAPTOP)
    with pytest.raises(PermissionError):
        auth.redeem(code, LAPTOP)                                    # the laptop cannot use the phone's code
    auth.redeem(code, PHONE)
    with pytest.raises(PermissionError):
        auth.redeem(code, PHONE)                                     # once
    late = auth.new_code("iphone")
    clock.t += 16 * 60
    with pytest.raises(PermissionError):
        auth.redeem(late, PHONE)


def test_an_app_code_is_good_once_and_five_failures_lock_the_address(door):
    auth, told, clock = door
    _bind(auth, clock, "iphone", PHONE, SoftKey())
    clock.t += 60
    otp = _otp(auth, clock)
    token = auth.login_totp(otp, PHONE)
    assert auth.session(token, PHONE).method == "totp" and any("по коду из приложения" in t for t in told)
    with pytest.raises(PermissionError):
        auth.login_totp(otp, PHONE)                                  # the same code again: no
    for _ in range(4):
        with pytest.raises(PermissionError):
            auth.login_totp("000000", PHONE)
    with pytest.raises(Locked):
        auth.login_totp(_otp(auth, clock), PHONE)
    assert any("заблокирован" in t for t in told)
    clock.t += 16 * 60
    auth.login_totp(_otp(auth, clock), PHONE)                        # after the lock


def test_a_passkey_of_another_device_or_without_presence_is_refused(door):
    auth, _, clock = door
    phone, laptop = SoftKey(), SoftKey()
    _bind(auth, clock, "iphone", PHONE, phone)
    clock.t += 60
    _bind(auth, clock, "laptop", LAPTOP, laptop)
    with pytest.raises(PermissionError):
        auth.login(phone.get(auth.login_options(LAPTOP)), LAPTOP)    # the phone's key from the laptop's address
    with pytest.raises(PermissionError):
        auth.login(laptop.get(auth.login_options(LAPTOP), uv=False), LAPTOP)   # no Face ID / Hello
    evil = SoftKey(origin="https://evil.example")
    evil.id = laptop.id
    with pytest.raises(PermissionError):
        auth.login(evil.get(auth.login_options(LAPTOP)), LAPTOP)


def test_binding_needs_the_app_before_the_passkey_and_revoking_ends_everything(door):
    auth, told, clock = door
    code = auth.add_device("iphone", PHONE)
    token = auth.redeem(code, PHONE)
    with pytest.raises(PermissionError):
        auth.register_options(token, PHONE)                          # the app first
    assert not auth.confirm_totp(token, "123456", PHONE)
    full = _bind(auth, clock, "laptop", LAPTOP, SoftKey())
    assert auth.revoke("laptop") and auth.session(full, LAPTOP) is None and auth.device_by_ip(LAPTOP) is None
    assert any("отвязано" in t for t in told)
    with pytest.raises(ValueError):
        auth.add_device("other", PHONE)                              # one address, one device


def test_b64url_round_trip():
    assert unb64url(b64url(b"\x00\xff\x10")) == b"\x00\xff\x10"


def test_a_device_reports_its_dialog_failures_but_cannot_flood_the_log(door):
    # 08.11 audit: /webauthn/client-error needs no session, so one address writes at most NOTES_PER_HOUR an hour
    from web.auth import NOTES_PER_HOUR
    auth, _, clock = door
    for _ in range(NOTES_PER_HOUR + 15):
        auth.note(PHONE, "iphone", "passkey dialog failed: NotAllowedError")
    assert sum(e["what"].startswith("passkey dialog failed") for e in auth.events(200)) == NOTES_PER_HOUR
    clock.t += 3601
    auth.note(PHONE, "iphone", "passkey dialog failed: again")
    assert auth.events(1)[0]["what"].endswith("again")


@pytest.mark.skipif(__import__("os").name == "nt", reason="unix file modes")
def test_the_auth_file_is_for_its_owner_alone(door):
    import os
    import stat
    auth, _, _ = door
    assert stat.S_IMODE(os.stat(auth.path).st_mode) == 0o600
