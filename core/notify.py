"""Messages to the owner through the test bot: the only way the learning jobs talk to anyone.

The token is read from .env.test inside ``notify_owner`` and never leaves it;
the chat is always TELEGRAM_OWNER_ID (the owner's whitelist), never a list of users.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER_ENV = ROOT / ".env.test"


PART = 3800                  # characters a message part: Telegram takes 4096 UTF-16 units, an emoji counts two


def units(text: str) -> int:
    """What Telegram counts: UTF-16 code units (an emoji is two)."""
    return len(text.encode("utf-16-le")) // 2


def _cut(line: str, size: int) -> int:
    """How many characters of ``line`` fit in ``size`` units."""
    n, used = 0, 0
    for ch in line:
        used += 2 if ord(ch) > 0xFFFF else 1
        if used > size:
            break
        n += 1
    return max(n, 1)


def parts(text: str, size: int = PART) -> list[str]:
    """The text cut at line ends into parts Telegram takes, counted in its own UTF-16 units (the night researcher,
    08.11: by characters, a part with ~300 emoji would pass 4096); a line longer than a part is cut where it must."""
    out, cur = [], ""
    for line in text.split("\n"):
        while units(line) > size:
            if cur:
                out.append(cur)
                cur = ""
            k = _cut(line, size)
            out.append(line[:k])
            line = line[k:]
        if cur and units(cur) + 1 + units(line) > size:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    return out + ([cur] if cur else [])


def notify_owner(text: str, env_path: Path = OWNER_ENV, silent: bool = False) -> bool:
    """One message to the owner through the test bot (in parts when it is longer than Telegram takes: a crash
    day's message is the one that must not be lost); ``silent``: no sound (the night)."""
    try:
        from dotenv import dotenv_values
        env = dotenv_values(env_path)
    except (ImportError, OSError):
        return False
    token, owner = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_OWNER_ID")
    if not token or not owner:
        return False
    ok = True
    for part in parts(text) or [""]:
        fields = {"chat_id": owner, "text": part, "disable_web_page_preview": "true"}
        if silent:
            fields["disable_notification"] = "true"
        data = urllib.parse.urlencode(fields).encode()
        sent = False
        for attempt in range(3):          # a part that failed is tried again here: a caller's retry would send the
            try:                          # parts that already went a second time (the night researcher, 08.11)
                with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data,
                                            timeout=20) as r:
                    sent = r.status == 200
            except OSError:
                sent = False
            if sent:
                break
            time.sleep(2 * (attempt + 1))
        ok = ok and sent
    return ok


def notify_owner_file(path, caption: str = "", env_path: Path = OWNER_ENV, silent: bool = False) -> bool:
    """One file to the owner through the test bot (a full report to read on the phone), ``caption`` under it."""
    try:
        from dotenv import dotenv_values
        env = dotenv_values(env_path)
    except (ImportError, OSError):
        return False
    token, owner, path = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_OWNER_ID"), Path(path)
    if not token or not owner or not path.is_file():
        return False
    boundary = uuid.uuid4().hex
    fields = {"chat_id": owner, "caption": caption[:1024]}
    if silent:
        fields["disable_notification"] = "true"
    body = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8")
                    for k, v in fields.items() if v)
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="document"; filename="{path.name}"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n").encode("utf-8")
    body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendDocument", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as r:
            return r.status == 200
    except OSError:
        return False


def owner_id(env_path: Path = OWNER_ENV) -> int | None:
    try:
        from dotenv import dotenv_values
        value = dotenv_values(env_path).get("TELEGRAM_OWNER_ID")
    except (ImportError, OSError):
        return None
    return int(value) if value and value.strip().lstrip("-").isdigit() else None
