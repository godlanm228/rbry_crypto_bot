"""Messages to the owner through the test bot: the only way the learning jobs talk to anyone.

The token is read from .env.test inside ``notify_owner`` and never leaves it;
the chat is always TELEGRAM_OWNER_ID (Vlad's whitelist), never a list of users.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER_ENV = ROOT / ".env.test"


def notify_owner(text: str, env_path: Path = OWNER_ENV, silent: bool = False) -> bool:
    """One message to the owner through the test bot; ``silent``: no sound (the night)."""
    try:
        from dotenv import dotenv_values
        env = dotenv_values(env_path)
    except (ImportError, OSError):
        return False
    token, owner = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_OWNER_ID")
    if not token or not owner:
        return False
    fields = {"chat_id": owner, "text": text, "disable_web_page_preview": "true"}
    if silent:
        fields["disable_notification"] = "true"
    data = urllib.parse.urlencode(fields).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20) as r:
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
