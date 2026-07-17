"""Central logging setup with defence-in-depth secret redaction."""

from __future__ import annotations

import logging
import os
import re
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


_SECRET_ENV_NAMES = (
    "TELEGRAM_BOT_TOKEN",
    "HUGGINGFACE_API_KEY",
    "GROQ_API_KEY",
    "COINGECKO_API_KEY",
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
)
_TELEGRAM_TOKEN = re.compile(
    r"(?i)(bot)?(\d{6,12}:[A-Za-z0-9_-]{20,})"
)
_BEARER_TOKEN = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~-]{8,}")
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(authorization|api[_-]?key|token|secret|password)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)


def _configured_secrets() -> tuple[str, ...]:
    values = []
    for name in _SECRET_ENV_NAMES:
        value = os.getenv(name, "").strip()
        if len(value) >= 8:
            values.append(value)
    return tuple(dict.fromkeys(values))


def redact_text(text: Any, secrets: Iterable[str] | None = None) -> str:
    """Return a printable value with credentials replaced by stable markers."""
    result = str(text)
    for secret in secrets if secrets is not None else _configured_secrets():
        if secret:
            result = result.replace(secret, "[redacted-secret]")
    result = _TELEGRAM_TOKEN.sub(r"\1[redacted-telegram-token]", result)
    result = _BEARER_TOKEN.sub(r"\1[redacted-token]", result)
    return _SENSITIVE_ASSIGNMENT.sub(r"\1\2[redacted-secret]", result)


def _redact_value(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, Mapping):
        return {
            key: _redact_value(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_redact_value(item, secrets) for item in value)
    if isinstance(value, list):
        return [_redact_value(item, secrets) for item in value]
    return value


class SecretRedactingFilter(logging.Filter):
    """Redact secrets in messages, interpolation args and tracebacks."""

    def __init__(self, secrets: Iterable[str] | None = None):
        super().__init__()
        self.secrets = tuple(secrets) if secrets is not None else _configured_secrets()

    def filter(self, record: logging.LogRecord) -> bool:
        # Render interpolation first. Redacting a format string such as
        # ``token=%s`` while retaining its args would break logging itself.
        try:
            rendered_message = record.getMessage()
        except Exception:
            rendered_message = f"{record.msg!s} {_redact_value(record.args, self.secrets)!s}"
        record.msg = redact_text(rendered_message, self.secrets)
        record.args = ()
        if record.exc_info:
            rendered = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = redact_text(rendered, self.secrets)
        if record.stack_info:
            record.stack_info = redact_text(record.stack_info, self.secrets)
        return True


def configure_logging(
    *,
    level: str | int = "INFO",
    log_file: str | os.PathLike[str] | None = "logs/bot.log",
) -> None:
    """Install the production handlers and apply redaction to every handler."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    redactor = SecretRedactingFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(redactor)

    # These clients include full request URLs at INFO. Telegram embeds the bot
    # token in the URL path, so request-level diagnostics must stay at WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
