import logging
import sys

from core.logging_config import SecretRedactingFilter, redact_text


def _format(record, secrets):
    SecretRedactingFilter(secrets).filter(record)
    return logging.Formatter("%(levelname)s %(message)s").format(record)


def test_redacts_interpolated_telegram_url_and_assignment():
    token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ_123456"
    record = logging.LogRecord(
        "httpx",
        logging.INFO,
        __file__,
        1,
        "POST https://api.telegram.org/bot%s/getMe token=%s",
        (token, token),
        None,
    )

    rendered = _format(record, (token,))

    assert token not in rendered
    assert "redacted" in rendered


def test_redacts_secret_from_traceback():
    secret = "gsk_super_private_value_123456"
    try:
        raise RuntimeError(f"Authorization=Bearer {secret}")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        "test",
        logging.ERROR,
        __file__,
        1,
        "request failed",
        (),
        exc_info,
    )

    rendered = _format(record, (secret,))

    assert secret not in rendered
    assert "redacted" in rendered


def test_generic_redaction_works_without_loaded_environment():
    token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ_123456"
    assert token not in redact_text(f"url=/bot{token}/getUpdates", secrets=())

