import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

from modules.payments.payment_manager import PaymentManager


def test_parse_payload_recovers_plan_after_restart():
    payment = PaymentManager.parse_payload("premium_sub_premium_1m_777")

    assert payment["user_id"] == 777
    assert payment["plan_id"] == "premium_1m"
    assert payment["plan"]["tier"] == "premium"


def test_parse_payload_rejects_invalid_payloads():
    assert PaymentManager.parse_payload("") is None
    assert PaymentManager.parse_payload("premium_sub_unknown_777") is None
    assert PaymentManager.parse_payload("premium_sub_premium_1m_notint") is None
    assert PaymentManager.parse_payload("other_premium_1m_777") is None
