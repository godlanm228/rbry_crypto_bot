import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_OWNER_ID", "123456")

from modules.payments.payment_ledger import PaymentLedger
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


def test_payment_ledger_rejects_charge_id_metadata_collision(tmp_path):
    ledger = PaymentLedger(tmp_path / "payments.db")
    terms = {
        "telegram_charge_id": "same-charge",
        "provider_charge_id": "",
        "invoice_payload": "premium_sub_premium_1m_777",
        "user_id": 777,
        "plan_id": "premium_1m",
        "currency": "XTR",
        "total_amount": 100,
    }
    ledger.claim(**terms)

    with pytest.raises(ValueError, match="metadata mismatch"):
        ledger.claim(**{**terms, "user_id": 888})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("currency", "amount"),
    [("USD", 100), ("XTR", 99), ("XTR", 101)],
)
async def test_pre_checkout_rejects_wrong_currency_or_amount(tmp_path, currency, amount):
    manager = PaymentManager(payment_ledger=PaymentLedger(tmp_path / "payments.db"))
    query = SimpleNamespace(
        invoice_payload="premium_sub_premium_1m_777",
        currency=currency,
        total_amount=amount,
        from_user=SimpleNamespace(id=777),
        answer=AsyncMock(),
    )

    await manager.handle_pre_checkout_query(
        SimpleNamespace(pre_checkout_query=query),
        SimpleNamespace(),
    )

    query.answer.assert_awaited_once()
    assert query.answer.await_args.kwargs["ok"] is False


@pytest.mark.asyncio
async def test_successful_payment_is_applied_once_by_charge_id(tmp_path, monkeypatch):
    import core.access.access_manager as access_mod

    class FakeAccessManager:
        def __init__(self):
            self.grants = 0

        def grant_access(self, **kwargs):
            self.grants += 1
            return "ok"

        def get_user_tier(self, user_id):
            return access_mod.AccessTier.PREMIUM

    access = FakeAccessManager()
    monkeypatch.setattr(access_mod, "get_access_manager", lambda: access)

    manager = PaymentManager(payment_ledger=PaymentLedger(tmp_path / "payments.db"))
    payment = SimpleNamespace(
        invoice_payload="premium_sub_premium_1m_777",
        currency="XTR",
        total_amount=100,
        telegram_payment_charge_id="tg-charge-1",
        provider_payment_charge_id="",
    )
    message = SimpleNamespace(successful_payment=payment, reply_text=AsyncMock())
    update = SimpleNamespace(
        message=message,
        effective_user=SimpleNamespace(id=777),
    )

    await manager.handle_successful_payment(update, SimpleNamespace())
    await manager.handle_successful_payment(update, SimpleNamespace())

    assert access.grants == 1
    assert manager.payment_ledger.get("tg-charge-1")["status"] == "applied"


@pytest.mark.asyncio
async def test_interrupted_payment_finalization_resumes_without_double_grant(tmp_path, monkeypatch):
    import core.access.access_manager as access_mod

    class RetryLedger(PaymentLedger):
        def __init__(self, path):
            super().__init__(path)
            self.fail_once = True

        def mark_applied(self, charge_id):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("temporary ledger failure")
            return super().mark_applied(charge_id)

    class IdempotentAccessManager:
        def __init__(self):
            self.keys = set()
            self.grants = 0

        def grant_access(self, **kwargs):
            key = kwargs["idempotency_key"]
            if key not in self.keys:
                self.keys.add(key)
                self.grants += 1
            return "ok"

        def get_user_tier(self, user_id):
            return access_mod.AccessTier.PREMIUM

    access = IdempotentAccessManager()
    monkeypatch.setattr(access_mod, "get_access_manager", lambda: access)
    ledger = RetryLedger(tmp_path / "payments.db")
    manager = PaymentManager(payment_ledger=ledger)
    payment = SimpleNamespace(
        invoice_payload="premium_sub_premium_1m_777",
        currency="XTR",
        total_amount=100,
        telegram_payment_charge_id="tg-charge-resume",
        provider_payment_charge_id="",
    )
    message = SimpleNamespace(successful_payment=payment, reply_text=AsyncMock())
    update = SimpleNamespace(message=message, effective_user=SimpleNamespace(id=777))
    context = SimpleNamespace()

    await manager.handle_successful_payment(update, context)
    assert ledger.get("tg-charge-resume")["status"] == "processing"
    await manager.handle_successful_payment(update, context)

    assert access.grants == 1
    assert ledger.get("tg-charge-resume")["status"] == "applied"
