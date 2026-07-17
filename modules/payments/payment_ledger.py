"""Durable payment processing ledger.

Telegram may redeliver updates after reconnects or restarts.  The charge id is
the idempotency key: once a charge has been claimed, it must never extend access
again automatically.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.state_db import DEFAULT_STATE_DB_PATH


@dataclass(frozen=True)
class PaymentClaim:
    created: bool
    status: str


class PaymentLedger:
    def __init__(self, db_path: Path = DEFAULT_STATE_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS payment_ledger (
                    telegram_charge_id TEXT PRIMARY KEY,
                    provider_charge_id TEXT,
                    invoice_payload TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    plan_id TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    total_amount INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('processing', 'applied', 'failed')),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    failure_code TEXT
                )
                """
            )

    def claim(
        self,
        *,
        telegram_charge_id: str,
        provider_charge_id: str,
        invoice_payload: str,
        user_id: int,
        plan_id: str,
        currency: str,
        total_amount: int,
    ) -> PaymentClaim:
        """Atomically claim a Telegram charge for at-most-once application."""
        charge_id = str(telegram_charge_id or "").strip()
        if not charge_id:
            raise ValueError("telegram charge id is required")

        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO payment_ledger (
                    telegram_charge_id, provider_charge_id, invoice_payload,
                    user_id, plan_id, currency, total_amount, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', ?, ?)
                """,
                (
                    charge_id,
                    provider_charge_id or "",
                    invoice_payload,
                    int(user_id),
                    plan_id,
                    currency,
                    int(total_amount),
                    now,
                    now,
                ),
            )
            if cursor.rowcount == 1:
                return PaymentClaim(created=True, status="processing")

            row = conn.execute(
                """
                SELECT status, provider_charge_id, invoice_payload, user_id,
                       plan_id, currency, total_amount
                FROM payment_ledger WHERE telegram_charge_id = ?
                """,
                (charge_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("payment claim disappeared after conflict")
            expected = (
                provider_charge_id or "",
                invoice_payload,
                int(user_id),
                plan_id,
                currency,
                int(total_amount),
            )
            stored = (
                row["provider_charge_id"],
                row["invoice_payload"],
                row["user_id"],
                row["plan_id"],
                row["currency"],
                row["total_amount"],
            )
            if stored != expected:
                raise ValueError("telegram charge id metadata mismatch")
            return PaymentClaim(created=False, status=row["status"])

    def mark_applied(self, telegram_charge_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE payment_ledger
                SET status = 'applied', updated_at = ?, failure_code = NULL
                WHERE telegram_charge_id = ? AND status = 'processing'
                """,
                (time.time(), telegram_charge_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("payment ledger was not in processing state")

    def mark_failed(self, telegram_charge_id: str, failure_code: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE payment_ledger
                SET status = 'failed', updated_at = ?, failure_code = ?
                WHERE telegram_charge_id = ? AND status = 'processing'
                """,
                (time.time(), failure_code[:80], telegram_charge_id),
            )

    def get(self, telegram_charge_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM payment_ledger WHERE telegram_charge_id = ?",
                (telegram_charge_id,),
            ).fetchone()
            return dict(row) if row else None
