"""SQLite storage for risk profiles inside the existing bot_state.db.

Follows the payment-ledger idiom: connection per operation, WAL, explicit
schema creation. Corrupt stored values fail loudly through RiskProfile
validation instead of being silently repaired.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from core.risk.models import RiskProfile
from core.state_db import DEFAULT_STATE_DB_PATH


class RiskProfileRepository:
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
                CREATE TABLE IF NOT EXISTS risk_profiles (
                    user_id INTEGER PRIMARY KEY,
                    account_currency TEXT NOT NULL,
                    risk_per_trade_pct REAL NOT NULL,
                    max_single_coin_pct REAL NOT NULL,
                    max_speculative_pct REAL NOT NULL,
                    max_stablecoin_issuer_pct REAL NOT NULL,
                    daily_loss_limit_pct REAL,
                    weekly_loss_limit_pct REAL,
                    fee_pct REAL,
                    slippage_pct REAL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def save(self, profile: RiskProfile) -> RiskProfile:
        """Upsert the profile; returns the stored copy with updated_at set."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO risk_profiles (
                    user_id, account_currency, risk_per_trade_pct,
                    max_single_coin_pct, max_speculative_pct,
                    max_stablecoin_issuer_pct, daily_loss_limit_pct,
                    weekly_loss_limit_pct, fee_pct, slippage_pct, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    account_currency = excluded.account_currency,
                    risk_per_trade_pct = excluded.risk_per_trade_pct,
                    max_single_coin_pct = excluded.max_single_coin_pct,
                    max_speculative_pct = excluded.max_speculative_pct,
                    max_stablecoin_issuer_pct = excluded.max_stablecoin_issuer_pct,
                    daily_loss_limit_pct = excluded.daily_loss_limit_pct,
                    weekly_loss_limit_pct = excluded.weekly_loss_limit_pct,
                    fee_pct = excluded.fee_pct,
                    slippage_pct = excluded.slippage_pct,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.user_id,
                    profile.account_currency,
                    profile.risk_per_trade_pct,
                    profile.max_single_coin_pct,
                    profile.max_speculative_pct,
                    profile.max_stablecoin_issuer_pct,
                    profile.daily_loss_limit_pct,
                    profile.weekly_loss_limit_pct,
                    profile.fee_pct,
                    profile.slippage_pct,
                    now,
                ),
            )
        return RiskProfile(
            user_id=profile.user_id,
            account_currency=profile.account_currency,
            risk_per_trade_pct=profile.risk_per_trade_pct,
            max_single_coin_pct=profile.max_single_coin_pct,
            max_speculative_pct=profile.max_speculative_pct,
            max_stablecoin_issuer_pct=profile.max_stablecoin_issuer_pct,
            daily_loss_limit_pct=profile.daily_loss_limit_pct,
            weekly_loss_limit_pct=profile.weekly_loss_limit_pct,
            fee_pct=profile.fee_pct,
            slippage_pct=profile.slippage_pct,
            updated_at=now,
        )

    def get(self, user_id: int) -> Optional[RiskProfile]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM risk_profiles WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
        if row is None:
            return None
        return RiskProfile(
            user_id=row["user_id"],
            account_currency=row["account_currency"],
            risk_per_trade_pct=row["risk_per_trade_pct"],
            max_single_coin_pct=row["max_single_coin_pct"],
            max_speculative_pct=row["max_speculative_pct"],
            max_stablecoin_issuer_pct=row["max_stablecoin_issuer_pct"],
            daily_loss_limit_pct=row["daily_loss_limit_pct"],
            weekly_loss_limit_pct=row["weekly_loss_limit_pct"],
            fee_pct=row["fee_pct"],
            slippage_pct=row["slippage_pct"],
            updated_at=row["updated_at"],
        )

    def get_or_default(self, user_id: int) -> RiskProfile:
        """Stored profile, or conservative defaults (NOT persisted)."""
        return self.get(user_id) or RiskProfile.conservative_default(user_id)
