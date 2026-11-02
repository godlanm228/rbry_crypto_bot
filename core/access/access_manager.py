"""
Access Manager - Tiered access control with admin-only management

Tier Definitions:
- FREE: 3 coins max (BTC/ETH/SOL), 1 request per 2 hours, basic analysis
- PREMIUM: 100 coins, 1 request per minute, full AI synthesis
- BETA: Same as premium + experimental features
- OWNER: Unlimited access + admin commands

Admin Commands:
- /grant <user_id> <tier> <days> - Grant access
- /revoke <user_id> - Revoke access
- /user_stats - Show active users
"""

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

from core.state_db import DEFAULT_STATE_DB_PATH, SQLiteStateDB

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    code: str = "ok"
    params: Dict[str, object] = field(default_factory=dict)


class AccessTier(Enum):
    """Access tier levels"""
    FREE = "free"
    PREMIUM = "premium"
    BETA = "beta"
    OWNER = "owner"


class AccessLimits:
    """Access limits for each tier"""

    FREE = {
        'max_coins': 3,
        'allowed_coins': ['bitcoin', 'ethereum', 'solana'],
        'request_cooldown': 7200,  # 1 request per 2 hours (seconds)
        'features': ['basic_analysis', 'portfolio', 'news'],  # Basic features work
        'ai_synthesis': False,  # AI synthesis requires premium
        'patterns': False  # Pattern detection requires premium
    }

    PREMIUM = {
        'max_coins': 100,
        'allowed_coins': None,  # All coins
        'request_cooldown': 60,  # 1 request per minute
        'features': ['basic_analysis', 'portfolio', 'news', 'top10_analysis', 'ai_synthesis', 'patterns', 'alerts'],
        'ai_synthesis': True,
        'patterns': True
    }

    BETA = {
        'max_coins': 100,
        'allowed_coins': None,
        'request_cooldown': 60,
        'features': ['basic_analysis', 'portfolio', 'news', 'top10_analysis', 'ai_synthesis', 'patterns', 'alerts', 'experimental'],
        'ai_synthesis': True,
        'patterns': True
    }

    OWNER = {
        'max_coins': None,  # Unlimited
        'allowed_coins': None,
        'request_cooldown': 0,  # No limits
        'features': ['all'],
        'admin_commands': True,
        'ai_synthesis': True,
        'patterns': True
    }


class AccessManager:
    """
    Access control manager with tiered permissions

    Storage: data/bot_state.db
    Legacy migration source: data/access_control.json
    Legacy format:
    {
        "users": {
            "user_id": {
                "tier": "premium",
                "granted_date": timestamp,
                "expires_date": timestamp or null,
                "granted_by": admin_user_id
            }
        },
        "access_log": [
            {
                "timestamp": timestamp,
                "action": "grant"/"revoke",
                "admin_id": user_id,
                "target_user_id": user_id,
                "tier": "premium",
                "days": 30
            }
        ]
    }
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.access_file = Path("data/access_control.json")
        self.access_file.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or DEFAULT_STATE_DB_PATH
        self.db = SQLiteStateDB(self.db_path)
        self._lock = threading.RLock()

        self._init_db()
        self.users = self._load_access_data()
        self.request_tracker: Dict[int, float] = {}  # user_id -> last_request_timestamp

    def close(self):
        """Close the SQLite connection owned by the current thread."""
        self.db.close()

    def _init_db(self):
        """Initialize SQLite schema for access control."""
        conn = self.db.conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_users (
                user_id INTEGER PRIMARY KEY,
                tier TEXT NOT NULL,
                granted_date REAL NOT NULL,
                expires_date REAL,
                granted_by INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                action TEXT NOT NULL,
                admin_id INTEGER,
                target_user_id INTEGER,
                tier TEXT,
                days INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state_migrations (
                source TEXT PRIMARY KEY,
                migrated_at REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_grant_idempotency (
                idempotency_key TEXT PRIMARY KEY,
                target_user_id INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.commit()

    def _migrate_json_if_needed(self):
        """Import legacy access_control.json once if SQLite has no access rows."""
        if not self.access_file.exists():
            return

        conn = self.db.conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM access_users")
        users_count = cursor.fetchone()["count"]
        cursor.execute("SELECT COUNT(*) AS count FROM access_log")
        log_count = cursor.fetchone()["count"]
        cursor.execute(
            "SELECT 1 FROM state_migrations WHERE source = ?",
            ("access_control_json",),
        )
        already_migrated = cursor.fetchone() is not None
        if already_migrated or users_count or log_count:
            return

        try:
            import json

            with open(self.access_file, "r", encoding="utf-8") as f:
                legacy = json.load(f)
        except Exception as e:
            logger.error(f"Error loading legacy access data for migration: {e!r}")
            return

        try:
            with conn:
                for user_id, data in legacy.get("users", {}).items():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO access_users (
                            user_id, tier, granted_date, expires_date, granted_by
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            int(user_id),
                            data.get("tier", "free"),
                            float(data.get("granted_date") or datetime.now().timestamp()),
                            data.get("expires_date"),
                            data.get("granted_by"),
                        ),
                    )

                for event in legacy.get("access_log", []):
                    cursor.execute(
                        """
                        INSERT INTO access_log (
                            timestamp, action, admin_id, target_user_id, tier, days
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            float(event.get("timestamp") or datetime.now().timestamp()),
                            event.get("action", "unknown"),
                            event.get("admin_id"),
                            event.get("target_user_id"),
                            event.get("tier"),
                            event.get("days"),
                        ),
                    )

                cursor.execute(
                    "INSERT OR REPLACE INTO state_migrations (source, migrated_at) VALUES (?, ?)",
                    ("access_control_json", datetime.now().timestamp()),
                )
            logger.info(f"Migrated access control from {self.access_file} to {self.db_path}")
        except Exception as e:
            logger.error(f"Error migrating access data to SQLite: {e}", exc_info=True)

    def _load_access_data(self) -> Dict:
        """Load access data from SQLite in the legacy in-memory shape."""
        self._migrate_json_if_needed()

        try:
            conn = self.db.conn()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT user_id, tier, granted_date, expires_date, granted_by
                FROM access_users
            """)
            users = {
                str(row["user_id"]): {
                    "tier": row["tier"],
                    "granted_date": row["granted_date"],
                    "expires_date": row["expires_date"],
                    "granted_by": row["granted_by"],
                }
                for row in cursor.fetchall()
            }

            cursor.execute("""
                SELECT timestamp, action, admin_id, target_user_id, tier, days
                FROM access_log
                ORDER BY id
            """)
            access_log = []
            for row in cursor.fetchall():
                event = {
                    "timestamp": row["timestamp"],
                    "action": row["action"],
                    "admin_id": row["admin_id"],
                    "target_user_id": row["target_user_id"],
                }
                if row["tier"] is not None:
                    event["tier"] = row["tier"]
                if row["days"] is not None:
                    event["days"] = row["days"]
                access_log.append(event)

            return {"users": users, "access_log": access_log}
        except Exception as e:
            logger.error(f"Error loading access data from SQLite: {e!r}")

        return {
            "users": {},  # user_id -> {tier, granted_date, expires_date}
            "access_log": []
        }

    def _persist_access_change(
        self,
        target_user_id: int,
        event: Dict,
        user_data: Optional[Dict] = None,
        idempotency_key: Optional[str] = None,
        extend_days: int = 0,
    ) -> bool:
        """Atomically persist one access change and its audit event.

        Returns ``False`` only when the supplied idempotency key was already
        committed. The marker and access update share a transaction.
        """
        conn = self.db.conn()
        cursor = conn.cursor()
        with conn:
            cursor.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                claimed = cursor.execute(
                    """
                    INSERT OR IGNORE INTO access_grant_idempotency (
                        idempotency_key, target_user_id, created_at
                    ) VALUES (?, ?, ?)
                    """,
                    (idempotency_key, target_user_id, event["timestamp"]),
                )
                if claimed.rowcount != 1:
                    existing = cursor.execute(
                        """
                        SELECT target_user_id FROM access_grant_idempotency
                        WHERE idempotency_key = ?
                        """,
                        (idempotency_key,),
                    ).fetchone()
                    if existing is None or existing["target_user_id"] != target_user_id:
                        raise ValueError("access grant idempotency metadata mismatch")
                    return False

            if user_data is not None and extend_days > 0:
                current = cursor.execute(
                    "SELECT tier, expires_date FROM access_users WHERE user_id = ?",
                    (target_user_id,),
                ).fetchone()
                current_expiry = current["expires_date"] if current else None
                if (
                    current is not None
                    and current["tier"] == user_data["tier"]
                    and current_expiry
                    and current_expiry > event["timestamp"]
                ):
                    user_data["expires_date"] = current_expiry + (extend_days * 86400)
                # Paid time never replaces permanent access (beta lifetime,
                # permanent premium, owner): buying one month used to cut a
                # lifetime grant down to 30 days.
                if (
                    current is not None
                    and current_expiry is None
                    and current["tier"] in (AccessTier.PREMIUM.value, AccessTier.BETA.value, AccessTier.OWNER.value)
                ):
                    user_data["tier"] = current["tier"]
                    user_data["expires_date"] = None

            if user_data is None:
                cursor.execute(
                    "DELETE FROM access_users WHERE user_id = ?",
                    (target_user_id,),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO access_users (
                        user_id, tier, granted_date, expires_date, granted_by
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        tier = excluded.tier,
                        granted_date = excluded.granted_date,
                        expires_date = excluded.expires_date,
                        granted_by = excluded.granted_by
                    """,
                    (
                        target_user_id,
                        user_data["tier"],
                        user_data["granted_date"],
                        user_data.get("expires_date"),
                        user_data.get("granted_by"),
                    ),
                )

            cursor.execute(
                """
                INSERT INTO access_log (
                    timestamp, action, admin_id, target_user_id, tier, days
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event["timestamp"],
                    event["action"],
                    event.get("admin_id"),
                    target_user_id,
                    event.get("tier"),
                    event.get("days"),
                ),
            )
        return True

    def _reload_user(self, user_id: int) -> Optional[Dict]:
        row = self.db.conn().execute(
            """
            SELECT tier, granted_date, expires_date, granted_by
            FROM access_users WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            self.users.setdefault("users", {}).pop(str(user_id), None)
            return None
        data = dict(row)
        self.users.setdefault("users", {})[str(user_id)] = data
        return data

    def has_permanent_access(self, user_id: int) -> bool:
        """Owner, or a premium/beta grant without an expiry date."""
        from config import OWNER_ID
        if user_id == OWNER_ID:
            return True
        user_data = self.users.get("users", {}).get(str(user_id)) or {}
        return (
            user_data.get("tier") in (AccessTier.PREMIUM.value, AccessTier.BETA.value, AccessTier.OWNER.value)
            and not user_data.get("expires_date")
        )

    def get_user_tier(self, user_id: int) -> AccessTier:
        """
        Get user's current access tier

        Args:
            user_id: Telegram user ID

        Returns:
            AccessTier enum
        """
        # OWNER always has OWNER tier (from config.py)
        from config import OWNER_ID
        if user_id == OWNER_ID:
            return AccessTier.OWNER

        user_data = self.users.get("users", {}).get(str(user_id))

        if not user_data:
            return AccessTier.FREE

        # Check expiration
        expires_date = user_data.get("expires_date")
        if expires_date:
            if datetime.now().timestamp() > expires_date:
                logger.info(f"User {user_id} access expired")
                return AccessTier.FREE

        tier_str = user_data.get("tier", "free")
        try:
            return AccessTier(tier_str)
        except ValueError:
            logger.warning(f"Invalid tier '{tier_str}' for user {user_id}, defaulting to FREE")
            return AccessTier.FREE

    def evaluate_access(self, user_id: int, feature: str) -> AccessDecision:
        """Return a structured access decision for presentation-layer localization."""
        feature_decision = self.evaluate_feature(user_id, feature)
        if not feature_decision.allowed:
            return feature_decision

        limits = self.get_limits(user_id)
        last_request = self.request_tracker.get(user_id, 0)
        cooldown = int(limits.get('request_cooldown', 0))
        remaining = max(0, math.ceil(cooldown - (time.time() - last_request)))
        if cooldown > 0 and remaining > 0:
            return AccessDecision(
                False,
                "cooldown",
                {
                    "minutes": remaining // 60,
                    "seconds": remaining % 60,
                },
            )
        return AccessDecision(True)

    @staticmethod
    def _legacy_reason(decision: AccessDecision) -> str:
        """Compatibility text for callers not yet migrated to structured decisions."""
        if decision.allowed:
            return "OK"
        if decision.code == "cooldown":
            return (
                f"Rate limit: wait {decision.params.get('minutes', 0)}m "
                f"{decision.params.get('seconds', 0)}s"
            )
        if decision.code == "coin_not_available":
            allowed = ", ".join(decision.params.get("allowed", ()))
            return f"Coin not available in this tier. Available: {allowed}"
        if decision.code == "portfolio_limit":
            return f"Portfolio limit reached ({decision.params.get('maximum')})"
        return "Feature requires PREMIUM tier"

    def check_access(self, user_id: int, feature: str) -> Tuple[bool, str]:
        """
        Check if user has access to feature

        Args:
            user_id: Telegram user ID
            feature: Feature name (e.g., 'ai_synthesis', 'patterns')

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        decision = self.evaluate_access(user_id, feature)
        return decision.allowed, self._legacy_reason(decision)

    def get_limits(self, user_id: int) -> Dict:
        """Return the single declarative policy for the user's effective tier."""
        tier = self.get_user_tier(user_id)
        return getattr(AccessLimits, tier.name)

    def evaluate_feature(self, user_id: int, feature: str) -> AccessDecision:
        features = self.get_limits(user_id).get('features', [])
        if 'all' in features or feature in features:
            return AccessDecision(True)
        return AccessDecision(False, "feature_requires_premium", {"feature": feature})

    def check_feature(self, user_id: int, feature: str) -> Tuple[bool, str]:
        """Check a feature without also consuming or evaluating rate limits."""
        decision = self.evaluate_feature(user_id, feature)
        return decision.allowed, self._legacy_reason(decision)

    def has_feature(self, user_id: int, feature: str) -> bool:
        return self.check_feature(user_id, feature)[0]

    def record_request(self, user_id: int):
        """Record successful request for rate limiting"""
        self.request_tracker[user_id] = time.time()
        logger.debug(f"Recorded request for user {user_id}")

    def evaluate_coin_limit(self, user_id: int, coin_id: str) -> AccessDecision:
        tier = self.get_user_tier(user_id)
        limits = self.get_limits(user_id)
        allowed_coins = limits.get('allowed_coins')
        if allowed_coins is not None and coin_id not in allowed_coins:
            return AccessDecision(
                False,
                "coin_not_available",
                {
                    "coin": coin_id,
                    "tier": tier.value.upper(),
                    "allowed": tuple(allowed_coins),
                },
            )
        return AccessDecision(True)

    def check_coin_limit(self, user_id: int, coin_id: str) -> Tuple[bool, str]:
        """
        Check if user can analyze this coin

        Args:
            user_id: Telegram user ID
            coin_id: Coin identifier

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        decision = self.evaluate_coin_limit(user_id, coin_id)
        return decision.allowed, self._legacy_reason(decision)

    def evaluate_portfolio_limit(
        self,
        user_id: int,
        existing_coin_ids,
        new_coin_id: str,
    ) -> AccessDecision:
        feature = self.evaluate_feature(user_id, "portfolio")
        if not feature.allowed:
            return feature
        coin = self.evaluate_coin_limit(user_id, new_coin_id)
        if not coin.allowed:
            return coin

        normalized = {str(coin_id).lower() for coin_id in existing_coin_ids}
        if new_coin_id.lower() in normalized:
            return AccessDecision(True)
        maximum = self.get_limits(user_id).get("max_coins")
        if maximum is not None and len(normalized) >= maximum:
            return AccessDecision(False, "portfolio_limit", {"maximum": maximum})
        return AccessDecision(True)

    def check_portfolio_limit(
        self,
        user_id: int,
        existing_coin_ids,
        new_coin_id: str,
    ) -> Tuple[bool, str]:
        """Enforce both allowed assets and unique portfolio size."""
        decision = self.evaluate_portfolio_limit(user_id, existing_coin_ids, new_coin_id)
        return decision.allowed, self._legacy_reason(decision)

    def grant_access(
        self,
        admin_id: int,
        target_user_id: int,
        tier: str,
        days: int,
        extend: bool = False,
        idempotency_key: Optional[str] = None,
    ) -> str:
        """
        Admin-only: Grant access to user

        Args:
            admin_id: Admin user ID (must be OWNER tier)
            target_user_id: Target user ID
            tier: Tier name ('free', 'premium', 'beta', 'owner')
            days: Days of access (0 = permanent)

        Returns:
            Result message
        """
        # Verify admin is OWNER
        if self.get_user_tier(admin_id) != AccessTier.OWNER:
            return "❌ Access denied: OWNER tier required"

        # Validate tier
        try:
            tier_enum = AccessTier(tier.lower())
        except ValueError:
            return f"❌ Invalid tier: {tier}. Valid: free, premium, beta"
        if tier_enum == AccessTier.OWNER:
            # Owner is the configured TELEGRAM_OWNER_ID only; a granted "owner"
            # tier could itself grant and revoke paid access.
            return "❌ The owner tier cannot be granted"

        if not isinstance(days, int) or isinstance(days, bool) or days < 0:
            return "❌ Days must be a non-negative integer"

        with self._lock:
            granted_date = datetime.now().timestamp()

            if days > 0:
                base = granted_date
                if extend:
                    # Paid renewals must ADD to the remaining time, not overwrite it.
                    existing = self.users.get("users", {}).get(str(target_user_id), {})
                    existing_expiry = existing.get("expires_date")
                    if (
                        existing.get("tier") == tier.lower()
                        and existing_expiry
                        and existing_expiry > granted_date
                    ):
                        base = existing_expiry
                expires_date = base + (days * 86400)
            else:
                expires_date = None

            user_data = {
                "tier": tier_enum.value,
                "granted_date": granted_date,
                "expires_date": expires_date,
                "granted_by": admin_id,
            }
            event = {
                "timestamp": granted_date,
                "action": "grant",
                "admin_id": admin_id,
                "target_user_id": target_user_id,
                "tier": tier_enum.value,
                "days": days,
            }

            # Commit durable state first. Never report a successful paid grant
            # that exists only in memory.
            applied = self._persist_access_change(
                target_user_id,
                event,
                user_data,
                idempotency_key=idempotency_key,
                extend_days=days if extend else 0,
            )
            if not applied:
                self._reload_user(target_user_id)
                logger.info(
                    "Idempotent access grant already applied key=%s user=%s",
                    idempotency_key,
                    target_user_id,
                )
                return f"✅ Access grant already applied for user {target_user_id}"
            self.users.setdefault("users", {})[str(target_user_id)] = user_data
            self.users.setdefault("access_log", []).append(event)

        expires_str = f"for {days} days" if days > 0 else "permanently"
        logger.info(f"Admin {admin_id} granted {tier_enum.value} to user {target_user_id} {expires_str}")

        return f"✅ Granted {tier_enum.value.upper()} access to user {target_user_id} {expires_str}"

    def revoke_access(self, admin_id: int, target_user_id: int) -> str:
        """
        Admin-only: Revoke user access

        Args:
            admin_id: Admin user ID (must be OWNER tier)
            target_user_id: Target user ID

        Returns:
            Result message
        """
        if self.get_user_tier(admin_id) != AccessTier.OWNER:
            return "❌ Access denied: OWNER tier required"

        with self._lock:
            if str(target_user_id) not in self.users.get("users", {}):
                return f"❌ User {target_user_id} has no special access"

            event = {
                "timestamp": datetime.now().timestamp(),
                "action": "revoke",
                "admin_id": admin_id,
                "target_user_id": target_user_id,
            }
            self._persist_access_change(target_user_id, event)
            del self.users["users"][str(target_user_id)]
            self.users.setdefault("access_log", []).append(event)

        logger.info(f"Admin {admin_id} revoked access for user {target_user_id}")
        return f"✅ Revoked access for user {target_user_id}"

    def get_user_stats(self, admin_id: int) -> str:
        """
        Admin-only: Get active users statistics

        Args:
            admin_id: Admin user ID (must be OWNER tier)

        Returns:
            Formatted statistics string
        """
        if self.get_user_tier(admin_id) != AccessTier.OWNER:
            return "❌ Access denied: OWNER tier required"

        users = self.users.get("users", {})

        stats = {
            "total_users": len(users),
            "by_tier": {},
            "active_premium": 0,
            "expired": 0
        }

        now = datetime.now().timestamp()

        for user_id, data in users.items():
            tier = data.get("tier", "free")
            stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + 1

            # Check expiration
            expires = data.get("expires_date")
            if expires and expires < now:
                stats["expired"] += 1
            elif tier in ["premium", "beta"]:
                stats["active_premium"] += 1

        output = "📊 **User Statistics**\n\n"
        output += f"Total users: {stats['total_users']}\n"
        output += f"Active premium: {stats['active_premium']}\n"
        output += f"Expired: {stats['expired']}\n\n"
        output += "**By tier:**\n"
        for tier, count in stats["by_tier"].items():
            output += f"  {tier.upper()}: {count}\n"

        return output

    def get_user_info(self, user_id: int) -> Dict:
        """
        Get user access information

        Args:
            user_id: Telegram user ID

        Returns:
            Dict with user info
        """
        tier = self.get_user_tier(user_id)
        limits = self.get_limits(user_id)
        user_data = self.users.get("users", {}).get(str(user_id), {})

        info = {
            "tier": tier.value,
            "limits": limits,
            "granted_date": user_data.get("granted_date"),
            "expires_date": user_data.get("expires_date")
        }

        return info


# Global singleton
_access_manager: Optional[AccessManager] = None


def get_access_manager() -> AccessManager:
    """Get global access manager instance"""
    global _access_manager
    if _access_manager is None:
        _access_manager = AccessManager()
    return _access_manager
