"""
Access Manager - Tiered access control with admin-only management

Tier Definitions:
- FREE: 3 coins max (BTC/ETH only), 1 request per 2 hours, basic analysis
- PREMIUM: 100 coins, 1 request per minute, full AI synthesis
- BETA: Same as premium + experimental features
- OWNER: Unlimited access + admin commands

Admin Commands:
- /grant <user_id> <tier> <days> - Grant access
- /revoke <user_id> - Revoke access
- /user_stats - Show active users
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


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
        'allowed_coins': ['bitcoin', 'ethereum'],  # BTC/ETH only
        'request_cooldown': 7200,  # 1 request per 2 hours (seconds)
        'features': ['basic_analysis', 'portfolio', 'news'],  # Basic features work
        'ai_synthesis': False,  # AI synthesis requires premium
        'patterns': False  # Pattern detection requires premium
    }

    PREMIUM = {
        'max_coins': 100,
        'allowed_coins': None,  # All coins
        'request_cooldown': 60,  # 1 request per minute
        'features': ['basic_analysis', 'ai_synthesis', 'patterns', 'alerts'],
        'ai_synthesis': True,
        'patterns': True
    }

    BETA = {
        'max_coins': 100,
        'allowed_coins': None,
        'request_cooldown': 60,
        'features': ['basic_analysis', 'ai_synthesis', 'patterns', 'alerts', 'experimental'],
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

    Storage: data/access_control.json
    Format:
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

    def __init__(self):
        self.access_file = Path("data/access_control.json")
        self.access_file.parent.mkdir(parents=True, exist_ok=True)

        self.users = self._load_access_data()
        self.request_tracker: Dict[int, float] = {}  # user_id -> last_request_timestamp

    def _load_access_data(self) -> Dict:
        """Load from data/access_control.json"""
        if self.access_file.exists():
            try:
                with open(self.access_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading access data: {e}")

        return {
            "users": {},  # user_id -> {tier, granted_date, expires_date}
            "access_log": []
        }

    def _save_access_data(self):
        """Persist access data to JSON"""
        try:
            with open(self.access_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            logger.debug("Access data saved")
        except Exception as e:
            logger.error(f"Error saving access data: {e}")

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

    def check_access(self, user_id: int, feature: str) -> Tuple[bool, str]:
        """
        Check if user has access to feature

        Args:
            user_id: Telegram user ID
            feature: Feature name (e.g., 'ai_synthesis', 'patterns')

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        tier = self.get_user_tier(user_id)
        limits = getattr(AccessLimits, tier.name)

        # Check feature access
        features = limits.get('features', [])
        if 'all' not in features and feature not in features:
            return False, f"Feature '{feature}' requires {AccessTier.PREMIUM.value.upper()} tier"

        # Check request cooldown
        last_request = self.request_tracker.get(user_id, 0)
        cooldown = limits.get('request_cooldown', 0)

        if cooldown > 0:
            time_since_last = time.time() - last_request
            if time_since_last < cooldown:
                remaining = int(cooldown - time_since_last)
                minutes = remaining // 60
                seconds = remaining % 60

                if minutes > 0:
                    time_str = f"{minutes}m {seconds}s"
                else:
                    time_str = f"{seconds}s"

                return False, f"Rate limit: wait {time_str} (cooldown: {cooldown//60}min)"

        return True, "OK"

    def record_request(self, user_id: int):
        """Record successful request for rate limiting"""
        self.request_tracker[user_id] = time.time()
        logger.debug(f"Recorded request for user {user_id}")

    def check_coin_limit(self, user_id: int, coin_id: str) -> Tuple[bool, str]:
        """
        Check if user can analyze this coin

        Args:
            user_id: Telegram user ID
            coin_id: Coin identifier

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        tier = self.get_user_tier(user_id)
        limits = getattr(AccessLimits, tier.name)

        # Check allowed coins
        allowed_coins = limits.get('allowed_coins')
        if allowed_coins is not None:
            if coin_id not in allowed_coins:
                return False, (
                    f"Coin '{coin_id}' not available in {tier.value.upper()} tier. "
                    f"Available: {', '.join(allowed_coins)}. "
                    f"Upgrade to premium: /premium"
                )

        return True, "OK"

    def grant_access(
        self,
        admin_id: int,
        target_user_id: int,
        tier: str,
        days: int,
        extend: bool = False
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
            return f"❌ Invalid tier: {tier}. Valid: free, premium, beta, owner"

        # Calculate expiration
        granted_date = datetime.now().timestamp()

        if days > 0:
            base = granted_date
            if extend:
                # Paid renewals must ADD to the remaining time, not overwrite it
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

        # Update access
        if "users" not in self.users:
            self.users["users"] = {}

        self.users["users"][str(target_user_id)] = {
            "tier": tier_enum.value,
            "granted_date": granted_date,
            "expires_date": expires_date,
            "granted_by": admin_id
        }

        # Log
        self.users.setdefault("access_log", []).append({
            "timestamp": granted_date,
            "action": "grant",
            "admin_id": admin_id,
            "target_user_id": target_user_id,
            "tier": tier_enum.value,
            "days": days
        })

        self._save_access_data()

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

        if str(target_user_id) in self.users.get("users", {}):
            del self.users["users"][str(target_user_id)]

            # Log
            self.users.setdefault("access_log", []).append({
                "timestamp": datetime.now().timestamp(),
                "action": "revoke",
                "admin_id": admin_id,
                "target_user_id": target_user_id
            })

            self._save_access_data()
            logger.info(f"Admin {admin_id} revoked access for user {target_user_id}")

            return f"✅ Revoked access for user {target_user_id}"
        else:
            return f"❌ User {target_user_id} has no special access"

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
        limits = getattr(AccessLimits, tier.name)
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
