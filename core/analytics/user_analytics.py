"""
User Analytics - Automatic tracking of user activity and usage statistics

Features:
- Auto-register users on first interaction
- Log all user actions (what, when, which function)
- Usage statistics (most popular functions)
- User list with activity history
"""

import atexit
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# Rewriting the whole JSON on every button press is a disk-IO tax on each
# user interaction; batch writes instead and flush on exit.
SAVE_INTERVAL_SECONDS = 15


class UserAnalytics:
    """
    Track user activity and provide usage statistics

    Storage: data/user_analytics.json
    Format:
    {
        "users": {
            "user_id": {
                "first_seen": timestamp,
                "last_seen": timestamp,
                "username": "optional",
                "first_name": "optional",
                "language_code": "en",
                "total_requests": 123
            }
        },
        "activity_log": [
            {
                "user_id": 123,
                "timestamp": 1234567890.123,
                "action": "analyze_coin",
                "details": {"coin": "BTC"}
            }
        ]
    }
    """

    def __init__(self):
        self.analytics_file = Path("data/user_analytics.json")
        self.data = self._load_data()
        self._dirty = False
        self._last_save = 0.0
        atexit.register(self.flush)
        logger.info("User analytics initialized")

    def _load_data(self) -> Dict:
        """Load analytics data from JSON file"""
        if self.analytics_file.exists():
            try:
                with open(self.analytics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load user analytics: {e}")
                return {"users": {}, "activity_log": []}
        return {"users": {}, "activity_log": []}

    def _save_data(self):
        """Mark data dirty and persist at most once per SAVE_INTERVAL_SECONDS."""
        self._dirty = True
        now = time.time()
        if now - self._last_save < SAVE_INTERVAL_SECONDS:
            return
        self.flush()

    def flush(self):
        """Write analytics to disk immediately (called on interval and at exit)."""
        if not self._dirty:
            return
        try:
            self.analytics_file.parent.mkdir(exist_ok=True)
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            self._dirty = False
            self._last_save = time.time()
        except Exception as e:
            logger.error(f"Failed to save user analytics: {e}")

    def register_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        language_code: Optional[str] = None
    ):
        """
        Register user (auto-called on first interaction)

        Args:
            user_id: Telegram user ID
            username: Telegram username (optional)
            first_name: User's first name (optional)
            language_code: User's language code (optional)
        """
        user_id_str = str(user_id)
        now = datetime.now().timestamp()

        if user_id_str not in self.data["users"]:
            # New user - register
            self.data["users"][user_id_str] = {
                "first_seen": now,
                "last_seen": now,
                "username": username,
                "first_name": first_name,
                "language_code": language_code or "en",
                "total_requests": 0
            }
            logger.info(f"New user registered: {user_id} (@{username})")
        else:
            # Existing user - update info
            user_data = self.data["users"][user_id_str]
            user_data["last_seen"] = now

            # Update username/name if changed
            if username and username != user_data.get("username"):
                user_data["username"] = username
            if first_name and first_name != user_data.get("first_name"):
                user_data["first_name"] = first_name
            if language_code and language_code != user_data.get("language_code"):
                user_data["language_code"] = language_code

        self._save_data()

    def log_activity(
        self,
        user_id: int,
        action: str,
        details: Optional[Dict] = None
    ):
        """
        Log user activity

        Args:
            user_id: Telegram user ID
            action: Action name (e.g., "analyze_coin", "view_portfolio", "check_news")
            details: Optional details dict (e.g., {"coin": "BTC"})
        """
        user_id_str = str(user_id)
        now = datetime.now().timestamp()

        # Increment total requests
        if user_id_str in self.data["users"]:
            self.data["users"][user_id_str]["total_requests"] += 1
            self.data["users"][user_id_str]["last_seen"] = now

        # Log activity
        activity_entry = {
            "user_id": user_id,
            "timestamp": now,
            "action": action,
            "details": details or {}
        }

        self.data["activity_log"].append(activity_entry)

        # Keep only last 10000 log entries to prevent file bloat
        if len(self.data["activity_log"]) > 10000:
            self.data["activity_log"] = self.data["activity_log"][-10000:]

        self._save_data()

    def get_user_list(self, limit: int = 100) -> List[Dict]:
        """
        Get list of all users with their activity stats

        Args:
            limit: Maximum number of users to return

        Returns:
            List of user dicts sorted by last_seen (most recent first)
        """
        users = []

        for user_id_str, user_data in self.data["users"].items():
            users.append({
                "user_id": int(user_id_str),
                "username": user_data.get("username"),
                "first_name": user_data.get("first_name"),
                "first_seen": user_data.get("first_seen"),
                "last_seen": user_data.get("last_seen"),
                "total_requests": user_data.get("total_requests", 0),
                "language_code": user_data.get("language_code", "en")
            })

        # Sort by last_seen (most recent first)
        users.sort(key=lambda x: x["last_seen"], reverse=True)

        return users[:limit]

    def get_user_activity(self, user_id: int, limit: int = 50) -> List[Dict]:
        """
        Get activity log for specific user

        Args:
            user_id: Telegram user ID
            limit: Maximum number of activities to return

        Returns:
            List of activity dicts (most recent first)
        """
        user_activities = [
            activity for activity in self.data["activity_log"]
            if activity["user_id"] == user_id
        ]

        # Sort by timestamp (most recent first)
        user_activities.sort(key=lambda x: x["timestamp"], reverse=True)

        return user_activities[:limit]

    def get_usage_statistics(self) -> Dict:
        """
        Get usage statistics (most popular functions)

        Returns:
            Dict with statistics:
            {
                "total_users": 150,
                "total_requests": 5000,
                "active_today": 25,
                "action_counts": {"analyze_coin": 1500, "view_portfolio": 800, ...},
                "top_actions": [("analyze_coin", 1500), ...]
            }
        """
        now = datetime.now().timestamp()
        today_start = now - 86400  # 24 hours ago

        # Count actions
        action_counts = defaultdict(int)
        active_today = set()

        for activity in self.data["activity_log"]:
            action_counts[activity["action"]] += 1

            if activity["timestamp"] > today_start:
                active_today.add(activity["user_id"])

        # Sort actions by count
        top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_users": len(self.data["users"]),
            "total_requests": sum(
                user.get("total_requests", 0)
                for user in self.data["users"].values()
            ),
            "active_today": len(active_today),
            "action_counts": dict(action_counts),
            "top_actions": top_actions[:10]  # Top 10 most used functions
        }

    def get_analytics_report(self) -> str:
        """
        Generate formatted analytics report for owner

        Returns:
            Formatted markdown report
        """
        stats = self.get_usage_statistics()

        report = "📊 **АНАЛИТИКА ПОЛЬЗОВАТЕЛЕЙ**\n\n"

        report += f"👥 **Всего пользователей:** {stats['total_users']}\n"
        report += f"📈 **Всего запросов:** {stats['total_requests']}\n"
        report += f"🔥 **Активных за 24ч:** {stats['active_today']}\n\n"

        report += "**🏆 ТОП ФУНКЦИЙ:**\n"
        for i, (action, count) in enumerate(stats['top_actions'], 1):
            # Translate action names to Russian
            action_names = {
                "analyze_coin": "Анализ монеты",
                "analyze_portfolio": "Анализ портфеля",
                "analyze_top10": "Анализ ТОП-10",
                "view_portfolio": "Просмотр портфеля",
                "view_stats": "Статистика",
                "view_movers": "Движения",
                "check_news": "Новости",
                "my_tier": "Мой тариф",
                "premium": "Премиум тарифы",
                "settings": "Настройки",
                "help": "Помощь"
            }

            action_display = action_names.get(action, action)
            report += f"{i}. {action_display}: {count} раз\n"

        return report


# Global singleton
_user_analytics: Optional[UserAnalytics] = None


def get_user_analytics() -> UserAnalytics:
    """Get global user analytics instance"""
    global _user_analytics
    if _user_analytics is None:
        _user_analytics = UserAnalytics()
    return _user_analytics

