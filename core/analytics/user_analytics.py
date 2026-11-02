"""
User Analytics - Automatic tracking of user activity and usage statistics

Features:
- Auto-register users on first interaction
- Log all user actions (what, when, which function)
- Usage statistics (most popular functions)
- User list with activity history
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from core.state_db import DEFAULT_STATE_DB_PATH, SQLiteStateDB

logger = logging.getLogger(__name__)


class UserAnalytics:
    """
    Track user activity and provide usage statistics

    Storage: data/bot_state.db
    Legacy migration source: data/user_analytics.json
    Legacy format:
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

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_STATE_DB_PATH
        self.analytics_file = (
            self.db_path.with_name("user_analytics.json")
            if db_path is not None
            else Path("data/user_analytics.json")
        )
        self.db = SQLiteStateDB(self.db_path)
        self._init_db()
        self._migrate_json_if_needed()
        logger.info("User analytics initialized")

    def _init_db(self):
        conn = self.db.conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_users (
                user_id INTEGER PRIMARY KEY,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                username TEXT,
                first_name TEXT,
                language_code TEXT,
                total_requests INTEGER NOT NULL DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '{}'
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_activity_user_ts
            ON analytics_activity(user_id, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_activity_ts
            ON analytics_activity(timestamp DESC)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state_migrations (
                source TEXT PRIMARY KEY,
                migrated_at REAL NOT NULL
            )
        """)
        conn.commit()

    def _migrate_json_if_needed(self):
        if not self.analytics_file.exists():
            return
        conn = self.db.conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM analytics_users")
        has_users = cursor.fetchone()["count"] > 0
        cursor.execute("SELECT COUNT(*) AS count FROM analytics_activity")
        has_activity = cursor.fetchone()["count"] > 0
        cursor.execute(
            "SELECT 1 FROM state_migrations WHERE source = ?",
            ("user_analytics_json",),
        )
        already_migrated = cursor.fetchone() is not None
        if already_migrated or has_users or has_activity:
            return

        try:
            with open(self.analytics_file, "r", encoding="utf-8") as f:
                legacy = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load legacy analytics for migration: {e!r}")
            return

        try:
            with conn:
                for user_id, data in legacy.get("users", {}).items():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO analytics_users (
                            user_id, first_seen, last_seen, username,
                            first_name, language_code, total_requests
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(user_id),
                            float(data.get("first_seen") or datetime.now().timestamp()),
                            float(data.get("last_seen") or datetime.now().timestamp()),
                            data.get("username"),
                            data.get("first_name"),
                            data.get("language_code") or "en",
                            int(data.get("total_requests") or 0),
                        ),
                    )

                for item in legacy.get("activity_log", []):
                    cursor.execute(
                        """
                        INSERT INTO analytics_activity (
                            user_id, timestamp, action, details
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            int(item.get("user_id")),
                            float(item.get("timestamp") or datetime.now().timestamp()),
                            item.get("action") or "unknown",
                            json.dumps(item.get("details") or {}, ensure_ascii=False),
                        ),
                    )

                cursor.execute(
                    "INSERT OR REPLACE INTO state_migrations (source, migrated_at) VALUES (?, ?)",
                    ("user_analytics_json", datetime.now().timestamp()),
                )
            logger.info(f"Migrated user analytics from {self.analytics_file} to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to migrate user analytics: {e}", exc_info=True)

    def flush(self):
        """Compatibility no-op: SQLite writes happen inside each transaction."""
        return None

    def close(self):
        """Close the SQLite connection owned by the current thread."""
        self.db.close()

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
        now = datetime.now().timestamp()

        conn = self.db.conn()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM analytics_users WHERE user_id = ?", (user_id,))
        is_new = cursor.fetchone() is None

        with conn:
            cursor.execute(
                """
                INSERT INTO analytics_users (
                    user_id, first_seen, last_seen, username,
                    first_name, language_code, total_requests
                ) VALUES (?, ?, ?, ?, ?, COALESCE(?, 'en'), 0)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    username = COALESCE(excluded.username, analytics_users.username),
                    first_name = COALESCE(excluded.first_name, analytics_users.first_name),
                    language_code = COALESCE(?, analytics_users.language_code)
                """,
                (user_id, now, now, username, first_name, language_code, language_code),
            )
            if is_new:
                logger.info(f"New user registered: {user_id} (@{username})")

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
        now = datetime.now().timestamp()

        details_json = json.dumps(details or {}, ensure_ascii=False)
        conn = self.db.conn()
        cursor = conn.cursor()
        with conn:
            # Be defensive: callers should register first, but analytics must not
            # create orphan activity or silently lose request counts if they do not.
            cursor.execute(
                """
                INSERT OR IGNORE INTO analytics_users (
                    user_id, first_seen, last_seen, language_code, total_requests
                ) VALUES (?, ?, ?, ?, 0)
                """,
                (user_id, now, now, "en"),
            )
            cursor.execute(
                """
                UPDATE analytics_users
                SET total_requests = total_requests + 1,
                    last_seen = ?
                WHERE user_id = ?
                """,
                (now, user_id),
            )
            cursor.execute(
                """
                INSERT INTO analytics_activity (user_id, timestamp, action, details)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    now,
                    action,
                    details_json,
                ),
            )
            cursor.execute("""
                DELETE FROM analytics_activity
                WHERE id NOT IN (
                    SELECT id FROM analytics_activity
                    ORDER BY id DESC
                    LIMIT 10000
                )
            """)

    def get_user_list(self, limit: int = 100) -> List[Dict]:
        """
        Get list of all users with their activity stats

        Args:
            limit: Maximum number of users to return

        Returns:
            List of user dicts sorted by last_seen (most recent first)
        """
        cursor = self.db.conn().cursor()
        cursor.execute(
            """
            SELECT user_id, username, first_name, first_seen, last_seen,
                   total_requests, language_code
            FROM analytics_users
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_user_activity(self, user_id: int, limit: int = 50) -> List[Dict]:
        """
        Get activity log for specific user

        Args:
            user_id: Telegram user ID
            limit: Maximum number of activities to return

        Returns:
            List of activity dicts (most recent first)
        """
        cursor = self.db.conn().cursor()
        cursor.execute(
            """
            SELECT user_id, timestamp, action, details
            FROM analytics_activity
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        activities = []
        for row in cursor.fetchall():
            item = dict(row)
            try:
                item["details"] = json.loads(item.get("details") or "{}")
            except Exception:
                item["details"] = {}
            activities.append(item)
        return activities

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

        cursor = self.db.conn().cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM analytics_users")
        total_users = cursor.fetchone()["count"]

        cursor.execute("SELECT COALESCE(SUM(total_requests), 0) AS total FROM analytics_users")
        total_requests = cursor.fetchone()["total"]

        cursor.execute(
            """
            SELECT COUNT(DISTINCT user_id) AS count
            FROM analytics_activity
            WHERE timestamp > ?
            """,
            (today_start,),
        )
        active_today = cursor.fetchone()["count"]

        cursor.execute("""
            SELECT action, COUNT(*) AS count
            FROM analytics_activity
            GROUP BY action
            ORDER BY count DESC
        """)
        rows = cursor.fetchall()
        action_counts = {row["action"]: row["count"] for row in rows}
        top_actions = [(row["action"], row["count"]) for row in rows[:10]]

        return {
            "total_users": total_users,
            "total_requests": total_requests,
            "active_today": active_today,
            "action_counts": action_counts,
            "top_actions": top_actions
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

