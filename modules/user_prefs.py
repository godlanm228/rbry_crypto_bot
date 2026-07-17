from __future__ import annotations
import json
import logging
import threading
import time
from pathlib import Path

from core.state_db import SQLiteStateDB

logger = logging.getLogger(__name__)

PREFS_FILE = Path("data/user_prefs.json")
PREFS_FILE.parent.mkdir(exist_ok=True)
PREFS_DB_FILE: Path | None = None

SUPPORTED_TIMEZONES = [
    "Europe/Berlin", "Europe/Kyiv", "Europe/Moscow",
    "Europe/London", "UTC", "America/New_York",
]
SUPPORTED_LANGS = ["ru", "uk", "de", "en"]

# In-memory cache: t() calls get_language() on every translated string.
_cache: dict | None = None
_cache_mtime: float | None = None
_db: SQLiteStateDB | None = None
_db_path: Path | None = None
_lock = threading.RLock()


def _default_profile() -> dict:
    return {"tz": None, "lang": "ru", "show_explanations": True, "ai_debug": False}


def _get_db() -> SQLiteStateDB:
    global _db, _db_path
    path = PREFS_DB_FILE or PREFS_FILE.with_name("bot_state.db")
    if _db is None or _db_path != path:
        _db = SQLiteStateDB(path)
        _db_path = path
        _init_db()
        _migrate_json_if_needed()
    return _db


def _init_db():
    conn = _db.conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_prefs (
            user_id INTEGER PRIMARY KEY,
            tz TEXT,
            lang TEXT NOT NULL DEFAULT 'ru',
            show_explanations INTEGER NOT NULL DEFAULT 1,
            ai_debug INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS state_migrations (
            source TEXT PRIMARY KEY,
            migrated_at REAL NOT NULL
        )
    """)
    conn.commit()


def _migrate_json_if_needed():
    if not PREFS_FILE.exists():
        return
    conn = _db.conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS count FROM user_prefs")
    has_prefs = cursor.fetchone()["count"] > 0
    cursor.execute(
        "SELECT 1 FROM state_migrations WHERE source = ?",
        ("user_prefs_json",),
    )
    already_migrated = cursor.fetchone() is not None
    if has_prefs or already_migrated:
        return
    try:
        with open(PREFS_FILE, "r", encoding="utf-8") as f:
            legacy = json.load(f)
    except Exception:
        return

    now = time.time()
    with conn:
        for user_id, profile in legacy.items():
            data = _default_profile()
            data.update(profile or {})
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_prefs (
                    user_id, tz, lang, show_explanations, ai_debug, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(user_id),
                    data.get("tz"),
                    data.get("lang") or "ru",
                    1 if data.get("show_explanations", True) else 0,
                    1 if data.get("ai_debug", False) else 0,
                    now,
                ),
            )
        cursor.execute(
            "INSERT OR REPLACE INTO state_migrations (source, migrated_at) VALUES (?, ?)",
            ("user_prefs_json", now),
        )


def _load():
    global _cache, _cache_mtime
    if _cache is not None:
        return _cache
    try:
        conn = _get_db().conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, tz, lang, show_explanations, ai_debug
            FROM user_prefs
        """)
        _cache = {}
        for row in cursor.fetchall():
            _cache[str(row["user_id"])] = {
                "tz": row["tz"],
                "lang": row["lang"] or "ru",
                "show_explanations": bool(row["show_explanations"]),
                "ai_debug": bool(row["ai_debug"]),
            }
        _cache_mtime = None
        return _cache
    except Exception as exc:
        logger.error("Failed to load user preferences from SQLite: %s", exc)
        return _cache or {}


def _save_profile(user_id: int, profile: dict):
    """Persist one profile without rewriting other users' rows."""
    global _cache, _cache_mtime
    conn = _get_db().conn()
    cursor = conn.cursor()
    now = time.time()
    row = _default_profile()
    row.update(profile or {})
    with conn:
        cursor.execute(
            """
            INSERT INTO user_prefs (
                user_id, tz, lang, show_explanations, ai_debug, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                tz = excluded.tz,
                lang = excluded.lang,
                show_explanations = excluded.show_explanations,
                ai_debug = excluded.ai_debug,
                updated_at = excluded.updated_at
            """,
            (
                int(user_id),
                row.get("tz"),
                row.get("lang") or "ru",
                1 if row.get("show_explanations", True) else 0,
                1 if row.get("ai_debug", False) else 0,
                now,
            ),
        )
    if _cache is None:
        _cache = {}
    _cache[str(user_id)] = row
    _cache_mtime = None


def _fetch_profile(user_id: int) -> dict | None:
    """Fetch one row to handle profiles created after the cache was loaded."""
    cursor = _get_db().conn().cursor()
    cursor.execute(
        """
        SELECT tz, lang, show_explanations, ai_debug
        FROM user_prefs
        WHERE user_id = ?
        """,
        (user_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "tz": row["tz"],
        "lang": row["lang"] or "ru",
        "show_explanations": bool(row["show_explanations"]),
        "ai_debug": bool(row["ai_debug"]),
    }


def get_profile(user_id: int) -> dict:
    with _lock:
        db = _load()
        key = str(user_id)
        if key not in db:
            profile = _fetch_profile(user_id)
            if profile is None:
                profile = _default_profile()
                _save_profile(user_id, profile)
            else:
                db[key] = profile
            return profile

        # Backfill defaults for older profiles.
        profile = _default_profile()
        profile.update(db[key])
        if profile != db[key]:
            _save_profile(user_id, profile)
        return profile

def set_timezone(user_id: int, tz: str):
    with _lock:
        profile = get_profile(user_id).copy()
        profile["tz"] = tz
        _save_profile(user_id, profile)

def set_language(user_id: int, lang: str):
    with _lock:
        profile = get_profile(user_id).copy()
        profile["lang"] = lang
        _save_profile(user_id, profile)

def get_timezone(user_id: int) -> str | None:
    return get_profile(user_id).get("tz")

def get_language(user_id: int) -> str:
    return get_profile(user_id).get("lang", "ru")

def set_ai_debug(user_id: int, enabled: bool):
    with _lock:
        profile = get_profile(user_id).copy()
        profile["ai_debug"] = bool(enabled)
        _save_profile(user_id, profile)

def get_ai_debug(user_id: int) -> bool:
    return bool(get_profile(user_id).get("ai_debug", False))

def set_show_explanations(user_id: int, enabled: bool):
    with _lock:
        profile = get_profile(user_id).copy()
        profile["show_explanations"] = bool(enabled)
        _save_profile(user_id, profile)

def get_show_explanations(user_id: int) -> bool:
    return bool(get_profile(user_id).get("show_explanations", True))
