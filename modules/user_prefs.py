from __future__ import annotations
import json, os
from pathlib import Path

PREFS_FILE = Path("data/user_prefs.json")
PREFS_FILE.parent.mkdir(exist_ok=True)

SUPPORTED_TIMEZONES = [
    "Europe/Berlin", "Europe/Kyiv", "Europe/Moscow",
    "Europe/London", "UTC", "America/New_York",
]
SUPPORTED_LANGS = ["ru", "uk", "de", "en"]

# In-memory cache: t() calls get_language() on every translated string, so an
# uncached _load() means dozens of disk reads per bot message.
_cache: dict | None = None
_cache_mtime: float | None = None


def _load():
    global _cache, _cache_mtime
    if not PREFS_FILE.exists():
        return {}
    try:
        mtime = PREFS_FILE.stat().st_mtime
        if _cache is not None and _cache_mtime == mtime:
            return _cache
        with open(PREFS_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f)
        _cache_mtime = mtime
        return _cache
    except Exception:
        return _cache or {}

def _save(data):
    global _cache, _cache_mtime
    with open(PREFS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    _cache = data
    try:
        _cache_mtime = PREFS_FILE.stat().st_mtime
    except Exception:
        _cache_mtime = None

def get_profile(user_id: int) -> dict:
    db = _load()
    key = str(user_id)
    if key not in db:
        db[key] = {"tz": None, "lang": "ru", "show_explanations": True}
        _save(db)
        return db[key]

    # Backfill defaults for older profiles
    updated = False
    if "tz" not in db[key]:
        db[key]["tz"] = None
        updated = True
    if "lang" not in db[key]:
        db[key]["lang"] = "ru"
        updated = True
    if "show_explanations" not in db[key]:
        db[key]["show_explanations"] = True
        updated = True
    if updated:
        _save(db)
    return db[key]

def set_timezone(user_id: int, tz: str):
    db = _load()
    db.setdefault(str(user_id), {"tz": None, "lang": "ru", "show_explanations": True})
    db[str(user_id)]["tz"] = tz
    _save(db)

def set_language(user_id: int, lang: str):
    db = _load()
    db.setdefault(str(user_id), {"tz": None, "lang": "ru", "show_explanations": True})
    db[str(user_id)]["lang"] = lang
    _save(db)

def get_timezone(user_id: int) -> str | None:
    return get_profile(user_id).get("tz")

def get_language(user_id: int) -> str:
    return get_profile(user_id).get("lang", "ru")

def set_ai_debug(user_id: int, enabled: bool):
    db = _load()
    db.setdefault(str(user_id), {"tz": None, "lang": "ru", "show_explanations": True})
    db[str(user_id)]["ai_debug"] = bool(enabled)
    _save(db)

def get_ai_debug(user_id: int) -> bool:
    return bool(get_profile(user_id).get("ai_debug", False))

def set_show_explanations(user_id: int, enabled: bool):
    db = _load()
    db.setdefault(str(user_id), {"tz": None, "lang": "ru", "show_explanations": True})
    db[str(user_id)]["show_explanations"] = bool(enabled)
    _save(db)

def get_show_explanations(user_id: int) -> bool:
    return bool(get_profile(user_id).get("show_explanations", True))
