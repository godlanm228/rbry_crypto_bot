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

def _load():
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save(data):
    with open(PREFS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_profile(user_id: int) -> dict:
    db = _load()
    key = str(user_id)
    if key not in db:
        db[key] = {"tz": None, "lang": "ru"}
        _save(db)
    return db[key]

def set_timezone(user_id: int, tz: str):
    db = _load()
    db.setdefault(str(user_id), {"tz": None, "lang": "ru"})
    db[str(user_id)]["tz"] = tz
    _save(db)

def set_language(user_id: int, lang: str):
    db = _load()
    db.setdefault(str(user_id), {"tz": None, "lang": "ru"})
    db[str(user_id)]["lang"] = lang
    _save(db)

def get_timezone(user_id: int) -> str | None:
    return get_profile(user_id).get("tz")

def get_language(user_id: int) -> str:
    return get_profile(user_id).get("lang", "ru")

def set_ai_debug(user_id: int, enabled: bool):
    db = _load()
    db.setdefault(str(user_id), {"tz": None, "lang": "ru"})
    db[str(user_id)]["ai_debug"] = bool(enabled)
    _save(db)

def get_ai_debug(user_id: int) -> bool:
    return bool(get_profile(user_id).get("ai_debug", False))
