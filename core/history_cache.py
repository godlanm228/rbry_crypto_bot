import time
from typing import Dict, Tuple, Any, Optional

# key: (coin_id, days)
_cache: Dict[Tuple[str, int], Tuple[float, Any]] = {}
TTL_SECONDS = 180  # 3 минуты — хватает, чтобы не долбить API в UI-кликах

def get(coin_id: str, days: int) -> Optional[Any]:
    key = (coin_id, days)
    item = _cache.get(key)
    if not item:
        return None
    ts, data = item
    if time.time() - ts > TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return data

def set(coin_id: str, days: int, data: Any) -> None:
    _cache[(coin_id, days)] = (time.time(), data)

