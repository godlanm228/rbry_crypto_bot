from __future__ import annotations

from typing import Dict, Iterable, Optional

def get_price_from_app(app, coin_id: str) -> Optional[float]:
    """Return current USD price from MarketSnapshotService if available."""
    try:
        snap = app.bot_data.get("market_snapshot") if app else None
        if not snap:
            return None
        s = snap.get_latest()
        if not s:
            return None
        row = s.markets.get(coin_id)
        return row.get("current_price") if row else None
    except Exception:
        return None

def get_prices_map(app, coin_ids: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        snap = app.bot_data.get("market_snapshot") if app else None
        s = snap.get_latest() if snap else None
        rows = s.markets if s else {}
        for cid in coin_ids:
            v = rows.get(cid, {}).get("current_price")
            if v is not None:
                out[cid] = v
    except Exception:
        pass
    return out
