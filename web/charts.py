"""What the charts are drawn from.

Big charts (candles, balances) run in the browser on TradingView Lightweight Charts, served by the site itself
(static/vendor): the page carries their data as JSON in a data- attribute, so the CSP stays 'self' only. Small
ones (sparklines, the panic hours) are SVG drawn here. Times go to the browser already shifted to Berlin.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")

# identity colours (validated for the dark surface with the dataviz checks: CVD ΔE ≥ 17, L 0.48–0.67)
SERIES = {"account": "#e8edf6", "nostop": "#d9732d", "selector": "#2c9fd0", "c1": "#a76cf0"}
SOURCE = {"паника": "nostop", "паника без стопа": "nostop", "модель выбора": "selector"}
UP, DOWN, INK, MUTED, WARN = "#3dae5b", "#f6465d", "#e8edf6", "#6c7a91", "#f5b941"


def local(ms: int) -> int:
    """Epoch seconds moved by Berlin's offset: the chart library shows UTC, the owner reads Berlin."""
    dt = datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone(BERLIN)
    return int(ms / 1000 + dt.utcoffset().total_seconds())


def attr(payload: dict) -> str:
    """JSON for a data- attribute (escaped, so no value can leave the attribute)."""
    return html.escape(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), quote=True)


def series_of(source: str) -> str:
    return SOURCE.get(source, "account")


def candles(rows: list[tuple], trades: list[dict], book: Optional[str] = None) -> dict:
    """Hourly candles, a marker at every entry and exit, and the levels of a single trade."""
    data = [{"time": local(ts), "open": o, "high": h, "low": low, "close": c} for ts, o, h, low, c in rows]
    first, last = (rows[0][0], rows[-1][0]) if rows else (0, 0)
    markers, lines = [], []
    for t in trades:
        moment, side = t.get("moment"), t.get("side", 1)
        colour = SERIES[series_of(t.get("source", ""))] if book is None else INK
        if moment and first - 3_600_000 <= moment <= last + 3_600_000:
            markers.append({"time": local(_floor(moment)), "position": "belowBar" if side > 0 else "aboveBar",
                            "color": colour, "shape": "arrowUp" if side > 0 else "arrowDown",
                            "text": ("лонг" if side > 0 else "шорт") + (f" · {t['base']}" if book is None else "")})
        end = t.get("exit_ts")
        if end and first <= end <= last + 3_600_000 and t.get("status") != "open":
            won = (t.get("net_pct") or 0) > 0
            markers.append({"time": local(_floor(end - 3_600_000)), "position": "aboveBar" if side > 0 else "belowBar",
                            "color": UP if won else DOWN, "shape": "circle",
                            "text": f"выход {t.get('net_pct', 0):+.1f}%"})
    if book is not None and trades:
        t = trades[0]
        for key, name, colour, style in (("entry", "вход", INK, 2), ("stop", "стоп", DOWN, 1),
                                         ("half", "половина", WARN, 3), ("target", "цель", UP, 1)):
            if t.get(key):
                lines.append({"price": t[key], "color": colour, "title": name, "style": style})
    markers.sort(key=lambda m: m["time"])
    return {"kind": "candles", "data": data, "markers": markers, "lines": lines}


def _floor(ms: int) -> int:
    return ms - ms % 3_600_000


def balance(points: list[tuple[int, float]], base: float = 100.0) -> dict:
    """A balance against where it started: above the start in green, below in red."""
    seen, data = set(), []
    for ms, value in points:
        t = local(ms)
        if t not in seen:
            seen.add(t)
            data.append({"time": t, "value": round(value, 4)})
    return {"kind": "balance", "base": base, "data": data}


def compare(books: dict[str, list[tuple[int, float]]], labels: dict[str, str], base: float = 100.0) -> dict:
    """Each book's return since its start, in %: one axis, one colour per book."""
    out = []
    for key, points in books.items():
        seen, data = set(), []
        for ms, value in points:
            t = local(ms)
            if t not in seen:
                seen.add(t)
                data.append({"time": t, "value": round((value / base - 1) * 100, 3)})
        if len(data) > 1:
            out.append({"key": key, "label": labels.get(key, key), "color": SERIES.get(key, INK), "data": data})
    return {"kind": "compare", "series": out}


def line_payload(points: list[tuple[int, float]], label: str, colour: str = INK) -> dict:
    seen, data = set(), []
    for ms, value in points:
        t = local(ms)
        if t not in seen and value is not None:
            seen.add(t)
            data.append({"time": t, "value": value})
    return {"kind": "line", "label": label, "color": colour, "data": data}


def bars_payload(points: list[tuple[int, float]], label: str, colour: str, threshold: Optional[float] = None) -> dict:
    return {"kind": "bars", "label": label, "color": colour, "threshold": threshold,
            "data": [{"time": local(ms), "value": v} for ms, v in points]}


def spark(values: list[float], key: str = "account", w: int = 120, h: int = 34) -> str:
    """A tiny line of a balance (no axes: the number beside it says the value)."""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    pts = " ".join(f"{i * (w - 2) / (len(values) - 1) + 1:.1f},{h - 3 - (v - lo) / span * (h - 6):.1f}"
                   for i, v in enumerate(values))
    return (f'<svg class="spark s-{key}" viewBox="0 0 {w} {h}" aria-hidden="true">'
            f'<polyline points="{pts}" pathLength="1"/></svg>')


def panic_hours(hours: list[dict], need: int = 4) -> str:
    """Today's 24 hours as cells: the hour, how many coins were in panic (deeper orange = more, lit from ``need``),
    a yellow mark on a nervous hour, hours still to come dashed."""
    by_hour = {datetime.fromtimestamp(x["hour"] / 1000, timezone.utc).astimezone(BERLIN).hour: x for x in hours}
    now = datetime.now(BERLIN).hour
    cells = []
    for hr in range(24):
        x = by_hour.get(hr)
        if x is None:
            cells.append(f'<span class="hc {"future" if hr > now else "none"}"><i>{hr:02d}</i><b>·</b></span>')
            continue
        n = len(x.get("panic") or [])
        names = ", ".join(map(str, (x.get("panic") or [])[:8])) or "никого"
        level = min(n, need)
        tip = f"{hr:02d}:00 — в панике {n}: {names}" + (" · рынок нервный" if x.get("nervous") else "")
        cells.append(f'<span class="hc l{level}{" nerv" if x.get("nervous") else ""}" title="{html.escape(tip)}">'
                     f'<i>{hr:02d}</i><b>{n}</b></span>')
    return f'<div class="hcells" role="img" aria-label="монет в панике по часам сегодня">{"".join(cells)}</div>'
