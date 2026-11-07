"""Charts drawn on the server as SVG: no script, no library, nothing loaded from elsewhere (the site's CSP)."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")
W, H, PAD = 640, 220, 34


def _when(ms: int, fmt: str = "%d.%m %H:%M") -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone(BERLIN).strftime(fmt)


def _scale(lo: float, hi: float, a: float, b: float):
    span = hi - lo or 1.0
    return lambda v: a + (v - lo) / span * (b - a)


def _money(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", " ")
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.6g}"


def line(points: list[tuple[int, float]], title: str, base: Optional[float] = None) -> str:
    """A value through time (a balance): the line, its first and last value, ``base`` dashed (the start)."""
    if len(points) < 2:
        return '<p class="muted">Пока мало точек для графика.</p>'
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    lo, hi = min(ys + ([base] if base is not None else [])), max(ys + ([base] if base is not None else []))
    pad = (hi - lo) * 0.08 or 1.0
    fx, fy = _scale(xs[0], xs[-1], PAD, W - 8), _scale(lo - pad, hi + pad, H - 22, 10)
    path = " ".join(f"{'M' if i == 0 else 'L'}{fx(x):.1f},{fy(y):.1f}" for i, (x, y) in enumerate(points))
    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    if base is not None:
        parts.append(f'<line class="base" x1="{PAD}" x2="{W - 8}" y1="{fy(base):.1f}" y2="{fy(base):.1f}"/>')
    parts.append(f'<path class="{"up" if ys[-1] >= (base if base is not None else ys[0]) else "down"}" d="{path}"/>')
    parts += [f'<text class="axis" x="{PAD}" y="{H - 6}">{_when(xs[0])}</text>',
              f'<text class="axis" x="{W - 8}" y="{H - 6}" text-anchor="end">{_when(xs[-1])}</text>',
              f'<text class="axis" x="2" y="{fy(hi + pad) + 10:.1f}">{_money(hi)}</text>',
              f'<text class="axis" x="2" y="{fy(lo - pad):.1f}">{_money(lo)}</text></svg>']
    return "".join(parts)


def trade(rows: list[tuple], entry_ms: int, entry: float, exit_ms: Optional[int], levels: dict, side: int) -> str:
    """A trade on the coin's hourly closes: the entry and the exit (or now), the plan's prices as lines."""
    if len(rows) < 2:
        return '<p class="muted">Свечей для графика нет в базе.</p>'
    xs, closes = [r[0] for r in rows], [r[4] for r in rows]
    marks = [v for v in (entry, *levels.values()) if v]
    lo, hi = min(min(r[3] for r in rows), *marks), max(max(r[2] for r in rows), *marks)
    pad = (hi - lo) * 0.06 or 1.0
    fx, fy = _scale(xs[0], xs[-1], PAD + 18, W - 8), _scale(lo - pad, hi + pad, H - 22, 10)
    held = [(x, c) for x, c in zip(xs, closes) if x >= entry_ms and (exit_ms is None or x < exit_ms)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{fx(x):.1f},{fy(c):.1f}" for i, (x, c) in enumerate(zip(xs, closes)))
    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="график сделки">']
    if held:
        x0, x1 = fx(held[0][0]), fx(held[-1][0])
        parts.append(f'<rect class="held" x="{x0:.1f}" y="10" width="{max(x1 - x0, 2):.1f}" height="{H - 32}"/>')
    names = {"stop": "стоп", "half": "половина", "target": "цель"}
    for key, value in (("entry", entry), *levels.items()):
        if not value:
            continue
        y = fy(value)
        parts.append(f'<line class="lvl {key}" x1="{PAD + 18}" x2="{W - 8}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="lvl-t {key}" x="{W - 10}" y="{y - 3:.1f}" text-anchor="end">'
                     f'{names.get(key, "вход")} {_money(value)}</text>')
    parts.append(f'<path class="price" d="{path}"/>')
    ex = fx(entry_ms) if xs[0] <= entry_ms <= xs[-1] else None
    if ex is not None:
        parts.append(f'<circle class="dot entry" cx="{ex:.1f}" cy="{fy(entry):.1f}" r="4"/>')
    if exit_ms and xs[0] <= exit_ms - 3_600_000 <= xs[-1]:
        last = [c for x, c in zip(xs, closes) if x < exit_ms][-1]
        parts.append(f'<circle class="dot exit" cx="{fx(exit_ms - 3_600_000):.1f}" cy="{fy(last):.1f}" r="4"/>')
    parts += [f'<text class="axis" x="{PAD + 18}" y="{H - 6}">{_when(xs[0])}</text>',
              f'<text class="axis" x="{W - 8}" y="{H - 6}" text-anchor="end">{_when(xs[-1])}</text></svg>']
    return "".join(parts)
