"""The allocator (plan item 6): how many of the account's five places each strategy gets.

A strategy's worth is its mean result a trade, believed from two sides: what history expects of it
(its prior) and what it did on paper since the account opened. Few paper trades: history speaks;
the more trades, the more paper does (PRIOR_TRADES paper trades weigh as much as history). Strategies
whose worth is positive share the places in proportion to it, the best first; the best of the
others keeps one place to explore, so nothing is written off for good and a strategy that starts to
earn can come back. A strategy that never passed the robustness check (the selector) explores until
its own paper trades vouch for it, whatever history's last years say. When nothing is worth anything, only that one place trades and the rest of the
account waits in USDT. With no paper trades yet the places follow history: the panic first, the
selector exploring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SLOTS = 5
PRIOR_TRADES = 20
EXPLORE = 1


@dataclass
class Source:
    label: str
    prior: float                                 # history's mean result a trade, as a share (0.02 = +2%)
    live: list = field(default_factory=list)     # its closed paper trades' results, shares
    proven: bool = True                          # passed the robustness check (core/brain/robust.py)


def worth(s: Source, n0: int = PRIOR_TRADES) -> float:
    """History and paper blended by their weights: n0 for history, one a paper trade."""
    live = [x for x in s.live if math.isfinite(x)]
    return (n0 * s.prior + sum(live)) / (n0 + len(live))


def allocate(sources: list[Source], slots: int = SLOTS, explore: int = EXPLORE) -> list[tuple[str, int, float]]:
    """[(label, places, worth)] best first: the positive share the places in proportion (largest
    remainders), the best of the rest keeps ``explore`` places."""
    scored = sorted(((s.label, worth(s)) for s in sources), key=lambda x: -x[1])
    # an unproven strategy explores until PRIOR_TRADES paper trades of its own say it earns
    trusted = {s.label for s in sources if s.proven or (len(s.live) >= PRIOR_TRADES and worth(s) > 0)}
    positive = [(l, w) for l, w in scored if w > 0 and l in trusted]
    rest = [(l, w) for l, w in scored if (l, w) not in positive]
    room = slots - (explore if rest else 0)
    places = {l: 0 for l, _ in scored}
    if positive and room > 0:
        total = sum(w for _, w in positive)
        raw = {l: room * w / total for l, w in positive}
        for l in raw:
            places[l] = int(raw[l])
        left = room - sum(places.values())
        for l, _ in sorted(raw.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:left]:
            places[l] += 1
    if rest:
        places[rest[0][0]] += explore
    return [(l, places[l], w) for l, w in scored]
