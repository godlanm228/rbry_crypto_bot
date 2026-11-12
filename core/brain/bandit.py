"""The RL experiment (plan item 10): an agent that learns by trading, trade by trade, from 2019.

A contextual bandit, the plain form of reinforcement learning when one decision does not change the
next situation: at every coin-hour of the event library the agent sees a coarse situation (BTC's trend
and mood, the market's breadth, the alts' season, the coin's first event) and picks one of the
selector's eight actions or none; its reward is the trade's net result, known only when the trade
closes, and only then does it learn from it. It picks by Thompson sampling: for each action a draw from
what it believes the action earns in this situation (the more trades behind a belief, the narrower),
taking the best draw if it is above zero, so it keeps trying what it is unsure of and drops what keeps
losing. A situation it knows too little about borrows the belief of the coarser one (event and
action, then the action alone). Booked as the selector's trades are: a $100 portfolio per year.
"""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

from core.brain import selector as se

PRIOR_N = 10                          # a belief starts as 10 trades of zero
PRIOR_SD = 0.06                       # a trade's net spread before anything is learned
MIN_N = 30                            # a situation needs this many closed trades before it speaks for itself
START = "2019-01-01"


def situation(c: pd.DataFrame) -> pd.DataFrame:
    """The coarse situation of every candidate, and its first event."""
    events = [col for col in se.EVENT_COLS if col in c]
    first = c[events].to_numpy().argmax(axis=1) if events else np.zeros(len(c), dtype=int)
    return pd.DataFrame({"fine": c["trend"].astype(str) + "|" + c["mood"].astype(str) + "|"
                                 + c.get("breadth_state", "").astype(str) + "|" + c.get("alt_state", "").astype(str)
                                 + "|" + np.asarray(events)[first],
                         "event": np.asarray(events)[first]}, index=c.index)


class Beliefs:
    """Sums of rewards per key, a running mean and spread with a zero prior."""

    def __init__(self):
        self.n, self.s, self.ss = {}, {}, {}

    def add(self, key, r: float) -> None:
        self.n[key] = self.n.get(key, 0) + 1
        self.s[key] = self.s.get(key, 0.0) + r
        self.ss[key] = self.ss.get(key, 0.0) + r * r

    def draw(self, keys: list, rng) -> float:
        """A draw from the first key with enough trades, else the last (the coarsest)."""
        for key in keys:
            if self.n.get(key, 0) >= MIN_N or key is keys[-1]:
                n, s, ss = self.n.get(key, 0), self.s.get(key, 0.0), self.ss.get(key, 0.0)
                mean = s / (n + PRIOR_N)
                var = (ss + PRIOR_N * PRIOR_SD ** 2) / (n + PRIOR_N) - mean ** 2
                return float(rng.normal(mean, np.sqrt(max(var, 1e-6) / (n + PRIOR_N))))
        return 0.0


def run(f: pd.DataFrame, cands: pd.DataFrame, start: str = START, seed: int = 21,
        clip: float = se.CLIP) -> pd.DataFrame:
    """The agent through history: per coin-hour its pick (or none), learning each trade's net when it closes."""
    rng = np.random.default_rng(seed)
    sit = situation(cands)
    keyed = cands[["history", "signal_ts"]].join(sit)
    f = f.merge(keyed, on=["history", "signal_ts"], how="inner").sort_values("signal_ts")
    beliefs, pending, out = Beliefs(), [], []
    begin = int(pd.Timestamp(start).value // 10**6)
    for ts, moment in f.groupby("signal_ts", sort=True):
        while pending and pending[0][0] <= ts:                  # trades closed by now teach
            _, _, keys, r = heapq.heappop(pending)
            for k in keys:
                beliefs.add(k, r)
        for (history,), coin in moment.groupby(["history"]):
            draws = []
            for r in coin.itertuples(index=False):
                keys = [(r.fine, r.action), (r.event, r.action), (r.action,)]
                draws.append((beliefs.draw(keys, rng), r, keys))
            best, r, keys = max(draws, key=lambda d: d[0])
            if ts >= begin and best > 0:
                out.append({"history": history, "signal_ts": int(ts), "action": r.action, "net": r.net,
                            "held": r.held, "year": r.year, "turnover": r.turnover, "forecast": best})
            # it learns from every action it could have taken, as the selector does: the market tells all
            for d_val, rr, kk in draws:
                heapq.heappush(pending, (int(rr.exit_ts), id(rr), kk, float(np.clip(rr.net, -clip, clip))))
    return pd.DataFrame(out)
