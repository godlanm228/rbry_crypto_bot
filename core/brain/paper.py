"""The paper journal: C1 on live prices, every decision written down before its outcome exists.

C1 (core/brain/playbook.py BLENDS): half the money in B1 (BTC on a day after a
close above its 50-day average, else USDT), half in N5 (each day the brain's 3
best coins bought and its 3 worst coins with a live perpetual shorted on 1x
margin, 14 days, at most 10 positions), rebalanced every day.

Every night after the data refresh (scripts/paper.py day):
1. decide: the champion places the newest day's coins; B1's switch, N5's six
   orders and the moment of the decision go into decisions.jsonl and are never
   changed. A day without a decision stays without one, as in real life;
2. book: simulator v2 runs both halves from the first decision to the newest
   close on market.db's hourly prices, live (core/brain/simulator.py): open
   positions are marked at market and an order enters the hour after its
   decision, not the hour after its signal.
The result is set against C1's usual weeks on history (EXPECT): a paper result
below their bad tenth means the market or our sums changed.

The journal only books prices after its own decisions, so the closed 2026 is
not looked at. Messages go to the owner only.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.brain import portfolio as pf
from core.brain import simulator as sim
from core.brain.dataset import DAY_MS, slippage_pct
from core.brain.model import Brain, day_place
from core.brain.playbook import BY_ID
from core.brain.walkforward import eligible
from core.sandbox import archive
from core.sandbox.market_store import HOUR_MS, MarketStore

PAPER_DIR = Path(__file__).resolve().parents[2] / "runtime" / "learning" / "paper"
STRATEGY = "N5"            # the coin half of C1; the other half is B1, BTC by its trend
B1_WEIGHT = 0.5
# C1 on history (simulator v2, exam-20261028-1014, 12.2020-2025, one path, every start day):
# weeks -> a window's result in %: bad tenth, middle, good tenth; windows in profit of 10
EXPECT = {1: (-3.7, 0.7, 5.7, 6.1), 4: (-5.7, 2.5, 14.9, 6.2), 13: (-8.7, 9.3, 35.7, 6.8)}


# -- prices ----------------------------------------------------------------------------------

class LiveMarket:
    """simulator.Market on market.db for the few histories a journal holds, prices up to now."""

    def __init__(self, store: MarketStore):
        self.store = store
        self.base_of: dict[str, str] = {}
        self._prices: dict[str, np.ndarray] = {}
        self._funding: dict[str, Optional[np.ndarray]] = {}
        self._notices = {b: np.sort(np.asarray(v, dtype=np.float64)) for b, v in store.delisting_notices().items()}
        spot = {b for _, b, _ in store.symbols()}
        with store.connect() as conn:
            perpetuals = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM funding")]
        self._perp = {base: pair for pair, base in archive.futures_pairs(perpetuals, spot).items()}

    def prices(self, history: str) -> np.ndarray:
        if history not in self._prices:
            base = history.split("@")[0]
            found = next((h for h in self.store.histories(min_days=0, symbols={base}) if h.name == history), None)
            self.base_of[history] = found.base if found else base
            self._prices[history] = (found.hourly[["ts", "open", "high", "low", "close"]].to_numpy(dtype=np.float64)
                                     if found else np.empty((0, 5)))
        return self._prices[history]

    def notices(self, history: str) -> np.ndarray:
        return self._notices.get(self.base_of.get(history, history.split("@")[0]), np.empty(0))

    def btc(self) -> np.ndarray:
        return self.prices("BTC")

    def funding(self, history: str) -> Optional[np.ndarray]:
        pair = self._perp.get(self.base_of.get(history, history.split("@")[0]))
        if pair is None:
            return None
        if pair not in self._funding:
            rates = self.store.funding(pair)
            self._funding[pair] = rates[["ts", "rate"]].to_numpy(dtype=np.float64) if len(rates) else None
        return self._funding[pair]

    def btc_closes(self) -> pd.Series:
        """BTC's close of every finished UTC day."""
        p = self.btc()
        last = pd.DataFrame({"day": p[:, 0] // DAY_MS, "ts": p[:, 0], "close": p[:, 4]}).groupby("day").last()
        last = last[last["ts"] >= last.index * DAY_MS + DAY_MS - HOUR_MS]          # its 23:00 candle is in
        return pd.Series(last["close"].to_numpy(), index=pd.to_datetime(last.index * DAY_MS, unit="ms"))


def live_until(store: MarketStore) -> int:
    """The end of the newest hourly candle."""
    return int(store.newest_hour() or 0) + HOUR_MS


def last_closed_day(until_ms: int) -> pd.Timestamp:
    return pd.Timestamp((until_ms // DAY_MS - 1) * DAY_MS, unit="ms")


# -- decisions --------------------------------------------------------------------------------

def decide(rows: pd.DataFrame, champion: Brain, decided_ms: int) -> dict:
    """The decision on the newest day of ``rows``: B1's switch and N5's orders, from that day's close.

    N5's own picker chooses, so the journal trades what was tested. Its outcome columns only say what
    could be traded: any coin the brain places can be bought; a short needs a perpetual that paid
    funding that day.
    """
    day = rows["day"].max()
    pool = eligible(rows[rows["day"] == day]).copy()
    pool["history"] = pool["history"].astype(str)
    pool["place"] = day_place(pool, champion.score(pool)).to_numpy()
    pool["slip_pct"] = slippage_pct(pool["turnover_30d"].to_numpy(dtype=float))
    pool["hold_ret"] = 0.0
    pool["short_liq_ret"] = np.where((pool["perp_listed"] > 0) & pool["funding_1d"].notna(), 0.0, np.nan)
    pool["short_liq_hours"] = np.nan                       # how long it lasts is not known yet
    orders = BY_ID[STRATEGY].orders(pool)
    btc = rows[(rows["day"] == day) & (rows["base"].astype(str) == "BTC")]
    btc_up = bool(len(btc) and float(btc["close"].iloc[0]) > float(btc["sma50"].iloc[0]))
    base_of = dict(zip(pool["history"], pool["base"].astype(str)))
    # every coin's place, so any other strategy can be booked on the same decisions later
    places = [[h, base_of[h], round(float(p), 4), float(a), float(s), bool(np.isfinite(sh))]
              for h, p, a, s, sh in zip(pool["history"], pool["place"], pool["atr"], pool["slip_pct"],
                                        pool["short_liq_ret"])]
    return {"day": str(day.date()), "decided_ms": int(decided_ms), "places": places,
            "decided_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(decided_ms / 1000)),
            "champion": champion.meta.get("model_id", "?"), "coins": int(len(pool)), "btc_up": btc_up,
            "orders": [{"history": o.history, "base": base_of.get(o.history, o.history), "side": int(o.side),
                        "place": float(pool.loc[pool["history"] == o.history, "place"].iloc[0]),
                        "priority": float(o.priority), "atr": float(o.atr), "slip_pct": float(o.slip_pct),
                        "exit": asdict(o.rule)} for o in orders.itertuples(index=False)]}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path: Path, decision: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")


# -- the book -----------------------------------------------------------------------------------

def orders_of(decisions: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"day": pd.Timestamp(d["day"]), "history": o["history"], "side": o["side"],
                          "priority": o["priority"], "atr": o["atr"], "slip_pct": o["slip_pct"],
                          "rule": sim.Exit(**o["exit"]), "decided_ms": d["decided_ms"]}
                         for d in decisions for o in d["orders"]])


def btc_switch(decisions: list[dict]) -> pd.Series:
    """B1 holds BTC on day d + 1 when d's close was above its 50-day average (portfolio applies d + 1's move)."""
    return pd.Series({pd.Timestamp(d["day"]) + pd.Timedelta(days=1): d["btc_up"] for d in decisions}, dtype=bool)


@dataclass
class Book:
    values: pd.Series                       # C1 at each day's close
    btc_half: pd.Series
    coin_half: pd.Series
    closed: pd.DataFrame
    still_open: list = field(default_factory=list)
    btc_close: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def book(decisions: list[dict], market, until_ms: int) -> Book:
    """C1 from the first decision to the newest finished day, open positions at market."""
    days = pd.date_range(pd.Timestamp(decisions[0]["day"]), last_closed_day(until_ms), freq="D")
    closes = market.btc_closes()
    still: list = []
    coins, closed = sim.run(orders_of(decisions), days, market, sim.Rules(), live_until=until_ms, still_open=still)
    btc, _ = sim.run(pd.DataFrame(), days, market, sim.Rules(), closes.pct_change().fillna(0.0),
                     btc_switch(decisions), live_until=until_ms)
    return Book(pf.blend(btc, coins, B1_WEIGHT), btc, coins, closed, still, closes.reindex(days))


# -- words ------------------------------------------------------------------------------------

def _pct(x: float, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{'+' if x >= 0 else '−'}{abs(x):.{digits}f}%".replace(".", ",")


def _usd(x: float) -> str:
    return f"${x:,.0f}".replace(",", " ")


def window_check(values: pd.Series, weeks: int) -> Optional[tuple[float, str]]:
    """The last ``weeks`` against C1's usual ones on history, None when the journal is younger."""
    if len(values) <= 7 * weeks:
        return None
    r = 100 * (values.iloc[-1] / values.iloc[-1 - 7 * weeks] - 1)
    bad, mid, good, up = EXPECT[weeks]
    verdict = ("⚠️ хуже 9 из 10 таких отрезков истории — разбираюсь, что не так" if r < bad
               else "лучше 9 из 10 отрезков истории" if r > good else "в обычных пределах")
    return r, f"{_pct(r)} (на истории обычно от {_pct(bad)} до {_pct(good)}, в плюсе {up:.0f} из 10): {verdict}"


def report_text(b: Book, decisions: list[dict], title: str = "неделя") -> str:
    v = b.values
    start, end = v.index[0], v.index[-1]
    btc = b.btc_close.dropna()
    btc_r = 100 * (btc.iloc[-1] / btc.iloc[0] - 1) if len(btc) > 1 else float("nan")
    lines = [f"📒 Бумажный C1 — {title} (по {end:%d.%m})",
             f"С {start:%d.%m}: {_usd(pf.START_USD)} → {_usd(v.iloc[-1])} ({_pct(100 * (v.iloc[-1] / v.iloc[0] - 1))}); "
             f"BTC за то же время {_pct(btc_r)}."]
    for weeks, name in ((1, "Неделя"), (4, "4 недели"), (13, "13 недель")):
        found = window_check(v, weeks)
        if found:
            lines.append(f"{name}: {found[1]}.")
    last = decisions[-1]
    lines.append(f"Половина в BTC: {'держит BTC' if last['btc_up'] else 'в USDT'} "
                 f"(BTC {'выше' if last['btc_up'] else 'ниже'} 50-дневной средней на {pd.Timestamp(last['day']):%d.%m}).")
    longs = [p for p in b.still_open if p["side"] > 0]
    shorts = [p for p in b.still_open if p["side"] < 0]

    def names(ps: list) -> str:
        base = {o["history"]: o["base"] for d in decisions for o in d["orders"]}
        return ", ".join(f"{base.get(p['history'], p['history'])} {_pct(100 * p['ret'])}" for p in ps) or "нет"

    lines.append(f"Вторая половина (лонг/шорт): открыто {len(b.still_open)} — лонг: {names(longs)}; "
                 f"шорт: {names(shorts)}.")
    week = b.closed[b.closed["closed"] > end - pd.Timedelta(days=7)] if len(b.closed) else b.closed
    if len(week):
        lines.append(f"Закрыто за неделю: {len(week)}, в плюс {int((week['ret'] > 0).sum())} из {len(week)}, "
                     f"в среднем {_pct(100 * week['ret'].mean(), 2)} на сделку.")
    missed = len(pd.date_range(start, end)) - len({d["day"] for d in decisions if pd.Timestamp(d["day"]) <= end})
    if missed > 0:
        lines.append(f"⚠️ Дней без решения: {missed}.")
    return "\n".join(lines)


def start_text(decision: dict) -> str:
    first = sorted(decision["orders"], key=lambda o: -o["priority"])       # the strongest pick of each side first
    longs = ", ".join(o["base"] for o in first if o["side"] > 0) or "нет"
    shorts = ", ".join(o["base"] for o in first if o["side"] < 0) or "нет"
    return ("📒 Бумажный журнал C1 начат. Каждую ночь после закрытия дня мозг решает, сделки записываются "
            "до того, как известен итог, и идут по живым ценам без денег. Отчёт — по воскресеньям в 9 утра.\n"
            f"Первое решение (закрытие {pd.Timestamp(decision['day']):%d.%m}, монет в оценке: {decision['coins']}): "
            f"купить {longs}; шорт {shorts}; половина в {'BTC' if decision['btc_up'] else 'USDT'}.")


def state(b: Book, decisions: list[dict]) -> dict:
    """What the journal holds now, for anything that wants to show it."""
    v = b.values
    return {"at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "from": str(v.index[0].date()),
            "through": str(v.index[-1].date()), "value_usd": round(float(v.iloc[-1]), 2),
            "return_pct": round(100 * float(v.iloc[-1] / v.iloc[0] - 1), 2),
            "btc_in": bool(decisions[-1]["btc_up"]), "decisions": len(decisions),
            "open": b.still_open, "closed": int(len(b.closed))}
