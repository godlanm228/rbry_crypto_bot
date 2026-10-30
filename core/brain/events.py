"""The event library (Vlad, 30.10.2026: a signal when something happens, not on a timer).

An event is something seen at an hour's close that a watcher could send at once:
a spike on heavy volume, a close beyond the week's range, crowded funding, open
interest jumping, ... Every event is tested on 2018-2026 hourly history:

- entry: the open of the next hour, then Vlad's reaction: 10 minutes, measured on
  5-minute candles where they exist (100 pairs since 10.2024) and applied to an
  event's trades as its average; also 1, 3 and 6 hours late, which says whether
  the event is urgent or can wait for the trading window;
- exit: after 4, 24 or 72 hours (better exits are stage 2);
- costs: spot fees and slippage for longs, futures fees, slippage and funding for
  shorts on 1x isolated margin; a short needs a live perpetual;
- timing edge: the gross result minus a random entry into the same coin in the
  same month (its twin), so an event is not credited with a month that went up;
  a listing's twin is BTC over the same hours (a coin's first month is the event);
- the direction comes from the data: the sign of an event's edge in 2018-2024 says
  whether it is bought or sold, 2025 and 2026 must agree;
- many tries make luck look like an edge: the bar on the 2018-2024 t rises with
  the number of (event, horizon) pairs ever tried (Bonferroni), and an event
  inside one condition (BTC trend, nervous market, time of day, weekend, coin
  size) is a try of its own, with its own, higher bar.
Everything an event reads is known at the close of its hour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.brain.dataset import DAY_MS, FEE_PCT, FUTURES_FEE_PCT, slippage_pct
from core.sandbox.evaluate import t_cluster
from core.sandbox.market_store import HOUR_MS, History

START_MS = int(pd.Timestamp("2018-01-01").value // 10**6)   # 08-12.2017 warm the indicators up (Vlad, 30.10.2026)
LEARN_END_MS = int(pd.Timestamp("2025-01-01").value // 10**6)     # outcomes known before it: 2018-2024
CHECK_END_MS = int(pd.Timestamp("2026-01-01").value // 10**6)     # 2025; later: 2026
PERIODS = ("2018–2024", "2025", "2026")
HORIZONS = (4, 24, 72)              # hours held
DELAYS = (0, 1, 3, 6)               # hours entered late (0: the next hour's open)
REACTION_MS = 10 * 60_000           # Vlad's reaction to a message
COOLDOWN_HOURS = 24                 # one event of a kind per coin per day: a cluster of hours is one event
MIN_HOURS = 120                     # of the 168 behind an hour, for its "usual" move and volume
WEEK_HOURS = 168
CROWD_HOURS = 720
TREND_HOURS = 1200                  # BTC's 50-day average
MIN_TURNOVER = 5e6                  # 30-day median USDT turnover a day: a $20 order moves nothing
BIG_TURNOVER = 50e6
LISTING_TURNOVER = 20e6             # a new listing's first 24 hours
LISTING_NEWS_DAYS = 7               # the announcement came at most this long before the first candle
FUNDING_FRESH_MS = 9 * HOUR_MS
MIN_EVENTS = 30                     # in 2018-2024
MIN_CHECK = 10                      # in each of 2025 and 2026
MIN_REACTION = 20                   # events with 5-minute candles to measure an event's own reaction
ALPHA = 0.05
WATCH_T = 2.0
CONDITIONS = {
    "btc": {True: "BTC в росте", False: "BTC в падении"},
    "nervous": {True: "рынок нервный", False: "рынок спокойный"},
    "daypart": {"ночь": "ночь 00–08", "утро": "утро 08–14", "день": "день 14–20", "вечер": "вечер 20–24"},
    "weekend": {True: "выходные", False: "будни"},
    "big": {True: "крупная монета (≥ $50M в день)", False: "средняя монета ($5–50M)"},
}
COND_VALUES = sum(len(v) for v in CONDITIONS.values())


@dataclass(frozen=True)
class Event:
    id: str
    title: str                                   # plain words for Vlad
    detect: Callable[[pd.DataFrame], pd.Series]
    edge: bool = False                           # a state (funding, crowd): only the hour it begins


def _big_move(x: pd.DataFrame, k: float, floor: float) -> pd.Series:
    return np.maximum(k * x["sigma"], floor)


EVENTS = (
    Event("SPIKE", "свеча +3% и больше за час (4 обычных хода монеты) на объёме ×5",
          lambda x: (x["r1"] >= _big_move(x, 4, 0.03)) & (x["vx"] >= 5)),
    Event("DUMP", "свеча −3% и больше за час (4 обычных хода) на объёме ×5",
          lambda x: (x["r1"] <= -_big_move(x, 4, 0.03)) & (x["vx"] >= 5)),
    Event("BREAKOUT", "закрытие выше максимума недели на объёме ×2",
          lambda x: (x["close"] > x["hi_week"]) & (x["vx"] >= 2) & (x["r1"] > 0)),
    Event("BREAKDOWN", "закрытие ниже минимума недели на объёме ×2",
          lambda x: (x["close"] < x["lo_week"]) & (x["vx"] >= 2) & (x["r1"] < 0)),
    Event("ABSORB-BUY", "объём ×5, а цена почти не сдвинулась; рыночных покупок ≥ 60%",
          lambda x: (x["vx"] >= 5) & (x["lr"].abs() <= x["sigma"]) & (x["taker"] >= 0.6)),
    Event("ABSORB-SELL", "объём ×5, а цена почти не сдвинулась; рыночных продаж ≥ 60%",
          lambda x: (x["vx"] >= 5) & (x["lr"].abs() <= x["sigma"]) & (x["taker"] <= 0.4)),
    Event("TAKER-BUY", "70% и больше объёма часа — рыночные покупки, объём ×3",
          lambda x: (x["taker"] >= 0.7) & (x["vx"] >= 3)),
    Event("TAKER-SELL", "70% и больше объёма часа — рыночные продажи, объём ×3",
          lambda x: (x["taker"] <= 0.3) & (x["vx"] >= 3)),
    Event("FUND-HIGH", "фандинг +0.1% и выше за выплату: толпа в лонгах переплачивает",
          lambda x: x["funding"] >= 0.001, edge=True),
    Event("FUND-LOW", "фандинг −0.1% и ниже: толпа в шортах переплачивает",
          lambda x: x["funding"] <= -0.001, edge=True),
    Event("OI-UP-UP", "открытый интерес +10% за 4 часа, цена +3%: новые деньги в рост",
          lambda x: (x["oi4"] >= 0.10) & (x["r4"] >= 0.03)),
    Event("OI-UP-DOWN", "открытый интерес +10% за 4 часа, цена −3%: новые шорты давят",
          lambda x: (x["oi4"] >= 0.10) & (x["r4"] <= -0.03)),
    Event("OI-FLUSH", "открытый интерес −10% за 4 часа, цена −5%: волна ликвидаций",
          lambda x: (x["oi4"] <= -0.10) & (x["r4"] <= -0.05)),
    Event("CROWD-LONG", "доля счетов в лонгах — максимум за месяц (на 2.5 обычных отклонения выше)",
          lambda x: x["crowd_z"] >= 2.5, edge=True),
    Event("CROWD-SHORT", "доля счетов в шортах — максимум за месяц (на 2.5 обычных отклонения)",
          lambda x: x["crowd_z"] <= -2.5, edge=True),
    Event("STRONG-VS-BTC", "монета +2% и больше за час (3 обычных хода), когда BTC за этот час −1%",
          lambda x: (x["r1"] >= _big_move(x, 3, 0.02)) & (x["btc_r1"] <= -0.01)),
    Event("CAPITULATION-TURN", "после −15% за сутки первая сильная зелёная свеча (2 обычных хода)",
          lambda x: (x["r24"].shift(1) <= -0.15) & (x["lr"] >= 2 * x["sigma"])),
    Event("FELL-WITH-BTC", "BTC −2% за час, а монета упала в 1.5 раза сильнее",
          lambda x: (x["btc_r1"] <= -0.02) & (x["r1"] <= 1.5 * x["btc_r1"])),
    Event("LISTING-24H", "сутки после листинга на Binance (оборот первых суток ≥ $20M)",
          lambda x: pd.Series(False, index=x.index)),         # found by listing_row, not by the hour
)
# on-chain events (core/brain/onchain.py): found from Binance's reserves and the stablecoins' supply a day,
# not from the hour's candles; coin_rows books them at the signals it is given
_NONE = lambda x: pd.Series(False, index=x.index)  # noqa: E731
EXTERNAL = ("CEX-IN", "CEX-OUT", "STABLE-IN", "STABLE-OUT", "STABLE-MINT",
            "CB-HI", "CB-LO", "KR-HI", "KR-LO", "FEAR-HI", "FEAR-LO")
EVENTS = EVENTS + (
    Event("CEX-IN", "занос токена на Binance: +10% резерва за день", _NONE),
    Event("CEX-OUT", "вывоз токена с Binance: −10% резерва за день", _NONE),
    Event("STABLE-IN", "занос стейблкоинов на Binance: +3% за день", _NONE),
    Event("STABLE-OUT", "вывоз стейблкоинов с Binance: −3% за день", _NONE),
    Event("STABLE-MINT", "выпуск стейблкоинов: +0.5% всего предложения за день", _NONE),
    # core/brain/sentiment.py: who is buying, by the hour
    Event("CB-HI", "на Coinbase дороже, чем на Binance, сильнее, чем за месяц (покупают США)", _NONE),
    Event("CB-LO", "на Coinbase дешевле обычного за месяц (США продают)", _NONE),
    Event("KR-HI", "в Корее переплачивают сильнее, чем за месяц (перегрев розницы)", _NONE),
    Event("KR-LO", "корейская переплата — минимум за месяц", _NONE),
    Event("FEAR-HI", "страх в опционах (DVOL) — выше всего за месяц", _NONE),
    Event("FEAR-LO", "страх в опционах — ниже всего за месяц (спокойствие)", _NONE),
)
BY_ID = {e.id: e for e in EVENTS}
PAIR_EVENTS = ("STRONG-VS-BTC", "FELL-WITH-BTC")             # meaningless on BTC itself


# -- one coin ---------------------------------------------------------------------------------

def grid(h: History) -> pd.DataFrame:
    """The history's hours on a full hourly grid; a missing hour is a row of NaN."""
    f = h.hourly
    full = np.arange(int(f["ts"].iloc[0]), int(f["ts"].iloc[-1]) + HOUR_MS, HOUR_MS, dtype=np.int64)
    return f.set_index("ts").reindex(full).rename_axis("ts").reset_index()


def btc_context(g: pd.DataFrame) -> pd.DataFrame:
    """BTC per hour: its move, above its 50-day average, a nervous week, and its own forward results."""
    c = g["close"]
    lr = np.log(c).diff()
    vol = lr.rolling(WEEK_HOURS, min_periods=MIN_HOURS).std()
    out = pd.DataFrame({"ts": g["ts"], "btc_r1": c / c.shift(1) - 1,
                        "btc_up": c > c.rolling(TREND_HOURS, min_periods=TREND_HOURS // 2).mean(),
                        "nervous": vol > vol.rolling(365 * 24, min_periods=2000).median()})
    for hz in HORIZONS:
        out[f"btc_g{hz}"] = c.shift(-hz) / g["open"].shift(-1) - 1
    return out


def features(g: pd.DataFrame, funding: Optional[pd.DataFrame] = None, futures: Optional[pd.DataFrame] = None,
             btc: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """What every event reads at an hour's close, from that hour and the ones before it only."""
    c, hi, lo, qv = g["close"], g["high"], g["low"], g["quote_volume"]
    lr = np.log(c).diff()
    x = pd.DataFrame({"ts": g["ts"], "close": c, "lr": lr, "r1": c / c.shift(1) - 1, "r4": c / c.shift(4) - 1,
                      "r24": c / c.shift(24) - 1})
    x["sigma"] = lr.rolling(WEEK_HOURS, min_periods=MIN_HOURS).std().shift(1)
    x["vx"] = qv / qv.rolling(WEEK_HOURS, min_periods=MIN_HOURS).median().shift(1)
    x["hi_week"] = hi.rolling(WEEK_HOURS, min_periods=MIN_HOURS).max().shift(1)
    x["lo_week"] = lo.rolling(WEEK_HOURS, min_periods=MIN_HOURS).min().shift(1)
    x["taker"] = g["taker_buy_quote"] / qv.where(qv > 0)
    day = g["ts"] // DAY_MS
    daily = qv.groupby(day).sum(min_count=1).where(qv.groupby(day).count() >= 20)
    x["turnover"] = day.map(daily.rolling(30, min_periods=20).median().shift(1)).to_numpy()
    x["funding"] = np.nan
    if funding is not None and len(funding):
        f_ts, rate = funding["ts"].to_numpy(np.int64), funding["rate"].to_numpy(float)
        close = x["ts"].to_numpy(np.int64) + HOUR_MS
        k = np.searchsorted(f_ts, close, side="right") - 1
        kk = np.clip(k, 0, None)
        x["funding"] = np.where((k >= 0) & (close - f_ts[kk] <= FUNDING_FRESH_MS), rate[kk], np.nan)
    x["oi4"], x["crowd_z"] = np.nan, np.nan
    if futures is not None and len(futures):
        m = futures.set_index("ts").reindex(x["ts"])          # ts is the hour's start, values as of its end
        oi = m["open_interest_value"].reset_index(drop=True)
        x["oi4"] = (oi / oi.shift(4) - 1).to_numpy()
        ls = m["ls_accounts"].reset_index(drop=True)
        mean = ls.rolling(CROWD_HOURS, min_periods=CROWD_HOURS // 2).mean().shift(1)
        std = ls.rolling(CROWD_HOURS, min_periods=CROWD_HOURS // 2).std().shift(1)
        x["crowd_z"] = ((ls - mean) / std.where(std > 0)).to_numpy()
    if btc is not None:
        x = x.merge(btc, on="ts", how="left")
    else:
        x["btc_r1"] = np.nan
    return x


def fire(mask: np.ndarray, edge: bool = False, cooldown: int = COOLDOWN_HOURS) -> np.ndarray:
    """Hours an event fires: (where a state begins,) at most once per ``cooldown`` hours."""
    m = np.asarray(mask, dtype=bool).copy()
    if edge:
        m &= ~np.r_[False, m[:-1]]
    keep, last = [], -10 ** 9
    for i in np.flatnonzero(m):
        if i - last >= cooldown:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.int64)


def _month(ts: np.ndarray) -> np.ndarray:
    return np.asarray(ts, dtype=np.int64).astype("datetime64[ms]").astype("datetime64[M]").astype(np.int64)


def _daypart(hours: np.ndarray) -> np.ndarray:
    return np.select([hours < 8, hours < 14, hours < 20], ["ночь", "утро", "день"], "вечер")


def forward(g: pd.DataFrame, funding: Optional[pd.DataFrame]) -> dict:
    """Per hour i (signal at its close) and horizon: long gross results entered 0/1/3/6 hours late,
    the funding a short would receive, the worst and best prices on the way; and the twins."""
    o, c, hi, lo, ts = g["open"], g["close"], g["high"], g["low"], g["ts"].to_numpy(np.int64)
    out: dict = {}
    if funding is not None and len(funding):
        f_ts = funding["ts"].to_numpy(np.int64)
        cum = np.r_[0.0, np.cumsum(funding["rate"].to_numpy(float))]
    for hz in HORIZONS:
        for d in DELAYS:
            entry = o.shift(-(1 + d))
            out[f"g{hz}_{d}"] = (c.shift(-(d + hz)) / entry - 1).to_numpy()
            if funding is not None and len(funding):
                start = ts + (1 + d) * HOUR_MS
                end = ts + (1 + d + hz) * HOUR_MS
                out[f"fr{hz}_{d}"] = (cum[np.searchsorted(f_ts, end, side="right")]
                                      - cum[np.searchsorted(f_ts, start, side="right")])
            else:
                out[f"fr{hz}_{d}"] = np.full(len(ts), np.nan)
        e0 = o.shift(-1)
        out[f"mae{hz}"] = (lo.rolling(hz, min_periods=hz).min().shift(-hz) / e0 - 1).to_numpy()
        out[f"mfe{hz}"] = (hi.rolling(hz, min_periods=hz).max().shift(-hz) / e0 - 1).to_numpy()
        twins = pd.Series(out[f"g{hz}_0"]).groupby(_month(ts + HOUR_MS)).mean()
        out[f"twin{hz}"] = pd.Series(_month(ts + HOUR_MS)).map(twins).to_numpy()
    return out


def perp_live(entry_ts: np.ndarray, funding: Optional[pd.DataFrame]) -> np.ndarray:
    """A perpetual paid funding within a day after the entry: the short could be opened."""
    if funding is None or not len(funding):
        return np.zeros(len(entry_ts), dtype=bool)
    f_ts = funding["ts"].to_numpy(np.int64)
    return np.searchsorted(f_ts, entry_ts + DAY_MS, side="right") > np.searchsorted(f_ts, entry_ts)


def reactions(rows: pd.DataFrame, close_exit: dict, five: Optional[pd.DataFrame]) -> dict:
    """Per horizon: the long gross entered 10 minutes after the hour's open minus at the open (5-minute candles)."""
    out = {hz: np.full(len(rows), np.nan) for hz in HORIZONS}
    if five is None or not len(five) or not len(rows):
        return out
    opens = five.set_index("ts")["open"]
    e0 = opens.reindex(rows["entry_ts"].to_numpy()).to_numpy()
    e10 = opens.reindex(rows["entry_ts"].to_numpy() + REACTION_MS).to_numpy()
    for hz in HORIZONS:
        out[hz] = close_exit[hz] / e10 - close_exit[hz] / e0
    return out


def coin_rows(h: History, funding: Optional[pd.DataFrame] = None, futures: Optional[pd.DataFrame] = None,
              five: Optional[pd.DataFrame] = None, btc: Optional[pd.DataFrame] = None,
              listed_ms: Optional[int] = None, external: Optional[list] = None) -> pd.DataFrame:
    """Every event of one history since 2018, with what followed each; ``external``: [(event, signal ms)] of
    events found elsewhere (on-chain), booked at the close of the hour they are known."""
    g = grid(h)
    if len(g) < MIN_HOURS + 2:
        return pd.DataFrame()
    x = features(g, funding, futures, btc)
    fwd = forward(g, funding)
    ts = g["ts"].to_numpy(np.int64)
    usual = (ts + HOUR_MS >= START_MS) & (x["turnover"].to_numpy() >= MIN_TURNOVER) & x["sigma"].notna().to_numpy()
    picked = []
    for e in EVENTS:
        if e.id == "LISTING-24H" or e.id in EXTERNAL or (e.id in PAIR_EVENTS and h.base == "BTC"):
            continue
        mask = e.detect(x).fillna(False).to_numpy(dtype=bool) & usual
        for i in fire(mask, e.edge):
            picked.append((e.id, i, float(x["turnover"].iat[i])))
    for event_id, signal_ms in external or ():
        i = int(np.searchsorted(ts, int(signal_ms) - HOUR_MS))
        if i < len(ts) and ts[i] == int(signal_ms) - HOUR_MS and usual[i]:
            picked.append((event_id, i, float(x["turnover"].iat[i])))
    if listed_ms is not None and ts[0] + HOUR_MS >= START_MS and len(ts) > 30:
        first = float(np.nansum(g["quote_volume"].to_numpy()[:24]))
        if first >= LISTING_TURNOVER:
            picked.append(("LISTING-24H", 23, first))
    if not picked:
        return pd.DataFrame()
    ids, idx, turnover = zip(*picked)
    idx = np.asarray(idx, dtype=np.int64)
    signal = ts[idx] + HOUR_MS
    berlin = pd.to_datetime(signal, unit="ms", utc=True).tz_convert("Europe/Berlin")
    rows = pd.DataFrame({
        "event": ids, "history": h.name, "base": h.base, "signal_ts": signal, "entry_ts": signal,
        "hour": berlin.hour.to_numpy(), "weekend": berlin.dayofweek.to_numpy() >= 5,
        "turnover": turnover, "slip": slippage_pct(np.asarray(turnover, dtype=float)) / 100.0})
    rows["daypart"] = _daypart(rows["hour"].to_numpy())
    rows["big"] = rows["turnover"] >= BIG_TURNOVER
    ctx = x.iloc[idx]
    rows["btc"] = ctx["btc_up"].to_numpy() if "btc_up" in ctx else np.nan
    rows["nervous"] = ctx["nervous"].to_numpy() if "nervous" in ctx else np.nan
    rows["perp"] = perp_live(rows["entry_ts"].to_numpy(np.int64), funding)
    for k, v in fwd.items():
        rows[k] = v[idx]
    listing = rows["event"].to_numpy() == "LISTING-24H"
    for hz in HORIZONS:
        if listing.any() and f"btc_g{hz}" in ctx:
            rows.loc[listing, f"twin{hz}"] = ctx[f"btc_g{hz}"].to_numpy()[listing]
    exits = {hz: g["close"].shift(-hz).to_numpy()[idx] for hz in HORIZONS}
    for hz, v in reactions(rows, exits, five).items():
        rows[f"react{hz}"] = v
    return rows


# -- results ----------------------------------------------------------------------------------

def period(signal_ts: np.ndarray, hours: int) -> np.ndarray:
    """By when the outcome is known (the exit), like the brain's exam."""
    known = np.asarray(signal_ts, dtype=np.int64) + hours * HOUR_MS
    return np.where(known < LEARN_END_MS, PERIODS[0], np.where(known < CHECK_END_MS, PERIODS[1], PERIODS[2]))


def net_long(gross, slip):
    fee = FEE_PCT / 100.0
    return (1 + gross) * (1 - slip) * (1 - fee) / ((1 + slip) * (1 + fee)) - 1


def net_short(gross, slip, received, best):
    """1x isolated short: fees and slippage both ways, funding received; liquidated once the price doubled."""
    fee = FUTURES_FEE_PCT / 100.0
    r = (1 - slip) * (1 - fee) - (1 + gross) * (1 + slip) * (1 + fee) + np.nan_to_num(received)
    return np.where(np.asarray(best) >= 1.0, -1.0, np.maximum(r, -1.0))


def reaction(rows: pd.DataFrame, hz: int) -> tuple[float, str]:
    """(mean change of the long gross from a 10-minute reaction, where it was measured)."""
    r = rows[f"react{hz}"].dropna()
    return (float(r.mean()), "своё") if len(r) >= MIN_REACTION else (float("nan"), "")


def sided(rows: pd.DataFrame, hz: int, side: int, react: float) -> pd.DataFrame:
    """One event at one horizon in one direction: per trade the timing edge, the net result after a
    10-minute reaction and after 1/3/6 hours, and the worst moment on the way."""
    r = rows[rows["perp"]] if side < 0 else rows
    r = r[np.isfinite(r[f"g{hz}_0"].to_numpy(dtype=float))]
    g0 = r[f"g{hz}_0"].to_numpy(dtype=float)
    react = 0.0 if not math.isfinite(react) else react
    out = pd.DataFrame({"period": period(r["signal_ts"].to_numpy(), hz),
                        "week": pd.to_datetime(r["signal_ts"].to_numpy(), unit="ms").strftime("%G-W%V"),
                        "excess": side * (g0 - r[f"twin{hz}"].to_numpy(dtype=float)),
                        "hour": r["hour"].to_numpy(), "month": _month(r["signal_ts"].to_numpy())},
                       index=r.index)
    for cond in CONDITIONS:
        out[cond] = r[cond].to_numpy()
    slip = r["slip"].to_numpy(dtype=float)
    for d in DELAYS:
        g = r[f"g{hz}_{d}"].to_numpy(dtype=float)
        if d == 0:
            g = g + react                    # the long gross after 10 minutes; the short loses what the long gains
        if side > 0:
            out[f"net{d}"] = net_long(g, slip)
        else:
            out[f"net{d}"] = net_short(g, slip, r[f"fr{hz}_{d}"].to_numpy(dtype=float), r[f"mfe{hz}"].to_numpy())
    out["worst"] = r[f"mae{hz}"].to_numpy(dtype=float) if side > 0 else -r[f"mfe{hz}"].to_numpy(dtype=float)
    return out


def describe(t: pd.DataFrame) -> dict:
    n = len(t)
    if not n:
        return {"n": 0, "months": 0, "excess_pct": float("nan"), "t": float("nan"), "usd_per_100": float("nan"),
                "up_of10": float("nan"), "worst_p10_pct": float("nan")}
    ex = t["excess"].to_numpy(dtype=float)
    ok = np.isfinite(ex)
    return {"n": n, "months": int(t["month"].nunique()),
            "excess_pct": 100 * float(np.nanmean(ex)) if ok.any() else float("nan"),
            "t": t_cluster(ex[ok], t["week"].to_numpy()[ok]) if ok.sum() > 1 else float("nan"),
            "usd_per_100": 100 * float(t["net0"].mean()), "up_of10": 10 * float((t["net0"] > 0).mean()),
            "worst_p10_pct": 100 * float(np.nanquantile(t["worst"], 0.1))}


def bar(tries: int) -> float:
    """The two-sided Bonferroni bar on t for ``tries`` tests at 5% in all."""
    return NormalDist().inv_cdf(1 - ALPHA / (2 * max(tries, 1)))


def verdict(parts: dict, bar_t: float) -> str:
    learn, check, late = (parts[p] for p in PERIODS)
    if learn["n"] < MIN_EVENTS or check["n"] < MIN_CHECK or late["n"] < MIN_CHECK:
        return "мало случаев"
    passed = math.isfinite(learn["t"]) and learn["t"] >= bar_t
    confirmed = all(math.isfinite(p["excess_pct"]) and p["excess_pct"] > 0 for p in (check, late))
    paid = all(p["usd_per_100"] > 0 for p in (learn, check, late))
    if passed and confirmed and paid:
        return "КАНДИДАТ"
    if passed and confirmed:
        return "перевес есть, но после комиссий и задержки в минусе"
    if passed:
        return "работало в 2018–2024, в 2025–2026 не подтвердилось"
    if math.isfinite(learn["t"]) and learn["t"] >= WATCH_T and confirmed and paid:
        return "наблюдать: похоже на перевес, доказательств мало"
    return "перевеса нет"


def direction(rows: pd.DataFrame, hz: int) -> int:
    """+1 buy, -1 sell: the sign of the timing edge in 2018-2024 (all events, shortable or not)."""
    g0 = rows[f"g{hz}_0"].to_numpy(dtype=float)
    learn = (period(rows["signal_ts"].to_numpy(), hz) == PERIODS[0]) & np.isfinite(g0)
    ex = g0[learn] - rows[f"twin{hz}"].to_numpy(dtype=float)[learn]
    ex = ex[np.isfinite(ex)]
    return 1 if not len(ex) or ex.mean() >= 0 else -1


def study(rows: pd.DataFrame, tries: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(every event and horizon, events that work only in one condition, the trades of each candidate)."""
    base_bar, cond_bar = bar(tries), bar(tries * COND_VALUES)
    table, conditional, trades = [], [], {}
    for e in EVENTS:
        ev = rows[rows["event"] == e.id]
        if not len(ev):
            continue
        for hz in HORIZONS:
            side = direction(ev, hz)
            react, _ = reaction(ev, hz)
            t = sided(ev, hz, side, react)
            parts = {p: describe(t[t["period"] == p]) for p in PERIODS}
            v = verdict(parts, base_bar)
            table.append({"event": e.id, "hours": hz, "side": side, "react_pct": 100 * react, "verdict": v,
                          **{f"{p}|{k}": val for p in PERIODS for k, val in parts[p].items()}})
            if v.startswith(("КАНДИДАТ", "наблюдать")):
                trades[(e.id, hz, "")] = t
            for cond, values in CONDITIONS.items():
                for value, label in values.items():
                    sub = t[t[cond] == value]
                    cparts = {p: describe(sub[sub["period"] == p]) for p in PERIODS}
                    cv = verdict(cparts, cond_bar)
                    if cv.startswith(("КАНДИДАТ", "наблюдать")) and not v.startswith("КАНДИДАТ"):
                        conditional.append({"event": e.id, "hours": hz, "side": side, "condition": label,
                                            "verdict": cv,
                                            **{f"{p}|{k}": val for p in PERIODS for k, val in cparts[p].items()}})
                        trades[(e.id, hz, label)] = sub
    return pd.DataFrame(table), pd.DataFrame(conditional), trades


def urgency(t: pd.DataFrame) -> tuple[str, dict]:
    """Whether a signal can wait: the mean net per $100 entered 10 minutes and 1/3/6 hours after it."""
    vals = {d: 100 * float(t[f"net{d}"].mean()) for d in DELAYS}
    now = vals[0]
    if now <= 0:
        return "не в плюсе даже сразу", vals
    waits = [d for d in DELAYS[1:] if vals[d] >= 0.5 * now]
    if not waits or waits[0] != 1:
        return "срочное: через час теряет больше половины", vals
    longest = max(d for d in waits if all(vals[k] >= 0.5 * now for k in DELAYS[1:] if k <= d))
    return f"может подождать окна до {longest} ч", vals


def best_window(trades: list[pd.DataFrame], waits: list[int], years: float, width: int = 3) -> pd.DataFrame:
    """$ per year, $100 a trade, that a window of ``width`` Berlin hours collects from these trades: a
    signal inside it entered at once, one up to ``wait`` hours before it at the window's start (1/3/6 h late)."""
    out = []
    for start in range(24):
        total = 0.0
        for t, wait in zip(trades, waits):
            gap = (start - t["hour"].to_numpy()) % 24              # hours from the signal to the window's start
            inside = ((t["hour"].to_numpy() - start) % 24) < width
            value = np.where(inside, t["net0"].to_numpy(), 0.0)
            for lo_h, d in ((0, 1), (1, 3), (3, 6)):
                late = ~inside & (gap > lo_h) & (gap <= d) & (d <= wait)
                value = np.where(late, t[f"net{d}"].to_numpy(), value)
            total += float(np.nansum(value))
        out.append({"start": start, "usd_per_100_trade_sum_per_year": 100 * total / years})
    return pd.DataFrame(out).sort_values("usd_per_100_trade_sum_per_year", ascending=False)
