"""The selector: at every moment, which trade fits the situation, learned only from trades already closed.

a strategy enters a moment knowing only what is known then, analyses everything,
decides and books; the next learns from the results of those before it, never from its own future.
Picking champions on years whose results are known failed exactly there (core/brain/regime.py's
switcher, walked forward). So the choice itself is a model:

- a candidate is a coin-hour of the event library (core/brain/events.py) with everything known at its
  close: which events fired, the coin's last day and usual move, its turnover, how many coins had
  events in the same hour, BTC's trend, mood and moves, the time of day;
- an action is one way to trade it: buy on spot or short a live perpetual on 1x margin, with one of
  four exits (4 hours; a day; 3 days, half out at +5% and the rest at the entry; 3 days, a stop at 2
  usual daily moves and a target at 2 stops), walked on hourly prices with costs (core/brain/factory.py);
- a LightGBM regression forecasts every action's net result; it is refit every month on the actions
  whose trades had closed before that month and forecasts only that month's (walk-forward, from 2022,
  the first year with a year of experience behind it);
- at every moment the selector takes, for each coin, the action with the best forecast if it clears a
  bar, and a $100 book takes the best forecasts first ($20 a trade, at most 5 open).
Every month the model has more experience: the self-improvement is in the refits. A forecast that does
not come true shows in the promise table (forecast against what happened).

v2 (plan item 4): a trade's net is mostly where the whole coin went that month, which no one
knows at the decision; learning it, the model learned good months (2022 -57%, 2024 -73% walked forward).
The "edge" target is the net minus the same action from random hours of the same coin and month (its
twin, 3 hours a coin-month): what timing adds. Both targets are walked forward and the live model takes
the one that did better over all years, not the last two.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.brain import events as ev
from core.brain import exits as ex
from core.brain import factory as fa
from core.brain import regime as rg
from core.sandbox.market_store import HOUR_MS

START = "2019-01-01"               # the first year with a year of trades behind it
CLIP = 0.3                           # a trade's net is learned between -30% and +30%: one squeeze is not the world
ACTIONS = (
    ("L4", 1, ex.Rule("L4", "", 4)),
    ("L24", 1, ex.Rule("L24", "", 24)),
    ("L72H5", 1, ex.Rule("L72H5", "", 72, half_pct=0.05)),
    ("L72S2T2", 1, ex.Rule("L72S2T2", "", 72, stop=2.0, take=2.0)),
    ("S4", -1, ex.Rule("S4", "", 4)),
    ("S24", -1, ex.Rule("S24", "", 24)),
    ("S72H5", -1, ex.Rule("S72H5", "", 72, half_pct=0.05)),
    ("S72S2T2", -1, ex.Rule("S72S2T2", "", 72, stop=2.0, take=2.0)),
)
ACTION_RU = {"L4": "купить, 4 ч", "L24": "купить, сутки", "L72H5": "купить, 3 дня, половину на +5%",
             "L72S2T2": "купить, 3 дня, стоп 2 хода, цель 2 стопа", "S4": "шорт, 4 ч", "S24": "шорт, сутки",
             "S72H5": "шорт, 3 дня, половину на +5%", "S72S2T2": "шорт, 3 дня, стоп 2 хода, цель 2 стопа"}
BARS = (0.005, 0.01, 0.02)           # a forecast must clear one of these to be taken
EVENT_COLS = tuple(f"ev_{e}" for e in fa.EVENT_IDS)
FEATURES = (*EVENT_COLS, "fell24", "sigma_day", "log_turnover", "big", "perp", "squeeze_chance", "moment_size",
            "moment_events", "btc_sma", "btc_vol", "btc_r1", "btc_r24", "btc_r168", "trend", "mood", "hour",
            "weekend", "breadth_state", "alt_state", "fund_state", "action", "side")
# the lesson of assessor A1: continuous market-wide features memorise good months; "coin" keeps
# the market only as its coarse regime
MARKET_MOVES = ("btc_sma", "btc_vol", "btc_r1", "btc_r24", "btc_r168")
FEATURE_SETS = {"full": FEATURES, "coin": tuple(f for f in FEATURES if f not in MARKET_MOVES)}
# the paper watcher's model: the coin set without the squeeze chance (the trade evaluator runs on the laptop),
# trained on the events the watcher sees live only (core/brain/factory.py LIVE_EVENTS)
FEATURE_SETS["live"] = tuple(f for f in FEATURE_SETS["coin"] if f != "squeeze_chance")
LIVE_BAR = 0.01
LIVE_DIR = Path(__file__).resolve().parents[2] / "runtime" / "learning" / "selector" / "live"
WORDS = {"fell24": "движение монеты за сутки", "sigma_day": "обычный ход монеты", "log_turnover": "оборот",
         "big": "крупная монета", "perp": "есть фьючерс", "squeeze_chance": "шанс выноса", "moment_size":
         "сколько монет в моменте", "moment_events": "сколько событий в часе", "btc_sma": "BTC против 50-дневной",
         "btc_vol": "нервность BTC", "btc_r1": "BTC за час", "btc_r24": "BTC за сутки", "btc_r168": "BTC за неделю",
         "trend": "тренд BTC", "mood": "настроение рынка", "hour": "время суток", "weekend": "выходные",
         "action": "способ сделки", "side": "купить или шорт", "breadth_state": "ширина рынка",
         "alt_state": "сезон альтов или BTC", "fund_state": "температура фандинга",
         **{f"ev_{e}": f"событие «{fa.SHORT_RU[e]}»" for e in fa.EVENT_IDS}}
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 300,
          "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 63,
          "verbose": -1, "seed": 7, "num_threads": max(1, (os.cpu_count() or 2) - 2)}
ROUNDS = 300


def btc_frame(market) -> pd.DataFrame:
    """BTC per hour (ts = the hour's start), as known at its close: against its 50-day average, nervousness
    against its year, its moves over an hour, a day and a week, trend and mood."""
    r = rg.btc_regimes(market.btc())
    c = r["close"]
    sma = c.rolling(rg.TREND_HOURS, min_periods=rg.TREND_HOURS // 2).mean()
    vol = np.log(c).diff().rolling(rg.WEEK_HOURS, min_periods=120).std()
    return pd.DataFrame({"ts": r["ts"], "btc_sma": c / sma - 1,
                         "btc_vol": vol / vol.rolling(rg.YEAR_HOURS, min_periods=2000).median() - 1,
                         "btc_r1": c / c.shift(1) - 1, "btc_r24": c / c.shift(24) - 1, "btc_r168": c / c.shift(168) - 1,
                         "trend": r["regime"].str.split("|").str[0], "mood": r["regime"].str.split("|").str[1]})


def candidates(table: pd.DataFrame, market, btc: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per coin-hour of the factory's trade table: its events as flags, its numbers, its moment, BTC
    (``btc``: the hours' BTC frame when there is no market cache, live)."""
    flags = pd.crosstab([table["history"], table["signal_ts"]], table["event"]).clip(upper=1)
    flags.columns = [f"ev_{c}" for c in flags.columns]
    base = table.drop_duplicates(["history", "signal_ts"]).set_index(["history", "signal_ts"])
    c = base.join(flags).reset_index()
    for col in EVENT_COLS:
        if col not in c:
            c[col] = 0
    c["moment_size"] = c.groupby("signal_ts")["history"].transform("size")
    c["moment_events"] = c["signal_ts"].map(table.groupby("signal_ts").size())
    c["log_turnover"] = np.log10(c["turnover"].clip(lower=1.0))
    c["hour"] = pd.to_datetime(c["signal_ts"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin").dt.hour
    for col in ("big", "perp", "weekend"):
        c[col] = c[col].map({True: 1.0, False: 0.0}).astype(float)
    btc = (btc_frame(market) if btc is None else btc).rename(columns={"ts": "hour_ts"})
    c["hour_ts"] = c["signal_ts"] - HOUR_MS
    c = c.merge(btc, on="hour_ts", how="left")
    return c.reset_index(drop=True)


def label_rows(c: pd.DataFrame, market, newest: int) -> pd.DataFrame:
    """Every candidate under every action it allows (a short needs a live perpetual): net, hours, exit time."""
    out = []
    for r in c.itertuples(index=False):
        prices = market.prices(r.history)
        funding = None
        for name, side, rule in ACTIONS:
            if side < 0 and not r.perp:
                continue
            if side < 0 and funding is None:
                funding = market.funding(r.history)
            res = fa.walk(prices, funding if side < 0 else None, int(r.signal_ts), side, rule, float(r.sigma_day),
                          float(r.delta), float(r.slip), newest)
            if res is not None:
                out.append((r.history, int(r.signal_ts), name, side, res[0], int(res[1])))
    t = pd.DataFrame(out, columns=["history", "signal_ts", "action", "side", "net", "held"])
    t["exit_ts"] = t["signal_ts"] + t["held"] * HOUR_MS
    return t


def frame(c: pd.DataFrame, labels: pd.DataFrame, twins: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    keep = ["history", "base", "signal_ts", "turnover", "year", *[f for f in FEATURES if f not in ("action", "side")]]
    f = labels.merge(c[keep], on=["history", "signal_ts"], how="inner")
    if twins is not None:
        f["month"] = _month(f["signal_ts"])
        f = f.merge(twins, on=["history", "month", "action"], how="left")
        f["edge"] = f["net"] - f["twin"]
    return f


def _month(ts) -> np.ndarray:
    return np.asarray(ts, dtype=np.int64).astype("datetime64[ms]").astype("datetime64[M]").astype(np.int64)


TWIN_HOURS = 3


def twin_rows(c: pd.DataFrame, market, newest: int, seed: int = 13) -> pd.DataFrame:
    """Per coin-month of the candidates and per action: the mean net of the action from random hours of it."""
    rng = np.random.default_rng(seed)
    out = []
    months = c.assign(month=_month(c["signal_ts"])).drop_duplicates(["history", "month"])
    for r in months.itertuples(index=False):
        prices = market.prices(r.history)
        ts = prices[:, 0]
        first = int(np.datetime64(int(r.month), "M").astype("datetime64[ms]").astype(np.int64))
        last = int((np.datetime64(int(r.month), "M") + 1).astype("datetime64[ms]").astype(np.int64))
        lo, hi = max(int(np.searchsorted(ts, first)), fa.ex.SIGMA_HOURS), int(np.searchsorted(ts, last))
        if hi <= lo:
            continue
        funding = market.funding(r.history) if r.perp else None
        nets: dict = {}
        for a in rng.integers(lo, hi, size=TWIN_HOURS):
            sigma = fa.ex.daily_sigma(prices[:, 4], int(a))
            if not math.isfinite(sigma):
                continue
            for name, side, rule in ACTIONS:
                if side < 0 and not r.perp:
                    continue
                res = fa.walk(prices, funding if side < 0 else None, int(ts[a]), side, rule, sigma, 0.0,
                              float(r.slip), newest)
                if res is not None:
                    nets.setdefault(name, []).append(res[0])
        out += [(r.history, int(r.month), name, float(np.mean(v))) for name, v in nets.items()]
    return pd.DataFrame(out, columns=["history", "month", "action", "twin"])


def _x(f: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    x = f[list(features)].copy()
    for col, cats in (("action", [a[0] for a in ACTIONS]), ("trend", list(rg.TRENDS)), ("mood", list(rg.MOODS)),
                      ("breadth_state", list(rg.STATES["breadth"])), ("alt_state", list(rg.STATES["alts"])),
                      ("fund_state", list(rg.STATES["fund"]))):
        x[col] = pd.Categorical(x[col], categories=cats)
    return x


def walk_forward(f: pd.DataFrame, start: str = START, on_refit=None, features=FEATURES,
                 target: str = "net") -> pd.DataFrame:
    """Every action from ``start`` on, forecast by a model of the trades closed before its month; ``target``
    "net" (the trade's result) or "edge" (its result over its twin's)."""
    if target == "edge":
        f = f[np.isfinite(f["edge"].to_numpy(dtype=float))]
    import lightgbm as lgb
    months = pd.date_range(start, pd.to_datetime(f["signal_ts"].max(), unit="ms"), freq="MS")
    out, importance = [], None
    for m in months:
        lo, hi = int(m.value // 10**6), int((m + pd.offsets.MonthBegin(1)).value // 10**6)
        train = f[f["exit_ts"] < lo]
        test = f[(f["signal_ts"] >= lo) & (f["signal_ts"] < hi)]
        if not len(test) or len(train) < 10_000:
            continue
        model = lgb.train(PARAMS, lgb.Dataset(_x(train, features), train[target].clip(-CLIP, CLIP),
                                              free_raw_data=True), num_boost_round=ROUNDS)
        out.append(test.assign(forecast=model.predict(_x(test, features))))
        gain = pd.Series(model.feature_importance("gain"), index=model.feature_name())
        importance = gain if importance is None else importance.add(gain, fill_value=0)
        if on_refit:
            on_refit(m, len(train))
    pred = pd.concat(out, ignore_index=True)
    pred.attrs["importance"] = (importance / importance.sum()).sort_values(ascending=False) if importance is not None \
        else pd.Series(dtype=float)
    return pred


def choose(pred: pd.DataFrame, bar: float) -> pd.DataFrame:
    """For every coin-hour the action with the best forecast, if it clears ``bar``."""
    best = pred.sort_values("forecast", ascending=False).drop_duplicates(["history", "signal_ts"])
    return best[best["forecast"] >= bar]


def promise_table(pred: pd.DataFrame, bins=(-1, 0, 0.005, 0.01, 0.02, 0.05, 1)) -> pd.DataFrame:
    """Forecast against what happened, by forecast band."""
    band = pd.cut(pred["forecast"], bins)
    g = pred.groupby(band, observed=True)
    return pd.DataFrame({"trades": g.size(), "forecast_pct": 100 * g["forecast"].mean(),
                         "real_pct": 100 * g["net"].mean(), "up_of10": 10 * g["net"].apply(lambda r: (r > 0).mean())})


def books(chosen: pd.DataFrame, years, newest: int) -> dict:
    """A $100 book in every year, the best forecasts of a moment first."""
    out = {}
    for y in years:
        t = chosen[chosen["year"] == y]
        b = ex.portfolio(t, *fa.year_span(y, newest), by="forecast")
        out[str(y)] = {"n": len(t), "up_of10": 10 * float((t["net"] > 0).mean()) if len(t) else float("nan"),
                       "usd_per_100": 100 * float(t["net"].mean()) if len(t) else float("nan"), **b}
    return out


# the live model: the paper watcher's

def live_table(table: pd.DataFrame) -> pd.DataFrame:
    """Only the events the paper watcher sees live, so a coin-hour's flags and moment are the same there."""
    return table[table["event"].isin(fa.LIVE_EVENTS)]


def train(f: pd.DataFrame, features=FEATURE_SETS["live"], target: str = "net"):
    """One model on every trade booked so far (each label is a closed trade)."""
    import lightgbm as lgb
    if target == "edge":
        f = f[np.isfinite(f["edge"].to_numpy(dtype=float))]
    return lgb.train(PARAMS, lgb.Dataset(_x(f, features), f[target].clip(-CLIP, CLIP), free_raw_data=True),
                     num_boost_round=ROUNDS)


def save(model, meta: dict, folder: Path = LIVE_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(str(folder / "model.txt"))
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load(folder: Path = LIVE_DIR):
    """(model, meta) of the newest live model, or None."""
    if not (folder / "model.txt").exists():
        return None
    import lightgbm as lgb
    return lgb.Booster(model_file=str(folder / "model.txt")), json.loads((folder / "meta.json").read_text("utf-8"))


def expand(c: pd.DataFrame) -> pd.DataFrame:
    """Every candidate under every action it allows (a short needs a live perpetual)."""
    rows = [c.assign(action=name, side=side) for name, side, _ in ACTIONS]
    out = pd.concat(rows, ignore_index=True)
    return out[(out["side"] > 0) | (out["perp"] == 1.0)].reset_index(drop=True)


def forecasts(c: pd.DataFrame, model, features=FEATURE_SETS["live"]) -> pd.DataFrame:
    """Every candidate under every action it allows, with the model's forecast of its net result."""
    if not len(c):
        return c.assign(action=[], side=[], forecast=[])
    x = expand(c)
    x["forecast"] = model.predict(_x(x, features))
    return x


def pick_best(x: pd.DataFrame, bar: float = LIVE_BAR) -> pd.DataFrame:
    """For every coin-hour of ``forecasts`` the action with the best forecast, if it clears ``bar``; the best first."""
    if not len(x):
        return x
    best = x.sort_values("forecast", ascending=False).drop_duplicates(["history", "signal_ts"])
    return best[best["forecast"] >= bar].reset_index(drop=True)


def best_actions(c: pd.DataFrame, model, features=FEATURE_SETS["live"], bar: float = LIVE_BAR) -> pd.DataFrame:
    """For every coin-hour the action with the best forecast, if it clears ``bar``; the best first."""
    if not len(c):
        return c.assign(action=[], side=[], forecast=[])
    return pick_best(forecasts(c, model, features), bar)


def contributions(model, rows: pd.DataFrame, features=FEATURE_SETS["live"]) -> list[dict]:
    """Each row's forecast taken apart by the model itself (LightGBM's own attribution): the base every forecast
    starts from, and what each feature added, in the forecast's units (+0.01 = one point of the net result)."""
    if not len(rows):
        return []
    parts = np.asarray(model.predict(_x(rows, features), pred_contrib=True), dtype=float)
    return [{"base": float(p[-1]), "by": {f: float(v) for f, v in zip(features, p[:-1])}} for p in parts]


def top_factors(split: dict, n: int = 3) -> tuple[list, list]:
    """The ``n`` features that raised a forecast most and the ``n`` that lowered it most: [(feature, added)]."""
    by = sorted(split.get("by", {}).items(), key=lambda kv: kv[1])
    return [kv for kv in reversed(by) if kv[1] > 0][:n], [kv for kv in by if kv[1] < 0][:n]


RULES = {name: (rule, side) for name, side, rule in ACTIONS}
