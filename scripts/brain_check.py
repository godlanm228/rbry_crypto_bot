"""Is the brain's textbook right? Recount it with deliberately simple, separate code.

    python scripts/brain_check.py [--samples 1000]

1. outcomes: random coin-days are recounted straight from the hourly candles in
   market.db with plain loops (own daily candles, own Wilder ATR, own costs)
   and compared with the dataset: the 14-day hold, the result against BTC and
   the fixed stop trade;
2. features: a few of them recounted the same way;
3. known effects: coins sorted each day by one number only (last week, last
   month, age, volatility, turnover, funding), best fifth against worst fifth
   over the next 14 days. No model: if these disagree with what the brain
   learned, one of the two is wrong.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.brain import dataset as ds  # noqa: E402
from core.sandbox.evaluate import t_cluster  # noqa: E402
from core.sandbox.market_store import MarketStore  # noqa: E402

DAY = 86_400_000
HOUR = 3_600_000


def hourly_of(store: MarketStore, row) -> pd.DataFrame:
    """The hourly candles of the row's own stretch of trading (a halt of over 72 h starts a new one)."""
    h = store.klines(f"{row.base}USDT", "1h")
    gaps = np.flatnonzero(np.diff(h["ts"].to_numpy()) > 72 * HOUR)
    cuts = np.r_[0, gaps + 1, len(h)]
    day = int((row.day - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1))
    for a, b in zip(cuts[:-1], cuts[1:]):
        if h["ts"].iloc[a] <= day <= h["ts"].iloc[b - 1]:
            return h.iloc[a:b].reset_index(drop=True)
    return h.iloc[0:0]


def daily_of(h: pd.DataFrame) -> pd.DataFrame:
    days: dict[int, list] = {}
    for ts, o, hi, lo, c, qv in h[["ts", "open", "high", "low", "close", "quote_volume"]].itertuples(index=False):
        d = ts // DAY * DAY
        if d not in days:
            days[d] = [o, hi, lo, c, qv]
        else:
            x = days[d]
            x[1], x[2], x[3], x[4] = max(x[1], hi), min(x[2], lo), c, x[4] + qv
    return pd.DataFrame([(d, *v) for d, v in sorted(days.items())], columns=["day", "open", "high", "low", "close", "qv"])


def wilder_atr(d: pd.DataFrame, n: int = 14) -> list[float]:
    out, prev_close, value = [], None, None
    for i, (hi, lo, c) in enumerate(zip(d["high"], d["low"], d["close"])):
        tr = hi - lo if prev_close is None else max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        value = tr if value is None else value + (tr - value) / n
        out.append(value if i >= n - 1 else float("nan"))
        prev_close = c
    return out


def slippage(turnover: float) -> float:
    for floor, pct in ds.SLIPPAGE_TIERS:
        if turnover >= floor:
            return pct
    return ds.SLIPPAGE_TIERS[-1][1]


def net(entry: float, exit_: float, slip_pct: float) -> float:
    s, f = slip_pct / 100, ds.FEE_PCT / 100
    return exit_ * (1 - s) * (1 - f) / (entry * (1 + s) * (1 + f)) - 1


def recount(store: MarketStore, row) -> dict:
    h = hourly_of(store, row)
    d = daily_of(h)
    t = int(np.flatnonzero(d["day"].to_numpy() == int((row.day - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)))[0])
    atr = wilder_atr(d)[t]
    turnover = float(np.median(d["qv"].to_numpy()[max(0, t - 29):t + 1]))
    slip = slippage(turnover)
    start = int(d["day"].iloc[t]) + DAY
    window = h[(h["ts"] >= start) & (h["ts"] < start + ds.TRADE_DAYS * DAY)]
    entry = float(window["open"].iloc[0])
    stop, target = entry - ds.STOP_ATR * atr, entry + ds.STOP_ATR * ds.TARGET_R * atr
    exit_ = float(window["close"].iloc[-1])
    for o, hi, lo in window[["open", "high", "low"]].itertuples(index=False):
        if o <= stop or (lo <= stop and not o >= target):
            exit_ = min(o, stop)
            break
        if hi >= target:
            exit_ = max(o, target)
            break
    hold = net(entry, float(window["close"].iloc[-1]), slip)
    return {"atr_pct": atr / float(d["close"].iloc[t]), "ret_7d": math.log(d["close"].iloc[t] / d["close"].iloc[t - 7]),
            "liquidity": math.log10(max(1.0, turnover)), "age": math.log1p(t), "hold_ret": hold,
            "r_net": net(entry, exit_, slip) / (ds.STOP_ATR * atr / entry), "turnover_30d": turnover,
            "dd_90": float(d["close"].iloc[t] / d["high"].iloc[max(0, t - 89):t + 1].max() - 1)}


def check_rows(rows: pd.DataFrame, samples: int, seed: int = 7) -> pd.DataFrame:
    store = MarketStore()
    pool = rows[rows["reason"].isin([-1.0, 0.0, 1.0]) & ~rows["history"].astype(str).str.contains("@")]
    pick = pool.sample(min(samples, len(pool)), random_state=seed)
    btc_hold: dict = {}
    out = []
    for row in pick.itertuples(index=False):
        mine = recount(store, row)
        if row.day not in btc_hold:
            btc_hold[row.day] = recount(store, pd.Series({"base": "BTC", "day": row.day}))["hold_ret"]
        excess = mine["hold_ret"] - btc_hold[row.day]
        out.append({"history": row.history, "day": row.day, **{f"{k}_ds": float(getattr(row, k)) for k in mine},
                    **{f"{k}_mine": v for k, v in mine.items()}, "excess_14d_ds": float(row.excess_14d),
                    "excess_14d_mine": excess})
    return pd.DataFrame(out)


def known_effects(rows: pd.DataFrame) -> pd.DataFrame:
    """Best fifth minus worst fifth by one number, every 7th day (two-week outcomes overlap less)."""
    g = rows[rows["excess_14d"].notna()]
    days = np.sort(g["day"].unique())[::7]
    g = g[g["day"].isin(days)]
    out = []
    for name, col in (("прошлая неделя", "ret_7d"), ("прошлый месяц", "ret_30d"), ("прошлые 3 месяца", "ret_90d"),
                      ("возраст на Binance", "age"), ("резкость (ATR)", "atr_pct"), ("оборот", "liquidity"),
                      ("фандинг за неделю", "funding_7d"), ("ниже пика за 90 дней", "dd_90")):
        q = g.groupby("day")[col].rank(pct=True)
        for per, mask in (("2020–2024", g["day"] <= "2024-12-31"), ("2025", (g["day"] > "2024-12-31") & (g["day"] <= "2025-12-31"))):
            top = g[mask & (q > 0.8)].groupby("day")["excess_14d"].mean()
            bottom = g[mask & (q <= 0.2)].groupby("day")["excess_14d"].mean()
            diff = (top - bottom).dropna()
            out.append({"по чему сортируем": name, "период": per, "дней": len(diff),
                        "верхняя 1/5 минус нижняя, $ на $100": 100 * diff.mean(),
                        "t": t_cluster(diff.to_numpy(), np.arange(len(diff)))})
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    rows = ds.load_table(ROOT / "runtime" / "learning" / "brain" / f"dataset-{ds.DATASET_VERSION}")
    checked = check_rows(rows, args.samples)
    print(f"1-2. пересчёт {len(checked)} случайных монето-дней простым кодом:")
    for k in ("hold_ret", "excess_14d", "r_net", "atr_pct", "ret_7d", "liquidity", "age", "dd_90", "turnover_30d"):
        a, b = checked[f"{k}_ds"].to_numpy(float), checked[f"{k}_mine"].to_numpy(float)
        ok = np.isclose(a, b, rtol=2e-3, atol=2e-4, equal_nan=True)
        worst = np.nanmax(np.abs(a - b)) if len(a) else float("nan")
        print(f"   {k:13s} совпало {ok.sum()}/{len(ok)}; наибольшее расхождение {worst:.2e}")
        if (~ok).any():
            print(checked.loc[~ok, ["history", "day", f"{k}_ds", f"{k}_mine"]].head(5).to_string(index=False))
    print("\n3. известные эффекты (без модели):")
    print(known_effects(ds_eligible(rows)).round(2).to_string(index=False))
    return 0


def ds_eligible(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["turnover_30d"] >= ds.MIN_TURNOVER]


if __name__ == "__main__":
    raise SystemExit(main())
