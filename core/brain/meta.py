"""The meta-brain: which of the brain's books to run, given the market of the day.

The brain ranks coins; strategies built on its places earn at different
times (pairs fairly evenly, single longs and shorts in their own regimes, BTC
in a bull market). risk management is learning which
strategy fits which situation and combining them, not running two that lose
in turn. The meta-brain learns that from the market state of each day.

Books, booked trade by trade on the exam's out-of-sample places (no portfolio
path, so no butterfly effect, see EXIT-PAIRED in the journal):
  long   the day's PICKS best coins, 14-day spot hold
  short  the day's PICKS worst coins with a perpetual, 14-day hold on 1x margin
  btc    BTC, 14-day hold
A book's outcome on a signal day is the mean of its trades; cash earns 0.

The state of day d is known at its close: BTC's trend and returns, breadth
(coins ahead of BTC, above their 50-day average), the alt median, dispersion
between coins, funding, macro, Fear & Greed, attention, and how each book did
on the signal days already graded (GAP_DAYS back and earlier): the brain's
recent skill is a marker too.

Exam: refit every month on days at least GAP_DAYS before it (MIN_TRAIN_DAYS
or more); a book is on when its predicted outcome is above 0. 2022-2024 to
choose, 2025 to confirm, 2026 closed; an outcome belongs to the period it is
known in, so the last signals of 2025 (graded on 2026 prices) are left out.

``short_books`` holds the same coins 1-3 days instead of 14, to test whether
the crowd's markers (FEAR) act faster than the books' two weeks.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from core.brain.dataset import FEE_PCT, FUTURES_FEE_PCT, slippage_pct
from core.brain.walkforward import GAP_DAYS
from core.sandbox.evaluate import DEV_END, VALID_END

BOOKS = ("long", "short", "btc")
PICKS = 3
MIN_TRAIN_DAYS = 365
RECENT_DAYS = 30
EXAM_START = "2022-01-01"
MARKET = ("btc_ret_7d", "btc_ret_30d", "btc_trend", "btc_atr_pct", "beat_btc_1d", "beat_btc_7d", "beat_btc_30d",
          "beat_btc_7d_vw", "above_sma50", "alt_median_7d", "btc_volume_share", "spx_4w", "dxy_4w", "vix",
          "us10y_4w", "gold_4w", "stables_4w", "fear_greed", "fear_greed_7d",
          "att_bitcoin_7d", "att_crypto_7d", "att_ethereum_7d")
DERIVED = ("dispersion_7d", "funding_median", "recent_long", "recent_short", "recent_btc", "recent_pair")
FEATURES = MARKET + DERIVED


def books(scored: pd.DataFrame, rows: pd.DataFrame, picks: int = PICKS) -> pd.DataFrame:
    """Per signal day the outcome of each book, NaN on a day it had no trade."""
    longs = scored[scored["hold_ret"].notna()]
    top = longs[longs.groupby("day")["place"].rank(ascending=False, method="first") <= picks]
    shorts = scored[scored["short_liq_ret"].notna()]
    bottom = shorts[shorts.groupby("day")["place"].rank(ascending=True, method="first") <= picks]
    out = pd.DataFrame({"long": top.groupby("day")["hold_ret"].mean(),
                        "short": bottom.groupby("day")["short_liq_ret"].mean()})
    btc = rows[rows["base"].astype(str) == "BTC"].drop_duplicates("day").set_index("day")["hold_ret"]
    out["btc"] = btc.reindex(out.index).astype(float)
    return out.sort_index()


FEAR = ("vix", "btc_volume_share", "funding_median", "fear_greed", "fear_greed_7d")


def short_books(scored: pd.DataFrame, rows: pd.DataFrame, hold_days: int, picks: int = PICKS) -> pd.DataFrame:
    """The books' coins held ``hold_days`` instead of 14: close of the signal day to the close ``hold_days`` later.

    Same picks as ``books``; costs as in the dataset (spot fees for longs, futures fees plus the funding received
    for shorts, slippage by turnover on both sides); a short on 1x margin loses at most its size.
    """
    keys = rows[["history", "day"]].assign(history=rows["history"].astype(str))
    later = keys.assign(day=keys["day"] - pd.Timedelta(days=hold_days), close_h=rows["close"].to_numpy(float))
    funding = 0.0
    for k in range(1, hold_days + 1):
        paid = keys.assign(day=keys["day"] - pd.Timedelta(days=k), f=rows["funding_1d"].to_numpy(float))
        funding = funding + scored[["history", "day"]].merge(paid, on=["history", "day"], how="left")["f"].fillna(0.0)
    s = scored.merge(later, on=["history", "day"], how="left")
    s["close_d"] = s[["history", "day"]].merge(keys.assign(c=rows["close"].to_numpy(float)),
                                               on=["history", "day"], how="left")["c"].to_numpy()
    slip, fee, ffee = s["slip_pct"] / 100.0, FEE_PCT / 100.0, FUTURES_FEE_PCT / 100.0
    s["long_h"] = s["close_h"] * (1 - slip) * (1 - fee) / (s["close_d"] * (1 + slip) * (1 + fee)) - 1.0
    s["short_h"] = ((s["close_d"] * (1 - slip) * (1 - ffee) - s["close_h"] * (1 + slip) * (1 + ffee)) / s["close_d"]
                    + funding.to_numpy()).clip(lower=-1.0)
    longs = s[s["hold_ret"].notna()]
    top = longs[longs.groupby("day")["place"].rank(ascending=False, method="first") <= picks]
    shorts = s[s["short_liq_ret"].notna()]
    bottom = shorts[shorts.groupby("day")["place"].rank(ascending=True, method="first") <= picks]
    out = pd.DataFrame({"long": top.groupby("day")["long_h"].mean(), "short": bottom.groupby("day")["short_h"].mean()})
    btc = rows[rows["base"].astype(str) == "BTC"].drop_duplicates("day").set_index("day").sort_index()
    c0, c1 = btc["close"].astype(float), btc["close"].astype(float).shift(-hold_days, freq="D")
    b_slip = pd.Series(slippage_pct(btc["turnover_30d"].to_numpy(float)) / 100.0, index=btc.index)
    btc_h = c1.reindex(c0.index) * (1 - b_slip) * (1 - fee) / (c0 * (1 + b_slip) * (1 + fee)) - 1.0
    out["btc"] = btc_h.reindex(out.index).astype(float)
    return out.sort_index()


def state(rows: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """The market of every day as known at its close, and the books' graded recent results."""
    by_day = rows.groupby("day")
    out = by_day[[c for c in MARKET if c in rows.columns]].first().astype(float)
    out["dispersion_7d"] = by_day["ret_7d"].std()
    out["funding_median"] = by_day["funding_7d"].median()
    calendar = pd.date_range(min(out.index.min(), outcomes.index.min()), max(out.index.max(), outcomes.index.max()))
    graded = outcomes.reindex(calendar).shift(GAP_DAYS)          # day d sees signal days d - GAP_DAYS and older
    recent = graded.rolling(RECENT_DAYS, min_periods=10).mean().reindex(out.index)
    for b in BOOKS:
        out[f"recent_{b}"] = recent[b].to_numpy()
    out["recent_pair"] = (out["recent_long"] + out["recent_short"]) / 2.0
    return out


# learners: state -> a book's 14-day outcome

def _standard(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean, std = train.mean(), train.std().replace(0, 1.0)
    return (((train - mean) / std).fillna(0.0).to_numpy(), ((test - mean) / std).fillna(0.0).to_numpy())


def fit_lgbm(x: pd.DataFrame, y: pd.Series, test: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb
    params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 4, "min_data_in_leaf": 40,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
              "verbose": -1, "seed": 7, "num_threads": 2}
    model = lgb.train(params, lgb.Dataset(x.to_numpy(float), y.to_numpy(float)), num_boost_round=200)
    return model.predict(test.to_numpy(float))


def fit_ridge(x: pd.DataFrame, y: pd.Series, test: pd.DataFrame) -> np.ndarray:
    a, b = _standard(x, test)
    lam = float(len(a))                                   # strong: a feature must earn its weight
    beta = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ (y.to_numpy(float) - y.mean()))
    return b @ beta + y.mean()


def fit_knn(x: pd.DataFrame, y: pd.Series, test: pd.DataFrame, k: int = 60) -> np.ndarray:
    """The mean outcome of the k most similar days in the past ("похожие ситуации")."""
    a, b = _standard(x, test)
    d = ((b[:, None, :] - a[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argpartition(d, min(k, len(a) - 1), axis=1)[:, :k]
    return y.to_numpy(float)[nearest].mean(axis=1)


MIN_BLOCK_DAYS = 90
MIN_MARKER_CORR = 0.05
MARKER_LOG: list = []      # (first test day, book, {marker: sign}) of every refit, for the report


def key_markers(x: pd.DataFrame, y: pd.Series) -> dict:
    """Features whose rank correlation with the outcome kept one sign in every past half-year (two or more)."""
    blocks = x.index.year * 2 + (x.index.month > 6)
    chosen = {}
    for f in x.columns:
        corr = []
        for _, idx in pd.Series(range(len(x)), index=x.index).groupby(blocks):
            fx, fy = x[f].iloc[idx.to_numpy()], y.iloc[idx.to_numpy()]
            ok = fx.notna() & fy.notna()
            if ok.sum() >= MIN_BLOCK_DAYS:
                corr.append(fx[ok].rank().corr(fy[ok].rank()))
        corr = np.asarray([c for c in corr if np.isfinite(c)])
        if len(corr) >= 2 and (np.all(corr > 0) or np.all(corr < 0)) and abs(corr.mean()) >= MIN_MARKER_CORR:
            chosen[f] = float(np.sign(corr.mean()))
    return chosen


def fit_markers(x: pd.DataFrame, y: pd.Series, test: pd.DataFrame) -> np.ndarray:
    """Only the key markers: their signed z-scores averaged into one score, the outcome fitted on it."""
    chosen = key_markers(x, y)
    MARKER_LOG.append((test.index.min(), y.name, chosen))
    if not chosen:
        return np.full(len(test), float(y.mean()))
    cols = list(chosen)
    a, b = _standard(x[cols], test[cols])
    signs = np.array([chosen[c] for c in cols])
    sa, sb = (a * signs).mean(axis=1), (b * signs).mean(axis=1)
    slope, intercept = np.polyfit(sa, y.to_numpy(float), 1)
    return intercept + slope * sb


LEARNERS: dict[str, Callable] = {"lgbm": fit_lgbm, "ridge": fit_ridge, "knn": fit_knn, "markers": fit_markers}


def exam(table: pd.DataFrame, learner: str, start: str = EXAM_START, end: str = VALID_END,
         features: tuple = FEATURES) -> pd.DataFrame:
    """Out-of-sample predictions of every book, refit monthly; ``table`` = state joined with outcomes."""
    fit = LEARNERS[learner]
    feats = [f for f in features if f in table.columns]
    parts = []
    for m0 in pd.date_range(start, end, freq="MS"):
        test = table[(table.index >= m0) & (table.index < m0 + pd.DateOffset(months=1))]
        learn = table[table.index <= m0 - pd.Timedelta(days=GAP_DAYS)]
        if test.empty or len(learn) < MIN_TRAIN_DAYS:
            continue
        part = pd.DataFrame(index=test.index)
        for b in BOOKS:
            ok = learn[b].notna()
            part[f"pred_{b}"] = fit(learn.loc[ok, feats], learn.loc[ok, b], test[feats])
        parts.append(part)
    return pd.concat(parts)


# switches and grades

def rule_switches(table: pd.DataFrame, name: str) -> pd.DataFrame:
    """Fixed rules to beat, decided by the same day's close."""
    on = pd.DataFrame(index=table.index)
    up, alt = table["btc_trend"] > 0, table["beat_btc_30d"] > 0.5
    if name == "all":
        on["long"], on["short"], on["btc"] = True, True, True
    elif name == "pairs":                                       # N1 / P1: the best long, the worst short
        on["long"], on["short"], on["btc"] = True, True, False
    elif name == "rule":                                        # N3 + B1: no shorts in alt season, BTC when up
        on["long"], on["short"], on["btc"] = True, ~alt, up
    elif name == "btc_trend":
        on["long"], on["short"], on["btc"] = False, False, up
    else:
        raise KeyError(name)
    return on.astype(bool)


def learned_switches(pred: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({b: pred[f"pred_{b}"] > 0 for b in BOOKS}, index=pred.index)


def tilted(pred: pd.DataFrame, outcomes: pd.DataFrame, cap: float = 2.0) -> pd.DataFrame:
    """Sizes instead of switches: a book's weight is its predicted outcome against its usual one (known then).

    1 when the forecast is the book's average, 0 at a forecast of 0 or less, ``cap`` at most; on average about
    as much money at work as with every book always on.
    """
    calendar = pd.date_range(outcomes.index.min(), outcomes.index.max())
    usual = outcomes.reindex(calendar).shift(GAP_DAYS).expanding(min_periods=90).mean().reindex(pred.index)
    out = pd.DataFrame(index=pred.index)
    for b in BOOKS:
        base = usual[b]
        w = (pred[f"pred_{b}"] / base).where(base > 0, (pred[f"pred_{b}"] > 0).astype(float))
        out[b] = w.clip(0.0, cap).fillna(1.0)
    return out


def mix(on: pd.DataFrame, outcomes: pd.DataFrame) -> pd.Series:
    """A day's result of a book of thirds: each switched-on book gets a third, the rest stays in cash."""
    out = outcomes.reindex(on.index)[list(BOOKS)].fillna(0.0)
    return (on[list(BOOKS)].astype(float) * out).sum(axis=1) / len(BOOKS)


def period_of(index: pd.DatetimeIndex) -> pd.Series:
    """By the day an outcome is known (GAP_DAYS after its signal), like walkforward.period."""
    known = index + pd.Timedelta(days=GAP_DAYS)
    return pd.Series(np.where(known <= pd.Timestamp(DEV_END), "2022–2024", "2025"), index=index)


def known_by(outcomes: pd.DataFrame, end: str = VALID_END) -> pd.DataFrame:
    """Only signal days whose outcome is known by ``end``: no price after it is booked."""
    return outcomes[outcomes.index + pd.Timedelta(days=GAP_DAYS) <= pd.Timestamp(end)]


def grade(on: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per period: $ per $100 per signal day, days in profit of 10, the bad tenth, how often each book was on."""
    result = mix(on, outcomes)
    rows = []
    for name, idx in result.groupby(period_of(result.index)).groups.items():
        r, o = result.loc[idx], on.loc[idx]
        active = o.any(axis=1)
        rows.append({"period": name, "usd_per_100": 100 * r.mean(), "up_of10": 10 * (r[active] > 0).mean(),
                     "bad_tenth": 100 * r.quantile(0.1), "days": len(r),
                     **{f"on_{b}": 10 * o[b].mean() for b in BOOKS}})
    return pd.DataFrame(rows).set_index("period")


def by_year(on: pd.DataFrame, outcomes: pd.DataFrame) -> pd.Series:
    result = mix(on, outcomes)
    return 100 * result.groupby(result.index.year).mean()


def switching_value(on: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per book and period: the mean outcome on the days it was on and off. A useful switch: on > 0 > off."""
    rows = []
    out = outcomes.reindex(on.index)
    for name, idx in on.groupby(period_of(on.index)).groups.items():
        for b in BOOKS:
            o, y = on.loc[idx, b].astype(float) > 0, out.loc[idx, b]
            rows.append({"period": name, "book": b, "on_of10": 10 * o.mean(),
                         "when_on": 100 * y[o].mean() if o.any() else np.nan,
                         "when_off": 100 * y[~o].mean() if (~o).any() else np.nan})
    return pd.DataFrame(rows)
