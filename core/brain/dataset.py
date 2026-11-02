"""The brain's textbook: one row per coin per day, and what happened next.

Every history in market.db counts, delisted and halted ones included, from the
day it has the 200-day warm-up the trade plan needs. A row holds what was known
at that day's close (00:00 UTC) and how one fixed trade ended:

  buy at the open of the next hour, stop 2 ATR below, target twice the risk
  above, sell after 14 days at the latest. Resolved on hourly candles: when
  one hour touches both levels the stop wins, an hour that opens through a
  level exits at that open. Fees and slippage are paid on both sides, and
  slippage grows as the coin's daily turnover shrinks.

The same trade is also kept the other way round, as a short on the coin's
USDT-M perpetual: stop 2 ATR above, target twice the risk below, futures fees,
and the funding a short receives (or pays) while it is open. A coin without a
perpetual at that time cannot be shorted and has no short outcome.

The trade is the same for every row on purpose: the brain learns *when* it
pays, not a new shape for every day. ``r_net`` is the result in units of risk
(-1 = the stop was hit), ``ret_net`` the same as a share of the position, so
100 * ret_net is the $ result per $100 bought. A trade still running at the
newest candle has no label; one whose coin stopped trading is closed at the
last price it had.

Left out on purpose:
- futures open interest and long/short ratios: stored for today's top 100
  pairs only, so "has the data" would tell the model which coins survived;
- 5-minute candles, for the same reason;
- headlines: the archive only starts.
Funding is in: its archive keeps the delisted perpetuals too.

Bump DATASET_VERSION whenever a feature or the trade changes.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from core.sandbox import archive
from core.sandbox.features import WARMUP, build_situations
from core.sandbox.market_store import HOUR_MS, History, MarketStore
from core.sandbox.simulate import Costs, net_return
from core.sandbox.strategies import STRATEGIES

logger = logging.getLogger(__name__)

DATASET_VERSION = "b6"
DAY_MS = 24 * HOUR_MS
STOP_ATR = 2.0
TARGET_R = 2.0
TRADE_DAYS = 14
ENTRY_WITHIN_HOURS = 6        # no candle this soon after the close: the coin was halted, no trade
FEE_PCT = 0.10                # Binance spot taker, per side
FUTURES_FEE_PCT = 0.05        # Binance USDT-M taker, per side
SHORT_CAP = 0.30              # the capped short hold is closed once the coin is this far above the entry
LIQUIDATION = 1.00            # a 1x isolated short is liquidated about here: it cannot lose more than its margin
PERP_SETTLE_MS = 8 * HOUR_MS  # a perpetual's last funding payment: it is settled within this time after it
SLIPPAGE_TIERS = ((20e6, 0.05), (5e6, 0.10), (1e6, 0.20), (0.0, 0.40))   # 30-day median turnover -> % per side
MIN_COINS = 20                # fewer coins trading that day: no market breadth
MIN_TURNOVER = 1e6            # 30-day median USDT turnover; below it a market order already moves the price
FUNDING_FROM = "2020-01-01"   # the funding archive's first month

SITUATION_FEATURES = (
    "ret_1d", "ret_7d", "ret_30d", "atr_pct", "vol_regime", "dist_sma20", "dist_sma50", "dist_sma200",
    "sma50_slope", "rsi14", "range_pos_30", "dd_90", "volume_ratio", "support_dist", "resistance_dist",
    "rs_30d", "plan_rr",
)
COIN_FEATURES = (
    "ret_90d", "ret_365d", "dd_365",          # longer memory
    "day_range", "close_pos",                 # the last candle's shape
    "liquidity", "taker_buy_7d", "trades_ratio", "age",
    "funding_1d", "funding_7d", "perp_listed", "delist_notice",
)
MARKET_FEATURES = (
    "btc_ret_7d", "btc_ret_30d", "btc_trend", "btc_atr_pct",
    "beat_btc_1d", "beat_btc_7d", "beat_btc_30d", "beat_btc_7d_vw",   # breadth: share of coins ahead of BTC
    "above_sma50", "alt_median_7d", "btc_volume_share", "coins_trading",
    "rs_rank", "ret_rank_7d", "liquidity_rank",
    "spx_4w", "dxy_4w", "vix", "us10y_4w", "gold_4w", "stables_4w", "fear_greed", "fear_greed_7d",
    "att_bitcoin_7d", "att_crypto_7d", "att_ethereum_7d",       # Wikipedia views: last week against 90 days
)
RULE_STRATEGIES = tuple(s for s in STRATEGIES if s.name not in ("control", "control_3r", "plan"))   # "plan" fires every day
RULE_FEATURES = tuple(f"rule_{s.name}" for s in RULE_STRATEGIES)
FEATURES = SITUATION_FEATURES + COIN_FEATURES + MARKET_FEATURES + RULE_FEATURES
# what the brain ranks coins by: everything about the coin, and where it stands among that day's coins;
# market-wide values are the same for every coin of a day and let a model memorise good months
CROSS_SECTION = ("rs_rank", "ret_rank_7d", "liquidity_rank")
RANKING_FEATURES = SITUATION_FEATURES + COIN_FEATURES + CROSS_SECTION + RULE_FEATURES
LABELS = ("r_net", "ret_net", "hold_ret", "excess_14d", "reason", "hours",
          "short_r_net", "short_ret_net", "short_hold_ret", "short_reason", "short_hours", "funding_hold",
          "short_cap_ret", "short_cap_hours", "short_liq_ret", "short_liq_hours")
SIDES = ("r_net", "ret_net", "hold_ret", "reason", "hours",
         "short_r_net", "short_ret_net", "short_hold_ret", "short_reason", "short_hours", "funding_hold",
         "short_cap_ret", "short_cap_hours", "short_liq_ret", "short_liq_hours")
REASONS = {1: "target", -1: "stop", 0: "time", 2: "ended"}
# situation columns the rules need, kept next to the features
RULE_INPUTS = ("close", "atr", "sma20", "sma50", "high20_prev", "support", "plan_stop", "better_entry", "verdict")
CATEGORIES = ("history", "base", "verdict")


def slippage_pct(turnover: np.ndarray) -> np.ndarray:
    out = np.full(len(turnover), SLIPPAGE_TIERS[-1][1])
    for floor, pct in reversed(SLIPPAGE_TIERS):
        out = np.where(np.nan_to_num(turnover) >= floor, pct, out)
    return out


# labels

def short_return(entry: float, exit_: float, slippage_pct: float, funding: float = 0.0) -> float:
    """A short's result as a share of its size: sold at ``entry``, bought back at ``exit_``, plus funding received."""
    s, f = slippage_pct / 100.0, FUTURES_FEE_PCT / 100.0
    return (entry * (1 - s) * (1 - f) - exit_ * (1 + s) * (1 + f)) / entry + funding


def _first(mask: np.ndarray, span: int) -> int:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else span


def trade_outcomes(hour_ts: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                   day_ts: np.ndarray, atr: np.ndarray, slippage: np.ndarray, ended: bool,
                   funding_ts: Optional[np.ndarray] = None, funding_rate: Optional[np.ndarray] = None
                   ) -> dict[str, np.ndarray]:
    """The fixed trade, long and short, for every day in ``day_ts`` (UTC midnight of the signal day).

    ``funding_ts`` / ``funding_rate``: the coin's perpetual; without them there is no short.
    """
    n = len(day_ts)
    out = {k: np.full(n, np.nan) for k in SIDES}
    if not len(hour_ts):
        return out
    has_perp = funding_ts is not None and len(funding_ts) > 0
    if has_perp:
        paid = np.r_[0.0, np.cumsum(funding_rate)]       # funding between two times: paid[j] - paid[i]

        def funding_between(t0: int, t1: int) -> float:
            return float(paid[np.searchsorted(funding_ts, t1, side="right")] - paid[np.searchsorted(funding_ts, t0)])
    starts = np.searchsorted(hour_ts, day_ts + DAY_MS)
    ends = np.searchsorted(hour_ts, day_ts + (1 + TRADE_DAYS) * DAY_MS)
    last_ts = hour_ts[-1]
    for i in range(n):
        a, b = int(starts[i]), int(ends[i])
        if a >= len(hour_ts) or hour_ts[a] >= day_ts[i] + DAY_MS + ENTRY_WITHIN_HOURS * HOUR_MS:
            continue
        if not (math.isfinite(atr[i]) and atr[i] > 0):
            continue
        complete = day_ts[i] + (1 + TRADE_DAYS) * DAY_MS - HOUR_MS <= last_ts
        if not complete and not ended:
            continue                      # still running at the newest candle
        entry = float(o[a])
        risk = STOP_ATR * float(atr[i])
        stop, target = entry - risk, entry + TARGET_R * risk
        if stop <= 0:
            continue
        span = b - a
        lo, hi, op = l[a:b], h[a:b], o[a:b]
        hits = np.flatnonzero(lo <= stop)
        ks = int(hits[0]) if len(hits) else span
        hits = np.flatnonzero(hi >= target)
        kt = int(hits[0]) if len(hits) else span
        if ks < kt or (ks == kt < span and op[ks] < target):
            exit_, reason, k = min(float(op[ks]), stop), -1, ks
        elif kt < span:
            exit_, reason, k = max(float(op[kt]), target), 1, kt
        else:
            exit_, reason, k = float(c[b - 1]), (0 if complete else 2), span - 1
        costs = Costs(fee_pct=FEE_PCT, slippage_pct=float(slippage[i]))
        ret = net_return(entry, exit_, costs)
        out["ret_net"][i] = ret
        out["r_net"][i] = ret / (risk / entry)
        out["hold_ret"][i] = net_return(entry, float(c[b - 1]), costs)
        out["reason"][i] = reason
        out["hours"][i] = k + 1

        if not has_perp or entry - TARGET_R * risk <= 0:
            continue                      # no perpetual, or the target would be below zero
        t0 = int(hour_ts[a])
        # the perpetual must be trading at the entry (a funding payment within a day of it), and a
        # perpetual that stops during the hold settles the short at its last price
        if np.searchsorted(funding_ts, t0 + DAY_MS, side="right") == np.searchsorted(funding_ts, t0):
            continue
        sb = min(b, int(np.searchsorted(hour_ts, int(funding_ts[-1]) + PERP_SETTLE_MS, side="right")))
        if sb <= a:
            continue
        s_span, s_complete = sb - a, complete and sb == b
        s_lo, s_hi, s_op = l[a:sb], h[a:sb], o[a:sb]
        s_stop, s_target = entry + risk, entry - TARGET_R * risk
        ks, kt = _first(s_hi >= s_stop, s_span), _first(s_lo <= s_target, s_span)
        if ks < kt or (ks == kt < s_span and s_op[ks] > s_target):
            s_exit, s_reason, sk = max(float(s_op[ks]), s_stop), -1, ks
        elif kt < s_span:
            s_exit, s_reason, sk = min(float(s_op[kt]), s_target), 1, kt
        else:
            s_exit, s_reason, sk = float(c[sb - 1]), (0 if s_complete else 2), s_span - 1
        slip = float(slippage[i])
        held = funding_between(t0, int(hour_ts[a + sk]) + HOUR_MS - 1)
        whole = funding_between(t0, int(hour_ts[sb - 1]) + HOUR_MS - 1)
        short = short_return(entry, s_exit, slip, held)
        out["short_ret_net"][i] = short
        out["short_r_net"][i] = short / (risk / entry)
        out["short_hold_ret"][i] = short_return(entry, float(c[sb - 1]), slip, whole)
        out["short_reason"][i] = s_reason
        out["short_hours"][i] = sk + 1
        out["funding_hold"][i] = whole
        # the 14-day short hold with one protection: out once the coin is SHORT_CAP above the entry
        for name, level in (("short_cap", SHORT_CAP), ("short_liq", LIQUIDATION)):
            cap = entry * (1 + level)
            kc = _first(s_hi >= cap, s_span)
            if kc < s_span:
                c_exit, ck = max(float(s_op[kc]), cap), kc
            else:
                c_exit, ck = float(c[sb - 1]), s_span - 1
            result = short_return(entry, c_exit, slip, funding_between(t0, int(hour_ts[a + ck]) + HOUR_MS - 1))
            # liquidated: the margin is gone, no more (the exchange closes it, not the market at the open)
            out[f"{name}_ret"][i] = max(result, -1.0) if name == "short_liq" else result
            out[f"{name}_hours"][i] = ck + 1
    return out


# per coin

def _daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"day": pd.to_datetime(daily["ts"], unit="ms"), "open": daily["open"],
                         "high": daily["high"], "low": daily["low"], "close": daily["close"],
                         "volume": daily["quote_volume"]})


def coin_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-day features beyond the situation vector; ``daily`` has all candle fields."""
    close, high, low = (daily[k].astype(float) for k in ("close", "high", "low"))
    logc = np.log(close)
    turnover = daily["quote_volume"].astype(float)
    trades = daily["trades"].astype(float)
    spread = (high - low).replace(0, np.nan)
    out = pd.DataFrame({
        "ret_90d": logc - logc.shift(90),
        "ret_365d": logc - logc.shift(365),
        "dd_365": close / high.rolling(365, min_periods=WARMUP).max() - 1.0,
        "day_range": (high - low) / close,
        "close_pos": (close - low) / spread,
        "liquidity": np.log10(turnover.rolling(30, min_periods=20).median().clip(lower=1.0)),
        "turnover_30d": turnover.rolling(30, min_periods=20).median(),
        "taker_buy_7d": daily["taker_buy_quote"].rolling(7).sum() / turnover.rolling(7).sum().replace(0, np.nan),
        "trades_ratio": np.log(trades.rolling(7).mean() / trades.rolling(90, min_periods=30).mean()),
        "age": np.log1p(np.arange(len(daily), dtype=float)),
    })
    return out.replace([np.inf, -np.inf], np.nan)


def funding_by_day(rates: pd.DataFrame) -> pd.DataFrame:
    """Funding paid per UTC day and over the last 7 days (sum of rates; positive = longs pay)."""
    if rates.empty:
        return pd.DataFrame(columns=["funding_1d", "funding_7d"])
    day = pd.to_datetime(rates["ts"] // DAY_MS * DAY_MS, unit="ms")
    per_day = rates.groupby(day)["rate"].sum()
    per_day = per_day.reindex(pd.date_range(per_day.index.min(), per_day.index.max(), freq="D"), fill_value=0.0)
    return pd.DataFrame({"funding_1d": per_day, "funding_7d": per_day.rolling(7, min_periods=1).sum()})


def light_rows(history: History) -> pd.DataFrame:
    """Every day of a history, a few numbers each: enough for breadth and the day's ranks."""
    frame = _daily_frame(history.daily)
    close = frame["close"].astype(float)
    logc = np.log(close)
    sma50 = close.rolling(50, min_periods=50).mean()
    turnover = frame["volume"].astype(float)
    out = pd.DataFrame({
        "day": frame["day"], "history": history.name, "base": history.base,
        "ret_1d": logc.diff(), "ret_7d": logc - logc.shift(7), "ret_30d": logc - logc.shift(30),
        "above_sma50": (close > sma50).astype(float).where(sma50.notna()), "turnover": turnover,
        "liquidity": np.log10(turnover.rolling(30, min_periods=20).median().clip(lower=1.0)),
        # the days a dataset row comes from: after the warm-up of a long enough history
        "warm": (np.arange(len(frame)) >= WARMUP - 1) & (len(frame) >= WARMUP + 1),
    })
    return out.astype({c: np.float32 for c in ("ret_1d", "ret_7d", "ret_30d", "above_sma50", "liquidity")})


def coin_rows(history: History, btc: pd.DataFrame, fng: Optional[pd.DataFrame],
              funding: Optional[pd.DataFrame], notices: Iterable[int]) -> pd.DataFrame:
    """Rows after the warm-up with the coin's own features and the trade's outcome."""
    daily = history.daily
    frame = _daily_frame(daily)
    if len(frame) < WARMUP + 1:
        return pd.DataFrame()
    sit = build_situations(frame, btc, fng)
    extra = coin_features(daily).iloc[WARMUP - 1:].reset_index(drop=True)
    rows = pd.concat([sit, extra], axis=1)
    rows.insert(0, "history", history.name)
    rows.insert(1, "base", history.base)

    if funding is not None and not funding.empty:
        rows = rows.join(funding_by_day(funding), on="day")
        listed_from = pd.to_datetime(funding["ts"].min(), unit="ms").normalize()
        rows["perp_listed"] = (rows["day"] >= listed_from).astype(float)
        rows[["funding_1d", "funding_7d"]] = rows[["funding_1d", "funding_7d"]].where(rows["perp_listed"] > 0)
    else:
        rows["funding_1d"], rows["funding_7d"], rows["perp_listed"] = np.nan, np.nan, 0.0
    early = rows["day"] < FUNDING_FROM
    rows.loc[early, ["funding_1d", "funding_7d", "perp_listed"]] = np.nan

    day_ts = ((rows["day"] - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).to_numpy(dtype=np.int64)
    published = np.sort(np.asarray(list(notices), dtype=np.int64))
    if len(published):
        # a notice published before the day's close, at most 30 days earlier
        closes = day_ts + DAY_MS
        recent = np.searchsorted(published, closes) - np.searchsorted(published, closes - 30 * DAY_MS)
        rows["delist_notice"] = (recent > 0).astype(float)
    else:
        rows["delist_notice"] = 0.0

    hourly = history.hourly
    perp = funding is not None and not funding.empty
    outcome = trade_outcomes(hourly["ts"].to_numpy(dtype=np.int64), *(hourly[k].to_numpy(dtype=float)
                             for k in ("open", "high", "low", "close")),
                             day_ts, rows["atr"].to_numpy(dtype=float),
                             slippage_pct(rows["turnover_30d"].to_numpy(dtype=float)), history.ended,
                             funding["ts"].to_numpy(dtype=np.int64) if perp else None,
                             funding["rate"].to_numpy(dtype=float) if perp else None)
    for k, v in outcome.items():
        rows[k] = v
    return rows


# the market around the coin

def breadth(light: pd.DataFrame) -> pd.DataFrame:
    """Per day: how the rest of the market did against BTC, from every coin trading that day."""
    btc = light[light["base"] == "BTC"].drop_duplicates("day").set_index("day")
    alts = light[light["base"] != "BTC"].copy()
    for k in ("ret_1d", "ret_7d", "ret_30d"):
        alts[f"beat_{k}"] = (alts[k] > alts["day"].map(btc[k])).astype(float).where(alts[k].notna())
    alts["w"] = alts["turnover"].where(alts["ret_7d"].notna())
    alts["w_beat"] = alts["w"] * alts["beat_ret_7d"]
    g = alts.groupby("day")
    out = pd.DataFrame({
        "beat_btc_1d": g["beat_ret_1d"].mean(), "beat_btc_7d": g["beat_ret_7d"].mean(),
        "beat_btc_30d": g["beat_ret_30d"].mean(),
        "beat_btc_7d_vw": g["w_beat"].sum() / g["w"].sum().replace(0, np.nan),
        "above_sma50": g["above_sma50"].mean(),
        "alt_median_7d": g["ret_7d"].median() - btc["ret_7d"].reindex(g["ret_7d"].median().index),
        "coins_trading": np.log(g["ret_1d"].count().clip(lower=1)),
    })
    total = light.groupby("day")["turnover"].sum().rolling(7, min_periods=1).sum()
    out["btc_volume_share"] = btc["turnover"].rolling(7, min_periods=1).sum().reindex(out.index) / total.reindex(out.index)
    few = g["ret_1d"].count() < MIN_COINS
    out.loc[few.reindex(out.index, fill_value=True)] = np.nan
    return out


def macro_context(store: MarketStore) -> pd.DataFrame:
    """Per calendar day; a US close comes before the crypto close, weekends carry Friday's value.

    Fear & Greed comes with the situation vector (build_situations).
    """
    def series(name: str) -> pd.Series:
        s = store.series(name)
        if s.empty:
            return pd.Series(dtype=float)
        s = s.drop_duplicates("day", keep="last").set_index("day")["value"].astype(float)
        return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill(limit=4)

    spx, dxy, vix, us10y, gold = (series(n) for n in ("spx", "dxy", "vix", "us10y", "gold"))
    stables = series("stablecoins_usd")
    attention = {name: series(f"wiki_{name}") for name in ("bitcoin", "crypto", "ethereum")}
    out = pd.DataFrame({
        "spx_4w": np.log(spx / spx.shift(28)), "dxy_4w": np.log(dxy / dxy.shift(28)), "vix": np.log(vix),
        "us10y_4w": us10y - us10y.shift(28), "gold_4w": np.log(gold / gold.shift(28)),
        "stables_4w": np.log(stables / stables.shift(28)),
        **{f"att_{name}_7d": np.log(v.rolling(7, min_periods=5).mean() / v.rolling(90, min_periods=60).mean())
           for name, v in attention.items()},
    })
    return out.replace([np.inf, -np.inf], np.nan)


def cross_section(light: pd.DataFrame) -> pd.DataFrame:
    """Each warm coin-day's place among that day's warm coins, indexed by (history, day)."""
    btc = light[light["base"] == "BTC"].drop_duplicates("day").set_index("day")
    warm = light[light["warm"]].copy()
    warm["rs_30d"] = warm["ret_30d"] - warm["day"].map(btc["ret_30d"])
    by_day = warm.groupby("day")
    out = pd.DataFrame({"history": warm["history"], "day": warm["day"]})
    for col, src in (("rs_rank", "rs_30d"), ("ret_rank_7d", "ret_7d"), ("liquidity_rank", "liquidity")):
        out[col] = by_day[src].rank(pct=True)
    out.loc[by_day["ret_7d"].transform("count") < MIN_COINS // 2, ["rs_rank", "ret_rank_7d", "liquidity_rank"]] = np.nan
    return out.set_index(["history", "day"])


def market_table(light: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    btc = light[light["base"] == "BTC"].drop_duplicates("day").set_index("day")
    out = breadth(light).join(macro, how="outer")
    out["btc_ret_7d"] = btc["ret_7d"].reindex(out.index)
    return out


def finish(rows: pd.DataFrame, market: pd.DataFrame, ranks: pd.DataFrame, btc_rows: pd.DataFrame) -> pd.DataFrame:
    """A coin's rows with each day's market, its place among the day's coins, and the rules."""
    rows = rows.join(market, on="day")
    rows = rows.join(ranks, on=["history", "day"])
    rows["btc_atr_pct"] = rows["day"].map(btc_rows["atr_pct"])
    # the coin against BTC over the same 14 days, both held without stops
    rows["excess_14d"] = rows["hold_ret"] - rows["day"].map(btc_rows["hold_ret"])
    for s in RULE_STRATEGIES:
        rows[f"rule_{s.name}"] = s.when(rows).fillna(False).astype(float)
    return rows


# everything

def build(store: MarketStore, symbols: Optional[Iterable[str]] = None,
          features: tuple[str, ...] = FEATURES) -> pd.DataFrame:
    """Every tradable coin-day after the warm-up: keys history/base/day, ``features``, labels.

    Two passes so that one coin's candles at a time are in memory: first a few
    numbers per day of every history (breadth, the day's ranks), then each coin
    in full, joined with them and cut down to what is kept.
    """
    fng = store.series("fear_greed")
    notices = store.delisting_notices()
    spot = {b for _, b, _ in store.symbols()}
    with store.connect() as conn:
        perpetuals = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM funding")]
    perp_of = {base: pair for pair, base in archive.futures_pairs(perpetuals, spot).items()}

    light = pd.concat([light_rows(h) for h in store.histories(min_days=60, symbols=symbols)], ignore_index=True)
    market, ranks = market_table(light, macro_context(store)), cross_section(light)
    long_enough = set(light.loc[light["warm"], "history"])
    del light
    _release_memory()
    logger.info("dataset: market of %d days, %d histories to build", len(market), len(long_enough))

    def rows_of(history: History, btc_daily: pd.DataFrame) -> pd.DataFrame:
        funding = store.funding(perp_of[history.base]) if history.base in perp_of else None
        return coin_rows(history, btc_daily, fng, funding, notices.get(history.base, ()))

    btc_history = next(h for h in store.histories(min_days=WARMUP + 1, symbols={"BTCUSDT"}) if h.name == "BTC")
    btc = _daily_frame(btc_history.daily)
    btc_rows = rows_of(btc_history, btc).set_index("day")
    keep = ["history", "base", "day", "turnover_30d", *features, *LABELS, *RULE_INPUTS]
    parts: list[dict[str, np.ndarray]] = []      # column -> values, one dict per history
    for history in store.histories(min_days=WARMUP + 1, symbols=symbols):
        if history.name not in long_enough:
            continue
        rows = finish(rows_of(history, btc), market, ranks, btc_rows)
        rows = _compact(rows.loc[rows["turnover_30d"] >= MIN_TURNOVER, keep])
        if not rows.empty:
            parts.append({c: rows[c].to_numpy() for c in keep})
        if len(parts) % 50 == 0 and parts:
            _release_memory()            # a coin's temporary frames come and go; keep the heap from growing
        if len(parts) % 100 == 0 and parts:
            logger.info("dataset: %d histories, %d rows", len(parts), sum(len(x["day"]) for x in parts))
    # one column at a time, freeing the pieces as it goes: the peak is the table plus one column, not two tables
    columns = {}
    for c in keep:
        pieces = [x.pop(c) for x in parts]
        columns[c] = pd.Categorical(np.concatenate(pieces)) if c in CATEGORIES else np.concatenate(pieces)
        del pieces
    del parts
    _release_memory()
    return pd.DataFrame(columns)


# storage: one .npy file per column, read straight from disk

def save_table(frame: pd.DataFrame, folder: Path) -> None:
    """Every column as its own .npy (categories as codes): a reader maps them instead of unpickling."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"version": DATASET_VERSION, "rows": len(frame), "columns": []}
    for name in frame.columns:
        col = frame[name]
        if isinstance(col.dtype, pd.CategoricalDtype):
            values, entry = col.cat.codes.to_numpy(), {"kind": "category", "categories": [str(c) for c in col.cat.categories]}
        elif col.dtype.kind == "M":
            values = col.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            entry = {"kind": "datetime", "dtype": str(col.dtype)}          # pandas 3 keeps microseconds
        else:
            values, entry = col.to_numpy(), {"kind": "plain"}
        np.save(folder / f"{name}.npy", values)
        meta["columns"].append({"name": name, **entry})
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_table(folder: Path, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """The table back; plain columns stay memory-mapped, so loading costs almost no memory."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    wanted = set(columns) if columns else None
    data = {}
    for entry in meta["columns"]:
        name = entry["name"]
        if wanted and name not in wanted:
            continue
        values = np.load(folder / f"{name}.npy", mmap_mode="r")
        if entry["kind"] == "category":
            data[name] = pd.Categorical.from_codes(np.asarray(values), categories=entry["categories"])
        elif entry["kind"] == "datetime":
            data[name] = np.asarray(values).astype("datetime64[ns]").astype(entry.get("dtype", "datetime64[ns]"))
        else:
            data[name] = values
    return pd.DataFrame(data, copy=False)


def _release_memory() -> None:
    """Hand freed heap back to the OS (glibc keeps it otherwise); a no-op elsewhere."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _compact(frame: pd.DataFrame) -> pd.DataFrame:
    floats = frame.select_dtypes(include="float64").columns
    return frame.astype({c: np.float32 for c in floats if c not in ("close", "atr", "support", "plan_stop",
                                                                     "better_entry", "sma20", "sma50", "high20_prev")})
