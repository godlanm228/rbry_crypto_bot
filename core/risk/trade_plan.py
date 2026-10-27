"""🎯 Trade plan: stop, target, size and the week ahead — computed, not guessed.

The bot has no proven edge on direction: the 14.10.2026 backtest found none of
33 signals beating a random entry after costs. So a plan never says "buy". It
answers "if you go in, here is the arithmetic":

  * stop under the nearest support with a volatility buffer (ATR), never
    closer than one ATR, never further than three — else a plain 2 ATR stop;
  * target at the next resistance at least one ATR away, else 2R;
  * size from the user's risk style, so the stop costs exactly the budget;
  * where the price ends a week from now in 1 of 6, 2 of 3 and 1 of 6 cases;
  * how often the target comes before the stop;
  * for a weak setup, a better level to wait for instead of a nudge to buy.

Both probabilities are calibrated on real candles (constants below), not
taken from the textbook normal distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from core.analysis.signal_generator import find_support_resistance
from core.analysis.technical_analysis import atr as atr_series
from core.risk.models import DrawdownState
from core.risk.service import RiskStyle, TradeSizing, size_trade

ATR_PERIOD = 14
MIN_CANDLES = 60                # S/R needs 50, the trend 60
STOP_BUFFER_ATR = 0.5           # below the support, so a wick does not stop us out
MIN_STOP_ATR = 1.0              # a closer stop is noise
MAX_STOP_ATR = 3.0              # a further support is not this trade's stop
FALLBACK_STOP_ATR = 2.0
MIN_TARGET_ATR = 1.0            # a resistance right overhead is not a target
FALLBACK_TARGET_R = 2.0
GOOD_RR, OK_RR = 2.0, 1.5
BETTER_ENTRY_MAX_PCT = 15.0     # do not suggest waiting for a level further away
BETTER_ENTRY_MIN_ATR = 0.5      # nor one we are practically at
WEEK_DAYS = 7
SIGMA_LOOKBACK = 90
# Where 1 in 6 weeks end, in units of the 7-day sigma (sigma from the prior 90
# daily log returns). Calibrated 26.10.2026 on 20 Binance coins x 2.7 years of
# daily candles, 5944 windows: the empirical 1/6 and 5/6 quantiles are -0.89 and
# +0.80 (the normal distribution would say +-0.97); both halves of the period
# agree within 1.5 points.
WEEK_K_DOWN, WEEK_K_UP = 0.89, 0.80
# The driftless formula ln(E/S) / ln(T/S) overstated how often the target came
# first on the same data (31% real vs 37% formula at 1.5/3 ATR, 30% vs 38% at
# 2/4 ATR; a day touching both counts as the stop): scaled down to match.
HIT_CALIBRATION = 0.86


class PlanUnavailable(Exception):
    """Not enough clean daily candles to place levels honestly."""


@dataclass(frozen=True)
class Candles:
    highs: tuple[float, ...]
    lows: tuple[float, ...]
    closes: tuple[float, ...]


@dataclass(frozen=True)
class WeekRange:
    low: float                  # 1 in 6 weeks end below
    high: float                 # 1 in 6 weeks end above


@dataclass(frozen=True)
class BetterEntry:
    price: float
    stop: float
    reward_risk: float
    hit_chance: float


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    price: float
    atr: float
    stop: float
    stop_basis: str             # "support" | "volatility"
    support: Optional[float]
    target: float
    target_basis: str           # "resistance" | "two_r"
    reward_risk: float
    hit_chance: float           # target before stop
    trend: str                  # "up" | "down" | "flat"
    verdict: str                # "good" | "ok" | "weak"
    week: Optional[WeekRange]
    better: Optional[BetterEntry]
    sizing: TradeSizing


def candles_from_frame(frame: Optional[pd.DataFrame], today: Optional[pd.Timestamp] = None) -> Optional[Candles]:
    """Closed daily candles from a history frame; None without high/low.

    The snapshot's last row is today's still-open candle: its high and low are
    not final, so it is dropped.
    """
    if frame is None or frame.empty or not {"high", "low"} <= set(frame.columns):
        return None
    frame = frame.dropna(subset=["high", "low", "price"])
    if "timestamp" in frame.columns:
        today = today if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        frame = frame[pd.to_datetime(frame["timestamp"]) < today]
    return Candles(tuple(map(float, frame["high"])), tuple(map(float, frame["low"])),
                   tuple(map(float, frame["price"])))


def candles_from_klines(rows: Optional[Sequence[dict]]) -> Optional[Candles]:
    """Binance klines (oldest first, last one open) -> closed candles."""
    if not rows or len(rows) < 2:
        return None
    rows = rows[:-1]
    return Candles(tuple(r["high"] for r in rows), tuple(r["low"] for r in rows),
                   tuple(r["close"] for r in rows))


def current_atr(c: Candles) -> float:
    series = atr_series(list(c.highs), list(c.lows), list(c.closes), period=ATR_PERIOD)
    value = float(series.iloc[-1])
    if not math.isfinite(value) or value <= 0:
        raise PlanUnavailable("atr")
    return value


def key_levels(c: Candles) -> tuple[list[float], list[float]]:
    frame = pd.DataFrame({"high": c.highs, "low": c.lows, "close": c.closes})
    levels = find_support_resistance(frame)
    return levels["support"], levels["resistance"]


def place_stop(price: float, atr: float, supports: Sequence[float]) -> tuple[float, str, Optional[float]]:
    below = [s for s in supports if s < price]
    support = max(below) if below else None
    if support is not None:
        stop = min(support - STOP_BUFFER_ATR * atr, price - MIN_STOP_ATR * atr)
        if price - stop <= MAX_STOP_ATR * atr:
            return stop, "support", support
    return price - FALLBACK_STOP_ATR * atr, "volatility", support


def place_target(price: float, atr: float, stop: float, resistances: Sequence[float]) -> tuple[float, str]:
    above = [r for r in resistances if r >= price + MIN_TARGET_ATR * atr]
    if above:
        return min(above), "resistance"
    return price + FALLBACK_TARGET_R * (price - stop), "two_r"


def hit_chance(entry: float, stop: float, target: float) -> float:
    """How often the target comes before the stop (calibrated, no edge assumed)."""
    raw = math.log(entry / stop) / math.log(target / stop)
    return max(0.0, min(1.0, HIT_CALIBRATION * raw))


def week_range(price: float, closes: Sequence[float]) -> Optional[WeekRange]:
    window = closes[-(SIGMA_LOOKBACK + 1):]
    if len(window) < 31:
        return None
    returns = [math.log(b / a) for a, b in zip(window, window[1:]) if a > 0 and b > 0]
    mean = sum(returns) / len(returns)
    sigma = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
    sigma_week = sigma * math.sqrt(WEEK_DAYS)
    return WeekRange(low=price * math.exp(-WEEK_K_DOWN * sigma_week),
                     high=price * math.exp(WEEK_K_UP * sigma_week))


def trend_of(price: float, closes: Sequence[float]) -> str:
    """Up/down when the price and the 50-day average's slope agree."""
    if len(closes) < 60:
        return "flat"
    sma_now = sum(closes[-50:]) / 50
    sma_before = sum(closes[-60:-10]) / 50
    if price > sma_now and sma_now > sma_before:
        return "up"
    if price < sma_now and sma_now < sma_before:
        return "down"
    return "flat"


def verdict_for(reward_risk: float, trend: str) -> str:
    verdict = "good" if reward_risk >= GOOD_RR else "ok" if reward_risk >= OK_RR else "weak"
    if trend == "down" and verdict == "good":
        verdict = "ok"
    return verdict


def better_entry(price: float, atr: float, support: Optional[float], target: float) -> Optional[BetterEntry]:
    """Just above the support, with the stop below it: a level worth waiting for."""
    if support is None:
        return None
    entry = support + 0.25 * atr
    if price - entry < BETTER_ENTRY_MIN_ATR * atr:
        return None
    if (price - entry) / price * 100.0 > BETTER_ENTRY_MAX_PCT:
        return None
    stop = min(support - STOP_BUFFER_ATR * atr, entry - MIN_STOP_ATR * atr)
    if stop <= 0 or target <= entry:
        return None
    reward_risk = (target - entry) / (entry - stop)
    if reward_risk < GOOD_RR:
        return None
    return BetterEntry(entry, stop, reward_risk, hit_chance(entry, stop, target))


def build_trade_plan(
    symbol: str,
    price: float,
    candles: Optional[Candles],
    *,
    user_id: int,
    style: RiskStyle,
    account_value: float,
    existing_position_value: float = 0.0,
    state: DrawdownState = DrawdownState.NORMAL,
) -> TradePlan:
    if candles is None or len(candles.closes) < MIN_CANDLES:
        raise PlanUnavailable("history")
    if not (math.isfinite(price) and price > 0):
        raise PlanUnavailable("price")
    atr = current_atr(candles)
    supports, resistances = key_levels(candles)
    stop, stop_basis, support = place_stop(price, atr, supports)
    if stop <= 0:
        raise PlanUnavailable("volatility")
    target, target_basis = place_target(price, atr, stop, resistances)
    reward_risk = (target - price) / (price - stop)
    trend = trend_of(price, candles.closes)
    verdict = verdict_for(reward_risk, trend)
    sizing = size_trade(user_id=user_id, style=style, symbol=symbol, entry=price, stop=stop,
                        take=target, account_value=account_value,
                        existing_position_value=existing_position_value, state=state)
    return TradePlan(
        symbol=symbol.upper(),
        price=price,
        atr=atr,
        stop=stop,
        stop_basis=stop_basis,
        support=support,
        target=target,
        target_basis=target_basis,
        reward_risk=reward_risk,
        hit_chance=hit_chance(price, stop, target),
        trend=trend,
        verdict=verdict,
        week=week_range(price, candles.closes),
        better=better_entry(price, atr, support, target) if verdict != "good" else None,
        sizing=sizing,
    )


def pick_alternatives(plans: Sequence[TradePlan], exclude: str, limit: int = 2,
                      beat: float = 0.0) -> list[TradePlan]:
    """Coins whose setup is good right now and beats ``beat`` reward/risk, best first."""
    good = [p for p in plans
            if p.verdict == "good" and p.symbol != exclude.upper() and p.reward_risk > beat]
    return sorted(good, key=lambda p: p.reward_risk, reverse=True)[:limit]
