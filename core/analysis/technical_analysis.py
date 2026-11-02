from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

def _to_series(x):
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(x)
    return pd.Series([x])

def sma(series, window: int):
    series = _to_series(series)
    if window <= 0:
        raise ValueError("window must be > 0")
    return series.rolling(window=window, min_periods=window).mean()

def ema(series, span: int):
    series = _to_series(series)
    if span <= 0:
        raise ValueError("span must be > 0")
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series, period: int = 14):
    series = _to_series(series)
    if period <= 0:
        raise ValueError("period must be > 0")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    # Only gains -> 100, flat -> 50 (was NaN exactly at the extreme). Warm-up
    # values stay NaN: the old bfill() copied future values backwards.
    rsi_val = rsi_val.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi_val = rsi_val.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi_val

def atr(high, low, close, period: int = 14):
    high  = _to_series(high)
    low   = _to_series(low)
    close = _to_series(close)
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, close must have same length")
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def macd(series, fast: int = 12, slow: int = 26, signal: int = 9):
    series = _to_series(series)
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def realized_volatility(series: pd.Series, window: int = 14, annualize: bool = True) -> float:
    """
    Calculate realized volatility (annualized by default)

    Args:
        series: Price series (close prices)
        window: Rolling window for calculation (default: 14 periods)
        annualize: If True, annualize volatility assuming 365 days per year

    Returns:
        Realized volatility as percentage (e.g., 50.0 for 50% annual volatility)
    """
    series = _to_series(series)

    if len(series) < 2:
        return 0.0

    # Calculate log returns
    log_returns = np.log(series / series.shift(1))

    # Calculate rolling standard deviation
    vol = log_returns.rolling(window=window, min_periods=max(2, window // 2)).std()

    # Get the most recent volatility
    current_vol = vol.iloc[-1] if not vol.empty else 0.0

    if np.isnan(current_vol):
        return 0.0

    # Annualize if requested (assuming daily data)
    if annualize:
        current_vol = current_vol * np.sqrt(365)

    # Convert to percentage
    return float(current_vol * 100)
