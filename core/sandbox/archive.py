"""Binance's public archive (data.binance.vision): every pair ever listed.

Monthly files cover finished months; all are zipped CSVs, so the whole
history is a few tens of thousands of small downloads and needs no API key.
Ingested files are recorded in market.db (``fetched``), so a re-run downloads
only what is new.

What is fetched how:
  candles  monthly files for finished months, then the public klines API from
           the last stored candle to now (one call per pair a day)
  metrics  daily files only (the API keeps 30 days): first day of last month
           to yesterday; a file missing three or more days back is recorded as
           missing (not trading yet, or halted), a fresher one is retried
Tokenized stocks (AAPLB, TSLAB, ...) and new pegged tokens are stored too: no
exchange field tells them apart, so the dataset filters them by behaviour.

Formats as found:
  spot klines      no header; open time in ms until 2024, in microseconds from 2025
  funding rate     header calc_time,funding_interval_hours,last_funding_rate
  futures metrics  header create_time,symbol,sum_open_interest,... every 5 minutes
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from datetime import date, timedelta
from typing import Iterable, Optional

import httpx
import numpy as np
import pandas as pd

from core.sandbox.data import EXCLUDED, KLINES_URLS, _get_json

ARCHIVE_URL = "https://data.binance.vision"
LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
SPOT_MONTHLY = "data/spot/monthly/klines/"
FUNDING_MONTHLY = "data/futures/um/monthly/fundingRate/"
METRICS_DAILY = "data/futures/um/daily/metrics/"
KLINE_CSV = ["ts", "open", "high", "low", "close", "volume", "close_ts", "quote_volume", "trades",
             "taker_buy_base", "taker_buy_quote"]
MISSING_AFTER_DAYS = 3
TF_MS = {"1h": 3_600_000, "5m": 300_000}

FIAT = {"EUR", "GBP", "AUD", "BRL", "TRY", "RUB", "UAH", "NGN", "ZAR", "IDRT", "BIDR", "BKRW", "JPY",
        "MXN", "ARS", "PLN", "RON", "CZK", "COP", "BVND"}
# Pegged to a currency or another coin: not separate bets.
PEGGED = set(EXCLUDED) | {"UST", "USTC", "SUSD", "USDS", "USDSB", "PAX", "VAI", "XUSD", "EURI", "GUSD", "FRAX",
                          "LUSD", "USDQ", "USDB"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


# universe

def _leveraged(base: str, bases: set[str]) -> bool:
    """BTCUP, ETHBEAR, plain BULL/BEAR (3x BTC); JUP or SYRUP are real coins."""
    if base in {"BULL", "BEAR"}:
        return True
    return any(base.endswith(s) and len(base) - len(s) >= 2 and base[:-len(s)] in bases for s in LEVERAGED_SUFFIXES)


def usdt_bases(pairs: Iterable[str]) -> dict[str, str]:
    """{pair: base} for USDT pairs that are separate bets: no stablecoins, fiat, wrapped or leveraged tokens."""
    found = {p: p[:-4] for p in pairs if p.endswith("USDT") and len(p) > 4}
    bases = set(found.values())
    return {p: b for p, b in sorted(found.items())
            if b not in PEGGED and b not in FIAT and not _leveraged(b, bases)}


def futures_pairs(futures: Iterable[str], spot_bases: Iterable[str]) -> dict[str, str]:
    """{USDT-M perpetual: spot base}; 1000PEPEUSDT is PEPE, 1000SATSUSDT is 1000SATS on spot too."""
    spot = set(spot_bases)
    out = {}
    for f in futures:
        if not f.endswith("USDT") or "_" in f:
            continue
        base = f[:-4]
        if base in spot:
            out[f] = base
        else:
            plain = re.sub(r"^(1000000|1000|1M)(?=[A-Z])", "", base)
            if plain in spot:
                out[f] = plain
    return out


# keys

def kline_monthly_key(pair: str, tf: str, month: str) -> str:
    return f"{SPOT_MONTHLY}{pair}/{tf}/{pair}-{tf}-{month}.zip"


def funding_monthly_key(pair: str, month: str) -> str:
    return f"{FUNDING_MONTHLY}{pair}/{pair}-fundingRate-{month}.zip"


def metrics_daily_key(pair: str, day: date) -> str:
    return f"{METRICS_DAILY}{pair}/{pair}-metrics-{day:%Y-%m-%d}.zip"


def key_period(key: str) -> str:
    """'2024-03' or '2024-03-01' from an archive key."""
    m = re.search(r"-(\d{4}-\d{2}(?:-\d{2})?)\.zip$", key)
    return m.group(1) if m else ""


def daily_window(today: date) -> list[date]:
    """First day of last month up to yesterday: what daily files cover."""
    first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    return [first + timedelta(days=i) for i in range((today - first).days)]


def plan_klines(listed: list[str], fetched: set[str], today: date, since_month: str = "") -> list[str]:
    """Monthly files of finished months still to fetch; the API covers the current one."""
    current = f"{today:%Y-%m}"
    return [k for k in listed
            if k.endswith(".zip") and since_month <= key_period(k) < current and k not in fetched]


def is_settled_missing(key: str, today: date) -> bool:
    """A daily file that should exist by now if the coin traded that day."""
    period = key_period(key)
    if len(period) != 10:
        return False
    return (today - date.fromisoformat(period)).days >= MISSING_AFTER_DAYS


# download

def list_dir(client: httpx.Client, prefix: str, dirs: bool = True) -> list[str]:
    """Child prefixes (``dirs``) or keys directly under an archive prefix."""
    out, marker = [], ""
    tag = "Prefix" if dirs else "Key"
    while True:
        resp = _get(client, LISTING_URL, params={"delimiter": "/", "prefix": prefix, "marker": marker})
        found = [f for f in re.findall(rf"<{tag}>([^<]+)</{tag}>", resp.text) if f != prefix]
        out += found
        if "<IsTruncated>true</IsTruncated>" not in resp.text or not found:
            return out
        nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", resp.text)
        marker = nxt.group(1) if nxt else found[-1]


def pair_names(prefixes: Iterable[str]) -> list[str]:
    return [p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes]


def _get(client: httpx.Client, url: str, params: Optional[dict] = None, attempts: int = 4) -> httpx.Response:
    last: object = None
    for attempt in range(attempts):
        try:
            resp = client.get(url, params=params)
            if resp.status_code in (200, 404):
                return resp
            last = f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            last = exc
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"{url}: {last}")


def download(client: httpx.Client, key: str) -> Optional[bytes]:
    """The CSV inside an archive zip, or None when the file does not exist."""
    resp = _get(client, f"{ARCHIVE_URL}/{key}")
    if resp.status_code == 404:
        return None
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        return z.read(z.namelist()[0])


# parsing

def _read_csv(raw: bytes) -> pd.DataFrame:
    has_header = not raw[:1].isdigit()
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)


def parse_klines(raw: bytes) -> pd.DataFrame:
    frame = _read_csv(raw).iloc[:, :len(KLINE_CSV)]
    frame.columns = KLINE_CSV
    ts = frame["ts"].astype("int64").to_numpy()
    out = pd.DataFrame({"ts": np.where(ts > 10 ** 14, ts // 1000, ts)})
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"):
        out[col] = frame[col].astype(float).to_numpy()
    out["trades"] = frame["trades"].astype("int64").to_numpy()
    return out


def parse_funding(raw: bytes) -> list[tuple[int, float]]:
    frame = _read_csv(raw)
    if "calc_time" in frame.columns:
        ts, rate = frame["calc_time"], frame["last_funding_rate"]
    else:
        ts, rate = frame.iloc[:, 0], frame.iloc[:, -1]
    return [(int(t), float(r)) for t, r in zip(ts.astype("int64"), rate.astype(float))]


METRICS_COLUMNS = {"sum_open_interest": "open_interest", "sum_open_interest_value": "open_interest_value",
                   "count_toptrader_long_short_ratio": "top_trader_ls_accounts",
                   "sum_toptrader_long_short_ratio": "top_trader_ls_positions",
                   "count_long_short_ratio": "ls_accounts", "sum_taker_long_short_vol_ratio": "taker_ls_volume"}


def parse_metrics(raw: bytes) -> pd.DataFrame:
    """5-minute futures metrics to hourly rows; ``ts`` is the hour's start, values as of its end.

    Open interest and the long/short ratios are snapshots (last value in the
    hour); the taker buy/sell ratio is a flow (hourly mean).
    """
    frame = _read_csv(raw)
    when = pd.to_datetime(frame["create_time"], errors="coerce")
    frame = frame.assign(when=when, hour=when.dt.floor("h")).dropna(subset=["hour"]).rename(columns=METRICS_COLUMNS)
    cols = list(METRICS_COLUMNS.values())
    for c in cols:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
        # a zero open interest or ratio is a hole in Binance's file (15.10.2026: every pair 0 for hours), not a value
        frame.loc[frame[c] <= 0, c] = np.nan
    # the files are not in time order (BTCUSDT 05.11.2026: 15:20, 15:25, 15:30, 14:45, 15:10, ...): sorted by the hour
    # alone, "last" was any 5-minute snapshot of the hour, up to 55 minutes early, from 2020 on (found 08.11.2026)
    grouped = frame.sort_values("when", kind="stable").groupby("hour")
    out = grouped[cols].last()
    out["taker_ls_volume"] = grouped["taker_ls_volume"].mean()
    out.insert(0, "ts", ((out.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).astype("int64"))
    return out.reset_index(drop=True)


def _api_frame(rows: list[list]) -> pd.DataFrame:
    out = pd.DataFrame({"ts": [int(r[0]) for r in rows]}, dtype="int64")
    for i, col in ((1, "open"), (2, "high"), (3, "low"), (4, "close"), (5, "volume"), (7, "quote_volume"),
                   (10, "taker_buy_quote")):
        out[col] = [float(r[i]) for r in rows]
    out["trades"] = [int(r[8]) for r in rows]
    return out


def fetch_recent(pair: str, tf: str, since_ms: int, now_ms: Optional[int] = None) -> pd.DataFrame:
    """Closed candles from ``since_ms`` to now from the public klines API."""
    now_ms = now_ms or int(time.time() * 1000)
    rows: list[list] = []
    start = since_ms
    while start < now_ms:
        page = _get_json(KLINES_URLS, {"symbol": pair, "interval": tf, "startTime": start, "limit": 1000})
        if not page:
            break
        rows += page
        start = int(page[-1][0]) + TF_MS[tf]
        if len(page) < 1000:
            break
    return _api_frame([r for r in rows if int(r[6]) < now_ms])
