"""market.db: everything the brain learns from, kept apart from its experiments.

Candles come from Binance's public archive (data.binance.vision), which keeps
every pair ever listed, delisted ones included: learning only on today's
survivors would teach the bot that crypto always recovers. Only 1h and 5m
candles are stored; 4h and daily are exact aggregates of 1h (Binance aligns
all of them to 00:00 UTC).

Tables
  symbols         every USDT pair seen: base asset, trading or delisted
  klines          symbol, tf ('1h' | '5m'), open time in ms, OHLC, base and quote
                  volume, number of trades, taker buys in USDT
  funding         USDT-M perpetual funding rate per payment
  futures_hourly  open interest and long/short ratios, hourly from 5-minute metrics
  daily_series    one value per day: macro indices, stablecoin supply, Fear & Greed
  news            headlines and Binance listing/delisting announcements
  fetched         every archive file already ingested or known to be missing
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "runtime" / "learning" / "market.db"
# Every pair's 5-minute candles since its listing: tens of GB, so a folder of its own beside the
# project, on the laptop only (the VM never opens it). RBRY_FINE_DB moves it.
FINE_DB_PATH = Path(os.environ.get("RBRY_FINE_DB") or
                    Path(__file__).resolve().parents[3] / "RBRY_история_цен_5мин" / "candles-5m.db")
LOCK_POLL_SECONDS = 5.0
HOUR_MS = 3_600_000
KLINE_FIELDS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]
FUTURES_FIELDS = ["open_interest", "open_interest_value", "top_trader_ls_accounts",
                  "top_trader_ls_positions", "ls_accounts", "taker_ls_volume"]
AGGREGATE = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
             "quote_volume": "sum", "trades": "sum", "taker_buy_quote": "sum"}
RESAMPLE = {"4h": "4h", "1d": "1D"}
STOCK_MARKERS = ("bStocks", "Tokenized Securities")
PEG_MEDIAN_MOVE = 0.003      # a coin's typical daily move is 1-5%; a peg's is under 0.1%
ENDED_AFTER_HOURS = 6        # a history this far behind the newest candle has stopped trading

SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT PRIMARY KEY, base TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'unknown',
    updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS products (
    base TEXT PRIMARY KEY, tags TEXT NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS klines (
    symbol TEXT NOT NULL, tf TEXT NOT NULL, ts INTEGER NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume REAL NOT NULL, quote_volume REAL NOT NULL, trades INTEGER NOT NULL, taker_buy_quote REAL NOT NULL,
    PRIMARY KEY (symbol, tf, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS funding (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL, rate REAL NOT NULL,
    PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS futures_hourly (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,
    open_interest REAL, open_interest_value REAL, top_trader_ls_accounts REAL,
    top_trader_ls_positions REAL, ls_accounts REAL, taker_ls_volume REAL,
    PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS daily_series (
    series TEXT NOT NULL, day TEXT NOT NULL, value REAL NOT NULL,
    PRIMARY KEY (series, day)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS hourly_series (
    series TEXT NOT NULL, ts INTEGER NOT NULL, value REAL NOT NULL,
    PRIMARY KEY (series, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS news (
    url TEXT PRIMARY KEY, source TEXT NOT NULL, kind TEXT NOT NULL, published_ts INTEGER NOT NULL,
    fetched_ts INTEGER NOT NULL, title TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS news_by_time ON news (published_ts);
CREATE TABLE IF NOT EXISTS fetched (
    key TEXT PRIMARY KEY, status TEXT NOT NULL, rows INTEGER NOT NULL, at REAL NOT NULL) WITHOUT ROWID;
"""


def split_segments(frame: pd.DataFrame, max_gap_hours: float = 72.0) -> list[pd.DataFrame]:
    """Cut a history where trading stopped for longer than ``max_gap_hours``.

    The same ticker can name two different coins (old LUNA until May 2022, the
    new one from its relisting), and after a long halt indicators would bridge
    the gap as if nothing happened. Each piece is its own history.
    """
    if frame.empty:
        return []
    gaps = frame["ts"].diff().fillna(0) > max_gap_hours * HOUR_MS
    return [part.reset_index(drop=True) for _, part in frame.groupby(gaps.cumsum())]


def resample(frame: pd.DataFrame, tf: str) -> pd.DataFrame:
    """4h or daily candles from 1h ones; a bucket with no hours stays absent."""
    if frame.empty:
        return frame
    out = (frame.set_index(pd.to_datetime(frame["ts"], unit="ms"))[KLINE_FIELDS]
           .resample(RESAMPLE[tf]).agg(AGGREGATE).dropna(subset=["open"]))
    out.insert(0, "ts", ((out.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).astype("int64"))
    return out.reset_index(drop=True)


@dataclass
class History:
    """One uninterrupted stretch of trading of one ticker."""
    name: str               # 'LUNA', or 'LUNA@2022-05' for a later stretch of the same ticker
    symbol: str             # the USDT pair
    base: str
    hourly: pd.DataFrame    # ts + KLINE_FIELDS
    daily: pd.DataFrame     # the same, aggregated to finished UTC days
    ended: bool             # stopped trading (delisted or halted) before the newest data


class MarketStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")   # safe with WAL; everything here can be fetched again
        return conn

    @contextmanager
    def fill_lock(self, wait: bool = True) -> Iterator[bool]:
        """One fill at a time; yields False when ``wait`` is off and another fill is running.

        Two fills at once download the same files twice and, on the 1 GB VM,
        need memory the bot does not have to spare. Windows has no flock: there
        the lock is one locked byte of the same file (the night job and a run
        by hand can meet on the laptop too).
        """
        try:
            import fcntl
        except ImportError:
            with self._byte_lock(wait) as free:
                yield free
            return
        with open(self.db_path.with_suffix(".fill.lock"), "w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if not wait:
                    yield False
                    return
                logger.info("another bootstrap or update is running; waiting for it to finish")
                fcntl.flock(handle, fcntl.LOCK_EX)
            yield True

    @contextmanager
    def _byte_lock(self, wait: bool) -> Iterator[bool]:
        import msvcrt
        with open(self.db_path.with_suffix(".fill.lock"), "a+") as handle:
            told = False
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if not wait:
                        yield False
                        return
                    if not told:
                        logger.info("another bootstrap or update is running; waiting for it to finish")
                        told = True
                    time.sleep(LOCK_POLL_SECONDS)
            try:
                yield True
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

    # universe
    def upsert_symbols(self, rows: Iterable[tuple[str, str, str]]) -> None:
        """(symbol, base, status) rows; status is 'trading', 'delisted' or 'unknown'."""
        now = time.time()
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO symbols (symbol, base, status, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET base = excluded.base, status = excluded.status, "
                "updated_at = excluded.updated_at", [(s, b, st, now) for s, b, st in rows])

    def symbols(self, status: Optional[str] = None) -> list[tuple[str, str, str]]:
        sql, params = "SELECT symbol, base, status FROM symbols", ()
        if status:
            sql, params = sql + " WHERE status = ?", (status,)
        with self.connect() as conn:
            return conn.execute(sql + " ORDER BY symbol", params).fetchall()

    # candles
    def save_klines(self, symbol: str, tf: str, frame: pd.DataFrame) -> int:
        rows = [(symbol, tf, int(ts), *map(float, vals[:6]), int(vals[6]), float(vals[7]))
                for ts, *vals in frame[["ts", *KLINE_FIELDS]].itertuples(index=False)]
        with self.connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO klines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    def first_hours(self, symbols: Iterable[str], tf: str = "1h") -> dict[str, int]:
        """The first stored candle of each pair (ms): how long a coin has been trading. One seek a pair."""
        out = {}
        with self.connect() as conn:
            for symbol in symbols:
                row = conn.execute("SELECT MIN(ts) FROM klines WHERE symbol = ? AND tf = ?", (symbol, tf)).fetchone()
                if row and row[0] is not None:
                    out[symbol] = int(row[0])
        return out

    def klines(self, symbol: str, tf: str = "1h", start_ts: int = 0) -> pd.DataFrame:
        """Stored 1h/5m candles, or 4h/1d aggregated from 1h; oldest first."""
        source = "1h" if tf in RESAMPLE else tf
        with self.connect() as conn:
            frame = pd.read_sql_query(
                f"SELECT ts, {', '.join(KLINE_FIELDS)} FROM klines WHERE symbol = ? AND tf = ? AND ts >= ? "
                "ORDER BY ts", conn, params=(symbol, source, int(start_ts)))
        return resample(frame, tf) if tf in RESAMPLE else frame

    def last_ts(self, symbol: str, tf: str) -> Optional[int]:
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(ts) FROM klines WHERE symbol = ? AND tf = ?", (symbol, tf)).fetchone()
        return row[0] if row and row[0] is not None else None

    def pairs_with(self, tf: str) -> list[str]:
        with self.connect() as conn:
            return [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM klines WHERE tf = ?", (tf,))]

    def save_products(self, rows: Iterable[tuple[str, Iterable[str]]]) -> int:
        """(base, tags) of Binance's product list; a base once seen keeps its row after it is delisted."""
        now = time.time()
        rows = [(b, ",".join(sorted(set(t))), now) for b, t in rows if b]
        with self.connect() as conn:
            conn.executemany("INSERT INTO products (base, tags, updated_at) VALUES (?, ?, ?) ON CONFLICT(base) DO "
                             "UPDATE SET tags = excluded.tags, updated_at = excluded.updated_at", rows)
        return len(rows)

    def tokenized_stocks(self) -> set[str]:
        """Tokenized shares (AAPLB, QQQB, ...), not crypto bets: Binance's own "bStocks" tag on its product list,
        and the tickers of its bStocks announcements that the list does not show as crypto (a share's ticker in
        a title, "(STX)", can be a coin's). The announcements alone missed the batches listed before the archive
        begins (NVDAB, TSLAB, AAPLB, ...): they traded as coins in the watcher until 05.11."""
        with self.connect() as conn:
            tagged = {b: set(t.split(",")) for b, t in conn.execute("SELECT base, tags FROM products")}
        titles = self.news(kind="listing")["title"]
        named = {t for title in titles if any(m in title for m in STOCK_MARKERS)
                 for t in re.findall(r"\(([A-Z0-9]{2,15})\)", title)}
        return {b for b, t in tagged.items() if "bStocks" in t} | {t for t in named if t not in tagged}

    def delisting_notices(self) -> dict[str, list[int]]:
        """{base: [publication ms]} from "Binance Will Delist A, B and C on <date>" (spot, since 2022)."""
        out: dict[str, list[int]] = {}
        for title, ts in self.news(kind="delisting")[["title", "published_ts"]].itertuples(index=False):
            m = re.match(r"Binance Will Delist (.+?) on \d{4}-\d{2}-\d{2}", title)
            if not m:
                continue
            for base in re.split(r",\s*|\s+and\s+|\s*&\s*", m.group(1)):
                if re.fullmatch(r"[A-Z0-9]{2,15}", base.strip()):
                    out.setdefault(base.strip(), []).append(int(ts))
        return out

    def newest_hour(self) -> Optional[int]:
        """The newest hourly candle of any pair (one index lookup per pair; a full MAX scans 19M rows)."""
        with self.connect() as conn:
            found = [conn.execute("SELECT MAX(ts) FROM klines WHERE symbol = ? AND tf = '1h'", (s,)).fetchone()[0]
                     for (s,) in conn.execute("SELECT symbol FROM symbols").fetchall()]
        found = [ts for ts in found if ts is not None]
        return max(found) if found else None

    def histories(self, min_days: int = 260, min_median_quote_volume: float = 0.0,
                  symbols: Optional[Iterable[str]] = None) -> Iterator[History]:
        """Every history of a real coin, one at a time (all of them do not fit in 1 GB).

        A ticker with a long halt gives several histories: the first keeps the
        base name, later ones get '@YYYY-MM' of their first day. Tokenized
        shares and anything that moves like a peg are left out. Daily candles
        stop at the last finished UTC day.
        """
        wanted = set(symbols) if symbols else None
        stocks = self.tokenized_stocks()
        newest = self.newest_hour() or 0
        for symbol, base, _ in self.symbols():
            if (wanted and symbol not in wanted and base not in wanted) or base in stocks:
                continue
            hourly = self.klines(symbol, "1h")
            for i, part in enumerate(split_segments(hourly)):
                daily = resample(part, "1d")
                daily = daily[daily["ts"] + 23 * HOUR_MS <= newest].reset_index(drop=True)
                if len(daily) < min_days or daily["quote_volume"].median() < min_median_quote_volume:
                    continue
                if daily["close"].pct_change().abs().median() < PEG_MEDIAN_MOVE:
                    continue
                name = base if i == 0 else f"{base}@{pd.to_datetime(part['ts'].iloc[0], unit='ms'):%Y-%m}"
                ended = int(part["ts"].iloc[-1]) < newest - ENDED_AFTER_HOURS * HOUR_MS
                yield History(name, symbol, base, part, daily, ended)

    def btc_pairs(self) -> list[str]:
        """The BTC-quoted pairs fetched from the archive (scripts/market_data.py btc-pairs)."""
        prefix = "data/spot/monthly/klines/"
        return sorted({k[len(prefix):].split("/", 1)[0] for k in self.fetched(prefix)
                       if k[len(prefix):].split("/", 1)[0].endswith("BTC") and "/1h/" in k})

    def btc_histories(self, min_days: int = 60, pairs: Optional[Iterable[str]] = None) -> Iterator[History]:
        """Altcoins' BTC pairs in dollars, for the hours before each coin's USDT pair (plan item 5).

        Until 2020 most altcoins traded against BTC, and their USDT pairs came late or thin. Prices and
        turnover are the pair's times BTCUSDT's close of the same hour (the hour's high and low through the
        close: near enough for hourly paths). Named '<base>~BTC'; a coin with a USDT pair keeps only the
        hours before it, so no hour is counted twice.
        """
        btc = self.klines("BTCUSDT", "1h")[["ts", "close"]].rename(columns={"close": "btc"})
        stocks = self.tokenized_stocks()
        for pair in pairs or self.btc_pairs():
            base = pair[:-3]
            if base in stocks or not base:
                continue
            frame = self.klines(pair, "1h")
            with self.connect() as conn:
                first_usdt = conn.execute("SELECT MIN(ts) FROM klines WHERE symbol = ? AND tf = '1h'",
                                          (base + "USDT",)).fetchone()[0]
            if first_usdt is not None:
                frame = frame[frame["ts"] < first_usdt]
            frame = frame.merge(btc, on="ts", how="inner")
            if frame.empty:
                continue
            for col in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote"):
                frame[col] = frame[col] * frame["btc"]
            frame = frame.drop(columns="btc").reset_index(drop=True)
            for i, part in enumerate(split_segments(frame)):
                daily = resample(part, "1d")
                if len(daily) < min_days or daily["close"].pct_change().abs().median() < PEG_MEDIAN_MOVE:
                    continue
                name = f"{base}~BTC" if i == 0 else f"{base}~BTC@{pd.to_datetime(part['ts'].iloc[0], unit='ms'):%Y-%m}"
                yield History(name, pair, base, part, daily, True)

    def daily_histories(self, min_days: int = 260, min_median_quote_volume: float = 0.0,
                        symbols: Optional[Iterable[str]] = None) -> dict[str, pd.DataFrame]:
        """Daily candles per history in the sandbox's format (volume = USDT volume)."""
        return {h.name: pd.DataFrame({
                    "day": pd.to_datetime(h.daily["ts"], unit="ms"), "open": h.daily["open"],
                    "high": h.daily["high"], "low": h.daily["low"], "close": h.daily["close"],
                    "volume": h.daily["quote_volume"]})
                for h in self.histories(min_days, min_median_quote_volume, symbols)}

    # futures
    def save_funding(self, symbol: str, rows: Iterable[tuple[int, float]]) -> int:
        rows = [(symbol, int(ts), float(rate)) for ts, rate in rows]
        with self.connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO funding VALUES (?, ?, ?)", rows)
        return len(rows)

    def funding(self, symbol: str, start_ts: int = 0) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("SELECT ts, rate FROM funding WHERE symbol = ? AND ts >= ? ORDER BY ts", conn,
                                     params=(symbol, int(start_ts)))

    def futures_tracked(self) -> list[str]:
        with self.connect() as conn:
            return [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM futures_hourly")]

    def save_futures_hourly(self, symbol: str, frame: pd.DataFrame) -> int:
        rows = [(symbol, int(ts), *(None if pd.isna(v) else float(v) for v in vals))
                for ts, *vals in frame[["ts", *FUTURES_FIELDS]].itertuples(index=False)]
        with self.connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO futures_hourly VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    def futures_hourly(self, symbol: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(f"SELECT ts, {', '.join(FUTURES_FIELDS)} FROM futures_hourly "
                                     "WHERE symbol = ? ORDER BY ts", conn, params=(symbol,))

    # daily series and news
    def save_series(self, series: str, rows: Iterable[tuple[str, float]]) -> int:
        rows = [(series, day, float(value)) for day, value in rows]
        with self.connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO daily_series VALUES (?, ?, ?)", rows)
        return len(rows)

    def series(self, series: str) -> pd.DataFrame:
        with self.connect() as conn:
            frame = pd.read_sql_query("SELECT day, value FROM daily_series WHERE series = ? ORDER BY day", conn,
                                      params=(series,))
        frame["day"] = pd.to_datetime(frame["day"])
        return frame

    def save_hourly(self, series: str, rows: Iterable[tuple[int, float]]) -> int:
        """An outside hourly series (ts = the hour's start, ms; the value as of its close)."""
        rows = [(series, int(ts), float(value)) for ts, value in rows]
        with self.connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO hourly_series VALUES (?, ?, ?)", rows)
        return len(rows)

    def hourly(self, series: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("SELECT ts, value FROM hourly_series WHERE series = ? ORDER BY ts", conn,
                                     params=(series,))

    def last_hourly(self, series: str) -> Optional[int]:
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(ts) FROM hourly_series WHERE series = ?", (series,)).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def save_news(self, items: Iterable[dict]) -> int:
        """Insert new items only; returns how many were new."""
        now = int(time.time() * 1000)
        rows = [(i["url"], i["source"], i["kind"], int(i["published_ts"]), now, i["title"], i.get("summary", ""))
                for i in items if i.get("url") and i.get("title")]
        with self.connect() as conn:
            before = conn.total_changes
            conn.executemany("INSERT OR IGNORE INTO news VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            return conn.total_changes - before

    def set_summary(self, url: str, text: str) -> None:
        """Fill an item's empty summary (the body of an announcement, read after its title)."""
        with self.connect() as conn:
            conn.execute("UPDATE news SET summary = ? WHERE url = ? AND (summary IS NULL OR summary = '')", (text, url))

    def news(self, since_ts: int = 0, kind: Optional[str] = None) -> pd.DataFrame:
        sql, params = "SELECT * FROM news WHERE published_ts >= ?", [int(since_ts)]
        if kind:
            sql, params = sql + " AND kind = ?", params + [kind]
        with self.connect() as conn:
            return pd.read_sql_query(sql + " ORDER BY published_ts", conn, params=params)

    # bookkeeping
    def mark_fetched(self, key: str, status: str, rows: int = 0) -> None:
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO fetched VALUES (?, ?, ?, ?)", (key, status, rows, time.time()))

    def fetched(self, prefix: str) -> set[str]:
        """Keys under ``prefix`` that need no new download (ingested or confirmed missing)."""
        with self.connect() as conn:
            rows = conn.execute("SELECT key FROM fetched WHERE key >= ? AND key < ?",
                                (prefix, prefix + "￿")).fetchall()
        return {r[0] for r in rows}

    def ingested(self, prefix: str) -> tuple[int, int]:
        """(files, rows) taken in from the archive under ``prefix``: a store's size without scanning its candles."""
        with self.connect() as conn:
            files, rows = conn.execute("SELECT COUNT(*), COALESCE(SUM(rows), 0) FROM fetched WHERE status = 'ok' "
                                       "AND key >= ? AND key < ?", (prefix, prefix + "￿")).fetchone()
        return int(files), int(rows)

    def coverage(self) -> dict:
        with self.connect() as conn:
            q = lambda sql: conn.execute(sql).fetchall()  # noqa: E731
            return {
                "symbols": dict(q("SELECT status, COUNT(*) FROM symbols GROUP BY status")),
                "klines": q("SELECT tf, COUNT(DISTINCT symbol), COUNT(*), MIN(ts), MAX(ts) FROM klines GROUP BY tf"),
                "funding": q("SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(ts), MAX(ts) FROM funding")[0],
                "futures": q("SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(ts), MAX(ts) FROM futures_hourly")[0],
                "series": q("SELECT series, COUNT(*), MIN(day), MAX(day) FROM daily_series GROUP BY series"),
                "news": q("SELECT kind, source, COUNT(*), MIN(published_ts), MAX(published_ts) FROM news "
                          "GROUP BY kind, source"),
                "size_mb": self.db_path.stat().st_size / 1e6 if self.db_path.exists() else 0.0,
            }
