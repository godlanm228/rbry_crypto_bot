"""Step 1 data layer: archive formats, the universe filter, aggregation and storage."""

from datetime import date

import pandas as pd
import pytest

from core.sandbox import archive, context_sources
from core.sandbox.market_store import MarketStore, resample, split_segments

HOUR = 3_600_000


def _hours(start_ms: int, n: int, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({"ts": [start_ms + i * HOUR for i in range(n)], "open": price, "high": price + 1,
                         "low": price - 1, "close": price, "volume": 1.0, "quote_volume": price,
                         "trades": 2, "taker_buy_quote": price / 2})


def test_universe_keeps_coins_and_drops_pegs_fiat_and_leveraged_tokens():
    pairs = ["BTCUSDT", "BTCUPUSDT", "ETHUSDT", "ETHBEARUSDT", "BULLUSDT", "USDCUSDT", "EURUSDT",
             "JUPUSDT", "SYRUPUSDT", "WBTCUSDT", "LUNABTC", "USTUSDT"]
    assert archive.usdt_bases(pairs) == {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "JUPUSDT": "JUP",
                                         "SYRUPUSDT": "SYRUP"}


def test_futures_map_to_spot_bases():
    spot = {"PEPE", "1000SATS", "1INCH", "BTC"}
    futures = ["1000PEPEUSDT", "1000SATSUSDT", "1INCHUSDT", "BTCUSDT", "BTCUSDT_230331", "ETHUSDT", "BTCUSDC"]
    assert archive.futures_pairs(futures, spot) == {"1000PEPEUSDT": "PEPE", "1000SATSUSDT": "1000SATS",
                                                     "1INCHUSDT": "1INCH", "BTCUSDT": "BTC"}


def test_klines_parse_milliseconds_and_the_2025_microseconds():
    raw = (b"1709251200000,61130.99,61749.0,61126.0,61501.11,1867.8,1709254799999,114776037.6,66804,1026.8,63094278.6,0\n"
           b"1740787200000000,84349.95,84628.93,83824.78,83857.92,636.6,1740790799999999,53599194.4,131136,249.1,20981965.2,0\n")
    frame = archive.parse_klines(raw)
    assert frame["ts"].tolist() == [1709251200000, 1740787200000]
    assert frame["quote_volume"].tolist() == [114776037.6, 53599194.4] and frame["trades"].tolist() == [66804, 131136]


def test_funding_and_metrics_parse_with_headers():
    funding = archive.parse_funding(b"calc_time,funding_interval_hours,last_funding_rate\n1709251200000,8,0.00053\n")
    assert funding == [(1709251200000, 0.00053)]
    head = (b"create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
            b"sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n")
    rows = (b"2024-03-01 00:00:00,BTCUSDT,100,1000,1.1,1.5,1.2,0.8\n"
            b"2024-03-01 00:55:00,BTCUSDT,110,1100,1.3,1.6,1.4,1.2\n"
            b"2024-03-01 01:00:00,BTCUSDT,120,1200,1.0,1.0,1.0,2.0\n")
    hourly = archive.parse_metrics(head + rows)
    assert hourly["ts"].tolist() == [1709251200000, 1709254800000]
    # snapshots: last value in the hour; taker ratio: hourly mean
    assert hourly["open_interest"].tolist() == [110, 120] and hourly["taker_ls_volume"].tolist() == [1.0, 2.0]


def test_plan_skips_the_current_month_fetched_files_and_old_months():
    listed = [archive.kline_monthly_key("X", "5m", m) for m in ("2024-09", "2024-10", "2026-09", "2026-10")]
    listed.append(listed[1] + ".CHECKSUM")
    fetched = {listed[1]}
    assert archive.plan_klines(listed, fetched, date(2026, 10, 28)) == [listed[0], listed[2]]
    assert archive.plan_klines(listed, set(), date(2026, 10, 28), since_month="2024-10") == listed[1:3]


def test_a_missing_daily_file_is_settled_after_three_days():
    key = archive.metrics_daily_key("BTCUSDT", date(2026, 10, 25))
    assert archive.is_settled_missing(key, date(2026, 10, 28))
    assert not archive.is_settled_missing(key, date(2026, 10, 27))
    assert not archive.is_settled_missing(archive.funding_monthly_key("BTCUSDT", "2026-09"), date(2026, 10, 28))


def test_daily_candles_are_exact_aggregates_of_hours():
    day = 1709251200000                      # 2024-03-01 00:00 UTC
    hours = _hours(day, 30)
    hours.loc[5, "high"] = 150.0
    hours.loc[23, "close"] = 99.0
    daily = resample(hours.drop(index=[10]), "1d")   # a missing hour does not break the day
    first = daily.iloc[0]
    assert daily["ts"].tolist() == [day, day + 24 * HOUR]
    assert (first.open, first.high, first.low, first.close, first.volume, first.trades) == (100, 150, 99, 99, 23, 46)
    assert resample(hours, "4h")["ts"].tolist()[:2] == [day, day + 4 * HOUR]


def test_a_long_halt_splits_the_history():
    old = _hours(0, 10, price=50)
    new = _hours(10 * HOUR + 18 * 24 * HOUR, 5, price=8)   # relisted 18 days later
    parts = split_segments(pd.concat([old, new], ignore_index=True))
    assert [len(p) for p in parts] == [10, 5] and parts[1]["close"].iloc[0] == 8


def test_store_roundtrip_and_histories_per_segment(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    store.upsert_symbols([("LUNAUSDT", "LUNA", "trading")])
    old = _hours(1577836800000, 24 * 300, price=50)               # 2020-01-01, 300 days
    new = _hours(1577836800000 + 24 * 330 * HOUR, 24 * 280, price=8)
    store.save_klines("LUNAUSDT", "1h", pd.concat([old, new]))
    assert store.last_ts("LUNAUSDT", "1h") == new["ts"].iloc[-1]
    hist = store.daily_histories(min_days=260)
    assert sorted(hist) == ["LUNA", "LUNA@2020-11"] and len(hist["LUNA"]) == 300
    assert list(hist["LUNA"].columns) == ["day", "open", "high", "low", "close", "volume"]
    assert store.daily_histories(min_days=290).keys() == {"LUNA"}

    store.mark_fetched("data/spot/monthly/klines/X/1h/a.zip", "ok", 3)
    store.mark_fetched("data/futures/um/x.zip", "ok", 1)
    assert store.fetched("data/spot/") == {"data/spot/monthly/klines/X/1h/a.zip"}
    item = {"url": "https://x/1", "source": "x", "kind": "article", "published_ts": 1, "title": "t"}
    assert store.save_news([item, {**item, "url": ""}]) == 1 and store.save_news([item]) == 0
    store.save_series("spx", [("2026-10-27", 7700.5)])
    assert store.series("spx")["value"].tolist() == [7700.5]


def test_recent_candles_leave_out_the_open_one(monkeypatch):
    def fake(urls, params, timeout=20):
        return [[0, "1", "2", "0.5", "1.5", "10", HOUR - 1, "15", 3, "5", "7", "0"],
                [HOUR, "1", "2", "0.5", "1.5", "10", 2 * HOUR - 1, "15", 3, "5", "7", "0"]]
    monkeypatch.setattr(archive, "_get_json", fake)
    frame = archive.fetch_recent("XUSDT", "1h", 0, now_ms=HOUR + 10)
    assert frame["ts"].tolist() == [0] and frame["taker_buy_quote"].tolist() == [7.0]


def test_context_sources_parse():
    yahoo = {"chart": {"result": [{"timestamp": [1761571800, 1761658200],
                                   "indicators": {"quote": [{"close": [6875.2, None]}]}}]}}
    assert context_sources.parse_yahoo(yahoo) == [("2025-10-27", 6875.2)]
    stables = [{"date": "1761523200", "totalCirculatingUSD": {"peggedUSD": 3.1e11}}, {"date": "1761609600"}]
    assert context_sources.parse_stablecoins(stables) == [("2025-10-27", 3.1e11)]
    assert context_sources.parse_fear_greed({"data": [{"timestamp": "1761523200", "value": "42"}]}) == \
        [("2025-10-27", 42.0)]
    cms = {"data": {"catalogs": [{"total": 2289, "articles": [
        {"code": "abc", "title": "Binance Will List X (X)", "releaseDate": 1792999800000}]}]}}
    items, total = context_sources.parse_announcements(cms, "listing")
    assert total == 2289 and items[0]["url"].endswith("/abc") and items[0]["kind"] == "listing"


def test_feed_items_carry_utc_time_and_clean_text():
    pytest.importorskip("feedparser")
    rss = (b"<?xml version='1.0'?><rss version='2.0'><channel><title>t</title>"
           b"<item><title>BTC &lt;b&gt;up&lt;/b&gt;</title><link>https://ex.com/a</link>"
           b"<description>&lt;p&gt;Some   text&lt;/p&gt;</description>"
           b"<pubDate>Tue, 27 Oct 2026 10:00:00 GMT</pubDate></item>"
           b"<item><title>no date</title><link>https://ex.com/b</link></item></channel></rss>")
    items = context_sources.parse_feed(rss, "https://www.ex.com/rss")
    assert len(items) == 1 and items[0]["source"] == "ex.com" and items[0]["summary"] == "Some text"
    assert items[0]["published_ts"] == 1793095200000
