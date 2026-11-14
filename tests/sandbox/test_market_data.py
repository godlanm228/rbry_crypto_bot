"""Step 1 data layer: archive formats, the universe filter, aggregation and storage."""

import threading
from datetime import date

import pandas as pd
import pytest

from core.sandbox import archive, context_sources, market_store
from core.sandbox.market_store import MarketStore, resample, split_segments

HOUR = 3_600_000


def _hours(start_ms: int, n: int, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({"ts": [start_ms + i * HOUR for i in range(n)], "open": price, "high": price + 1,
                         "low": price - 1, "close": price, "volume": 1.0, "quote_volume": price,
                         "trades": 2, "taker_buy_quote": price / 2})


def _moving(frame: pd.DataFrame) -> pd.DataFrame:
    """Close alternates +-2% by day, so the history does not look like a peg."""
    day = (frame["ts"] // (24 * HOUR)) % 2
    return frame.assign(close=frame["close"] * (1 + 0.02 * (2 * day - 1)))


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


def test_the_hours_last_snapshot_is_the_latest_in_time_not_in_the_files_order():
    # 08.11.2026: Binance's files are not in time order (BTCUSDT 05.11, the hour 15:00 as the file has it); the hour
    # kept whatever row came last, 15:50 here, and OI-FLUSH of BTC that hour did not show on history
    head = (b"create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
            b"sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n")
    order = (("15:20", 7.119722e9), ("15:45", 7.049942e9), ("15:00", 7.303441e9), ("15:55", 6.962886e9),
             ("15:50", 6.929604e9))
    rows = b"".join(f"2026-11-05 {t}:00,BTCUSDT,1,{v},1.1,1.2,1.3,0.9\n".encode() for t, v in order)
    hourly = archive.parse_metrics(head + rows)
    assert hourly["open_interest_value"].tolist() == [6.962886e9]


def test_a_zero_in_the_file_is_a_hole_not_a_value():
    # 15.10.2026: Binance wrote 0 open interest for every pair for hours; 0 made "+inf %" and false OI-UP-UP events
    head = (b"create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
            b"sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n")
    rows = (b"2026-10-15 16:40:00,XUSDT,10,500,1.1,1.2,1.3,0.9\n"
            b"2026-10-15 16:55:00,XUSDT,0,0,0,0,0,0\n"
            b"2026-10-15 17:55:00,XUSDT,0,0,0,0,0,0\n")
    hourly = archive.parse_metrics(head + rows)
    assert hourly["open_interest_value"].iloc[0] == 500             # the hour's last real snapshot
    assert hourly["open_interest_value"].isna().iloc[1] and hourly["ls_accounts"].isna().iloc[1]


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
    store.save_klines("LUNAUSDT", "1h", _moving(pd.concat([old, new])))
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


def test_histories_leave_out_tokenized_stocks_and_pegs(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    store.upsert_symbols([("AAPLBUSDT", "AAPLB", "trading"), ("KGSTUSDT", "KGST", "trading"),
                          ("SOLUSDT", "SOL", "trading")])
    store.save_news([{"url": "u1", "source": "binance", "kind": "listing", "published_ts": 1,
                      "title": "Binance Exchange Adds bStocks Apple (AAPLB) Trading Pair(s) on Binance Spot"}])
    moving = _moving(_hours(0, 24 * 300))
    peg = _hours(0, 24 * 300, price=1.0)
    for pair, frame in (("AAPLBUSDT", moving), ("KGSTUSDT", peg), ("SOLUSDT", moving)):
        store.save_klines(pair, "1h", frame)
    assert store.tokenized_stocks() == {"AAPLB"}
    assert list(store.daily_histories()) == ["SOL"]


# Binance's product list, 05.11: NVDAB, CRCLB and SOXLB were listed before the archive's announcements begin
PRODUCTS = {"data": [{"s": "NVDABUSDT", "b": "NVDAB", "q": "USDT", "tags": ["bStocks"]},
                     {"s": "STXUSDT", "b": "STX", "q": "USDT", "tags": ["Monitoring", "Layer1_Layer2"]},
                     {"s": "BNBUSDT", "b": "BNB", "q": "USDT", "tags": ["Layer1_Layer2", "pos", "BSC"]},
                     {"s": "NVDABUSDC", "b": "NVDAB", "q": "USDC", "tags": ["bStocks"]}]}


def test_binance_tags_of_usdt_pairs():
    from core.sandbox import context_sources as cs
    assert cs.parse_products(PRODUCTS) == [("NVDAB", ["bStocks"]), ("STX", ["Monitoring", "Layer1_Layer2"]),
                                           ("BNB", ["Layer1_Layer2", "pos", "BSC"])]


def test_tokenized_shares_by_tag_and_by_title_unless_the_tag_says_crypto(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    store.save_news([{"url": "u1", "source": "binance", "kind": "listing", "published_ts": 1,
                      "title": "Binance Exchange Adds Seagate Technology (STX) and Apple (AAPLB) bStocks Trading Pairs"}])
    assert store.tokenized_stocks() == {"STX", "AAPLB"}              # titles only: Stacks counts as a share
    from core.sandbox import context_sources as cs
    store.save_products(cs.parse_products(PRODUCTS))
    assert store.tokenized_stocks() == {"NVDAB", "AAPLB"}             # a share's ticker in a title is not enough


def test_a_second_fill_waits_for_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(market_store, "LOCK_POLL_SECONDS", 0.05)      # Windows polls its byte lock
    store = MarketStore(tmp_path / "m.db")
    entered = threading.Event()

    def second():
        with store.fill_lock():
            entered.set()

    with store.fill_lock():
        waiting = threading.Thread(target=second)
        waiting.start()
        assert not entered.wait(0.3)
    assert entered.wait(5)
    waiting.join()


def test_the_hourly_refresh_skips_while_a_fill_runs(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    with store.fill_lock():
        with store.fill_lock(wait=False) as free:
            assert free is False
    with store.fill_lock(wait=False) as free:
        assert free is True


def test_dominance_is_saved_once_for_the_day_that_just_closed(tmp_path, monkeypatch):
    import importlib.util
    from datetime import datetime, timezone
    from pathlib import Path
    script = Path(__file__).resolve().parents[2] / "scripts" / "market_data.py"
    spec = importlib.util.spec_from_file_location("market_data_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    payload = {"data": {"market_cap_percentage": {"btc": 59.5, "eth": 10.5, "usdt": 6.6, "usdc": 2.6, "bnb": 3},
                        "total_market_cap": {"usd": 3.9e12}}}
    assert context_sources.parse_global(payload) == {"btc_dominance": 59.5, "eth_dominance": 10.5,
                                                     "usdt_dominance": 6.6, "usdc_dominance": 2.6,
                                                     "total_mcap_usd": 3.9e12}
    calls = []
    monkeypatch.setattr(cli.context_sources, "fetch_global",
                        lambda client: calls.append(1) or context_sources.parse_global(payload))
    store = MarketStore(tmp_path / "m.db")
    cli.snapshot_step(store, None, datetime(2026, 10, 29, 0, 1, tzinfo=timezone.utc))
    cli.snapshot_step(store, None, datetime(2026, 10, 29, 0, 16, tzinfo=timezone.utc))   # already saved
    cli.snapshot_step(store, None, datetime(2026, 10, 29, 14, 0, tzinfo=timezone.utc))   # not the close any more
    assert len(calls) == 1
    saved = store.series("btc_dominance")
    assert saved["day"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-28"] and saved["value"].tolist() == [59.5]


def test_wikipedia_page_views_parse_by_day():
    payload = {"items": [{"timestamp": "2021010100", "views": 13774}, {"timestamp": "2021010200", "views": 30157}]}
    assert context_sources.parse_pageviews(payload) == [("2021-01-01", 13774.0), ("2021-01-02", 30157.0)]


def test_the_first_hour_of_each_pair(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    store.save_klines("AAAUSDT", "1h", _hours(1577836800000, 48))
    store.save_klines("BBBUSDT", "1h", _hours(1577836800000 + 100 * HOUR, 5))
    assert store.first_hours(["AAAUSDT", "BBBUSDT", "NONEUSDT"]) == {
        "AAAUSDT": 1577836800000, "BBBUSDT": 1577836800000 + 100 * HOUR}
