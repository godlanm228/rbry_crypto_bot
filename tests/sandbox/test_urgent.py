"""Urgent events: real headlines of 26-30.10.2026 from the news archive, and the market checks."""

from datetime import datetime, timezone

import numpy as np

from core.sandbox import urgent as ur


def _ms(text):
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _h(when, source, title, kind="article"):
    return ur.Headline(f"https://{source}/{abs(hash(title))}", source, kind, _ms(when), title)


BINANCE_NIGHT = [
    _h("2026-10-29 21:12", "coindesk.com", "Potential multi-billion Binance hack hitting crypto markets"),
    _h("2026-10-29 21:24", "cointelegraph.com", "Binance investigates potential multibillion-dollar hack"),
    _h("2026-10-29 21:36", "decrypt.co", "Binance Drained of $5.5 Billion in Suspected Hack, Tether and Circle Freeze $1.8 Billion"),
    _h("2026-10-29 21:49", "theblock.co", "Binance investigating hack as suspicious outflows reach up to $5.8 billion"),
    _h("2026-10-29 22:57", "decrypt.co", "Binance Hacker Hunts for Exit Ramps as DeFi Exchanges Pause Markets"),
    _h("2026-10-29 23:24", "cointelegraph.com", "Binance hack rattles crypto markets as BNB falls 7%, Bitcoin slides"),
    _h("2026-10-30 00:11", "cointelegraph.com", "Binance hack may involve private key ‘misuse,’ not theft: Blockaid CEO"),
    _h("2026-10-30 00:17", "coindesk.com", "Binance hack drained wallets backing customer assets, reserve data shows"),
]
QUIET = [
    _h("2026-10-26 12:27", "cryptonews.com", "Quant Crypto Soars 18% as QNT Charges Toward $250"),
    _h("2026-10-27 01:55", "cointelegraph.com", "Keystone breach deepens: Hackers stole key used to verify wallets as genuine"),
    _h("2026-10-28 02:18", "cointelegraph.com", "Bithumb discloses data leak after internal system breach: Report"),
    _h("2026-10-28 05:10", "coindesk.com", "Bithumb hacked through employee system as South Korean financial firms face cyberattacks"),
    _h("2026-10-28 10:57", "decrypt.co", "Bithumb Says Hackers Accessed Internal System, Leaking Personal Data"),
    _h("2026-10-28 07:42", "cointelegraph.com", "Coinbase denies systems breach, says vendor exposed customer data"),
    _h("2026-10-28 10:10", "decrypt.co", "Coinbase Customer Data Hit in Analytics Vendor Breach, Funds ‘Not Impacted’"),
    _h("2026-10-29 17:08", "cointelegraph.com", "KuCoin suspends services in Russia and Belarus"),
    _h("2026-10-29 18:55", "coindesk.com", "Polymarket suspends trading after third-party breach leads to user losses"),
    _h("2026-10-29 19:00", "theblock.co", "Polymarket pauses trading after $2.65 million exploit, says third-party provider was compromised"),
]


def test_amounts_in_headlines():
    assert ur.amount_usd("Binance Drained of $5.5 Billion in Suspected Hack") == (5.5e9, True)
    assert ur.amount_usd("Bloom bot hackers loot $953K faster than you can microwave popcorn") == (953e3, True)
    assert ur.amount_usd("Polymarket pauses trading after $2.65 million exploit") == (2.65e6, True)
    assert ur.amount_usd("Binance pauses withdrawals after suspected $3B+ hack") == (3e9, True)
    assert ur.amount_usd("Potential multi-billion Binance hack hitting crypto markets") == (1e9, False)
    assert ur.amount_usd("Bitcoin falls to $76,837") == (0.0, False)          # a price is not an amount


def test_the_binance_night_alarms_at_the_first_headline_and_confirms_at_the_second():
    first = ur.stories(BINANCE_NIGHT[:1])
    assert len(first) == 1 and first[0].entity == "Binance" and first[0].urgent()
    assert ur.due(first[0], None, _ms("2026-10-29 21:13")) == "new"
    text = ur.text_new(first[0], translate=lambda t: "Возможный взлом Binance на миллиарды бьёт по рынку")
    assert "СРОЧНО · Binance: взлом / кража" in text and "пока 1 источник" in text
    assert "«Возможный взлом Binance на миллиарды бьёт по рынку»" in text and "миллиарды долларов" in text
    record = ur.record_of(first[0], _ms("2026-10-29 21:13"))
    two = ur.stories(BINANCE_NIGHT[:2], {"Binance": (first[0].key, record["last_ts"])})
    assert two[0].key == first[0].key and ur.due(two[0], record, _ms("2026-10-29 21:25")) == "update"
    assert "подтверждают 2 источника" in ur.text_update(two[0], record)
    record = ur.record_of(two[0], _ms("2026-10-29 21:25"))
    rest = ur.stories(BINANCE_NIGHT, {"Binance": (two[0].key, record["last_ts"])})
    assert len(rest) == 1 and ur.due(rest[0], record, _ms("2026-10-30 00:20")) is None   # more of the same: quiet
    assert rest[0].usd == (5.8e9, True)


def test_leaks_small_exploits_and_unrelated_words_stay_quiet():
    found = ur.stories(QUIET)
    assert not [s.entity for s in found if s.urgent()]
    assert {s.entity for s in found} <= {"Bithumb", "Coinbase"}      # Polymarket, Keystone, KuCoin, Quant: not watched


def test_paused_withdrawals_and_relief():
    paused = [_h("2026-10-29 23:40", "investinglive.com",
                 "Binance pauses withdrawals after suspected $3B+ hack as Tether, Circle freeze $2.4B")]
    s = ur.stories(paused)[0]
    assert {"withdrawals", "theft"} <= s.kinds and s.urgent()
    record = ur.record_of(s, _ms("2026-10-29 23:41"))
    later = paused + [_h("2026-10-30 09:00", "coindesk.com", "Binance resumes withdrawals after security review")]
    s = ur.stories(later, {"Binance": (s.key, record["last_ts"])})[0]
    assert ur.due(s, record, _ms("2026-10-30 09:01")) == "update"
    assert "🟢 Binance — новое: хорошая новость" in ur.text_update(s, {**record, "sources": ["a", "b"]})


def test_old_news_does_not_start_a_story_and_a_late_one_is_not_sent():
    old = [_h("2026-10-29 10:00", "coindesk.com", "Bybit hacker launders $300 million through THORChain")]
    assert ur.stories(old) == []
    late = ur.stories(BINANCE_NIGHT[:1])[0]
    assert ur.due(late, None, _ms("2026-10-30 08:00")) == "silent"


def test_routine_binance_notices_are_not_incidents_and_official_ones_need_no_second_source():
    routine = _h("2026-10-27 02:00", "binance", "Binance Will Temporarily Suspend Deposits and Withdrawals on Ethereum "
                 "Network for the Scheduled Upgrade", "maintenance")
    assert ur.stories([routine]) == []
    incident = _h("2026-10-29 22:00", "binance", "Temporary Suspension of Withdrawals on Solana and XRP Networks",
                  "binance_news")
    s = ur.stories([incident])
    assert s[0].entity == "Binance" and s[0].urgent()


def test_a_delisting_of_a_held_coin():
    h = _h("2025-04-25 08:00", "binance", "Binance Will Delist ALPACA, PDA, VIB and WING on 2025-05-02", "delisting")
    assert ur.delisted(h, {"VIB", "BTC"}) == ["VIB"]
    assert ur.delisted(h, {"BTC", "ETH"}) == []


def test_a_fall_counts_in_typical_moves():
    rng = np.random.default_rng(0)
    closes = 80_000 * np.exp(np.cumsum(rng.normal(0, 0.0037, 90 * 24)))   # about 0.9% a 6-hour move
    sigma = ur.sigma_of(closes)
    assert 0.007 < sigma < 0.011
    assert ur.fall(80_891.82, 76_837.04, 0.0091) > ur.SHOCK_SIGMAS          # the night of 29.10: -5.0% = 5.7 moves
    assert ur.fall(80_000, 78_000, 0.0091) is None                          # -2.5%: an ordinary day
    text = ur.text_fall("BTCUSDT", 80_891.82, 76_837.04, 0.0091, 5.7, {"ETHUSDT": -0.061}, {"BTC", "ETH"})
    assert "BTC −5,0% за 6 часов (в портфеле)" in text and "ETH −6,1% (в портфеле)" in text


def test_depegs():
    assert ur.depegs({"USDCUSDT": 1.001, "FDUSDUSDT": 0.997}) == []
    off = ur.depegs({"USDCUSDT": 0.962, "FDUSDUSDT": 0.999})
    assert off == [("USDCUSDT", 0.962)] and "USDC = 0,962 USDT (−3,8%)" in ur.text_depeg(off, {"USDCUSDT": 0.962})
    every = {"USDCUSDT": 1.04, "FDUSDUSDT": 1.035}
    assert "USDT дешевеет" in ur.text_depeg(ur.depegs(every), every)


def test_binance_own_word_goes_out_at_once_with_what_its_text_says():
    body = ("Binance has temporarily suspended deposits and withdrawals while security checks are completed. "
            "The estimated total impact sits at around $6.3B. Binance will cover all losses from the SAFU fund.")
    official = _h("2026-10-30 02:23", "binance", "Security Incident Update", "binance_news")
    official = ur.Headline(official.url, official.source, official.kind, official.ts, official.title, body)
    first = ur.stories(BINANCE_NIGHT)[0]
    record = ur.record_of(first, _ms("2026-10-30 00:20"))
    s = ur.stories(BINANCE_NIGHT + [official], {"Binance": (first.key, record["last_ts"])})[0]
    assert {"withdrawals", "relief"} <= s.kinds and s.usd == (6.3e9, True)
    assert ur.due(s, record, _ms("2026-10-30 02:24")) == "update"
    text = ur.text_update(s, record, summarize=lambda t: "Binance приостановила депозиты и выводы.")
    assert text.startswith("🏛 Binance: Binance официально") and "Binance приостановила депозиты и выводы." in text
    assert "до $6,3 млрд" in text


def test_announcement_text_from_its_node_tree():
    from core.sandbox.context_sources import parse_article
    body = '{"node":"root","child":[{"node":"element","child":[{"node":"text","text":"Fellow Binancians,"}]},' \
           '{"node":"text","text":"User funds are&nbsp;safe."}]}'
    assert parse_article({"data": {"body": body}}) == "Fellow Binancians, User funds are safe."
