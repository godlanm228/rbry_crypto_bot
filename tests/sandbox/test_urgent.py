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


def test_paused_withdrawals_and_their_reopening():
    paused = [_h("2026-10-29 23:40", "investinglive.com",
                 "Binance pauses withdrawals after suspected $3B+ hack as Tether, Circle freeze $2.4B")]
    s = ur.stories(paused)[0]
    assert {"withdrawals", "theft"} <= s.kinds and s.urgent()
    record = ur.record_of(s, _ms("2026-10-29 23:41"))
    ur.follow(record, "Binance", paused, set(), _ms("2026-10-29 23:41"), quiet=True)       # the first message told it
    assert {"access:closed", "risk:theft"} <= set(record["told"]) and record["usd_told"] == 3e9
    record["sources"] = ["investinglive.com", "theblock.co"]                    # confirmed already
    later = paused + [_h("2026-10-30 09:00", "coindesk.com", "Binance resumes withdrawals after security review")]
    s = ur.stories(later, {"Binance": (s.key, record["last_ts"])})[0]
    assert ur.due(s, record, _ms("2026-10-30 09:01")) is None                  # not a confirmation: a new fact
    ur.follow(record, "Binance", later, set(), _ms("2026-10-30 09:01"))
    assert [p["key"] for p in record["pending"]] == ["access:open"]
    assert not ur.follow_due(record, _ms("2026-10-30 09:15")) and ur.follow_due(record, _ms("2026-10-30 09:16"))
    text = ur.text_follow("Binance", record, later)
    assert text.startswith("🔄 Binance — что нового\n🔑 Вывод: «Binance resumes withdrawals after security review» "
                           "(CoinDesk, 10:00 · пока 1 источник)")
    ur.follow_told(record)
    assert "access:open" in record["told"] and "access:closed" not in record["told"] and not record["pending"]
    ur.follow(record, "Binance", later + [_h("2026-10-30 09:30", "decrypt.co", "Binance resumes withdrawals")],
              set(), _ms("2026-10-30 09:31"))
    assert not record["pending"]                                                # the same from another outlet


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


def test_binance_own_word_goes_out_with_what_its_text_says():
    body = ("Binance has temporarily suspended deposits and withdrawals while security checks are completed. "
            "The estimated total impact sits at around $6.3B. Binance will cover all losses from the SAFU fund.")
    official = _h("2026-10-30 02:23", "binance", "Security Incident Update", "binance_news")
    official = ur.Headline(official.url, official.source, official.kind, official.ts, official.title, body)
    first = ur.stories(BINANCE_NIGHT)[0]
    record = ur.record_of(first, _ms("2026-10-30 00:20"))
    ur.follow(record, "Binance", BINANCE_NIGHT, set(), _ms("2026-10-30 00:20"), quiet=True)
    ur.follow(record, "Binance", BINANCE_NIGHT + [official], set(), _ms("2026-10-30 02:24"))
    assert {p["key"] for p in record["pending"]} == {"access:closed", "safe", f"official:{official.url}"}
    text = ur.text_follow("Binance", record, BINANCE_NIGHT + [official],
                          summarize=lambda t: "Binance приостановила депозиты и выводы.")
    assert "🔑🟢📢 Binance официально, 03:23: «Security Incident Update»\nBinance приостановила депозиты и выводы." in text


def test_the_binance_hack_after_its_confirmation_tells_each_new_fact_once():
    # every headline of 29.10 21:00 - 31.10 15:00 UTC in the news archive (210, 92 of them about Binance),
    # replayed minute by minute after the first message (21:12) and its confirmation (21:24)
    import json
    from pathlib import Path
    rows = json.loads((Path(__file__).parent / "data" / "binance_hack_news.json").read_text(encoding="utf-8"))
    hs = [ur.Headline(f"https://{s}/{i}", s, k, _ms(t), title) for i, (t, s, k, title) in enumerate(rows)]

    def replay(held):
        first = [h for h in hs if h.ts <= _ms("2026-10-29 21:24")]
        record = ur.record_of(ur.stories([h for h in first if h.kind != "onchain"])[0], _ms("2026-10-29 21:25"))
        ur.follow(record, "Binance", first, held, _ms("2026-10-29 21:25"), quiet=True)
        sent = []
        for now in range(_ms("2026-10-29 21:26"), _ms("2026-10-31 15:30"), 60_000):
            seen = [h for h in hs if h.ts <= now]
            ur.follow(record, "Binance", seen, held, now)
            if ur.follow_due(record, now):
                keys = sorted(p["key"] if not p["key"].startswith("official:") else "official" for p in record["pending"])
                sent.append((datetime.fromtimestamp(now / 1000, timezone.utc).strftime("%d.%m %H:%M"), keys,
                             ur.text_follow("Binance", record, seen)))
                ur.follow_told(record)
        return sent

    sent = replay(set())
    assert [(at, keys) for at, keys, _ in sent] == [
        ("29.10 22:18", ["safe", "scale"]),                                  # $5.5B; CZ: user funds safe
        ("30.10 00:30", ["access:closed", "access:date:within 48 hours", "risk:customers"]),
        ("30.10 02:38", ["official"]),                                       # Security Incident Update
        ("30.10 09:46", ["flag:wider"]),                                     # additional points of breach
        ("30.10 11:23", ["safe"]),                                           # $6.34B of its own funds set aside
        ("30.10 16:46", ["access:date:sunday"]),
        ("30.10 19:48", ["official"]),                                       # Security Incident Update #2
        ("31.10 08:51", ["flag:inside", "flag:por"]),
        ("31.10 14:39", ["official", "safe"]),                               # 90% replenished
    ]
    assert "🟠 Тревожный знак: «Binance faces inside-job allegations as it delays proof-of-reserves report after " \
           "$6.3 billion hack» (Whale Alert, 09:36 · пока 1 источник)" in sent[7][2]
    held = replay({"PEPE", "UNI", "ETH"})                                     # the hacker sold them for $ETH
    assert ("29.10 23:03", ["held:PEPE", "held:UNI"]) in [(at, keys) for at, keys, _ in held]
    assert "🪙 Твои монеты PEPE, UNI: «Attention! The Binance hacker is now dumping" in held[1][2]


def test_announcement_text_from_its_node_tree():
    from core.sandbox.context_sources import parse_article
    body = '{"node":"root","child":[{"node":"element","child":[{"node":"text","text":"Fellow Binancians,"}]},' \
           '{"node":"text","text":"User funds are&nbsp;safe."}]}'
    assert parse_article({"data": {"body": body}}) == "Fellow Binancians, User funds are safe."


def test_a_hack_belongs_to_the_name_it_stands_next_to():
    # real headlines of 30-31.10.2026; the first sent "СРОЧНО · Coinbase: взлом / кража" on 30.10 at 17:52 UTC
    named = {
        "Coinbase leads crypto stock selloff following $6.3B Binance hack": "Binance",
        "Binance hacker’s Bitcoin haul tops Coinbase’s corporate treasury": "Binance",
        "Secret Service, FBI Join Binance Hack Probe as Circle Freezes $823 Million": "Binance",
        "Morning Minute: Binance’s $6.4B Hack Overshadows Hyperliquid’s Biggest Day": "Binance",
        "Breaking: The hackers behind the $6B #Binance hack, $1.5B #Bybit hack, and $388M #Bitget hack may all be "
        "linked to North Korea's Lazarus Group.": "Binance",
        "CZ: Binance May Have Been Hacked as Tether and Circle Freeze Over $2.3B in Stablecoins": "Binance",
    }
    for title, whom in named.items():
        assert ur.read(_h("2026-10-30 17:48", "cointelegraph.com", title)).entity == whom, title
    record = ur.record_of(ur.stories(BINANCE_NIGHT)[0], _ms("2026-10-30 00:20"))
    known = {"Binance": (ur.stories(BINANCE_NIGHT)[0].key, record["last_ts"])}
    later = BINANCE_NIGHT + [_h("2026-10-30 17:48", "cointelegraph.com",
                                "Coinbase leads crypto stock selloff following $6.3B Binance hack")]
    found = ur.stories(later, known)
    assert [s.entity for s in found] == ["Binance"] and ur.due(found[0], record, _ms("2026-10-30 17:52")) is None
