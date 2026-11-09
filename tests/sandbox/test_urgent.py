"""Urgent events: real headlines of from the news archive, and the market checks."""

import json
from datetime import datetime, timezone
from pathlib import Path

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


DATA = Path(__file__).parent / "data"
# the bot's model's answers (Groq) for every danger headline of the archive and of Bybit's hack
SUBJECTS = json.loads((DATA / "subjects.json").read_text(encoding="utf-8"))


def _named(when, source, title, kind="article", url=None):
    """A real headline with the model's recorded answer (None: not asked, or no answer)."""
    key = title if title in SUBJECTS else next((k for k in SUBJECTS if k.startswith(title)), title)
    answer = SUBJECTS.get(key)
    return ur.Headline(url or f"https://{source}/{abs(hash(key))}", source, kind, _ms(when), key, "",
                       *(answer if answer else (None, "")))


def _replay(watch, name, held=()):
    """The watcher's own news round over a fixture at every minute a headline comes out or a fact waits:
    [(UTC time, message)], and the watcher."""
    rows = json.loads((DATA / name).read_text(encoding="utf-8"))
    hs = [_named(t, s, title, k, f"https://{s}/{i}") for i, (t, s, k, title) in enumerate(rows)]
    sent, out, last = [], [], -1
    w = watch(sent)
    first, end = min(h.ts for h in hs), max(h.ts for h in hs) + 4 * 3_600_000
    for now in range(first - first % 60_000, end, 60_000):
        count = sum(h.ts <= now for h in hs)
        if count == last and not any(r.get("pending") for r in w.state["stories"].values()):
            continue
        last, n = count, len(sent)
        seen = [h for h in hs if h.ts <= now]
        w.news_round(None, now, set(held), [h for h in seen if h.kind != "onchain"], seen)
        out += [(datetime.fromtimestamp(now / 1000, timezone.utc).strftime("%d.%m %H:%M"), text) for text in sent[n:]]
    return out, w


def test_the_binance_hack_is_told_once_and_then_each_new_fact_once(watch):
    # every headline of 29.10 21:00 - 31.10 15:00 UTC in the news archive (210, 92 of them about Binance)
    sent, w = _replay(watch, "binance_hack_news.json")
    expected = [
        ("29.10 21:12", ["🚨 СРОЧНО · Binance: взлом / кража", "«Potential multi-billion Binance hack"]),
        ("29.10 21:24", ["✅ Binance: подтверждают 2 источника — CoinDesk, Cointelegraph"]),
        ("29.10 22:18", ["🟢 Деньги клиентов: «CZ: Binance May Have Been Hacked"]),
        ("30.10 00:30", ["🔴 Деньги под угрозой: «Binance hack drained wallets backing customer assets",
                         "🔑 Вывод: «He Yi: Binance will cover all hacking losses, withdrawals expected to gradually "
                         "resume within 48 hours", "📈 Сумма уже $5,8 млрд: «Binance has been hacked for over $5.8B"]),
        ("30.10 02:38", ["📢 Binance официально, 03:23: «Security Incident Update»"]),
        ("30.10 09:46", ["🟠 Тревожный знак: «Binance Finds 'Additional Points of Breach' in $6.3B Hack» (Decrypt"]),
        ("30.10 11:23", ["🟢 Деньги клиентов: «Binance Reserves Over $6.34B in Company Funds to Cover Hack Losses"]),
        # CoinDesk's "targets Sunday withdrawal restart" (16:31) is no news: Sunday is He Yi's "within 48 hours"
        ("30.10 19:48", ["📢 Binance официально, 20:33: «Security Incident Update #2 (2026-10-30)»"]),
        ("31.10 08:51", ["🟠 Тревожный знак: «Binance faces inside-job allegations as it delays proof-of-reserves "
                         "report after $6.3 billion hack» (Whale Alert, 09:36 · пока 1 источник)"]),
        ("31.10 14:39", ["🟢📢 Binance официально, 15:24: «Security Incident | Recovery Update #3: 90% of Stolen"]),
    ]
    assert [at for at, _ in sent] == [at for at, _ in expected]
    for (at, text), (_, parts) in zip(sent, expected):
        assert all(part in text for part in parts), (at, text)
    assert {r["entity"] for r in w.state["stories"].values()} == {"Binance"}
    held, _ = _replay(watch, "binance_hack_news.json", {"PEPE", "UNI", "ETH"})   # the hacker sold them for $ETH
    assert any("🪙 Твои монеты PEPE, UNI: «Attention! The Binance hacker is now dumping" in text for _, text in held)


def test_the_bybit_hack_of_february_2025_with_no_stories_of_the_attacker(watch):
    # 133 headlines of from Google News (any outlet, not just ours); some outlets' times are only
    # the day, so the first day comes as one message
    sent, w = _replay(watch, "bybit_hack_news.json")
    assert [at for at, _ in sent] == ["21.02 08:00", "22.02 08:15", "24.02 08:15", "26.02 08:15"]
    for (at, text), part in zip(sent, [
            "🚨 СРОЧНО · Bybit: взлом / кража\nbinance.com, 09:00 — пишут 18 источников: binance.com, bloomberg.com, "
            "businesspost.ng, cnbc.com, CoinDesk и ещё 13",
            "🟠 Тревожный знак: «Bybit Faces $4 Billion Withdrawal Wave",
            "🟢 Деньги клиентов: «Bybit say it's fully recovered after record-breaking $1.5 billion hack»",
            "🟠 Мошенники на волне новости: «Binance SMS Scam Preys on Bybit Hack Fallout, Tricks Users»"]):
        assert part in text, (at, text)
    # not North Korea or Lazarus (the attackers), nor the names the model made up for "a crypto exchange"
    assert {r["entity"] for r in w.state["stories"].values()} == {"Bybit"}


def test_a_story_is_whoever_the_model_names_and_of_strangers_only_the_big_or_the_confirmed_is_sent():
    # a US prediction market charged in Utah, a $2.65M exploit there, an Indonesian exchange stopping
    # withdrawals; the names alone saw none of them, and none is where our money is
    kalshi = [_named("2026-10-29 05:02", "coindesk.com", "Utah charges predictions platform Kalshi with illegal gambling"),
              _named("2026-10-29 06:40", "theblock.co",
                     "Utah files criminal charges against Kalshi, executives over alleged illegal gambling"),
              _named("2026-10-29 11:06", "decrypt.co", "Utah Files Felony Charges Against Kalshi, Co-Founders Over Sports Bets")]
    poly = [_named("2026-10-29 18:53", "decrypt.co", "Polymarket Pauses Trading as Prediction Market Investigates Security Incident"),
            _named("2026-10-29 18:55", "coindesk.com", "Polymarket suspends trading after third-party breach leads to user losses"),
            _named("2026-10-29 19:00", "theblock.co",
                   "Polymarket pauses trading after $2.65 million exploit, says third-party provider was compromised"),
            _named("2026-10-29 19:23", "cointelegraph.com", "Polymarket suspended trading amid reports of user funds being drained")]
    indodax = [_named("2026-10-29 11:52", "wublockchain",
                      "Indonesia's Largest Crypto Exchange Indodax Suspends Deposits and Withdrawals")]
    infini = [_named("2025-02-24 08:00", "tradingview.com", "Infini loses $50M in exploit; developer deception suspected")]
    found = {s.entity: s for s in ur.stories(kalshi + poly + indodax + infini)}
    assert set(found) == {"Kalshi", "Polymarket", "Indodax", "Infini"}
    assert not [name for name, s in found.items() if s.urgent()]
    assert found["Kalshi"].kinds == {"enforcement"} and found["Polymarket"].usd == (2.65e6, True)
    assert "withdrawals" in found["Indodax"].kinds and found["Indodax"].sources == ["wublockchain"]
    unnamed = _named("2025-02-21 08:00", "apnews.com", "Cryptocurrency exchange says it was victim of $1.5 billion hack")
    assert unnamed.subject == "Bitfinex" and ur.subject_of(unnamed) is None and ur.read(unnamed) is None
    attacker = _named("2025-02-26 08:00", "fortune.com",
                      "North Korean hackers pull off the biggest heist of all time, stealing $1.5 billion in crypto")
    assert ur.read(attacker) is None
    silent = _named("2025-02-26 08:00", "bleepingcomputer.com", "Lazarus hacked Bybit via breached Safe{Wallet} developer machine")
    assert silent.subject is None and ur.read(silent).entity == "Bybit"           # no answer: the names decide


def test_dates_of_a_restart_are_days():
    friday = _ms("2026-10-30 00:15")
    assert {ur.when(p, friday) for p in ("within 48 hours", "Sunday", "Nov. 1", "November 1st", "1st Nov")} == {"2026-11-01"}
    assert ur.when("tomorrow", friday) == "2026-10-31" and ur.when("in 3 days", friday) == "2026-11-02"


def test_announcement_text_from_its_node_tree():
    from core.sandbox.context_sources import parse_article
    body = '{"node":"root","child":[{"node":"element","child":[{"node":"text","text":"Fellow Binancians,"}]},' \
           '{"node":"text","text":"User funds are&nbsp;safe."}]}'
    assert parse_article({"data": {"body": body}}) == "Fellow Binancians, User funds are safe."


def test_a_hack_belongs_to_the_name_it_stands_next_to():
    # real headlines of; the first sent "СРОЧНО · Coinbase: взлом / кража" on 30.10 at 17:52 UTC
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


def _said(when, source, title, subject, ticker=""):
    """A real headline with the model's answer as the watcher recorded it."""
    return ur.Headline(f"https://{source}/{abs(hash(title))}", source, "article", _ms(when), title, "", subject, ticker)


NOVEMBER = [   # 02-04.11; sent were Mercado Bitcoin (right), Solana, Liquid and Liquid Network (wrong), Arkéa (the rule)
    _said("2026-11-02 15:53", "coindesk.com",
          "Brazil’s Mercado Bitcoin halts withdrawals after $40 million drained across 10 chains", "Mercado Bitcoin"),
    _said("2026-11-02 16:46", "decrypt.co",
          "Brazil's Largest Exchange Mercado Bitcoin Halts Withdrawals After $40 Million Hack", "Mercado Bitcoin"),
    _said("2026-11-02 21:16", "cointelegraph.com",
          "Crypto investigators flag suspected $106M theft, $786K Solana bot drain", "Solana", "SOL"),
    _said("2026-11-03 01:35", "lookonchain", "Another exchange, @MercadoBitcoin , has been hacked for over $46M! "
          "Over $9M in stolen assets has already been frozen.", "MercadoBitcoin"),
    _said("2026-11-03 11:12", "coindesk.com",
          "Liquid’s bitcoin IOUs are fully backed again after a $320 million exploit", "Liquid", "LDO"),
    _said("2026-11-03 11:22", "decrypt.co",
          "Liquid Network Restores Full Bitcoin Backing Two Months After $320M Exploit", "Liquid Network", "L-BTC"),
    _said("2026-11-03 11:49", "theblock.co",
          "Liquid Network restores full bitcoin backing following $320 million exploit", "Liquid Network", "LQDR"),
    _said("2026-11-04 04:57", "theblock.co",
          "French bank Crédit Mutuel Arkéa exploited for over $133 million, ZachXBT says", "Crédit Mutuel Arkéa"),
]


def test_good_news_of_an_old_exploit_one_party_under_two_names_and_an_amount_that_is_someone_elses():
    found = {s.entity: s for s in ur.stories(NOVEMBER)}
    assert set(found) == {"Mercado Bitcoin", "Solana", "Crédit Mutuel Arkéa"}       # no Liquid: it is backed again
    assert found["Mercado Bitcoin"].sources == ["coindesk.com", "decrypt.co", "lookonchain"]   # @MercadoBitcoin too
    assert found["Mercado Bitcoin"].urgent() and found["Crédit Mutuel Arkéa"].urgent()
    assert found["Solana"].usd == (786e3, True) and not found["Solana"].urgent()    # the $106M is not Solana's
    assert ur.amount_usd("Binance Drained of $5.5 Billion in Suspected Hack, Tether and Circle Freeze $1.8 Billion",
                         near="Binance") == (5.5e9, True)
    assert ur.same_party("Liquid", "Liquid Network") and ur.same_party("MercadoBitcoin", "Mercado Bitcoin")
    assert not ur.same_party("Liquid", "Lido") and not ur.same_party("", "Liquid")
    # a Liquid exploit still open joins one story under either name
    fresh = [_said("2026-11-03 11:12", "coindesk.com", "Liquid hit by $320 million exploit", "Liquid"),
             _said("2026-11-03 11:49", "theblock.co", "Liquid Network drained of $320 million", "Liquid Network")]
    assert [(s.entity, s.sources) for s in ur.stories(fresh)] == [("Liquid", ["coindesk.com", "theblock.co"])]


def test_a_token_the_model_names_must_be_the_partys():
    listed = {"SOL", "AAVE", "ADA", "LDO", "BNB"}
    title = "Liquid’s bitcoin IOUs are fully backed again after a $320 million exploit"
    assert ur.plausible_ticker("Liquid", "LDO", title, listed) == ""                 # Lido's
    assert ur.plausible_ticker("Liquid Network", "LQDR", title, listed) == ""        # not on Binance
    assert ur.plausible_ticker("Liquid Network", "L-BTC", title) == ""
    assert ur.plausible_ticker("Solana", "SOL", "Solana bot drained", listed) == "SOL"
    assert ur.plausible_ticker("Aave", "aave", "Aave exploited", listed) == "AAVE"
    assert ur.plausible_ticker("Cardano", "ADA", "Cardano halts", listed) == "ADA"
    assert ur.plausible_ticker("Binance Smart Chain", "BNB", "BNB Chain halts blocks", listed) == "BNB"


def test_a_coin_with_another_word_is_another_party():
    # the night researcher, 08.11.2026: Ethereum = Ethereum Classic made an ETC theft Ethereum's story
    for a, b in (("Ethereum", "Ethereum Classic"), ("Bitcoin", "Bitcoin Cash"), ("Terra", "Terra Classic"),
                 ("Coinbase", "Coinbase Wallet"), ("Bitcoin", "Bitcoin SV")):
        assert not ur.same_party(a, b) and not ur.same_party(b, a), (a, b)
    assert ur.same_party("Kraken", "Kraken Exchange") and ur.same_party("Lido", "Lido Finance")


def test_an_etc_headline_does_not_join_ethereums_story():
    t0 = 1_794_000_000_000
    a = ur.Headline("u1", "coindesk", "news", t0, "Ethereum network halts block production after bug",
                    subject="Ethereum", ticker="ETH")
    b = ur.Headline("u2", "theblock", "news", t0 + 3_600_000, "Ethereum Classic exploited, $5 million stolen",
                    subject="Ethereum Classic", ticker="ETC")
    stories = ur.stories([a, b])
    assert len(stories) == 2 and {s.entity for s in stories} == {"Ethereum", "Ethereum Classic"}
