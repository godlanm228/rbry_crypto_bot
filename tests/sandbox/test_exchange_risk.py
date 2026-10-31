"""Trouble at an exchange before the news: real Whale Alert posts of the Binance hack night (29.10.2026, UTC)."""

from datetime import datetime, timezone

import pytest

from core.sandbox import exchange_risk as er


def _ms(text):
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


NIGHT = [
    ("2026-10-29 16:36", "500,000,000 $USDT (499,576,000 USD) transferred from #Binance to Tether Treasury Details"),
    ("2026-10-29 20:06", "185,017,485 $XRP (235,816,974 USD) transferred from #Binance to unknown new wallet Details"),
    ("2026-10-29 20:11", "965,455 $SOL (95,436,212 USD) transferred from #Binance to unknown new wallet Details"),
    ("2026-10-29 20:23", "612,729,468 $USDT (612,130,218 USD) transferred from #Binance to unknown wallet Details"),
    ("2026-10-29 20:26", "28,005 $ETH (64,863,623 USD) transferred from #Binance to unknown wallet Details"),
    ("2026-10-29 20:30", "7,636 $BTC (613,984,871 USD) transferred from #Binance to unknown new wallet Details"),
    ("2026-10-29 20:31", "2,946 $BTC (236,925,416 USD) transferred from #Binance to unknown new wallet Details"),
    ("2026-10-29 21:50", "7,772,977,017,415 $PEPE (37,481,346 USD) transferred from Binance Hack 2026 to unknown "
                         "wallet Details"),
    ("2026-10-29 23:55", "2,210,833 $HYPE (197,165,480 USD) transferred from unknown wallet to #Binance Details"),
]
TRANSFERS = [er.parse_whale(text, _ms(at)) for at, text in NIGHT]


def test_whale_alert_posts():
    xrp = TRANSFERS[1]
    assert (xrp.usd, xrp.token, xrp.source, xrp.to_unknown, xrp.fresh) == (235_816_974, "XRP", "Binance", True, True)
    assert TRANSFERS[0].source == "Binance" and not TRANSFERS[0].to_unknown        # to Tether's treasury: known
    assert TRANSFERS[7].source == "" and TRANSFERS[8].source == ""               # the hacker's wallet; an inflow
    assert er.parse_whale("Binance has been hacked for over $5.8B!", 0) is None
    assert er.exchange_name("#OKEx") == "OKX" and er.exchange_name("unknown wallet") == ""


def test_the_outflow_alarm_rings_before_the_news():
    rang = None
    for t in TRANSFERS:
        usd, n, first = er.outflows(TRANSFERS, t.ts).get("Binance", (0.0, 0, 0))
        if er.outflow_alarm(usd, n):
            rang = t.ts
            break
    assert rang is not None and rang < _ms("2026-10-29 21:12")                 # the first headline
    assert rang <= _ms("2026-10-29 20:30")
    quiet = er.outflows(TRANSFERS, _ms("2026-10-29 23:59"))
    assert "Binance" not in quiet or not er.outflow_alarm(*quiet["Binance"][:2])   # hours later it is over


def test_the_canary():
    assert er.canary(-0.070, -0.015, 9e6, 1.5e6) == pytest.approx(-0.055)       # BNB on its own, 6x volume
    assert er.canary(-0.070, -0.050, 9e6, 1.5e6) is None                         # the whole market fell
    assert er.canary(-0.070, -0.015, 3e6, 1.5e6) is None                         # on ordinary volume
    assert er.canary(-0.070, -0.015, 9e6, 0.0) is None


def test_stuck_withdrawals():
    payload = {"data": [
        {"coin": "BTC", "networkList": [{"network": "BTC", "withdrawEnable": False}]},
        {"coin": "USDT", "networkList": [{"network": "TRX", "withdrawEnable": False},
                                         {"network": "ETH", "withdrawEnable": False}]},
        {"coin": "ETH", "networkList": [{"network": "ETH", "withdrawEnable": True}]},
        {"coin": "PEPE", "networkList": [{"network": "ETH", "withdrawEnable": False}]}]}
    assert er.stuck(payload) == ["BTC", "USDT"]


def test_the_card_says_what_is_seen_what_it_may_be_and_what_to_do():
    sit = er.Situation("Binance", _ms("2026-10-29 20:26"))
    sit.add("outflow", _ms("2026-10-29 20:06"), "с Binance ушло $1,0 млрд на неизвестные кошельки за 60 мин")
    assert er.level(sit.signals) == "средняя"
    text = er.text_situation(sit, lambda ms: "21:06")
    assert text.startswith("🆘 Binance: похоже на беду · уверенность средняя")
    assert "новостей пока нет" in text and "взлом горячих кошельков" in text
    assert "монеты на споте этой биржи" in text and "свой кошелёк" in text and "не совет" in text
    assert sit.add("canary", _ms("2026-10-29 20:32"), "BNB −5,5% за час сверх движения BTC") is True
    assert sit.add("canary", _ms("2026-10-29 20:40"), "again") is False
    assert er.level(sit.signals) == "высокая"
    assert "уверенность теперь: высокая".capitalize() in er.text_more(sit, "canary", lambda ms: "21:32")
    assert er.hypotheses(["withdrawals"])[0].startswith("биржа остановила выводы")


def test_telegram_preview_posts():
    from core.sandbox.context_sources import parse_telegram
    page = ('<div class="tgme_widget_message_wrap js-widget_message_wrap"><div class="tgme_widget_message" '
            'data-post="whale_alert_io/91234"><div class="tgme_widget_message_text js-message_text" dir="auto">'
            '&#128680; 185,017,485 <a href="x">#XRP</a> (235,816,974 USD) transferred from #Binance to unknown '
            'new wallet<br/><a href="y">Details</a></div><time datetime="2026-10-29T20:06:11+00:00" '
            'class="time">20:06</time></div></div>')
    items = parse_telegram(page, "whale_alert_io", "onchain")
    assert items[0]["url"] == "https://t.me/whale_alert_io/91234" and items[0]["kind"] == "onchain"
    assert items[0]["published_ts"] == _ms("2026-10-29 20:06") + 11_000
    t = er.parse_whale(items[0]["title"], items[0]["published_ts"])
    assert t is not None and t.source == "Binance" and t.usd == 235_816_974


def test_withdrawals_closed_for_days_are_told_once_then_their_reopening_and_closing_again(watch):
    # 30-31.10.2026: Binance's withdrawals closed after the hack; the watcher told it again a day later
    sent, closed = [], {"Binance": {"withdrawals": (_ms("2026-10-31 09:52"), "на Binance закрыт вывод 7 из 7")}}
    w = watch(sent)
    found = closed
    w.risk_signals = lambda client, now_ms: found
    for hour in range(0, 60):                           # two and a half days, still closed
        w.risk_round(None, _ms("2026-10-31 09:52") + hour * 3_600_000, [])
    assert len(sent) == 1 and sent[0].startswith("🆘 Binance")
    found = {}
    w.risk_round(None, _ms("2026-11-02 22:00"), [])
    w.risk_round(None, _ms("2026-11-02 23:00"), [])
    assert len(sent) == 2 and sent[1].startswith("🟢 Binance: выводы снова открыты")
    found = closed
    w.risk_round(None, _ms("2026-11-03 01:00"), [])
    assert len(sent) == 3 and sent[2].startswith("⚠️ Binance: новое")


def test_reopened_withdrawals_are_told_once_by_the_news_or_by_the_check(watch):
    from core.sandbox import urgent as ur
    sent = []
    w = watch(sent)
    closed = {"Binance": {"withdrawals": (_ms("2026-10-31 09:52"), "на Binance закрыт вывод 7 из 7")}}
    found = closed
    w.risk_signals = lambda client, now_ms: found
    story = {"entity": "Binance", "sent_at": _ms("2026-10-29 21:13"), "sources": ["coindesk.com", "cointelegraph.com"],
             "kinds": ["theft"], "last_ts": _ms("2026-10-31 09:00"), "told": {}, "pending": [], "looked": []}
    w.state["stories"]["Binance|1"] = story
    w.risk_round(None, _ms("2026-10-31 09:52"), [])
    assert len(sent) == 1 and "access:closed" in story["told"]         # the SOS: the headlines of it are no news
    ur.tell(story, "access:open")                                       # the news told the reopening first ...
    found = {}
    w.risk_round(None, _ms("2026-11-01 10:00"), [])
    assert len(sent) == 1                                               # ... so the check keeps quiet
    found = closed
    w.risk_round(None, _ms("2026-11-01 12:00"), [])
    assert len(sent) == 2 and sent[1].startswith("⚠️ Binance: новое") and "access:open" not in story["told"]
    found = {}
    w.risk_round(None, _ms("2026-11-01 14:00"), [])
    assert len(sent) == 3 and sent[2].startswith("🟢 Binance: выводы снова открыты") and "access:open" in story["told"]


def test_the_model_is_asked_once_a_danger_headline_and_without_its_answer_the_names_decide(watch):
    from core.sandbox import urgent as ur
    w = watch([])
    asked = []
    answers = {"Utah charges predictions platform Kalshi with illegal gambling": ["Kalshi", ""]}
    w.name = lambda title: asked.append(title) or answers.get(title)
    hs = [ur.Headline("u1", "coindesk.com", "article", _ms("2026-10-31 10:00"),
                      "Utah charges predictions platform Kalshi with illegal gambling"),
          ur.Headline("u2", "cryptonews.com", "article", _ms("2026-10-31 10:01"),
                      "Copilot AI Predicts PI Network Binance Price"),                   # no danger words: not asked
          ur.Headline("u3", "cointelegraph.com", "article", _ms("2026-10-31 10:02"),
                      "Coinbase leads crypto stock selloff following $6.3B Binance hack")]   # the model is silent
    for _ in range(4):
        out = w.subjects(hs)
    assert asked.count(hs[0].title) == 1 and hs[1].title not in asked and asked.count(hs[2].title) == 3
    assert out[0].subject == "Kalshi" and ur.read(out[0]).entity == "Kalshi"
    assert out[2].subject is None and ur.read(out[2]).entity == "Binance"
