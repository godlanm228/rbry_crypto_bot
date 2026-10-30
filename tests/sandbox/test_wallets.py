"""Our own look at exchanges' hot wallets: outside flows in dollars, and a signal only past the week's worst hour."""

from core.sandbox import exchange_risk as er
from core.sandbox import wallets as wl

HOUR = 3_600_000
HOT = {"0xaaa", "0xbbb"}


def test_only_money_crossing_the_exchange_wall_counts_in_dollars():
    rows = [{"from": "0xAAA", "to": "0xccc", "value": str(2 * 10 ** 18), "timeStamp": "100"},
            {"from": "0xddd", "to": "0xbbb", "value": str(10 ** 18), "timeStamp": "200"},
            {"from": "0xaaa", "to": "0xBBB", "value": str(5 * 10 ** 18), "timeStamp": "300"}]
    assert wl.flows(rows, HOT, 18, 3000.0) == [(100_000, 6000.0, 0.0), (200_000, 0.0, 3000.0)]


def test_a_signal_needs_two_days_of_learning_a_floor_and_half_again_the_weeks_worst_hour():
    now = 1000 * HOUR
    hours = {str((now // HOUR - k) * HOUR): [80e6, 20e6] for k in range(2, 60)}     # 60M net an hour, usually
    since = now - 100 * HOUR
    assert wl.alarm([(now - 10, 95e6, 0.0)], hours, now, since) is None              # above 60M, below 90M
    net, worst = wl.alarm([(now - 10, 120e6, 10e6)], hours, now, since)
    assert net == 110e6 and worst == 60e6
    assert wl.alarm([(now - 10, 500e6, 0.0)], {}, now, now - 10 * HOUR) is None     # still learning
    assert wl.alarm([(now - 10, 500e6, 0.0)], {}, now, since) == (500e6, 0.0)        # quiet week, a big hour


def test_our_own_wallet_signal_counts_as_an_outflow():
    assert er.level(["wallets", "canary"]) == "высокая"
    assert er.hypotheses(["wallets"])[0] == "взлом горячих кошельков"
