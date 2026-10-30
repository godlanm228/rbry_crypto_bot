"""The strategy factory: genomes, the trades they take, a short on the mirrored path, goals and breeding."""

import numpy as np
import pandas as pd

from core.brain import exits as ex
from core.brain import factory as fa

HOUR = 3_600_000
T0 = int(pd.Timestamp("2023-03-01 10:00").value // 10**6)


def test_one_strategy_has_one_id_and_settings_that_cannot_act_are_dropped():
    a = fa.Genome(events=("DUMP", "OI-FLUSH"), side=1, take=2.0, squeeze=0.2, stop=("pct", 0.1))
    b = fa.Genome(events=("OI-FLUSH", "DUMP"), side=1, stop=("pct", 0.1))
    assert fa.normal(a) == fa.normal(b) and fa.normal(a).id == fa.normal(b).id   # take needs a sigma stop,
    assert fa.Genome.from_dict(fa.normal(a).to_dict()) == fa.normal(a)          # a squeeze filter a short


def test_breeding_makes_only_valid_new_genomes():
    rng = np.random.default_rng(3)
    pool = {g.id: {"id": g.id, "genome": g.to_dict(), "stamp": "old", "learn_n": 0, "years": {}} for g in fa.seeds()}
    fresh, again = fa.breed(pool, "new", rng, batch=40, retry=3)
    assert len(fresh) == 40 and not {g.id for g in fresh} & set(pool)
    assert all(g == fa.normal(g) and 1 <= len(g.events) <= 4 for g in fresh)
    assert len(again) == 3                                      # booked on old data: booked again


def _table():
    rows = []
    for k in range(5):                                         # one moment of 5 coins, then a lone coin
        rows.append({"event": "DUMP", "history": f"C{k}", "signal_ts": T0, "turnover": 1e7 * (k + 1),
                     "fell24": -0.1 * k, "nervous": True, "btc": False, "weekend": False, "big": True,
                     "daypart": "утро", "perp": k % 2 == 0, "squeeze_chance": 0.1 * k})
    rows.append({"event": "SPIKE", "history": "C9", "signal_ts": T0 + HOUR, "turnover": 1e9, "fell24": 0.2,
                 "nervous": False, "btc": True, "weekend": False, "big": True, "daypart": "утро", "perp": True,
                 "squeeze_chance": 0.5})
    return pd.DataFrame(rows).astype({"nervous": object, "btc": object, "weekend": object, "big": object,
                                      "perp": object})


def test_a_genome_takes_its_events_conditions_moment_and_picks():
    t = _table()
    g = fa.Genome(events=("DUMP",), nervous=True, mass=4, pick="liquid", k=2)
    assert fa.chosen(g, t)["history"].tolist() == ["C3", "C4"]
    g = fa.Genome(events=("DUMP",), mass=4, pick="extreme", k=1)
    assert fa.chosen(g, t)["history"].tolist() == ["C4"]                     # fell the most
    g = fa.normal(fa.Genome(events=("DUMP",), side=-1, pick="all", squeeze=0.25))
    assert fa.chosen(g, t)["history"].tolist() == ["C0", "C2"]              # a live perpetual, a low squeeze
    assert fa.chosen(fa.Genome(events=("DUMP", "SPIKE"), mass=4, pick="all"), t)["history"].nunique() == 5


def _prices(closes, start=T0):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    ts = start + np.arange(len(c), dtype=np.int64) * HOUR
    return np.column_stack([ts, o, np.maximum(o, c), np.minimum(o, c), c])


def test_a_short_gains_what_the_price_loses_pays_fees_and_takes_funding():
    prices = _prices([100, 100, 95, 90, 90])
    funding = np.array([[T0 + 2 * HOUR, 0.001], [T0 + 10 * HOUR, 0.5]])
    rule = ex.Rule("x", "", 4)
    net, held, worst = fa.walk(prices, funding, T0, -1, rule, 0.05, 0.0, 0.0, T0 + 10 * HOUR)
    fee = 0.0005
    assert held == 4 and abs(net - ((1 - fee) - 0.9 * (1 + fee) + 0.001)) < 1e-12
    long_net, _, _ = fa.walk(prices, None, T0, 1, rule, 0.05, 0.0, 0.0, T0 + 10 * HOUR)
    assert long_net < 0 < net


def test_a_short_is_gone_once_the_price_doubled_and_its_stop_works_upwards():
    up = _prices([100, 100, 150, 210, 220])
    assert fa.walk(up, None, T0, -1, ex.Rule("x", "", 4), 0.05, 0.0, 0.0, T0 + 10 * HOUR)[0] == -1.0
    net, held, _ = fa.walk(up, None, T0, -1, ex.Rule("x", "", 4, stop_pct=0.1), 0.05, 0.0, 0.0, T0 + 10 * HOUR)
    assert held == 3 and abs(net - ((1 - 0.0005) - 1.1 * (1 + 0.0005))) < 1e-12   # out at 110


def _stats(growths, drawdowns, n=20):
    return {"learn_n": n * 4, "years": {str(y): {"n": n, "growth_pct": g, "drawdown_pct": d, "up_of10": 6}
                                        for y, g, d in zip(fa.YEARS, growths, drawdowns)}}


def test_goals_by_their_drawdown_and_three_years_of_four_without_a_loss():
    s = _stats([40, -10, 30, 20, 10, 5], [-30, -40, -20, -25, -25, -10])
    assert fa.fitness(s, "razgon") > 0 and fa.fitness(s, "seif") == float("-inf")
    assert fa.holds(s, "razgon") and not fa.holds(s, "seif")
    s = _stats([40, -10, -5, 20, 10, 5], [-10, -10, -10, -10, -10, -10])
    assert fa.fitness(s, "seif") == float("-inf")                               # two losing years
    s = _stats([10, 10, 10, 10, 10, -1], [-5] * 6)
    assert fa.fitness(s, "seif") > 0 and not fa.holds(s, "seif")                 # 2026 lost


def test_the_bar_rises_with_tries():
    assert fa.bar(100) < fa.bar(1000) < fa.bar(10000)


def test_a_genome_in_words():
    g = fa.seeds()[1]
    text = fa.describe(g)
    assert text.startswith("купить после:") and "момент от 4 монет" in text and "стоп −30%" in text
