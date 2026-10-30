"""The allocator: history first, paper more with every trade, the positive share, the best of the rest explores."""

from core.brain import allocator as al


def test_without_paper_trades_history_decides_and_the_unproven_explores():
    out = al.allocate([al.Source("паника", 0.02), al.Source("модель", 0.0)])
    assert out == [("паника", 4, 0.02), ("модель", 1, 0.0)]


def test_paper_trades_move_the_places_and_twenty_weigh_as_much_as_history():
    s = al.Source("паника", 0.02, [-0.01] * 20)
    assert abs(al.worth(s) - 0.005) < 1e-12
    losing = al.Source("паника", 0.02, [-0.05] * 40)
    out = dict((l, n) for l, n, _ in al.allocate([losing, al.Source("модель", 0.0, [0.03] * 30)]))
    assert out == {"модель": 4, "паника": 1}                          # the selector earned, the panic explores


def test_the_positive_share_in_proportion_and_nothing_positive_trades_one_place():
    out = dict((l, n) for l, n, _ in al.allocate([al.Source("a", 0.03), al.Source("b", 0.01), al.Source("c", -0.01)]))
    assert out == {"a": 3, "b": 1, "c": 1}
    out = dict((l, n) for l, n, _ in al.allocate([al.Source("a", -0.01), al.Source("b", -0.02)]))
    assert out == {"a": 1, "b": 0}


def test_an_unproven_strategy_explores_until_its_own_trades_vouch_for_it():
    hopeful = al.Source("модель", 0.009, proven=False)
    out = dict((l, n) for l, n, _ in al.allocate([al.Source("паника", 0.02), hopeful]))
    assert out == {"паника": 4, "модель": 1}
    earned = al.Source("модель", 0.009, [0.03] * 25, proven=False)
    out = dict((l, n) for l, n, _ in al.allocate([al.Source("паника", 0.02), earned]))
    assert out["модель"] >= 2
