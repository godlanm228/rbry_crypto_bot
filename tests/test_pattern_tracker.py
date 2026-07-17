import json
import time

from core.analysis.pattern_tracker import PatternTracker


def test_empty_pattern_history_never_invents_winrate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = PatternTracker()

    stats = tracker.get_pattern_stats("breakout")

    assert stats["resolved"] == 0
    assert stats["winrate"] is None


def test_pending_signals_are_not_counted_as_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = PatternTracker()
    tracker.save_signal("bitcoin", "breakout", "bullish", 120.0, 100.0)

    tracker.check_results({"bitcoin": 105.0})
    stats = tracker.get_pattern_stats("breakout")

    assert stats["total"] == 1
    assert stats["resolved"] == 0
    assert stats["winrate"] is None


def test_favorable_move_below_target_is_not_marked_as_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = PatternTracker()
    tracker.save_signal("bitcoin", "breakout", "bullish", 150.0, 100.0)

    tracker.check_results({"bitcoin": 115.0})

    assert tracker.history["signals"][0]["result"] == "pending"


def test_timeout_is_a_resolved_non_hit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = PatternTracker()
    tracker.save_signal("bitcoin", "breakout", "bullish", 120.0, 100.0)
    tracker.history["signals"][0]["timestamp"] = time.time() - 8 * 86400

    tracker.check_results({"bitcoin": 105.0})
    stats = tracker.get_pattern_stats("breakout")

    assert stats["resolved"] == 1
    assert stats["timeouts"] == 1
    assert stats["winrate"] == 0.0
    with open(tmp_path / "data" / "pattern_history.json", encoding="utf-8") as handle:
        assert json.load(handle)["signals"][0]["result"] == "timeout"
