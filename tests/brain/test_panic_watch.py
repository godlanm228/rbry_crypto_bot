"""The paper watcher's script: one price path a symbol, the account's places through time."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HOUR = 3_600_000
T0 = int(pd.Timestamp("2026-11-02 13:00").value // 10**6)


@pytest.fixture
def pw(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts" / "panic_watch.py"
    spec = importlib.util.spec_from_file_location("panic_watch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PLACES", tmp_path / "places.jsonl")
    return module


class _Store:
    def __init__(self, frame):
        self.frame, self.calls = frame, []

    def klines(self, symbol, tf, start_ts=0):
        self.calls.append((symbol, start_ts))
        return self.frame[self.frame["ts"] >= start_ts].reset_index(drop=True)


def test_one_path_a_symbol_sliced_for_every_moment(pw):
    ts = T0 + np.arange(10) * HOUR
    frame = pd.DataFrame({"ts": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": np.arange(10.0)})
    store = _Store(frame)
    prices = pw.prices_fn(store, int(ts[-1]), live=False)
    a = prices("BTCUSDT", int(ts[4]))
    b = prices("BTCUSDT", int(ts[6]))
    c = prices("BTCUSDT", int(ts[2]))                 # earlier than the path held: read again from there
    d = prices("BTCUSDT", int(ts[8]))
    assert [int(x[0, 0]) for x in (a, b, c, d)] == [ts[4], ts[6], ts[2], ts[8]]
    assert len(a) == 6 and len(b) == 4 and len(c) == 8 and len(d) == 2
    assert store.calls == [("BTCUSDT", int(ts[4])), ("BTCUSDT", int(ts[2]))]


def test_the_places_history_keeps_order_changes_and_rewrites_a_lost_opening(pw):
    account = {"start_ms": T0 - 100 * HOUR}
    places = [("паника", 5, 0.02), ("модель выбора", 3, 0.019)]
    history = pw.places_history(account, places, T0)
    assert [h["from"] for h in history] == [T0 - 100 * HOUR, T0] and history[1]["order"] == ["паника", "модель выбора"]
    assert pw.places_history(account, places, T0 + HOUR) == history                 # nothing changed: no entry
    flipped = [("модель выбора", 3, 0.03), ("паника", 5, 0.02)]
    assert pw.places_history(account, flipped, T0 + 2 * HOUR)[-1]["order"] == ["модель выбора", "паника"]
    pw.PLACES.write_text('{"from": 1, "pla', encoding="utf-8")                       # only a torn line left
    again = pw.places_history(account, places, T0 + 3 * HOUR)
    assert [h["from"] for h in pw.watch.load(pw.PLACES)] == [T0 - 100 * HOUR, T0 + 3 * HOUR]
    assert again[0]["places"] == pw.FIRST_PLACES
