"""The urgent watcher (scripts/urgent_watch.py) as a module for tests: no state file, no Etherscan key, no
network for prices; what it sends goes into a list."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def watch(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts" / "urgent_watch.py"
    spec = importlib.util.spec_from_file_location("urgent_watch", path)
    uw = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = uw
    spec.loader.exec_module(uw)
    monkeypatch.setattr(uw, "load_state", lambda: {})
    monkeypatch.setattr(uw, "etherscan_key", lambda: "")
    monkeypatch.setattr(uw, "price_gap", lambda client: "")

    def make(sent: list):
        w = uw.Watch(store=None, send=lambda text: sent.append(text) or True, translate=None, summarize=None,
                     name=lambda title: None)
        w.first, w.state["risk_seeded"] = False, True
        w._market = lambda client, story, held: ""
        return w
    return make
