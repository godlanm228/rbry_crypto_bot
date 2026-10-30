"""The paper account screen of the owner's test bot: only for the owner with the flag, read from the watcher's state."""

import json

from modules import paper_view as pv


def _write(tmp_path):
    pos = {"base": "SOL", "side": -1, "source": "модель выбора", "size": 20.0, "entry": 150.0, "net_pct": 3.2,
           "pnl_usd": 0.64, "half_sold": False, "status": "open", "due": "02.11 15:00", "exit": None,
           "stop": 165.0, "half": 142.5}
    done = {**pos, "base": "DOGE", "side": 1, "status": "breakeven", "net_pct": 2.4, "pnl_usd": 0.48,
            "exit": "31.10 09:00", "source": "паника"}
    book = {"label": "Счёт Разгон", "value": 101.12, "cash": 80.0, "open": [pos], "last": [done], "closed": 1,
            "up_of10": 10.0, "usd_per_100": 2.4, "trades": 2}
    state = {"updated": "31.10 10:01", "regime": "down|nervous", "account": {"opened": "30.10.2026 17:20"},
             "books": {"account": book, "nostop": {**book, "label": "паника без стопа", "value": 99.5}}}
    (tmp_path / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "equity.jsonl").write_text("\n".join(json.dumps({"books": {"account": v}})
                                                     for v in (100, 99, 101.12)), encoding="utf-8")


def test_only_the_owner_with_the_flag_sees_it(monkeypatch):
    monkeypatch.setenv("RBRY_PAPER_VIEW", "1")
    assert pv.enabled(7, 7) and not pv.enabled(8, 7)
    monkeypatch.delenv("RBRY_PAPER_VIEW")
    assert not pv.enabled(7, 7)


def test_the_account_screen_shows_balance_positions_levels_and_trades(tmp_path, monkeypatch):
    _write(tmp_path)
    monkeypatch.setattr(pv, "WATCH_DIR", tmp_path)
    text, _ = pv.account_screen()
    assert "$101.12" in text and "падение · нервно" in text and "▁" in text
    assert "SOL</b> шорт $20.00" in text and "стоп $165.00" in text and "половина на $142.50" in text
    assert "✅ DOGE лонг +2.4%" in text and "без убытка" in text
    text, _ = pv.all_screen()
    assert "Счёт Разгон (общий)" in text and "паника без стопа" in text


def test_before_the_first_hour_it_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "WATCH_DIR", tmp_path)
    assert "ещё не открыт" in pv.account_screen()[0]


def test_the_brains_verdict_under_a_coins_analysis(tmp_path, monkeypatch):
    _write(tmp_path)
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    state["market"] = {"breadth_state": "narrow", "alt_state": "btc", "fund_state": "cold"}
    (tmp_path / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "seen.jsonl").write_text(json.dumps({"hour": 1793300000000 - 3600000, "coins": [
        {"base": "SOL", "events": ["DUMP", "OI-FLUSH"]}]}) + "\n", encoding="utf-8")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "decisions.jsonl").write_text(json.dumps({"places": [["SOL", "SOL", 0.9], ["BTC", "BTC", 0.5],
                                                                  ["DOGE", "DOGE", 0.1]]}) + "\n", encoding="utf-8")
    monkeypatch.setattr(pv, "WATCH_DIR", tmp_path)
    monkeypatch.setattr(pv, "PAPER_DIR", paper)
    text = pv.brain_block("SOLUSDT")
    assert "падение · нервно" in text and "узкий рынок" in text and "фандинг холодный" in text
    assert "1 (в верхней трети)" in text and "волна ликвидаций" in text and "шорт $20" in text
    assert "Сигналов по ней сейчас нет" in pv.brain_block("XRP")
