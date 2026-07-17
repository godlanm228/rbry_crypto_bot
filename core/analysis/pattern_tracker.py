import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PatternTracker:
    def __init__(self):
        self.history_file = Path("data/pattern_history.json")
        self.history_file.parent.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("signals"), list):
                    data.setdefault("stats", {})
                    return data
            except (OSError, json.JSONDecodeError):
                logger.exception("Pattern history is unreadable; starting with empty history")
        return {"signals": [], "stats": {}}

    def save_signal(self, coin_id: str, pattern: str, prediction: str, target_price: float, current_price: float):
        """Save pattern signal for tracking"""
        signal = {
            "coin_id": coin_id,
            "pattern": pattern,
            "prediction": prediction,
            "target_price": target_price,
            "entry_price": current_price,
            "timestamp": time.time(),
            "result": "pending"
        }
        with self._lock:
            self.history["signals"].append(signal)
            self._save_history()
        return signal

    def check_results(self, current_prices: Dict[str, float]):
        """Check pending signals against current prices"""
        with self._lock:
            for signal in self.history["signals"]:
                if signal["result"] != "pending":
                    continue

                price = current_prices.get(signal["coin_id"])
                if not price:
                    continue

                # Check if target hit or the move went 10% against the signal.
                profit_pct = (price - signal["entry_price"]) / signal["entry_price"] * 100

                if signal["prediction"] == "bullish" and price >= signal["target_price"]:
                    signal["result"] = "win"
                    signal["profit_pct"] = profit_pct
                elif signal["prediction"] == "bearish" and price <= signal["target_price"]:
                    signal["result"] = "win"
                    signal["profit_pct"] = abs(profit_pct)
                elif signal["prediction"] == "bullish" and profit_pct <= -10:
                    signal["result"] = "loss"
                    signal["profit_pct"] = profit_pct
                elif signal["prediction"] == "bearish" and profit_pct >= 10:
                    signal["result"] = "loss"
                    signal["profit_pct"] = -profit_pct
                elif time.time() - signal["timestamp"] > 7 * 86400:
                    signal["result"] = "timeout"
                    signal["profit_pct"] = profit_pct

            self._update_stats()
            self._save_history()

    def _update_stats(self):
        """Calculate pattern effectiveness stats"""
        stats = {}
        for signal in self.history["signals"]:
            pattern = signal["pattern"]
            if pattern not in stats:
                stats[pattern] = {
                    "total": 0,
                    "resolved": 0,
                    "wins": 0,
                    "losses": 0,
                    "timeouts": 0,
                    "winrate": None,
                }

            stats[pattern]["total"] += 1
            if signal["result"] == "win":
                stats[pattern]["wins"] += 1
                stats[pattern]["resolved"] += 1
            elif signal["result"] == "loss":
                stats[pattern]["losses"] += 1
                stats[pattern]["resolved"] += 1
            elif signal["result"] == "timeout":
                stats[pattern]["timeouts"] += 1
                stats[pattern]["resolved"] += 1

            if stats[pattern]["resolved"] > 0:
                stats[pattern]["winrate"] = (
                    stats[pattern]["wins"] / stats[pattern]["resolved"] * 100
                )

        self.history["stats"] = stats

    def get_pattern_stats(self, pattern: str) -> Dict:
        """Get effectiveness stats for specific pattern"""
        return self.history["stats"].get(
            pattern,
            {"winrate": None, "total": 0, "resolved": 0, "wins": 0, "losses": 0, "timeouts": 0},
        )

    def _save_history(self):
        temporary = self.history_file.with_suffix(self.history_file.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, self.history_file)

pattern_tracker = PatternTracker()
