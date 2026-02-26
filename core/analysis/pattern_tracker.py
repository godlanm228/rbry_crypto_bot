import json
import time
from pathlib import Path
from typing import Dict, List, Optional

class PatternTracker:
    def __init__(self):
        self.history_file = Path("data/pattern_history.json")
        self.history_file.parent.mkdir(exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
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
        self.history["signals"].append(signal)
        self._save_history()
        return signal

    def check_results(self, current_prices: Dict[str, float]):
        """Check pending signals against current prices"""
        for signal in self.history["signals"]:
            if signal["result"] != "pending":
                continue

            price = current_prices.get(signal["coin_id"])
            if not price:
                continue

            # Check if target hit or stop loss
            profit_pct = (price - signal["entry_price"]) / signal["entry_price"] * 100

            if signal["prediction"] == "bullish" and price >= signal["target_price"]:
                signal["result"] = "win"
                signal["profit_pct"] = profit_pct
            elif signal["prediction"] == "bearish" and price <= signal["target_price"]:
                signal["result"] = "win"
                signal["profit_pct"] = abs(profit_pct)
            elif abs(profit_pct) > 10:  # Stop loss at 10%
                signal["result"] = "loss"
                signal["profit_pct"] = profit_pct
            elif time.time() - signal["timestamp"] > 7 * 86400:  # 7 days timeout
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
                stats[pattern] = {"total": 0, "wins": 0, "losses": 0, "winrate": 0}

            stats[pattern]["total"] += 1
            if signal["result"] == "win":
                stats[pattern]["wins"] += 1
            elif signal["result"] == "loss":
                stats[pattern]["losses"] += 1

            if stats[pattern]["total"] > 0:
                stats[pattern]["winrate"] = stats[pattern]["wins"] / stats[pattern]["total"] * 100

        self.history["stats"] = stats

    def get_pattern_stats(self, pattern: str) -> Dict:
        """Get effectiveness stats for specific pattern"""
        return self.history["stats"].get(pattern, {"winrate": 50, "total": 0})

    def _save_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

pattern_tracker = PatternTracker()