import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.analyzer_full import analyze_coin_full

def test_analyze_coin_full_basic():
    result = analyze_coin_full("ethereum", 2500, {"amount": 1.0, "price": 2200})
    assert "RSI" in result
    assert "Рекомендация" in result

