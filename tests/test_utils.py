import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils import get_avg_price_and_amount

def test_avg_price_and_amount():
    entries = [
        {"date": "2024-01-01", "amount": 0.5, "price": 2000},
        {"date": "2024-01-15", "amount": 0.5, "price": 3000}
    ]
    avg_price, total_amount = get_avg_price_and_amount(entries)
    assert round(avg_price, 2) == 2500.00
    assert total_amount == 1.0
