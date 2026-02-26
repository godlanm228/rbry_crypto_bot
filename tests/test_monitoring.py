import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.monitoring import Monitor
from unittest.mock import MagicMock
import time

def test_monitor_should_alert():
    m = Monitor(["eth"], {}, MagicMock())
    coin = "eth"
    assert m.should_alert(coin) is True
    assert m.should_alert(coin) is False
    time.sleep(1)
    m.last_alert_time[coin] -= 9999
    assert m.should_alert(coin) is True
