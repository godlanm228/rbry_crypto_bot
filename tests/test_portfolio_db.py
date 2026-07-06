"""
Tests for SQLite-based portfolio manager.
"""

import pytest
import tempfile
import os
from pathlib import Path
from core.portfolio_db import PortfolioManager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield Path(path)
    # Cleanup
    try:
        os.unlink(path)
    except:
        pass


def test_portfolio_manager_init(temp_db):
    """Test portfolio manager initialization."""
    pm = PortfolioManager(temp_db)
    assert temp_db.exists()
    
    # Check tables exist
    conn = pm._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert 'users' in tables
    assert 'portfolio' in tables
    assert 'entries' in tables
    assert 'sales' in tables


def test_add_entry(temp_db):
    """Test adding portfolio entries."""
    pm = PortfolioManager(temp_db)
    user_id = 12345
    
    # Add first entry
    result = pm.add_entry(user_id, 'BTC', 0.1, 50000.0)
    assert result['avg_price'] == 50000.0
    assert result['total_amount'] == 0.1
    
    # Add second entry
    result = pm.add_entry(user_id, 'BTC', 0.2, 60000.0)
    assert abs(result['total_amount'] - 0.3) < 0.0001  # Handle floating point precision
    # Average should be (0.1*50000 + 0.2*60000) / 0.3 = 56666.67
    assert abs(result['avg_price'] - 56666.67) < 0.01


def test_load_portfolio(temp_db):
    """Test loading user portfolio."""
    pm = PortfolioManager(temp_db)
    user_id = 12345
    
    # Add entries
    pm.add_entry(user_id, 'BTC', 0.1, 50000.0)
    pm.add_entry(user_id, 'ETH', 2.0, 3000.0)
    
    # Load portfolio
    portfolio = pm.load_portfolio(user_id)
    assert 'BTC' in portfolio
    assert 'ETH' in portfolio
    assert portfolio['BTC']['amount'] == 0.1
    assert portfolio['BTC']['buy_price'] == 50000.0
    assert len(portfolio['BTC']['entries']) == 1
    assert portfolio['ETH']['amount'] == 2.0


def test_sell_partial(temp_db):
    """Test partial sell."""
    pm = PortfolioManager(temp_db)
    user_id = 12345
    
    # Add entry
    pm.add_entry(user_id, 'BTC', 1.0, 50000.0)
    
    # Sell half
    result = pm.sell(user_id, 'BTC', 0.5, 60000.0)
    assert result['pnl'] == (60000.0 - 50000.0) * 0.5  # 5000.0
    assert result['remaining_amount'] == 0.5
    assert result['was_full_sale'] is False
    
    # Check portfolio
    portfolio = pm.load_portfolio(user_id)
    assert portfolio['BTC']['amount'] == 0.5


def test_sell_full(temp_db):
    """Test full sell."""
    pm = PortfolioManager(temp_db)
    user_id = 12345
    
    # Add entry
    pm.add_entry(user_id, 'BTC', 1.0, 50000.0)
    
    # Sell all
    result = pm.sell(user_id, 'BTC', 1.0, 60000.0)
    assert result['pnl'] == 10000.0
    assert result['remaining_amount'] == 0.0
    assert result['was_full_sale'] is True
    
    # Check portfolio - coin should be gone
    portfolio = pm.load_portfolio(user_id)
    assert 'BTC' not in portfolio


def test_sell_too_much(temp_db):
    """Test selling more than available."""
    pm = PortfolioManager(temp_db)
    user_id = 12345
    
    # Add entry
    pm.add_entry(user_id, 'BTC', 1.0, 50000.0)
    
    # Try to sell more than available
    with pytest.raises(ValueError):
        pm.sell(user_id, 'BTC', 2.0, 60000.0)


def test_get_all_portfolio_coins(temp_db):
    """Test getting all coins across users."""
    pm = PortfolioManager(temp_db)
    
    # Add entries for different users
    pm.add_entry(12345, 'BTC', 0.1, 50000.0)
    pm.add_entry(12345, 'ETH', 2.0, 3000.0)
    pm.add_entry(67890, 'SOL', 10.0, 100.0)
    pm.add_entry(67890, 'BTC', 0.5, 55000.0)
    
    # Get all coins
    coins = pm.get_all_portfolio_coins()
    assert set(coins) == {'BTC', 'ETH', 'SOL'}


def test_get_eligible_portfolio_coins(temp_db):
    """Test getting coins from eligible users only."""
    pm = PortfolioManager(temp_db)
    
    # Add entries for different users
    pm.add_entry(12345, 'BTC', 0.1, 50000.0)
    pm.add_entry(12345, 'ETH', 2.0, 3000.0)
    pm.add_entry(67890, 'SOL', 10.0, 100.0)
    pm.add_entry(99999, 'DOGE', 1000.0, 0.1)
    
    # Get coins from specific users
    coins = pm.get_eligible_portfolio_coins([12345, 67890])
    assert set(coins) == {'BTC', 'ETH', 'SOL'}
    
    # DOGE should not be included
    assert 'DOGE' not in coins


def test_get_all_users(temp_db):
    """Test getting all users with portfolios."""
    pm = PortfolioManager(temp_db)
    
    # Add entries for different users
    pm.add_entry(12345, 'BTC', 0.1, 50000.0)
    pm.add_entry(67890, 'ETH', 2.0, 3000.0)
    pm.add_entry(99999, 'SOL', 10.0, 100.0)
    
    users = pm.get_all_users()
    assert set(users) == {12345, 67890, 99999}


def test_get_pnl_history(temp_db):
    """Test getting PnL history."""
    pm = PortfolioManager(temp_db)
    user_id = 12345
    
    # Add and sell entries
    pm.add_entry(user_id, 'BTC', 1.0, 50000.0)
    pm.sell(user_id, 'BTC', 0.5, 60000.0)
    
    pm.add_entry(user_id, 'ETH', 2.0, 3000.0)
    pm.sell(user_id, 'ETH', 1.0, 3500.0)
    
    # Get history
    history = pm.get_pnl_history(user_id)
    assert len(history) == 2
    
    # Most recent first
    assert history[0]['coin'] == 'ETH'
    assert history[0]['pnl'] == 500.0
    assert history[1]['coin'] == 'BTC'
    assert history[1]['pnl'] == 5000.0


def test_get_total_pnl(temp_db):
    """Test getting total realized PnL."""
    pm = PortfolioManager(temp_db)
    user_id = 12345
    
    # Add and sell entries
    pm.add_entry(user_id, 'BTC', 1.0, 50000.0)
    pm.sell(user_id, 'BTC', 0.5, 60000.0)  # PnL: +5000
    
    pm.add_entry(user_id, 'ETH', 2.0, 3000.0)
    pm.sell(user_id, 'ETH', 1.0, 2500.0)  # PnL: -500
    
    # Total PnL should be 4500
    total_pnl = pm.get_total_pnl(user_id)
    assert total_pnl == 4500.0


def test_multi_user_isolation(temp_db):
    """Test that users' portfolios are isolated."""
    pm = PortfolioManager(temp_db)
    
    # User 1
    pm.add_entry(12345, 'BTC', 0.1, 50000.0)
    
    # User 2
    pm.add_entry(67890, 'BTC', 0.5, 55000.0)
    
    # Check isolation
    portfolio1 = pm.load_portfolio(12345)
    portfolio2 = pm.load_portfolio(67890)
    
    assert portfolio1['BTC']['amount'] == 0.1
    assert portfolio1['BTC']['buy_price'] == 50000.0
    
    assert portfolio2['BTC']['amount'] == 0.5
    assert portfolio2['BTC']['buy_price'] == 55000.0


def test_rebuy_after_full_sale(temp_db):
    """Buying again after a full sale must NOT resurrect sold entries in the average."""
    pm = PortfolioManager(temp_db)
    user_id = 12345

    pm.add_entry(user_id, 'BTC', 1.0, 100.0)
    pm.sell(user_id, 'BTC', 1.0, 200.0)

    # Re-buy at a new price: position must be exactly 1.0 @ 300, not 2.0 @ 200
    result = pm.add_entry(user_id, 'BTC', 1.0, 300.0)
    assert abs(result['total_amount'] - 1.0) < 1e-9
    assert abs(result['avg_price'] - 300.0) < 1e-9


def test_add_after_partial_sale(temp_db):
    """Average after a partial sale must be based on remaining position, not full history."""
    pm = PortfolioManager(temp_db)
    user_id = 12345

    pm.add_entry(user_id, 'BTC', 2.0, 100.0)
    pm.sell(user_id, 'BTC', 1.0, 150.0)          # remaining: 1.0 @ 100
    result = pm.add_entry(user_id, 'BTC', 1.0, 200.0)

    assert abs(result['total_amount'] - 2.0) < 1e-9
    assert abs(result['avg_price'] - 150.0) < 1e-9   # (1*100 + 1*200) / 2


def test_sell_full_position_with_float_dust(temp_db):
    """Selling the entire position accumulated from odd floats must close it cleanly."""
    pm = PortfolioManager(temp_db)
    user_id = 12345

    pm.add_entry(user_id, 'ETH', 0.1, 3000.0)
    pm.add_entry(user_id, 'ETH', 0.2, 3000.0)
    total = pm.load_portfolio(user_id)['ETH']['amount']

    result = pm.sell(user_id, 'ETH', total, 3500.0)
    assert result['was_full_sale'] is True
    assert 'ETH' not in pm.load_portfolio(user_id)


def test_remove_coin(temp_db):
    """Test removing a coin from portfolio."""
    pm = PortfolioManager(temp_db)
    user_id = 12345
    
    # Add entries
    pm.add_entry(user_id, 'BTC', 0.1, 50000.0)
    pm.add_entry(user_id, 'ETH', 2.0, 3000.0)
    
    # Remove BTC
    result = pm.remove_coin(user_id, 'BTC')
    assert result is True
    
    # Check portfolio
    portfolio = pm.load_portfolio(user_id)
    assert 'BTC' not in portfolio
    assert 'ETH' in portfolio
    
    # Try to remove non-existent coin
    result = pm.remove_coin(user_id, 'SOL')
    assert result is False
