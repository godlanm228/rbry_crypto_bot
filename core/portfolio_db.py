"""
SQLite-based multi-user portfolio manager.
Replaces the single shared portfolio.json with per-user database storage.
"""

import sqlite3
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

DB_PATH = Path("data/portfolio.db")

class PortfolioManager:
    """Thread-safe SQLite-based portfolio manager for multi-user support."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """Initialize database schema."""
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Portfolio table - current holdings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                coin TEXT NOT NULL,
                amount REAL NOT NULL,
                buy_price REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, coin)
            )
        """)
        
        # Entries table - purchase history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                coin TEXT NOT NULL,
                amount REAL NOT NULL,
                price REAL NOT NULL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Sales table - sell history with PnL
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                coin TEXT NOT NULL,
                amount REAL NOT NULL,
                sell_price REAL NOT NULL,
                avg_buy_price REAL NOT NULL,
                pnl REAL NOT NULL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Coin registry table - CoinGecko mapping cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coin_registry (
                symbol TEXT PRIMARY KEY,
                coingecko_id TEXT NOT NULL,
                binance_symbol TEXT,
                name TEXT,
                updated_at TIMESTAMP
            )
        """)
        
        conn.commit()
        logger.info(f"Portfolio database initialized at {self.db_path}")
    
    def _ensure_user(self, user_id: int):
        """Ensure user exists in users table."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
        conn.commit()
    
    def load_portfolio(self, user_id: int) -> dict:
        """
        Load user portfolio in old format for backward compatibility.
        Returns: {COIN: {amount, buy_price, entries}}
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Get current holdings
        cursor.execute("""
            SELECT coin, amount, buy_price 
            FROM portfolio 
            WHERE user_id = ?
        """, (user_id,))
        
        portfolio = {}
        for row in cursor.fetchall():
            coin = row['coin']
            
            # Get entries for this coin
            cursor.execute("""
                SELECT amount, price, date 
                FROM entries 
                WHERE user_id = ? AND coin = ?
                ORDER BY date
            """, (user_id, coin))
            
            entries = []
            for entry_row in cursor.fetchall():
                entries.append({
                    "amount": entry_row['amount'],
                    "price": entry_row['price'],
                    "date": entry_row['date']
                })
            
            portfolio[coin] = {
                "amount": row['amount'],
                "buy_price": row['buy_price'],
                "entries": entries
            }
        
        return portfolio
    
    def add_entry(self, user_id: int, coin: str, amount: float, price: float) -> dict:
        """
        Add a purchase entry and update portfolio.
        Returns: {avg_price, total_amount}
        """
        self._ensure_user(user_id)
        conn = self._get_conn()
        cursor = conn.cursor()
        
        coin = coin.upper()
        
        # Add to entries table
        cursor.execute("""
            INSERT INTO entries (user_id, coin, amount, price, date)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, coin, amount, price, datetime.utcnow().isoformat() + "Z"))
        
        # Get all entries for this coin to calculate new average
        cursor.execute("""
            SELECT amount, price FROM entries 
            WHERE user_id = ? AND coin = ?
        """, (user_id, coin))
        
        entries = cursor.fetchall()
        total_cost = sum(e['amount'] * e['price'] for e in entries)
        total_amount = sum(e['amount'] for e in entries)
        avg_price = total_cost / total_amount if total_amount > 0 else 0
        
        # Update or insert into portfolio table
        cursor.execute("""
            INSERT INTO portfolio (user_id, coin, amount, buy_price)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, coin) 
            DO UPDATE SET amount = ?, buy_price = ?
        """, (user_id, coin, total_amount, avg_price, total_amount, avg_price))
        
        conn.commit()
        
        return {
            "avg_price": avg_price,
            "total_amount": total_amount
        }
    
    def sell(self, user_id: int, coin: str, amount: float, sell_price: float) -> dict:
        """
        Sell partial or full position with PnL tracking.
        Returns: {pnl, remaining_amount, was_full_sale}
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        coin = coin.upper()
        
        # Get current position
        cursor.execute("""
            SELECT amount, buy_price FROM portfolio 
            WHERE user_id = ? AND coin = ?
        """, (user_id, coin))
        
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Coin {coin} not found in portfolio")
        
        current_amount = row['amount']
        avg_buy_price = row['buy_price']
        
        if amount > current_amount:
            raise ValueError(f"Cannot sell {amount} {coin}, only {current_amount} available")
        
        # Calculate PnL
        pnl = (sell_price - avg_buy_price) * amount
        remaining = current_amount - amount
        
        # Record sale
        cursor.execute("""
            INSERT INTO sales (user_id, coin, amount, sell_price, avg_buy_price, pnl, date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, coin, amount, sell_price, avg_buy_price, pnl, datetime.utcnow().isoformat() + "Z"))
        
        # Update portfolio
        if remaining > 0:
            cursor.execute("""
                UPDATE portfolio 
                SET amount = ? 
                WHERE user_id = ? AND coin = ?
            """, (remaining, user_id, coin))
        else:
            # Full sale - remove from portfolio
            cursor.execute("""
                DELETE FROM portfolio 
                WHERE user_id = ? AND coin = ?
            """, (user_id, coin))
        
        conn.commit()
        
        return {
            "pnl": pnl,
            "remaining_amount": remaining,
            "was_full_sale": remaining == 0
        }
    
    def remove_coin(self, user_id: int, coin: str) -> bool:
        """
        Remove coin from portfolio entirely (no PnL tracking).
        Returns: True if coin was removed, False if not found.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        coin = coin.upper()
        
        cursor.execute("""
            DELETE FROM portfolio 
            WHERE user_id = ? AND coin = ?
        """, (user_id, coin))
        
        conn.commit()
        return cursor.rowcount > 0
    
    def get_all_portfolio_coins(self) -> List[str]:
        """Get all unique coins across all users for snapshot optimization."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT coin FROM portfolio")
        return [row['coin'] for row in cursor.fetchall()]
    
    def get_eligible_portfolio_coins(self, eligible_user_ids: List[int]) -> List[str]:
        """
        Get coins only from premium/beta/owner users for snapshot optimization.
        """
        if not eligible_user_ids:
            return []
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(eligible_user_ids))
        cursor.execute(f"""
            SELECT DISTINCT coin FROM portfolio 
            WHERE user_id IN ({placeholders})
        """, eligible_user_ids)
        
        return [row['coin'] for row in cursor.fetchall()]
    
    def get_all_users(self) -> List[int]:
        """Get all user IDs with portfolios."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT user_id FROM portfolio")
        return [row['user_id'] for row in cursor.fetchall()]
    
    def get_pnl_history(self, user_id: int) -> List[dict]:
        """Get realized PnL history for user."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT coin, amount, sell_price, avg_buy_price, pnl, date 
            FROM sales 
            WHERE user_id = ?
            ORDER BY date DESC
        """, (user_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_total_pnl(self, user_id: int) -> float:
        """Get total realized PnL for user."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COALESCE(SUM(pnl), 0) as total_pnl 
            FROM sales 
            WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        return row['total_pnl'] if row else 0.0
    
    def migrate_from_json(self, json_path: str, owner_id: int):
        """
        Best-effort migration from old portfolio.json to SQLite for owner.
        This is optional since current data is test data.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                old_portfolio = json.load(f)
            
            logger.info(f"Migrating {len(old_portfolio)} coins from {json_path} to user {owner_id}")
            
            for coin, data in old_portfolio.items():
                amount = data.get('amount', 0)
                buy_price = data.get('buy_price', data.get('price', 0))
                entries = data.get('entries', [])
                
                # If no entries, create one synthetic entry
                if not entries:
                    entries = [{
                        'amount': amount,
                        'price': buy_price,
                        'date': datetime.utcnow().isoformat() + "Z"
                    }]
                
                # Add each entry
                for entry in entries:
                    entry_amount = entry.get('amount', 0)
                    entry_price = entry.get('price', buy_price)
                    if entry_amount > 0:
                        self.add_entry(owner_id, coin, entry_amount, entry_price)
            
            logger.info(f"Migration completed successfully")
            
        except FileNotFoundError:
            logger.info(f"No portfolio.json found at {json_path}, skipping migration")
        except Exception as e:
            logger.error(f"Migration error: {e}", exc_info=True)


# Singleton instance
_portfolio_manager: Optional[PortfolioManager] = None
_manager_lock = threading.Lock()


def get_portfolio_manager() -> PortfolioManager:
    """Get singleton PortfolioManager instance."""
    global _portfolio_manager
    if _portfolio_manager is None:
        with _manager_lock:
            if _portfolio_manager is None:
                _portfolio_manager = PortfolioManager()
    return _portfolio_manager
