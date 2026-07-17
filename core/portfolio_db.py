"""
SQLite-based multi-user portfolio manager.
Replaces the single shared portfolio.json with per-user database storage.
"""

import sqlite3
import logging
import math
import threading
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timezone
import json


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn.execute("PRAGMA busy_timeout=10000")
        return self._local.conn

    def close(self):
        """Close the connection owned by the current thread."""
        connection = getattr(self._local, "conn", None)
        if connection is not None:
            connection.close()
            self._local.conn = None

    @staticmethod
    def _normalize_coin(coin: str) -> str:
        normalized = str(coin).strip().upper()
        if not normalized:
            raise ValueError("Coin symbol cannot be empty")
        return normalized

    @staticmethod
    def _positive_number(value: float, field: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a number") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{field} must be greater than zero")
        return number
    
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
                amount REAL NOT NULL CHECK(amount > 0),
                buy_price REAL NOT NULL CHECK(buy_price > 0),
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
                amount REAL NOT NULL CHECK(amount > 0),
                price REAL NOT NULL CHECK(price > 0),
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
                amount REAL NOT NULL CHECK(amount > 0),
                sell_price REAL NOT NULL CHECK(sell_price > 0),
                avg_buy_price REAL NOT NULL CHECK(avg_buy_price > 0),
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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_migrations (
                source TEXT PRIMARY KEY,
                migrated_at TIMESTAMP NOT NULL,
                details TEXT
            )
        """)
        
        conn.commit()
        logger.info(f"Portfolio database initialized at {self.db_path}")
    
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
        coin = self._normalize_coin(coin)
        amount = self._positive_number(amount, "Amount")
        price = self._positive_number(price, "Price")

        conn = self._get_conn()
        cursor = conn.cursor()

        # The history entry and current holding must succeed or fail together.
        # The first statement is a write, so concurrent writers cannot both read
        # the same stale portfolio row before updating it.
        with conn:
            avg_price, total_amount = self._add_entry_in_transaction(
                cursor, user_id, coin, amount, price
            )
        
        return {
            "avg_price": avg_price,
            "total_amount": total_amount
        }

    def _add_entry_in_transaction(
        self,
        cursor: sqlite3.Cursor,
        user_id: int,
        coin: str,
        amount: float,
        price: float,
    ) -> tuple[float, float]:
        """Add an entry using the caller's transaction."""
        cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
        cursor.execute("""
            INSERT INTO entries (user_id, coin, amount, price, date)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, coin, amount, price, _utc_now_iso()))
        cursor.execute("""
            SELECT amount, buy_price FROM portfolio
            WHERE user_id = ? AND coin = ?
        """, (user_id, coin))
        row = cursor.fetchone()

        if row:
            current_amount = float(row["amount"])
            current_average = float(row["buy_price"])
            total_amount = current_amount + amount
            average_price = (
                (current_amount * current_average) + (amount * price)
            ) / total_amount
        else:
            total_amount = amount
            average_price = price

        cursor.execute("""
            INSERT INTO portfolio (user_id, coin, amount, buy_price)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, coin)
            DO UPDATE SET amount = excluded.amount, buy_price = excluded.buy_price
        """, (user_id, coin, total_amount, average_price))
        return average_price, total_amount
    
    def sell(self, user_id: int, coin: str, amount: float, sell_price: float) -> dict:
        """
        Sell partial or full position with PnL tracking.
        Returns: {pnl, remaining_amount, was_full_sale}
        """
        coin = self._normalize_coin(coin)
        amount = self._positive_number(amount, "Amount")
        sell_price = self._positive_number(sell_price, "Sell price")

        conn = self._get_conn()
        cursor = conn.cursor()

        # Acquire the write lock before reading the position. This prevents two
        # simultaneous sells from both validating against the same balance.
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor.execute("""
                SELECT amount, buy_price FROM portfolio
                WHERE user_id = ? AND coin = ?
            """, (user_id, coin))

            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Coin {coin} not found in portfolio")

            current_amount = row['amount']
            avg_buy_price = row['buy_price']

            if amount > current_amount * (1 + 1e-9):
                raise ValueError(f"Cannot sell {amount} {coin}, only {current_amount} available")

            pnl = (sell_price - avg_buy_price) * amount
            remaining = current_amount - amount
            if remaining < 1e-9:
                remaining = 0.0

            cursor.execute("""
                INSERT INTO sales (user_id, coin, amount, sell_price, avg_buy_price, pnl, date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, coin, amount, sell_price, avg_buy_price, pnl, _utc_now_iso()))

            if remaining > 0:
                cursor.execute("""
                    UPDATE portfolio
                    SET amount = ?
                    WHERE user_id = ? AND coin = ?
                """, (remaining, user_id, coin))
            else:
                cursor.execute("""
                    DELETE FROM portfolio
                    WHERE user_id = ? AND coin = ?
                """, (user_id, coin))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        
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
        
        coin = self._normalize_coin(coin)

        with conn:
            cursor.execute("""
                DELETE FROM portfolio
                WHERE user_id = ? AND coin = ?
            """, (user_id, coin))
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
            ORDER BY date DESC, id DESC
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
    
    def migrate_from_json(self, json_path: str, owner_id: int) -> bool:
        """
        Best-effort migration from old portfolio.json to SQLite for owner.
        This is optional since current data is test data.
        """
        source = f"legacy_portfolio_json:{int(owner_id)}"
        path = Path(json_path)
        if not path.exists():
            logger.info(f"No portfolio.json found at {json_path}, skipping migration")
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                old_portfolio = json.load(f)
            if not isinstance(old_portfolio, dict):
                raise ValueError("legacy portfolio root must be an object")

            conn = self._get_conn()
            cursor = conn.cursor()
            conn.execute("BEGIN IMMEDIATE")
            cursor.execute(
                "SELECT 1 FROM portfolio_migrations WHERE source = ?",
                (source,),
            )
            if cursor.fetchone():
                conn.rollback()
                logger.info("Legacy portfolio migration already applied for owner %s", owner_id)
                return False

            imported_entries = 0
            for coin, data in old_portfolio.items():
                if not isinstance(data, dict):
                    continue
                normalized_coin = self._normalize_coin(coin)
                buy_price = data.get('buy_price', data.get('price', 0))
                entries = data.get('entries', []) or []
                if not entries:
                    entries = [{"amount": data.get("amount", 0), "price": buy_price}]

                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        entry_amount = self._positive_number(entry.get('amount'), "Amount")
                        entry_price = self._positive_number(entry.get('price', buy_price), "Price")
                    except ValueError:
                        logger.warning("Skipping invalid legacy entry for %s", normalized_coin)
                        continue
                    self._add_entry_in_transaction(
                        cursor,
                        int(owner_id),
                        normalized_coin,
                        entry_amount,
                        entry_price,
                    )
                    imported_entries += 1

            cursor.execute(
                """
                INSERT INTO portfolio_migrations (source, migrated_at, details)
                VALUES (?, ?, ?)
                """,
                (source, _utc_now_iso(), f"path={path};entries={imported_entries}"),
            )
            conn.commit()
            logger.info(
                "Legacy portfolio migration completed: owner=%s entries=%s",
                owner_id,
                imported_entries,
            )
            return True
        except Exception as e:
            connection = getattr(self._local, "conn", None)
            if connection is not None:
                connection.rollback()
            logger.error(f"Migration error: {e}", exc_info=True)
            return False


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
