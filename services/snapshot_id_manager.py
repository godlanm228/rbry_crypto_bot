"""
Snapshot ID Manager - Optimizes market snapshot by collecting unique coins
only from eligible users (premium/beta/owner) with automatic caching.
"""

import logging
import time
import threading
from typing import List, Set
from core.access.access_manager import get_access_manager, AccessTier
from services.coin_registry_service import get_coin_registry_service

logger = logging.getLogger(__name__)

class SnapshotIDManager:
    """
    Manages the list of coin IDs for market snapshots.
    - Collects coins only from eligible users (premium/beta/owner)
    - Includes TOP20 coins for alerts
    - Caches the list and rebuilds every 5 minutes
    - Deduplicates coins across all users
    """
    
    def __init__(self, top20_coins: List[str], cache_ttl: int = 300):
        """
        Args:
            top20_coins: List of top 20 coin IDs for alerts (always included)
            cache_ttl: Cache time-to-live in seconds (default: 300 = 5 minutes)
        """
        self.top20_coins = set(top20_coins)
        self.cache_ttl = cache_ttl
        
        self._cached_ids: List[str] = []
        self._cache_time: float = 0
        self._lock = threading.Lock()
        
        logger.info(f"SnapshotIDManager initialized with {len(self.top20_coins)} TOP20 coins")
    
    def get_snapshot_ids(self) -> List[str]:
        """
        Get the list of coin IDs for market snapshot.
        Returns cached list if still valid, otherwise rebuilds.
        """
        with self._lock:
            now = time.time()
            
            # Check if cache is still valid
            if self._cached_ids and (now - self._cache_time) < self.cache_ttl:
                return self._cached_ids
            
            # Rebuild cache
            logger.info("Rebuilding snapshot coin list...")
            self._cached_ids = self._build_snapshot_ids()
            self._cache_time = now
            
            logger.info(f"Snapshot IDs rebuilt: {len(self._cached_ids)} unique coins")
            return self._cached_ids
    
    def _build_snapshot_ids(self) -> List[str]:
        """Build the list of coin IDs from eligible users and TOP20."""
        coin_ids: Set[str] = set()
        
        # Always include TOP20 for alerts
        coin_ids.update(self.top20_coins)
        
        # Get coins from eligible users' portfolios
        try:
            from core.portfolio_db import get_portfolio_manager
            pm = get_portfolio_manager()
            
            # Get eligible user IDs
            eligible_user_ids = self._get_eligible_user_ids()
            
            if eligible_user_ids:
                # Get coins from eligible users
                eligible_coins = pm.get_eligible_portfolio_coins(eligible_user_ids)
                registry = get_coin_registry_service()
                resolved_count = 0
                for symbol in eligible_coins:
                    resolved = registry.resolve(symbol)
                    if not resolved:
                        logger.warning(f"Unknown coin ticker: {symbol}")
                        continue
                    coin_ids.add(resolved["coingecko_id"])
                    resolved_count += 1
                logger.info(f"Added {resolved_count} coins from {len(eligible_user_ids)} eligible users")
            else:
                logger.info("No eligible users found, using TOP20 only")
        
        except Exception as e:
            logger.warning(f"Could not get portfolio coins: {e}")
        
        return sorted(coin_ids)
    
    def _get_eligible_user_ids(self) -> List[int]:
        """
        Get list of user IDs eligible for snapshot inclusion.
        Eligible tiers: PREMIUM, BETA, OWNER
        """
        access_manager = get_access_manager()
        eligible_tiers = {AccessTier.PREMIUM, AccessTier.BETA, AccessTier.OWNER}
        
        try:
            from core.portfolio_db import get_portfolio_manager
            pm = get_portfolio_manager()
            all_users = pm.get_all_users()
            
            eligible_users = []
            for user_id in all_users:
                tier = access_manager.get_user_tier(user_id)
                if tier in eligible_tiers:
                    eligible_users.append(user_id)
            
            logger.info(f"Found {len(eligible_users)} eligible users out of {len(all_users)} total")
            return eligible_users
        
        except Exception as e:
            logger.error(f"Error getting eligible users: {e}", exc_info=True)
            return []
    
    def invalidate_cache(self):
        """Force cache invalidation on next get_snapshot_ids() call."""
        with self._lock:
            self._cache_time = 0
            logger.info("Snapshot ID cache invalidated")


def create_snapshot_id_manager(top20_coins: List[str]) -> SnapshotIDManager:
    """
    Factory function to create SnapshotIDManager instance.
    
    Args:
        top20_coins: List of top 20 coin IDs for alerts
    Returns:
        SnapshotIDManager instance
    """
    return SnapshotIDManager(top20_coins)
