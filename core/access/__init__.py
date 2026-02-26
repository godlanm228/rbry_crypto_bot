"""
Access Control Module

Manages tiered access control with admin-only management.
"""

from .access_manager import (
    AccessManager,
    AccessTier,
    AccessLimits,
    get_access_manager
)

__all__ = [
    'AccessManager',
    'AccessTier',
    'AccessLimits',
    'get_access_manager',
]
