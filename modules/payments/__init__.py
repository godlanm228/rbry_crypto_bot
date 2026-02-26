"""
Payments Module - Telegram Stars Integration

Provides payment processing for premium subscriptions using Telegram Stars (XTR).
"""

from .payment_manager import PaymentManager, PricingPlans, get_payment_manager

__all__ = ['PaymentManager', 'PricingPlans', 'get_payment_manager']
