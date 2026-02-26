"""
Payment Manager - Telegram Stars Integration

Handles premium subscription payments via Telegram Stars (XTR).
Integrates with access_manager to grant access after successful payment.
"""

import logging
from typing import Dict, Optional
from telegram import Update, LabeledPrice, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from locales.translator import t

logger = logging.getLogger(__name__)


class PricingPlans:
    """
    Pricing structure for premium subscriptions

    Prices are in Telegram Stars (XTR)
    1 Star ≈ $0.015 USD (may vary by region)
    """

    PLANS = {
        'premium_1m': {
            'id': 'premium_1m',
            'name': 'Premium - 1 месяц',
            'name_en': 'Premium - 1 Month',
            'price_stars': 100,  # ~$1.50 USD
            'duration_days': 30,
            'tier': 'premium',
            'description': '30 дней полного доступа к AI анализу'
        },
        'premium_3m': {
            'id': 'premium_3m',
            'name': 'Premium - 3 месяца',
            'name_en': 'Premium - 3 Months',
            'price_stars': 250,  # ~$3.75 USD (17% discount)
            'duration_days': 90,
            'tier': 'premium',
            'description': '90 дней + скидка 17%',
            'discount': '17%'
        },
        'premium_6m': {
            'id': 'premium_6m',
            'name': 'Premium - 6 месяцев',
            'name_en': 'Premium - 6 Months',
            'price_stars': 450,  # ~$6.75 USD (25% discount)
            'duration_days': 180,
            'tier': 'premium',
            'description': '180 дней + скидка 25%',
            'discount': '25%'
        },
        'premium_12m': {
            'id': 'premium_12m',
            'name': 'Premium - 1 год',
            'name_en': 'Premium - 1 Year',
            'price_stars': 800,  # ~$12 USD (33% discount)
            'duration_days': 365,
            'tier': 'premium',
            'description': '365 дней + скидка 33%',
            'discount': '33%'
        },
        'beta_lifetime': {
            'id': 'beta_lifetime',
            'name': 'Beta - Навсегда',
            'name_en': 'Beta - Lifetime',
            'price_stars': 1500,  # ~$22.50 USD
            'duration_days': 0,  # 0 = permanent
            'tier': 'beta',
            'description': 'Пожизненный доступ + экспериментальные функции'
        }
    }

    @classmethod
    def get_plan(cls, plan_id: str) -> Optional[Dict]:
        """Get plan details by ID"""
        return cls.PLANS.get(plan_id)

    @classmethod
    def get_all_plans(cls) -> Dict[str, Dict]:
        """Get all available plans"""
        return cls.PLANS.copy()


class PaymentManager:
    """
    Manages Telegram Stars payments for premium subscriptions

    Flow:
    1. User requests payment → create_invoice()
    2. User pays → Telegram sends successful_payment update
    3. Bot processes → handle_successful_payment()
    4. Grant access via access_manager
    """

    def __init__(self):
        self.pending_payments: Dict[str, Dict] = {}  # payment_id -> {user_id, plan_id, timestamp}
        logger.info("Payment manager initialized")

    async def show_premium_options(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Show premium subscription options with pricing

        Displays all available plans with interactive "Buy" buttons
        """
        user_id = update.effective_user.id if update.effective_user else None

        # Build message with pricing table
        message = t("premium.title", user_id=user_id) + "\n\n"
        message += t("premium.features_title", user_id=user_id) + "\n"
        message += t("premium.feature_all_coins", user_id=user_id) + "\n"
        message += t("premium.feature_ai_synthesis", user_id=user_id) + "\n"
        message += t("premium.feature_patterns", user_id=user_id) + "\n"
        message += t("premium.feature_alerts", user_id=user_id) + "\n"
        message += t("premium.feature_fast_access", user_id=user_id) + "\n\n"

        message += t("premium.prices_title", user_id=user_id) + "\n\n"

        # Build keyboard with plan buttons
        keyboard = []

        for plan_id, plan in PricingPlans.PLANS.items():
            discount_label = f" 🔥 {plan.get('discount', '')}" if plan.get('discount') else ""

            # Get translated duration label
            if plan['duration_days'] == 0:
                duration_label = t("premium.duration_lifetime", user_id=user_id)
            elif plan['duration_days'] == 30:
                duration_label = t("premium.duration_1m", user_id=user_id)
            elif plan['duration_days'] == 90:
                duration_label = t("premium.duration_3m", user_id=user_id)
            elif plan['duration_days'] == 180:
                duration_label = t("premium.duration_6m", user_id=user_id)
            elif plan['duration_days'] == 365:
                duration_label = t("premium.duration_1y", user_id=user_id)
            else:
                duration_label = f"{plan['duration_days']} дней"

            # Get translated plan name and description
            plan_name = t(f"premium.plan_{plan_id.split('_')[1]}", user_id=user_id)
            plan_desc = t(f"premium.desc_{plan_id.split('_')[1]}", user_id=user_id)

            message += f"**{plan_name}** - {plan['price_stars']} ⭐{discount_label}\n"
            message += f"  ↳ {plan_desc}\n\n"

            # Add buy button
            keyboard.append([
                InlineKeyboardButton(
                    f"💳 {duration_label} - {plan['price_stars']} ⭐",
                    callback_data=f"buy_plan:{plan_id}"
                )
            ])

        # Manual payment option (fallback)
        message += "\n" + t("premium.payment_issues", user_id=user_id) + "\n"
        message += t("premium.payment_contact", user_id=user_id)

        keyboard.append([InlineKeyboardButton(t("premium.button_contact", user_id=user_id), url="https://t.me/your_username")])
        keyboard.append([InlineKeyboardButton(t("premium.button_back_tier", user_id=user_id), callback_data="my_tier")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        # Send or edit message
        if update.callback_query:
            await update.callback_query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

    async def create_invoice(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        plan_id: str
    ) -> bool:
        """
        Create Telegram Stars payment invoice

        Args:
            update: Telegram update
            context: Bot context
            plan_id: Plan ID from PricingPlans

        Returns:
            True if invoice sent successfully
        """
        plan = PricingPlans.get_plan(plan_id)

        if not plan:
            user_id = update.effective_user.id
            await update.callback_query.answer(t("premium.invalid_plan", user_id=user_id), show_alert=True)
            return False

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        try:
            # Generate unique payload for tracking
            payload = f"premium_sub_{plan_id}_{user_id}"

            # Track pending payment
            self.pending_payments[payload] = {
                'user_id': user_id,
                'plan_id': plan_id,
                'plan': plan
            }

            # Create invoice description
            plan_name = t(f"premium.plan_{plan_id.split('_')[1]}", user_id=user_id)
            if plan['duration_days'] == 0:
                duration_text = t("premium.duration_lifetime", user_id=user_id)
            else:
                duration_text = t(f"premium.duration_{plan_id.split('_')[1]}", user_id=user_id)

            description = f"{plan_name}: {duration_text}"

            # Create LabeledPrice for Telegram Stars
            # For Telegram Stars, currency is "XTR" and provider_token is empty
            prices = [
                LabeledPrice(
                    label=plan_name,
                    amount=plan['price_stars']  # Amount in Telegram Stars (no need to multiply by 100)
                )
            ]

            # Send invoice
            await context.bot.send_invoice(
                chat_id=chat_id,
                title=plan_name,
                description=description,
                payload=payload,
                provider_token="",  # Empty for Telegram Stars
                currency="XTR",  # Telegram Stars currency code
                prices=prices,
                start_parameter=f"premium_{plan_id}",
                # Optional: Add photo
                # photo_url="https://your-domain.com/premium-badge.png",
                # photo_size=512,
                # photo_width=512,
                # photo_height=512
            )

            logger.info(f"Invoice created for user {user_id}, plan {plan_id}")

            # Answer callback query
            if update.callback_query:
                await update.callback_query.answer(t("premium.invoice_sent", user_id=user_id))

            return True

        except Exception as e:
            logger.error(f"Failed to create invoice for user {user_id}, plan {plan_id}: {e}", exc_info=True)

            # Fallback: Show manual payment option
            keyboard = [
                [InlineKeyboardButton(t("premium.button_contact", user_id=user_id), url="https://t.me/your_username")],
                [InlineKeyboardButton(t("common.back", user_id=user_id), callback_data="premium")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            plan_name = t(f"premium.plan_{plan_id.split('_')[1]}", user_id=user_id)
            error_message = (
                f"{t('premium.payment_unavailable', user_id=user_id)}\n\n"
                f"{t('premium.payment_plan_info', user_id=user_id, plan=plan_name, stars=plan['price_stars'])}\n\n"
                f"{t('premium.payment_contact_info', user_id=user_id)}"
            )

            if update.callback_query:
                await update.callback_query.edit_message_text(
                    error_message,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )

            return False

    async def handle_pre_checkout_query(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle pre-checkout query (before payment is confirmed)

        This is called by Telegram to verify the order before charging the user
        """
        query = update.pre_checkout_query

        # Verify payload is valid
        payload = query.invoice_payload

        if payload not in self.pending_payments:
            await query.answer(
                ok=False,
                error_message="Ошибка: неверный платеж. Попробуйте снова."
            )
            return

        # Everything is OK, proceed with payment
        await query.answer(ok=True)

        logger.info(f"Pre-checkout approved for payload: {payload}")

    async def handle_successful_payment(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle successful payment notification from Telegram

        This is called after user completes payment successfully
        Grant access to user via access_manager
        """
        payment = update.message.successful_payment
        payload = payment.invoice_payload

        logger.info(f"Successful payment received: {payload}")

        # Get pending payment info
        payment_info = self.pending_payments.get(payload)

        if not payment_info:
            logger.error(f"Payment info not found for payload: {payload}")
            await update.message.reply_text(
                "⚠️ Ошибка обработки платежа. Свяжитесь с поддержкой."
            )
            return

        user_id = payment_info['user_id']
        plan = payment_info['plan']
        plan_id = payment_info['plan_id']

        # Grant access via access_manager
        try:
            from core.access.access_manager import get_access_manager

            access_manager = get_access_manager()

            # Grant access (use owner ID as admin)
            from config import OWNER_ID
            result = access_manager.grant_access(
                admin_id=OWNER_ID,  # System grant
                target_user_id=user_id,
                tier=plan['tier'],
                days=plan['duration_days']
            )

            logger.info(f"Access granted to user {user_id}: {result}")

            # Remove from pending payments
            del self.pending_payments[payload]

            # Send success message
            plan_name = t(f"premium.plan_{plan_id.split('_')[1]}", user_id=user_id)
            if plan['duration_days'] == 0:
                duration_text = t("premium.duration_lifetime", user_id=user_id)
            else:
                duration_text = t(f"premium.duration_{plan_id.split('_')[1]}", user_id=user_id)

            success_message = (
                f"{t('premium.payment_success', user_id=user_id)}\n\n"
                f"{t('premium.payment_congrats', user_id=user_id)}\n"
                f"{t('premium.payment_plan', user_id=user_id, plan=plan_name)}\n"
                f"{t('premium.payment_duration', user_id=user_id, duration=duration_text)}\n\n"
                f"{t('premium.payment_features', user_id=user_id)}\n"
                f"{t('premium.feature_all_coins', user_id=user_id)}\n"
                f"{t('premium.feature_ai_synthesis', user_id=user_id)}\n"
                f"{t('premium.feature_patterns', user_id=user_id)}\n"
                f"{t('premium.feature_alerts', user_id=user_id)}\n"
                f"{t('premium.feature_fast_access', user_id=user_id)}\n\n"
                f"{t('premium.payment_start', user_id=user_id)}"
            )

            keyboard = [[InlineKeyboardButton(t("premium.button_start_analysis", user_id=user_id), callback_data="analyze")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                success_message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

        except Exception as e:
            logger.error(f"Failed to grant access after payment: {e}", exc_info=True)
            await update.message.reply_text(
                f"⚠️ Оплата получена, но произошла ошибка при активации.\n"
                f"Свяжитесь с поддержкой: payment ID = {payload}"
            )


# Global singleton
_payment_manager: Optional[PaymentManager] = None


def get_payment_manager() -> PaymentManager:
    """Get global payment manager instance"""
    global _payment_manager
    if _payment_manager is None:
        _payment_manager = PaymentManager()
    return _payment_manager

