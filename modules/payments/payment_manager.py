"""
Payment Manager - Telegram Stars Integration

Handles premium subscription payments via Telegram Stars (XTR).
Integrates with access_manager to grant access after successful payment.
"""

import logging
from typing import Dict, Optional
from telegram import Update, LabeledPrice, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from config import SUPPORT_URL
from locales.translator import t
from modules.payments.payment_ledger import PaymentLedger
from modules.callback_utils import answer_once
from core.incidents import report_incident

logger = logging.getLogger(__name__)

# Locale suffix per plan id ("premium_12m" is named "1y" in the locales).
_PLAN_LOCALE_SUFFIX = {"premium_12m": "1y"}


def _plan_key(plan_id: str) -> str:
    return _PLAN_LOCALE_SUFFIX.get(plan_id, plan_id.split("_", 1)[1])


class PricingPlans:
    """
    Pricing structure for premium subscriptions

    Prices are in Telegram Stars (XTR)
    Star-to-fiat pricing is intentionally not advertised here because it varies.
    """

    PLANS = {
        'premium_1m': {
            'id': 'premium_1m',
            'name': 'Premium - 1 месяц',
            'name_en': 'Premium - 1 Month',
            'price_stars': 100,
            'duration_days': 30,
            'tier': 'premium',
            'description': '30 дней полного доступа к многофакторному анализу'
        },
        'premium_3m': {
            'id': 'premium_3m',
            'name': 'Premium - 3 месяца',
            'name_en': 'Premium - 3 Months',
            'price_stars': 250,
            'duration_days': 90,
            'tier': 'premium',
            'description': '90 дней + скидка 17%',
            'discount': '17%'
        },
        'premium_6m': {
            'id': 'premium_6m',
            'name': 'Premium - 6 месяцев',
            'name_en': 'Premium - 6 Months',
            'price_stars': 450,
            'duration_days': 180,
            'tier': 'premium',
            'description': '180 дней + скидка 25%',
            'discount': '25%'
        },
        'premium_12m': {
            'id': 'premium_12m',
            'name': 'Premium - 1 год',
            'name_en': 'Premium - 1 Year',
            'price_stars': 800,
            'duration_days': 365,
            'tier': 'premium',
            'description': '365 дней + скидка 33%',
            'discount': '33%'
        },
        'beta_lifetime': {
            'id': 'beta_lifetime',
            'name': 'Beta - Навсегда',
            'name_en': 'Beta - Lifetime',
            'price_stars': 1500,
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

    PAYLOAD_PREFIX = "premium_sub_"

    def __init__(self, payment_ledger: Optional[PaymentLedger] = None):
        self.pending_payments: Dict[str, Dict] = {}  # payment_id -> {user_id, plan_id, timestamp}
        self.payment_ledger = payment_ledger or PaymentLedger()
        logger.info("Payment manager initialized")

    @staticmethod
    def _validate_payment_terms(payment_info: Dict, currency: str, total_amount: int) -> bool:
        plan = payment_info["plan"]
        return currency == "XTR" and total_amount == int(plan["price_stars"])

    def _support_keyboard(self, user_id: Optional[int], back_callback: str) -> InlineKeyboardMarkup:
        keyboard = []
        if SUPPORT_URL:
            keyboard.append([InlineKeyboardButton(t("premium.button_contact", user_id=user_id), url=SUPPORT_URL)])
        keyboard.append([InlineKeyboardButton(t("common.back", user_id=user_id), callback_data=back_callback)])
        return InlineKeyboardMarkup(keyboard)

    @classmethod
    def parse_payload(cls, payload: str) -> Optional[Dict]:
        """
        Recover order info from the invoice payload itself.

        The in-memory pending_payments dict does not survive a restart, but a user
        can still pay an already-issued invoice. The payload
        "premium_sub_{plan_id}_{user_id}" carries everything needed to honor it.
        """
        if not payload or not payload.startswith(cls.PAYLOAD_PREFIX):
            return None
        rest = payload[len(cls.PAYLOAD_PREFIX):]
        plan_id, sep, user_id_str = rest.rpartition("_")
        if not sep or not plan_id:
            return None
        try:
            user_id = int(user_id_str)
        except ValueError:
            return None
        plan = PricingPlans.get_plan(plan_id)
        if not plan:
            return None
        return {"user_id": user_id, "plan_id": plan_id, "plan": plan}

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
            plan_name = t(f"premium.plan_{_plan_key(plan_id)}", user_id=user_id)
            plan_desc = t(f"premium.desc_{_plan_key(plan_id)}", user_id=user_id)

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

        if SUPPORT_URL:
            keyboard.append([InlineKeyboardButton(t("premium.button_contact", user_id=user_id), url=SUPPORT_URL)])
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
            await answer_once(update.callback_query, t("premium.invalid_plan", user_id=user_id), show_alert=True)
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
            plan_name = t(f"premium.plan_{_plan_key(plan_id)}", user_id=user_id)
            if plan['duration_days'] == 0:
                duration_text = t("premium.duration_lifetime", user_id=user_id)
            else:
                duration_text = t(f"premium.duration_{_plan_key(plan_id)}", user_id=user_id)

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
                await answer_once(update.callback_query, t("premium.invoice_sent", user_id=user_id))

            return True

        except Exception as e:
            logger.error(f"Failed to create invoice for user {user_id}, plan {plan_id}: {e}", exc_info=True)
            await report_incident(
                context,
                e,
                subsystem="payment_invoice",
                user_id=user_id,
            )

            # Fallback: Show manual payment option
            reply_markup = self._support_keyboard(user_id, "premium")

            plan_name = t(f"premium.plan_{_plan_key(plan_id)}", user_id=user_id)
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

        payment_info = self.pending_payments.get(payload) or self.parse_payload(payload)

        if not payment_info:
            payer_id = query.from_user.id if query.from_user else None
            await query.answer(
                ok=False,
                error_message=t("errors.payment_invalid", user_id=payer_id)
            )
            return

        payer_id = query.from_user.id if query.from_user else None
        if payer_id is not None:
            from core.access.access_manager import get_access_manager
            if get_access_manager().has_permanent_access(payer_id):
                # Nothing to buy: permanent access already covers every plan.
                await query.answer(
                    ok=False,
                    error_message=t("premium.already_permanent", user_id=payer_id),
                )
                return
        if payer_id != payment_info["user_id"]:
            await query.answer(
                ok=False,
                error_message=t("errors.payment_other_user", user_id=payer_id)
            )
            logger.warning(
                "Rejected pre-checkout payload %s: payer %s != payload user %s",
                payload,
                payer_id,
                payment_info["user_id"],
            )
            return

        if not self._validate_payment_terms(
            payment_info,
            getattr(query, "currency", ""),
            getattr(query, "total_amount", -1),
        ):
            await query.answer(
                ok=False,
                error_message=t("errors.payment_terms_changed", user_id=payer_id)
            )
            logger.error(
                "Rejected pre-checkout terms: payload=%s currency=%s amount=%s expected=%s",
                payload,
                getattr(query, "currency", None),
                getattr(query, "total_amount", None),
                payment_info["plan"]["price_stars"],
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

        # Prefer in-memory data, but recover from payload after restarts.
        payment_info = self.pending_payments.get(payload) or self.parse_payload(payload)

        if not payment_info:
            logger.error(f"Payment info not found for payload: {payload}")
            await report_incident(
                context,
                RuntimeError("successful payment payload is invalid"),
                subsystem="payment_payload",
                user_id=getattr(update.effective_user, "id", None),
            )
            await update.message.reply_text(
                t(
                    "errors.payment_error",
                    user_id=getattr(update.effective_user, "id", None),
                )
            )
            return

        user_id = payment_info['user_id']
        plan = payment_info['plan']
        plan_id = payment_info['plan_id']

        currency = getattr(payment, "currency", "")
        total_amount = getattr(payment, "total_amount", -1)
        if not self._validate_payment_terms(payment_info, currency, total_amount):
            logger.critical(
                "Successful payment terms mismatch: payload=%s currency=%s amount=%s expected=%s",
                payload,
                currency,
                total_amount,
                plan["price_stars"],
            )
            await report_incident(
                context,
                RuntimeError(
                    f"successful payment terms mismatch currency={currency} amount={total_amount}"
                ),
                subsystem="payment_terms",
                user_id=user_id,
            )
            await update.message.reply_text(
                t("errors.payment_review", user_id=user_id)
            )
            return

        payer_id = update.effective_user.id if update.effective_user else None
        if payer_id != user_id:
            logger.error(
                "Payment payload user mismatch for %s: payer %s != payload user %s",
                payload,
                payer_id,
                user_id,
            )
            await report_incident(
                context,
                RuntimeError(f"successful payment payer mismatch payer={payer_id}"),
                subsystem="payment_payer",
                user_id=user_id,
            )
            await update.message.reply_text(
                t(
                    "errors.payment_review",
                    user_id=getattr(update.effective_user, "id", None),
                )
            )
            return

        charge_id = str(getattr(payment, "telegram_payment_charge_id", "") or "").strip()
        provider_charge_id = str(getattr(payment, "provider_payment_charge_id", "") or "")
        try:
            claim = self.payment_ledger.claim(
                telegram_charge_id=charge_id,
                provider_charge_id=provider_charge_id,
                invoice_payload=payload,
                user_id=user_id,
                plan_id=plan_id,
                currency=currency,
                total_amount=total_amount,
            )
        except Exception as e:
            logger.critical("Could not claim successful payment: %s", e, exc_info=True)
            await report_incident(
                context,
                e,
                subsystem="payment_ledger_claim",
                user_id=user_id,
            )
            await update.message.reply_text(
                t("errors.payment_review", user_id=user_id)
            )
            return

        if not claim.created and claim.status == "applied":
            logger.warning("Duplicate payment update ignored: charge=%s status=%s", charge_id, claim.status)
            await update.message.reply_text(t("premium.payment_success", user_id=user_id))
            return
        if not claim.created and claim.status == "failed":
            logger.warning("Failed payment update requires review: charge=%s", charge_id)
            await update.message.reply_text(
                t("errors.payment_processing", user_id=user_id)
            )
            return

        if not claim.created:
            logger.warning("Resuming interrupted payment grant: charge=%s", charge_id)

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
                days=plan['duration_days'],
                extend=True,
                idempotency_key=f"telegram-stars:{charge_id}",
            )

            logger.info(f"Access granted to user {user_id}: {result}")

            if access_manager.get_user_tier(user_id).value != plan['tier']:
                raise RuntimeError("access grant verification failed")

        except Exception as e:
            try:
                self.payment_ledger.mark_failed(charge_id, "access_grant_failed")
            except Exception:
                logger.exception("Could not mark failed payment charge=%s", charge_id)
            logger.error(f"Failed to grant access after payment: {e}", exc_info=True)
            await report_incident(
                context,
                e,
                subsystem="payment_access_grant",
                user_id=user_id,
            )
            await update.message.reply_text(
                t("errors.payment_review", user_id=user_id)
            )
            return

        try:
            # Safe to retry: the access grant and this charge share the same
            # durable idempotency key in AccessManager.
            self.payment_ledger.mark_applied(charge_id)
            self.pending_payments.pop(payload, None)
        except Exception as e:
            logger.critical("Could not finalize payment ledger charge=%s", charge_id, exc_info=True)
            await report_incident(
                context,
                e,
                subsystem="payment_ledger_finalize",
                user_id=user_id,
            )
            await update.message.reply_text(
                t("errors.payment_review", user_id=user_id)
            )
            return

        # Receipt delivery is deliberately outside the access transaction.
        plan_name = t(f"premium.plan_{_plan_key(plan_id)}", user_id=user_id)
        if plan['duration_days'] == 0:
            duration_text = t("premium.duration_lifetime", user_id=user_id)
        else:
            duration_text = t(f"premium.duration_{_plan_key(plan_id)}", user_id=user_id)

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
        try:
            await update.message.reply_text(
                success_message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error("Payment receipt delivery failed for user %s: %s", user_id, e, exc_info=True)
            await report_incident(
                context,
                e,
                subsystem="payment_receipt",
                user_id=user_id,
            )


# Global singleton
_payment_manager: Optional[PaymentManager] = None


def get_payment_manager() -> PaymentManager:
    """Get global payment manager instance"""
    global _payment_manager
    if _payment_manager is None:
        _payment_manager = PaymentManager()
    return _payment_manager

