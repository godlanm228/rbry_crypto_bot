# Payment System

Premium is sold for Telegram Stars (XTR). Stars need no payment provider
token: the bot's own token is enough.

Flow (`modules/payments/payment_manager.py`):
- `/premium` shows the plans (`PricingPlans`: 1, 3, 6 and 12 months; the
  lifetime plan is kept only so old payloads parse, it is not for sale);
- the bot sends an invoice whose payload is `premium_sub_{plan_id}_{user_id}`,
  so a payment still resolves after a restart;
- the pre-checkout query is checked against the plan and the price;
- a successful payment is written once per Telegram charge id
  (`payment_ledger.py`) and extends the user's access
  (`core/access/access_manager.py`); a renewal keeps the remaining days.

Support and refunds: `/paysupport` for users, `/refund USER_ID CHARGE_ID` for
the owner (`modules/privacy.py`).

Config: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_OWNER_ID`, optional `SUPPORT_URL`.
