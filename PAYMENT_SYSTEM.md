# Payment System

The bot supports Telegram Stars payments for premium tiers.

Flow:
- User opens premium menu
- Bot creates invoice
- Successful payment upgrades access tier

Config:
- TOKEN
- PAYMENT_PROVIDER_TOKEN (if applicable)
- OWNER_ID

See modules/payments/ for implementation details.
