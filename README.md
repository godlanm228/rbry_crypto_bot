# RBRY CryptoBot

Telegram assistant for crypto holders: portfolio and profit/loss tracking,
a market overview per coin (trend, RSI, volatility, levels) and a news digest.
It never asks for exchange API keys or wallet access; positions are entered
manually.

## Commands

| Command | What it does |
|---|---|
| `/start` | Main menu (first run: language from Telegram, time zone) |
| `/portfolio` | Positions, value and P&L in USD |
| `/add_entry COIN AMOUNT PRICE` | Record a purchase (price in USD, `0,5` and `0.5` both work) |
| `/sell COIN AMOUNT PRICE` | Record a sale with realized P&L |
| `/analyze` | Market overview for a coin |
| `/news` | Crypto news digest |
| `/settings` | Language, time zone, explanations |
| `/my_tier`, `/premium` | Plan, limits and purchase (Telegram Stars) |
| `/privacy`, `/delete_me` | What is stored; erase your data |
| `/paysupport` | Payment help and refunds |

Owner only: `/grant`, `/revoke`, `/user_stats`, `/check_user`, `/health`,
`/debug`, `/scan`, `/refund USER_ID CHARGE_ID`.

## Configuration

Copy `.env.example` to `.env` and set at least:

- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_OWNER_ID` — required.
- `ALERTS_ENABLED` — alert scans run only when `true` (default `false`).
  `PATTERN_ALERTS_ENABLED` and `NEWS_ALERTS_ENABLED` stay `false`: neither
  family showed value in validation.
- `ALLOWED_USER_IDS` — optional comma-separated allow-list; when set, every
  other user is ignored and alerts go only to these ids.
- `GROQ_API_KEY` / `HUGGINGFACE_API_KEY` — optional; without them the bot
  shows no AI explanations or news tone (it never invents one).

## Run locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Python 3.11 or 3.12
.venv/bin/python main.py
```

State lives in `data/` and logs in `logs/` relative to the working
directory.

### Owner-only test bot

Create a second bot in BotFather and put its token in `.env.test` together
with `TELEGRAM_OWNER_ID`, `ALLOWED_USER_IDS=<your id>` and
`ALERTS_ENABLED=true`, then:

```bash
.venv/bin/python scripts/run_test_bot.py
```

It keeps its own state under `runtime/test/` and never touches `data/`.

## Deploy (Linux VM, systemd)

```bash
git clone <repo> ~/rbry_crypto_bot && cd ~/rbry_crypto_bot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in the values
sudo cp deploy/rbry-bot.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now rbry-bot
journalctl -u rbry-bot -f
```

Choose an EU region: Binance public market data is not served to US IP
addresses. A revoked or wrong token makes the process exit with status 1.

Library updates run weekly on their own (`deploy/autoupdate`): a fresh venv
with the newest versions inside the requirement ranges and never a new major,
the test suite and a vulnerability check before the swap, five minutes of
watching after the restart, and a rollback on any error. `.venv` is then a
symlink to the live generation in `~/.venvs/`; install packages with
`.venv/bin/python -m pip`, so the path inside the venv never matters.

## Tests

```bash
.venv/bin/python -m pytest -q
```
