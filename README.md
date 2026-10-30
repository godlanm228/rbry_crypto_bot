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

## Learning data

The learning sandbox (`core/sandbox/`) keeps its market history in
`runtime/learning/market.db`, apart from user data. It holds 1h candles of every
USDT pair Binance ever listed, delisted ones included, plus 5m candles for the
top 100, futures funding and open interest, daily macro series, stablecoin
supply, Fear & Greed, BTC/ETH/stablecoin dominance (a daily snapshot from
28.10.2026 on) and a news archive. All sources are public and keyless.

```bash
.venv/bin/python scripts/market_data.py bootstrap   # first fill, about an hour
.venv/bin/python scripts/market_data.py report
```

On the VM `market-data.timer` refreshes it daily, `market-hourly.timer` adds
every closed hour of the trading pairs, and `news-archive.timer` collects
headlines every 15 minutes (`deploy/`).

## Urgent events

`rbry-urgent.service` (`scripts/urgent_watch.py`, rules in
`core/sandbox/urgent.py`) checks the news feeds, Binance's own announcements
and Binance's 6-hour tickers every minute and tells the owner at once, through
the test bot and nobody else: a hack, paused withdrawals, a depeg or
insolvency at a big exchange or stablecoin, a Binance delisting of a held
coin, BTC or ETH falling 3.5 times their usual 6-hour move, a stablecoin 2%
away from 1 USDT. Warnings, not trading signals.

```bash
.venv/bin/python scripts/urgent_watch.py --once   # one round
.venv/bin/python scripts/urgent_watch.py --test   # the newest urgent story as it would have looked
```

## The brain

`core/brain/` learns from market.db which coins tend to do better than the
others over the next two weeks (LightGBM on one row per coin per day, no LLM).
It is research until its exam says otherwise:

```bash
.venv/bin/python scripts/brain.py dataset   # market.db -> runtime/learning/brain/dataset-<version>.pkl
.venv/bin/python scripts/brain.py exam      # monthly walk-forward up to 2025, report next to the dataset
```

### Paper journal

`rbry-paper.timer` (01:00 UTC, after the data refresh of 00:20) rebuilds the
dataset when it lacks the newest closed day and runs `scripts/paper.py day`:
the champion places that day's coins and strategy C1 (half in BTC while it is
above its 50-day average, half in the 3 best coins bought and the 3 worst
shorted for 14 days) is written to `runtime/learning/paper/decisions.jsonl`
before any outcome exists, then booked on live hourly prices
(`core/brain/paper.py`). `rbry-paper-report.timer` sends the week to the owner
on Sunday at 09:00 Berlin time, against C1's usual weeks on history. No money
is traded.

```bash
.venv/bin/python scripts/paper.py day --dry-run   # tonight's decision and book, nothing written
.venv/bin/python scripts/paper.py report          # the journal now
```

### The laptop's night

The heavy learning runs on the owner's laptop, not on the 1 GB VM:
`deploy/night-task.ps1` registers the Windows task RBRY-Night (every day at
00:00, waking the laptop, at most 7 hours), which runs `scripts/night.py`:
market.db's update, the dataset and the simulator's hourly cache every night,
and in the night to Sunday the brain's exam, the trade evaluator
(`scripts/brain_evaluator.py`) and the paper candidates' robustness. Logs go to
`runtime/learning/night/`; the owner hears only about a failed step.

```bash
.venv/Scripts/python.exe scripts/night.py --dry-run --weekly   # what tonight would run
```
