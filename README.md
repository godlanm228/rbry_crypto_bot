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
(`scripts/brain_evaluator.py`), the paper candidates' robustness and the event
library. Logs go to `runtime/learning/night/`; the owner hears only about a
failed step.

```bash
.venv/Scripts/python.exe scripts/night.py --dry-run --weekly   # what tonight would run
```

### Event library

Stage 1 of the watcher (signals when something happens, for the owner to trade
by hand): `core/brain/events.py` finds events at an hour's close (spikes on
volume, closes beyond the week's range, crowded funding, open interest jumps,
new listings, ...) on 2021–2026 hourly history and grades what followed: entry
10 minutes after the signal (measured on 5-minute candles), exits after 4, 24
or 72 hours, fees, slippage and funding, against a random entry into the same
coin in the same month. The direction comes from 2021–2024, 2025 and 2026 must
agree, and the bar rises with every event ever tried. Nothing is sent yet.

```bash
.venv/Scripts/python.exe scripts/brain_events.py   # report runtime/learning/brain/events-<time>.md
```

Stage 2 (`core/brain/exits.py`, `scripts/brain_panic.py`): the panic buy (four
events that all buy forced selling, in a nervous market only) under 13 exit
rules and 5 picks of a moment, walked hour by hour on spot and booked as a
$100 portfolio ($20 a trade, at most 5 open). The pick and the exit are chosen
on 2021–2024; a candidate's trades must beat random entries and its portfolio
must grow in 2021–2024, 2025 and 2026 with a drawdown within 50%.

```bash
.venv/Scripts/python.exe scripts/brain_panic.py   # report runtime/learning/brain/panic-<time>.md
```

Stage 3, the watcher on paper (`core/brain/watch.py`, `scripts/panic_watch.py`):
`rbry-panic.timer` runs a minute after every hour closes on the test VM. It
reads 45 days of hours from market.db and the newest closed hours from the
klines API, finds the four panic events with the library's own code, and in a
mass moment (4+ coins in panic, nervous market) writes the 3 most liquid to
`runtime/learning/watch/decisions.jsonl` before any outcome exists. Two $100
paper books (half out at +5%, the rest at the entry or after 3 days; without a
stop and with a disaster stop at −30%) replay every decision on hourly prices.
The owner gets a card for each moment and a line for each half sold or trade
closed, without sound from 00:00 to 08:00 Berlin; `rbry-panic-report.timer`
sends the week on Sunday at 09:10. No money is traded.

### Strategy factory

`core/brain/factory.py`, `scripts/brain_factory.py`: every night on the laptop
the factory breeds 500 strategies from the event library's bricks (1–4 events,
buy on spot or short on 1x margin, conditions, the moment's size, the picks,
the exit, a squeeze filter for shorts) and books each as a $100 portfolio in
every year. Two goals: «Разгон» (fastest growth, drawdown within 50% in every
year) and «Сейф» (within 20%). Only 2021–2024 guide the search: new genomes are
bred from the best of each goal, a fifth are random, and genomes booked on
older data are booked again. Every genome is a try in
`runtime/learning/factory/pool.jsonl`; a finalist must beat random entries past
the tries bar and grow in 2025 and 2026 within its goal's drawdown. Candidates
go to `promoted.json`, which the night copies to the VM, where the paper
watcher runs each as its own $100 book with its own cards.

Regimes (`core/brain/regime.py`): BTC's trend (up / flat / down against its
50-day average ±3%) × its mood (calm / nervous), a new regime counting after 6
hours in a row. Every regime is a niche of its own in the factory (a genome may
trade only there), and the switcher holds one $100 account that each hour runs
the champion of the current regime: a specialist, holding BTC or ETH, a BTC
short, or USDT. The factory's report walks the switcher forward (each year's
champions chosen on the years before it) against the single best strategy and
holding BTC; a switcher goes to paper only when it holds in 2025 and 2026.

### Selector

`core/brain/selector.py`, `scripts/brain_selector.py`: the choice of a trade
as a model that learns trade by trade. Every coin-hour of the event library is
a candidate with what is known at its close (its events, the coin, its
moment, BTC's trend, mood and moves, the time); every candidate has up to 8
actions (spot long or 1x short, four exits) walked on hourly prices. A
LightGBM regression of the net result is refit every month from 2022 on the
trades closed before that month and forecasts only that month; per coin-hour
the best action is taken when its forecast clears a bar. Booked as a $100
portfolio per year against the panic and holding BTC, with forecasts set
against what happened.

```bash
.venv/Scripts/python.exe scripts/brain_selector.py   # report runtime/learning/selector/selector-<time>.md
```

```bash
.venv/Scripts/python.exe scripts/brain_factory.py --batch 500   # report runtime/learning/factory/factory-<time>.md
.venv/bin/python scripts/panic_watch.py hour --dry-run             # the hour that just closed, nothing written
.venv/bin/python scripts/panic_watch.py hour --at "2026-09-23 15:00"   # a past hour (UTC start), db only
.venv/bin/python scripts/panic_watch.py report                     # the paper books now
```
