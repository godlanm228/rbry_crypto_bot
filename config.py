import os
from pathlib import Path

# Load the env file if it exists. RBRY_ENV_FILE selects another file (e.g. the
# owner-only test bot runs with RBRY_ENV_FILE=.env.test); real environment
# variables always win over the file.
ENV_FILE = Path(os.getenv("RBRY_ENV_FILE", ".env"))
if not ENV_FILE.is_absolute():
    ENV_FILE = Path(__file__).parent / ENV_FILE
try:
    from dotenv import load_dotenv
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
except ImportError:
    pass  # dotenv not installed, continue with os.getenv

# CRITICAL: Telegram Bot Configuration
# NEVER commit your actual token to git! Use .env file
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set! ")

OWNER_ID = os.getenv('TELEGRAM_OWNER_ID')
if not OWNER_ID:
    raise ValueError("TELEGRAM_OWNER_ID not set! ")
OWNER_ID = int(OWNER_ID)


def _parse_id_list(raw: str) -> frozenset:
    return frozenset(int(part) for part in raw.replace(";", ",").split(",") if part.strip())


# Optional allow-list of Telegram user ids. When set, updates from anyone else
# are dropped before any handler runs and alerts are delivered only to these
# ids. Empty means the bot is open to everyone (production).
ALLOWED_USER_IDS = _parse_id_list(os.getenv("ALLOWED_USER_IDS", ""))


def is_user_allowed(user_id) -> bool:
    return not ALLOWED_USER_IDS or int(user_id) in ALLOWED_USER_IDS


COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# File paths (PORTFOLIO_FILE: the pre-SQLite portfolio, migrated once at startup if it still exists)
PORTFOLIO_FILE = "data/portfolio.json"
LOG_FILE = "logs/bot.log"

# Performance / freshness
SNAPSHOT_REFRESH_SECONDS = int(os.getenv("SNAPSHOT_REFRESH_SECONDS", "60"))   # Market snapshot period
HISTORY_REFRESH_SECONDS = int(os.getenv("HISTORY_REFRESH_SECONDS", "3600"))   # Daily history changes slowly
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "200"))  # Daily points; RSI(14) needs ~100+ to converge (30 gave ~5pt error)
SNAPSHOT_MIN_COVERAGE = float(os.getenv("SNAPSHOT_MIN_COVERAGE", "0.90"))
HISTORY_MIN_COVERAGE = float(os.getenv("HISTORY_MIN_COVERAGE", "0.90"))
SNAPSHOT_WARMUP_GRACE_SECONDS = int(os.getenv("SNAPSHOT_WARMUP_GRACE_SECONDS", "180"))
HISTORY_WARMUP_GRACE_SECONDS = int(os.getenv("HISTORY_WARMUP_GRACE_SECONDS", "600"))

# Alerts
# Kill switch: alert scans and reminders are scheduled only when explicitly
# enabled. Off by default so a fresh deploy never starts pushing on its own.
ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "false").lower() == "true"
ALERTS_INTERVAL_SECONDS = int(os.getenv("ALERTS_INTERVAL_SECONDS", "900"))    # 15 min by default

# AI Model APIs (Optional - graceful degradation if missing)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# LLM prose under /analyze. Off: in testing gpt-oss-20b wrote "MACD is bullish"
# for MACD=bearish, and the card already explains every number in words.
# Groq is still used for news translation.
AI_EXPLANATIONS_ENABLED = os.getenv("AI_EXPLANATIONS_ENABLED", "false").lower() == "true"

# FinBERT/Model Settings
ENABLE_FINBERT = os.getenv("ENABLE_FINBERT", "True").lower() == "true"
# REQUIRE_FINBERT=True would crash startup when no AI backend is configured - keep opt-in.
REQUIRE_FINBERT = os.getenv("REQUIRE_FINBERT", "False").lower() == "true"
FINBERT_MODEL_ID = os.getenv("FINBERT_MODEL_ID", "ProsusAI/finbert")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")  # llama-3.x models were retired by Groq

# CoinGecko (Optional - free tier works without key)
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")

# Support / contact
# Shown on payment screens; leave empty to hide the contact button.
SUPPORT_URL = os.getenv("SUPPORT_URL", "")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Healthchecks
HEALTHCHECK_ENABLED = os.getenv("HEALTHCHECK_ENABLED", "True").lower() == "true"
HEALTHCHECK_INTERVAL_SECONDS = int(os.getenv("HEALTHCHECK_INTERVAL_SECONDS", "3600"))
# Remote model checks can be rate-limited/costly
HEALTHCHECK_REMOTE_MODELS = os.getenv("HEALTHCHECK_REMOTE_MODELS", "True").lower() == "true"
