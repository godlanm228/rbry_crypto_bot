import os
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, continue with os.getenv

# === CRITICAL: Telegram Bot Configuration ===
# NEVER commit your actual token to git! Use .env file
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set! ")

OWNER_ID = os.getenv('TELEGRAM_OWNER_ID')
if not OWNER_ID:
    raise ValueError("TELEGRAM_OWNER_ID not set! ")
OWNER_ID = int(OWNER_ID)
USER_ID = OWNER_ID  # USER_ID is alias for OWNER_ID
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
ALERT_COOLDOWN_MINUTES = 60

# Пути для файлов
PORTFOLIO_FILE = "data/portfolio.json"
LOG_FILE = "logs/bot.log"

# Настройки алертов
SIGNAL_THRESHOLD = 3.0
PRICE_CHECK_INTERVAL = 60

# Ключевые слова для новостей
SIGNIFICANT_NEWS_KEYWORDS = [
    "ban", "hacked", "etf", "blackrock", "regulation",
    "fraud", "lawsuit", "listing", "integration"
]

# === Performance / freshness ===
SNAPSHOT_REFRESH_SECONDS = int(os.getenv("SNAPSHOT_REFRESH_SECONDS", "60"))   # Market snapshot period
HISTORY_REFRESH_SECONDS = int(os.getenv("HISTORY_REFRESH_SECONDS", "300"))    # History snapshot period
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "30"))                           # Days for technicals

# Alerts
ALERTS_INTERVAL_SECONDS = int(os.getenv("ALERTS_INTERVAL_SECONDS", "900"))    # 15 min by default
MAX_ALERTS_PER_CYCLE = int(os.getenv("MAX_ALERTS_PER_CYCLE", "3"))
ALERT_DEDUP_WINDOW_SECONDS = int(os.getenv("ALERT_DEDUP_WINDOW_SECONDS", "7200"))  # 2h dedup

# === AI Model APIs (Optional - graceful degradation if missing) ===
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Local FinBERT needs torch (~2.5GB install, 500MB+ RAM) — cloud inference is the default.
USE_LOCAL_FINBERT = os.getenv("USE_LOCAL_FINBERT", "False").lower() == "true"

# === FinBERT/Model Settings ===
ENABLE_FINBERT = os.getenv("ENABLE_FINBERT", "True").lower() == "true"
# REQUIRE_FINBERT=True would crash startup when no AI backend is configured — keep opt-in.
REQUIRE_FINBERT = os.getenv("REQUIRE_FINBERT", "False").lower() == "true"
ALLOW_MODEL_DOWNLOAD = os.getenv("ALLOW_MODEL_DOWNLOAD", "False").lower() == "true"
FINBERT_MODEL_ID = os.getenv("FINBERT_MODEL_ID", "ProsusAI/finbert")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
VALIDATION_MIN_AGE_HOURS = int(os.getenv("VALIDATION_MIN_AGE_HOURS", "12"))
VALIDATION_INTERVAL_SECONDS = int(os.getenv("VALIDATION_INTERVAL_SECONDS", "3600"))

# === Exchange APIs (Optional) ===
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# === CoinGecko (Optional - free tier works without key) ===
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")

# === Learning System Configuration ===
FEEDBACK_LOOP_LOOKBACK_DAYS = int(os.getenv("FEEDBACK_LOOP_LOOKBACK_DAYS", "90"))
FEEDBACK_LOOP_MIN_SIGNALS = int(os.getenv("FEEDBACK_LOOP_MIN_SIGNALS", "5"))
FEEDBACK_LOOP_CROSS_COIN = os.getenv("FEEDBACK_LOOP_CROSS_COIN", "False").lower() == "true"

# === Support / contact ===
# Shown on payment screens; leave empty to hide the contact button.
SUPPORT_URL = os.getenv("SUPPORT_URL", "")

# === Logging ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# === Healthchecks ===
HEALTHCHECK_ENABLED = os.getenv("HEALTHCHECK_ENABLED", "True").lower() == "true"
HEALTHCHECK_INTERVAL_SECONDS = int(os.getenv("HEALTHCHECK_INTERVAL_SECONDS", "3600"))
# Remote model checks can be rate-limited/costly
HEALTHCHECK_REMOTE_MODELS = os.getenv("HEALTHCHECK_REMOTE_MODELS", "True").lower() == "true"
