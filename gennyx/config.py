"""Configuration settings using environment variables, config files, or database."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def get_env(key: str, default: str = None, required: bool = False) -> str:
    """Get environment variable with optional default."""
    value = os.environ.get(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def load_credentials_from_file(config_dir: Path) -> dict:
    """Load Schwab credentials from config file (for local development)."""
    creds_file = config_dir / "credentials.json"
    token_file = config_dir / "schwab_token.json"

    result = {}

    if creds_file.exists():
        with open(creds_file) as f:
            data = json.load(f)
            schwab = data.get("schwab", {})
            result["api_key"] = schwab.get("api_key")
            result["api_secret"] = schwab.get("api_secret")
            result["callback_url"] = schwab.get("callback_url", "https://127.0.0.1")

    if token_file.exists():
        with open(token_file) as f:
            result["token_json"] = f.read()

    return result


def load_token_from_database(database_url: str) -> Optional[str]:
    """Load Schwab token from database (for Heroku deployment)."""
    if not database_url:
        return None

    try:
        import psycopg2
        conn = psycopg2.connect(database_url)
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM system_config WHERE key = 'schwab_token'")
            row = cur.fetchone()
        conn.close()

        if row:
            token_data = row[0]
            logger.info("Loaded Schwab token from database")
            return json.dumps(token_data)
        return None
    except Exception as e:
        logger.warning(f"Failed to load token from database: {e}")
        return None


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.environ.get(key, str(default)).lower()
    return value in ("true", "1", "yes")


def get_env_float(key: str, default: float) -> float:
    """Get float environment variable."""
    return float(os.environ.get(key, default))


def get_env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    return int(os.environ.get(key, default))


def get_env_list_int(key: str, default: list = None) -> list:
    """Get list of integers from comma-separated environment variable."""
    value = os.environ.get(key)
    if not value:
        return default or []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


@dataclass
class Config:
    """Trading strategy configuration from environment variables."""

    # Schwab API Credentials (from environment)
    schwab_api_key: str = None
    schwab_api_secret: str = None
    schwab_callback_url: str = None
    schwab_token_json: str = None  # Token stored as JSON string in env var

    # Schwab Symbol
    schwab_symbol: str = "/MNQ"

    # Database URL for state persistence
    database_url: str = None

    # UT Bot Settings
    ut_sensitivity: float = 3.5
    ut_atr_period: int = 10
    use_heikin_ashi: bool = False
    use_ha_atr: bool = None  # None = follow use_heikin_ashi; True/False = override

    # Supertrend Settings
    st_atr_period: int = 8
    st_multiplier: float = 2.5

    # ADX Settings
    adx_period: int = 14
    adx_threshold: int = 25
    adx_min_trend: int = 20

    # EMA Settings
    ema_periods: Tuple[int, int, int] = (9, 20, 50)

    # Bollinger Bands Settings
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_percentile: int = 25

    # Timeframes
    primary_tf: str = "5m"
    htf: str = "1h"

    # Trading Hours (ET)
    trading_start: str = "09:30"
    trading_end: str = "16:00"
    timezone: str = "America/New_York"
    blocked_hours: tuple = ()  # Hours to skip entries (e.g., (1, 5, 7, 8, 9, 21, 22, 23))

    # Risk Management
    initial_capital: float = 30000.0
    risk_per_trade: float = 0.03
    hard_stop_atr_mult: float = 2.0

    # Trade Costs
    commission_per_trade: float = 2.50
    slippage_points: float = 0.25

    # Contract Specifications
    symbol: str = "MNQ=F"  # yfinance symbol
    point_value: float = 2.0
    tick_size: float = 0.25

    # Margin Settings
    margin_per_contract: float = 2100.0
    use_intraday_margin: bool = False
    margin_buffer: float = 0.80

    # Live Trading Settings
    poll_interval: int = 30
    save_interval: int = 60
    simple_mode: bool = False
    session_type: str = "overnight"

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, config_dir: Optional[Path] = None) -> "Config":
        """
        Create configuration from environment variables or config files.

        For local development, credentials are loaded from config/ directory.
        For Heroku, credentials come from environment variables.
        """
        # Try to load credentials from files first (local development)
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"

        file_creds = load_credentials_from_file(config_dir)

        # Get database URL first (needed for token lookup)
        database_url = get_env("DATABASE_URL")

        # Get Schwab credentials (env vars take precedence over files, then database)
        schwab_api_key = get_env("SCHWAB_API_KEY") or file_creds.get("api_key")
        schwab_api_secret = get_env("SCHWAB_API_SECRET") or file_creds.get("api_secret")
        schwab_callback_url = get_env("SCHWAB_CALLBACK_URL") or file_creds.get("callback_url", "https://127.0.0.1")

        # Token: env var -> file -> database
        schwab_token_json = get_env("SCHWAB_TOKEN") or file_creds.get("token_json")
        if not schwab_token_json and database_url:
            schwab_token_json = load_token_from_database(database_url)

        if not schwab_api_key or not schwab_api_secret:
            raise ValueError(
                "Schwab credentials not found. Set SCHWAB_API_KEY and SCHWAB_API_SECRET "
                "environment variables, or create config/credentials.json"
            )

        config = cls(
            # Schwab credentials
            schwab_api_key=schwab_api_key,
            schwab_api_secret=schwab_api_secret,
            schwab_callback_url=schwab_callback_url,
            schwab_token_json=schwab_token_json,
            schwab_symbol=get_env("SCHWAB_SYMBOL", "/MNQ"),

            # Database
            database_url=database_url,

            # Strategy settings (with defaults)
            ut_sensitivity=get_env_float("UT_SENSITIVITY", 3.5),
            ut_atr_period=get_env_int("UT_ATR_PERIOD", 10),
            st_atr_period=get_env_int("ST_ATR_PERIOD", 8),
            st_multiplier=get_env_float("ST_MULTIPLIER", 2.5),
            adx_period=get_env_int("ADX_PERIOD", 14),
            adx_threshold=get_env_int("ADX_THRESHOLD", 25),
            adx_min_trend=get_env_int("ADX_MIN_TREND", 20),

            # Trading settings
            initial_capital=get_env_float("INITIAL_CAPITAL", 30000.0),
            risk_per_trade=get_env_float("RISK_PER_TRADE", 0.03),
            commission_per_trade=get_env_float("COMMISSION_PER_TRADE", 2.50),
            slippage_points=get_env_float("SLIPPAGE_POINTS", 0.25),

            # Session settings
            trading_start=get_env("TRADING_START", "09:30"),
            trading_end=get_env("TRADING_END", "16:00"),
            session_type=get_env("SESSION_TYPE", "overnight"),
            simple_mode=get_env_bool("SIMPLE_MODE", False),
            blocked_hours=tuple(get_env_list_int("BLOCKED_HOURS", [])),

            # Polling
            poll_interval=get_env_int("POLL_INTERVAL", 30),
            save_interval=get_env_int("SAVE_INTERVAL", 60),

            # Logging
            log_level=get_env("LOG_LEVEL", "INFO"),
        )

        return config

    def to_dict(self) -> dict:
        """Convert config to dictionary (excluding secrets)."""
        return {
            "ut_sensitivity": self.ut_sensitivity,
            "ut_atr_period": self.ut_atr_period,
            "st_atr_period": self.st_atr_period,
            "st_multiplier": self.st_multiplier,
            "adx_period": self.adx_period,
            "adx_threshold": self.adx_threshold,
            "initial_capital": self.initial_capital,
            "risk_per_trade": self.risk_per_trade,
            "trading_start": self.trading_start,
            "trading_end": self.trading_end,
            "session_type": self.session_type,
            "simple_mode": self.simple_mode,
            "poll_interval": self.poll_interval,
            "blocked_hours": self.blocked_hours,
        }
