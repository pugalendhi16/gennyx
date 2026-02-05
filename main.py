#!/usr/bin/env python3
"""
GenNyx - Live Paper Trading System for MNQ Futures

Designed for Heroku deployment with Neon PostgreSQL for state persistence.

Environment Variables Required:
    SCHWAB_API_KEY      - Schwab API key
    SCHWAB_API_SECRET   - Schwab API secret
    SCHWAB_TOKEN        - Schwab OAuth token JSON
    DATABASE_URL        - PostgreSQL connection URL (Neon)

Optional Environment Variables:
    SCHWAB_SYMBOL       - Symbol to trade (default: /MNQ)
    POLL_INTERVAL       - Seconds between polls (default: 30)
    INITIAL_CAPITAL     - Starting capital (default: 30000)
    SESSION_TYPE        - Trading session: auto, rth, overnight, 24h (default: auto)
    LOG_LEVEL           - Logging level (default: INFO)
"""

import logging
import sys
from datetime import datetime

import pytz

from gennyx.config import Config
from gennyx.live import LiveTradingEngine


class ETFormatter(logging.Formatter):
    """Formatter that outputs timestamps in Eastern Time."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.et_tz = pytz.timezone("America/New_York")

    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, tz=self.et_tz)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime("%Y-%m-%d %H:%M:%S")


def setup_logging(log_level: str = "INFO"):
    """Configure logging for Heroku with ET timestamps."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ETFormatter(
        fmt="%(asctime)s ET | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logging.basicConfig(
        level=getattr(logging, log_level),
        handlers=[handler],
    )

    # Suppress verbose HTTP request logs from httpx/httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def main():
    """Main entry point."""
    # Load configuration from environment
    try:
        config = Config.from_env()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nRequired environment variables:")
        print("  SCHWAB_API_KEY")
        print("  SCHWAB_API_SECRET")
        print("  SCHWAB_TOKEN")
        print("\nOptional:")
        print("  DATABASE_URL (for state persistence)")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(config.log_level)

    logger.info("=" * 60)
    logger.info("GenNyx - MNQ Live Paper Trading System")
    logger.info("=" * 60)

    # Create and start engine
    engine = LiveTradingEngine(config)

    try:
        engine.start(restore_state=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
