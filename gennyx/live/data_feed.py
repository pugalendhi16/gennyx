"""Schwab API data feed for live quotes."""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from schwab import auth

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Represents a market quote."""

    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    volume: int
    timestamp: datetime

    @classmethod
    def from_schwab_response(cls, symbol: str, data: dict) -> "Quote":
        """Create Quote from Schwab API response."""
        quote_data = data.get("quote", data)
        return cls(
            symbol=symbol,
            last_price=quote_data.get("lastPrice", 0.0),
            bid_price=quote_data.get("bidPrice", 0.0),
            ask_price=quote_data.get("askPrice", 0.0),
            bid_size=quote_data.get("bidSize", 0),
            ask_size=quote_data.get("askSize", 0),
            volume=quote_data.get("totalVolume", 0),
            timestamp=datetime.now(),
        )


class SchwabDataFeed:
    """Fetches real-time quotes from Schwab API."""

    def __init__(self, config=None):
        """
        Initialize Schwab data feed.

        Args:
            config: Config object with Schwab credentials
        """
        self.config = config
        self._client = None
        self._token_file = None
        self._token_file_path = None  # For local file-based token
        self._last_token_hash = None  # Track token changes for refresh persistence
        self._database_url = config.database_url if config else os.environ.get("DATABASE_URL")

    def _get_credentials(self) -> dict:
        """Get Schwab credentials from config or environment."""
        if self.config:
            return {
                "api_key": self.config.schwab_api_key,
                "api_secret": self.config.schwab_api_secret,
                "callback_url": self.config.schwab_callback_url,
                "token_json": self.config.schwab_token_json,
            }
        return {
            "api_key": os.environ.get("SCHWAB_API_KEY"),
            "api_secret": os.environ.get("SCHWAB_API_SECRET"),
            "callback_url": os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1"),
            "token_json": os.environ.get("SCHWAB_TOKEN"),
        }

    def connect(self) -> bool:
        """
        Establish connection to Schwab API.

        Supports both:
        - Local development: token from config/schwab_token.json file
        - Heroku: token from SCHWAB_TOKEN environment variable

        Returns:
            True if connection successful
        """
        creds = self._get_credentials()

        if not creds["api_key"] or not creds["api_secret"]:
            raise ValueError(
                "Schwab API credentials not found. "
                "Set environment variables or create config/credentials.json"
            )

        # Check for local token file first
        config_dir = Path(__file__).parent.parent.parent / "config"
        local_token_file = config_dir / "schwab_token.json"

        if local_token_file.exists():
            # Use local token file directly
            try:
                self._client = auth.client_from_token_file(
                    token_path=str(local_token_file),
                    api_key=creds["api_key"],
                    app_secret=creds["api_secret"],
                )
                self._token_file_path = local_token_file
                self._last_token_hash = self._get_token_hash()
                return True
            except Exception as e:
                logger.warning(f"Local token invalid: {e}")

        # If we have a token JSON string (from env var or database), write it to a temp file
        if creds["token_json"]:
            try:
                self._token_file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', delete=False
                )
                self._token_file.write(creds["token_json"])
                self._token_file.close()

                self._client = auth.client_from_token_file(
                    token_path=self._token_file.name,
                    api_key=creds["api_key"],
                    app_secret=creds["api_secret"],
                )
                self._last_token_hash = self._get_token_hash()
                logger.info("Connected to Schwab API using token from environment/database")
                return True
            except Exception as e:
                logger.error(f"Token from env var/database invalid: {e}")
                if self._token_file:
                    Path(self._token_file.name).unlink(missing_ok=True)

        raise ValueError(
            "No valid Schwab token found. Either:\n"
            "  1. Create config/schwab_token.json (run auth flow locally), or\n"
            "  2. Set SCHWAB_TOKEN environment variable with token JSON"
        )

    def get_quote(self, symbol: str = "/MNQ") -> Quote:
        """
        Fetch current quote for a symbol.

        Args:
            symbol: Symbol to fetch (default /MNQ for micro nasdaq futures)

        Returns:
            Quote object with current market data
        """
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        response = self._client.get_quotes([symbol])
        response.raise_for_status()
        data = response.json()

        # Check for errors
        if "errors" in data:
            raise ValueError(f"Invalid symbol {symbol}: {data['errors']}")

        # For futures like /MNQ, Schwab returns the front-month contract (e.g., /MNQH26)
        if symbol in data:
            return Quote.from_schwab_response(symbol, data[symbol])

        # If exact symbol not found, look for a matching futures contract
        for resp_symbol, quote_data in data.items():
            if resp_symbol.startswith(symbol) or (
                symbol.startswith("/") and resp_symbol.startswith(symbol[:4])
            ):
                return Quote.from_schwab_response(resp_symbol, quote_data)

        raise ValueError(
            f"No data returned for {symbol}. "
            "Markets may be closed or symbol is invalid."
        )

    def get_updated_token(self) -> Optional[str]:
        """
        Get the updated token JSON after refresh.

        Returns:
            Token JSON string if available
        """
        if self._token_file and Path(self._token_file.name).exists():
            with open(self._token_file.name) as f:
                return f.read()
        return None

    def is_connected(self) -> bool:
        """Check if connected to Schwab API."""
        return self._client is not None

    def cleanup(self):
        """Clean up temporary files."""
        # Save any token changes before cleanup
        self._persist_token_if_changed()
        if self._token_file:
            Path(self._token_file.name).unlink(missing_ok=True)

    def _get_token_hash(self) -> Optional[str]:
        """Get hash of current token file for change detection."""
        token_path = None
        if self._token_file and Path(self._token_file.name).exists():
            token_path = self._token_file.name
        elif self._token_file_path and self._token_file_path.exists():
            token_path = str(self._token_file_path)

        if token_path:
            try:
                with open(token_path) as f:
                    content = f.read()
                return str(hash(content))
            except Exception:
                pass
        return None

    def _persist_token_if_changed(self):
        """Check if token was refreshed and save to database if changed."""
        if not self._database_url:
            return

        current_hash = self._get_token_hash()
        if current_hash and current_hash != self._last_token_hash:
            token_json = self.get_updated_token()
            if token_json:
                self._save_token_to_database(token_json)
                self._last_token_hash = current_hash
                logger.info("Token refreshed and saved to database")

    def _save_token_to_database(self, token_json: str) -> bool:
        """Save token to database for persistence across restarts."""
        if not self._database_url:
            return False

        try:
            import psycopg2
            from psycopg2.extras import Json

            token_data = json.loads(token_json)
            conn = psycopg2.connect(self._database_url)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO system_config (key, value, description, updated_at)
                    VALUES ('schwab_token', %s, 'Schwab OAuth token (auto-refreshed)', CURRENT_TIMESTAMP)
                    ON CONFLICT (key)
                    DO UPDATE SET value = %s, updated_at = CURRENT_TIMESTAMP
                """, (Json(token_data), Json(token_data)))
                conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to save token to database: {e}")
            return False

    def check_and_persist_token(self):
        """Public method to check and persist token changes. Call periodically."""
        self._persist_token_if_changed()
