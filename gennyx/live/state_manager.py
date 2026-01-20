"""State persistence using PostgreSQL (Neon) for Heroku deployment."""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

import psycopg2
from psycopg2.extras import Json


logger = logging.getLogger(__name__)


class StatePersistence:
    """
    Manages state persistence using PostgreSQL (Neon database).

    Saves and restores:
    - Trading state (position, capital, daily stats)
    - Candle history (for indicator warmup on restart)
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize state persistence.

        Args:
            database_url: PostgreSQL connection URL (from DATABASE_URL env var)
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        self._conn = None
        self._last_save_time: Optional[datetime] = None

        if self.database_url:
            self._init_database()

    def _get_connection(self):
        """Get database connection with retry logic."""
        if self._conn is None or self._conn.closed:
            max_retries = 3
            retry_delay = 1  # seconds

            for attempt in range(max_retries):
                try:
                    # sslmode is already in the URL, no need to pass separately
                    self._conn = psycopg2.connect(self.database_url)
                    return self._conn
                except psycopg2.OperationalError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to connect after {max_retries} attempts: {e}")
                        raise

        return self._conn

    def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute a database operation with retry on connection failure."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database operation failed, reconnecting: {e}")
                    self._conn = None  # Force reconnection
                    time.sleep(0.5)
                else:
                    raise

    def _init_database(self):
        """Initialize database tables if they don't exist."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trading_state (
                        id SERIAL PRIMARY KEY,
                        state_type VARCHAR(50) UNIQUE NOT NULL,
                        data JSONB NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def save_trading_state(self, state: Dict[str, Any]) -> bool:
        """
        Save trading state to database.

        Args:
            state: Trading state dictionary

        Returns:
            True if save successful
        """
        if not self.database_url:
            logger.warning("No database URL configured, state not saved")
            return False

        try:
            state["_saved_at"] = datetime.now().isoformat()
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trading_state (state_type, data, updated_at)
                    VALUES ('trading', %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (state_type)
                    DO UPDATE SET data = %s, updated_at = CURRENT_TIMESTAMP
                """, (Json(state), Json(state)))
                conn.commit()

            self._last_save_time = datetime.now()
            logger.debug("Trading state saved to database")
            return True

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Connection error - try to reconnect and retry once
            logger.warning(f"Connection error during save, retrying: {e}")
            self._conn = None
            try:
                conn = self._get_connection()
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trading_state (state_type, data, updated_at)
                        VALUES ('trading', %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (state_type)
                        DO UPDATE SET data = %s, updated_at = CURRENT_TIMESTAMP
                    """, (Json(state), Json(state)))
                    conn.commit()
                self._last_save_time = datetime.now()
                return True
            except Exception as retry_error:
                logger.error(f"Failed to save trading state after retry: {retry_error}")
                return False
        except Exception as e:
            logger.error(f"Failed to save trading state: {e}")
            return False

    def load_trading_state(self) -> Optional[Dict[str, Any]]:
        """
        Load trading state from database.

        Returns:
            Trading state dictionary or None if not found
        """
        if not self.database_url:
            logger.info("No database URL configured")
            return None

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT data FROM trading_state WHERE state_type = 'trading'
                """)
                row = cur.fetchone()

            if row:
                state = row[0]
                saved_at = state.pop("_saved_at", None)
                if saved_at:
                    logger.info(f"Loaded trading state from {saved_at}")
                return state

            logger.info("No existing trading state found")
            return None

        except Exception as e:
            logger.error(f"Failed to load trading state: {e}")
            return None

    def save_candle_history(self, candle_data: Dict[str, Any]) -> bool:
        """
        Save candle history to database.

        Args:
            candle_data: Candle history dictionary

        Returns:
            True if save successful
        """
        if not self.database_url:
            return False

        try:
            candle_data["_saved_at"] = datetime.now().isoformat()
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trading_state (state_type, data, updated_at)
                    VALUES ('candles', %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (state_type)
                    DO UPDATE SET data = %s, updated_at = CURRENT_TIMESTAMP
                """, (Json(candle_data), Json(candle_data)))
                conn.commit()

            logger.debug("Candle history saved to database")
            return True

        except Exception as e:
            logger.error(f"Failed to save candle history: {e}")
            return False

    def load_candle_history(self) -> Optional[Dict[str, Any]]:
        """
        Load candle history from database.

        Returns:
            Candle history dictionary or None if not found
        """
        if not self.database_url:
            return None

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT data FROM trading_state WHERE state_type = 'candles'
                """)
                row = cur.fetchone()

            if row:
                data = row[0]
                saved_at = data.pop("_saved_at", None)
                if saved_at:
                    logger.info(f"Loaded candle history from {saved_at}")
                return data

            logger.info("No existing candle history found")
            return None

        except Exception as e:
            logger.error(f"Failed to load candle history: {e}")
            return None

    def save_all(
        self,
        trading_state: Dict[str, Any],
        candle_data: Dict[str, Any],
    ) -> bool:
        """Save all state."""
        trading_ok = self.save_trading_state(trading_state)
        candle_ok = self.save_candle_history(candle_data)
        return trading_ok and candle_ok

    def clear_state(self) -> bool:
        """Clear all persisted state."""
        if not self.database_url:
            return True

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM trading_state")
                conn.commit()
            logger.info("State cleared")
            return True

        except Exception as e:
            logger.error(f"Failed to clear state: {e}")
            return False

    def has_state(self) -> bool:
        """Check if any persisted state exists."""
        if not self.database_url:
            return False

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM trading_state")
                count = cur.fetchone()[0]
            return count > 0

        except Exception:
            return False

    def should_save(self, interval_seconds: int = 60) -> bool:
        """Check if enough time has passed since last save."""
        if self._last_save_time is None:
            return True
        elapsed = (datetime.now() - self._last_save_time).total_seconds()
        return elapsed >= interval_seconds

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
