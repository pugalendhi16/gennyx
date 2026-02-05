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
        """Get a fresh database connection."""
        # Always close existing connection and create fresh one
        # This avoids transaction state issues with connection pooling
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                self._conn = psycopg2.connect(self.database_url)
                self._conn.autocommit = False
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

    def _reset_connection(self):
        """Reset the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

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
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS candle_signals (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL DEFAULT '/MNQ',
                        timestamp TIMESTAMP NOT NULL,
                        open DECIMAL(12, 4) NOT NULL,
                        high DECIMAL(12, 4) NOT NULL,
                        low DECIMAL(12, 4) NOT NULL,
                        close DECIMAL(12, 4) NOT NULL,
                        volume BIGINT,
                        ut_signal VARCHAR(10) NOT NULL DEFAULT 'NONE',
                        ut_trend INTEGER,
                        ut_trailing_stop DECIMAL(12, 4),
                        atr DECIMAL(10, 4),
                        ha_open DECIMAL(12, 4),
                        ha_high DECIMAL(12, 4),
                        ha_low DECIMAL(12, 4),
                        ha_close DECIMAL(12, 4),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp)
                    )
                """)
                # Add HA columns to existing table (safe for production)
                for col in ("ha_open", "ha_high", "ha_low", "ha_close"):
                    cur.execute(f"ALTER TABLE candle_signals ADD COLUMN IF NOT EXISTS {col} DECIMAL(12,4)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candle_signals_time ON candle_signals(timestamp DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candle_signals_signal ON candle_signals(ut_signal)")
                # Ensure candles table exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS candles (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL DEFAULT '/MNQ',
                        timeframe VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        open DECIMAL(12, 4) NOT NULL,
                        high DECIMAL(12, 4) NOT NULL,
                        low DECIMAL(12, 4) NOT NULL,
                        close DECIMAL(12, 4) NOT NULL,
                        volume BIGINT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timeframe, timestamp)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_time ON candles(symbol, timeframe, timestamp DESC)")
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

        except Exception as e:
            logger.error(f"Failed to save trading state: {e}")
            self._reset_connection()
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
            self._reset_connection()
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
            self._reset_connection()
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
            self._reset_connection()
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

    def save_trade(self, trade: Dict[str, Any], symbol: str = "/MNQ") -> bool:
        """
        Save a completed trade to the trades table.

        Args:
            trade: Trade dictionary with entry/exit details
            symbol: Trading symbol

        Returns:
            True if save successful
        """
        if not self.database_url:
            logger.warning("No database URL configured, trade not saved")
            return False

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades (
                        symbol, entry_time, exit_time, entry_price, exit_price,
                        quantity, pnl, pnl_percent, entry_reason, exit_reason,
                        entry_atr, stop_loss
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    symbol,
                    trade.get("entry_time"),
                    trade.get("exit_time"),
                    trade.get("entry_price"),
                    trade.get("exit_price"),
                    trade.get("quantity"),
                    trade.get("pnl"),
                    trade.get("pnl_percent"),
                    trade.get("entry_reason", ""),
                    trade.get("exit_reason", ""),
                    trade.get("entry_atr"),
                    trade.get("stop_loss"),
                ))
                trade_id = cur.fetchone()[0]
                conn.commit()

            logger.info(f"Trade saved to database (id={trade_id}): PnL=${trade.get('pnl', 0):+.2f}")
            return True

        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            self._reset_connection()
            return False

    def save_signal(self, signal: Dict[str, Any], symbol: str = "/MNQ") -> bool:
        """
        Save a trading signal to the signals table for audit.

        Args:
            signal: Signal dictionary
            symbol: Trading symbol

        Returns:
            True if save successful
        """
        if not self.database_url:
            return False

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO signals (
                        symbol, timestamp, signal_type, price, stop_loss, atr, reason, executed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    symbol,
                    signal.get("timestamp"),
                    signal.get("signal_type"),
                    signal.get("price"),
                    signal.get("stop_loss"),
                    signal.get("atr"),
                    signal.get("reason", ""),
                    signal.get("executed", False),
                ))
                conn.commit()

            logger.debug(f"Signal saved: {signal.get('signal_type')}")
            return True

        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            self._reset_connection()
            return False

    def save_candle_signal(self, candle: Dict[str, Any], symbol: str = "/MNQ") -> bool:
        """
        Save a completed candle with its UT Bot signal to candle_signals table.

        Uses UPSERT to handle duplicate timestamps gracefully.

        Args:
            candle: Dict with timestamp, open, high, low, close, volume,
                    ut_signal, ut_trend, ut_trailing_stop, atr,
                    ha_open, ha_high, ha_low, ha_close
            symbol: Trading symbol

        Returns:
            True if save successful
        """
        if not self.database_url:
            return False

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO candle_signals (
                        symbol, timestamp, open, high, low, close, volume,
                        ut_signal, ut_trend, ut_trailing_stop, atr,
                        ha_open, ha_high, ha_low, ha_close
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        ut_signal = EXCLUDED.ut_signal,
                        ut_trend = EXCLUDED.ut_trend,
                        ut_trailing_stop = EXCLUDED.ut_trailing_stop,
                        atr = EXCLUDED.atr,
                        ha_open = EXCLUDED.ha_open,
                        ha_high = EXCLUDED.ha_high,
                        ha_low = EXCLUDED.ha_low,
                        ha_close = EXCLUDED.ha_close
                """, (
                    symbol,
                    candle.get("timestamp"),
                    candle.get("open"),
                    candle.get("high"),
                    candle.get("low"),
                    candle.get("close"),
                    candle.get("volume"),
                    candle.get("ut_signal", "NONE"),
                    candle.get("ut_trend"),
                    candle.get("ut_trailing_stop"),
                    candle.get("atr"),
                    candle.get("ha_open"),
                    candle.get("ha_high"),
                    candle.get("ha_low"),
                    candle.get("ha_close"),
                ))
                conn.commit()

            logger.debug(f"Candle signal saved: {candle.get('timestamp')} {candle.get('ut_signal')}")
            return True

        except Exception as e:
            logger.error(f"Failed to save candle signal: {e}")
            self._reset_connection()
            return False

    def save_candle(self, candle: Dict[str, Any], symbol: str = "/MNQ", timeframe: str = "5m") -> bool:
        """
        Save a raw OHLCV candle to the candles table.

        Uses UPSERT to handle duplicate timestamps gracefully.

        Args:
            candle: Dict with timestamp, open, high, low, close, volume
            symbol: Trading symbol
            timeframe: Candle timeframe (e.g. '5m', '1h')

        Returns:
            True if save successful
        """
        if not self.database_url:
            return False

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO candles (
                        symbol, timeframe, timestamp, open, high, low, close, volume
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timeframe, timestamp)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, (
                    symbol,
                    timeframe,
                    candle.get("timestamp"),
                    candle.get("open"),
                    candle.get("high"),
                    candle.get("low"),
                    candle.get("close"),
                    candle.get("volume"),
                ))
                conn.commit()

            logger.debug(f"Candle saved: {timeframe} {candle.get('timestamp')}")
            return True

        except Exception as e:
            logger.error(f"Failed to save candle: {e}")
            self._reset_connection()
            return False

    def save_daily_stats(self, stats: Dict[str, Any], symbol: str = "/MNQ") -> bool:
        """
        Save or update daily statistics.

        Args:
            stats: Daily statistics dictionary
            symbol: Trading symbol

        Returns:
            True if save successful
        """
        if not self.database_url:
            return False

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO daily_stats (
                        trade_date, symbol, starting_capital, ending_capital,
                        pnl, pnl_percent, trade_count, winning_trades, losing_trades,
                        largest_win, largest_loss
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trade_date)
                    DO UPDATE SET
                        ending_capital = EXCLUDED.ending_capital,
                        pnl = EXCLUDED.pnl,
                        pnl_percent = EXCLUDED.pnl_percent,
                        trade_count = EXCLUDED.trade_count,
                        winning_trades = EXCLUDED.winning_trades,
                        losing_trades = EXCLUDED.losing_trades,
                        largest_win = GREATEST(daily_stats.largest_win, EXCLUDED.largest_win),
                        largest_loss = LEAST(daily_stats.largest_loss, EXCLUDED.largest_loss),
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    stats.get("date"),
                    symbol,
                    stats.get("starting_capital"),
                    stats.get("ending_capital"),
                    stats.get("pnl"),
                    stats.get("pnl_percent"),
                    stats.get("trade_count"),
                    stats.get("winning_trades"),
                    stats.get("losing_trades"),
                    stats.get("largest_win"),
                    stats.get("largest_loss"),
                ))
                conn.commit()

            logger.debug(f"Daily stats saved for {stats.get('date')}")
            return True

        except Exception as e:
            logger.error(f"Failed to save daily stats: {e}")
            self._reset_connection()
            return False

    def get_trades(self, limit: int = 100, symbol: str = "/MNQ") -> list:
        """
        Get recent trades from database.

        Args:
            limit: Maximum number of trades to return
            symbol: Trading symbol

        Returns:
            List of trade dictionaries
        """
        if not self.database_url:
            return []

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, symbol, entry_time, exit_time, entry_price, exit_price,
                           quantity, pnl, pnl_percent, entry_reason, exit_reason,
                           entry_atr, stop_loss, created_at
                    FROM trades
                    WHERE symbol = %s
                    ORDER BY exit_time DESC
                    LIMIT %s
                """, (symbol, limit))
                rows = cur.fetchall()

            trades = []
            for row in rows:
                trades.append({
                    "id": row[0],
                    "symbol": row[1],
                    "entry_time": row[2].isoformat() if row[2] else None,
                    "exit_time": row[3].isoformat() if row[3] else None,
                    "entry_price": float(row[4]) if row[4] else None,
                    "exit_price": float(row[5]) if row[5] else None,
                    "quantity": row[6],
                    "pnl": float(row[7]) if row[7] else 0,
                    "pnl_percent": float(row[8]) if row[8] else 0,
                    "entry_reason": row[9],
                    "exit_reason": row[10],
                    "entry_atr": float(row[11]) if row[11] else None,
                    "stop_loss": float(row[12]) if row[12] else None,
                    "created_at": row[13].isoformat() if row[13] else None,
                })

            return trades

        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            self._reset_connection()
            return []

    def get_total_stats(self, symbol: str = "/MNQ") -> Dict[str, Any]:
        """
        Get aggregated statistics from all trades.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with total statistics
        """
        if not self.database_url:
            return {}

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE pnl >= 0) as winning_trades,
                        COUNT(*) FILTER (WHERE pnl < 0) as losing_trades,
                        COALESCE(SUM(pnl), 0) as total_pnl,
                        COALESCE(AVG(pnl), 0) as avg_pnl,
                        COALESCE(MAX(pnl), 0) as largest_win,
                        COALESCE(MIN(pnl), 0) as largest_loss
                    FROM trades
                    WHERE symbol = %s
                """, (symbol,))
                row = cur.fetchone()

            if row:
                total_trades = row[0] or 0
                winning_trades = row[1] or 0
                return {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": row[2] or 0,
                    "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
                    "total_pnl": float(row[3]),
                    "avg_pnl": float(row[4]),
                    "largest_win": float(row[5]),
                    "largest_loss": float(row[6]),
                }

            return {}

        except Exception as e:
            logger.error(f"Failed to get total stats: {e}")
            self._reset_connection()
            return {}

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
