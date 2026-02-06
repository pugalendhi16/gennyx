#!/usr/bin/env python3
"""Create database tables for GenNyx trading system."""

import os

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

SCHEMA_SQL = """
-- ============================================
-- GenNyx Trading System Database Schema
-- ============================================

-- 1. Trading State (JSONB blob for quick state persistence)
-- Used by state_manager.py for state restore on restart
CREATE TABLE IF NOT EXISTS trading_state (
    id SERIAL PRIMARY KEY,
    state_type VARCHAR(50) UNIQUE NOT NULL,
    data JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Trades - Completed trade history
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL DEFAULT '/MNQ',
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP NOT NULL,
    entry_price DECIMAL(12, 4) NOT NULL,
    exit_price DECIMAL(12, 4) NOT NULL,
    quantity INTEGER NOT NULL,
    pnl DECIMAL(12, 2) NOT NULL,
    pnl_percent DECIMAL(8, 4) NOT NULL,
    entry_reason VARCHAR(255),
    exit_reason VARCHAR(255),
    entry_atr DECIMAL(10, 4),
    stop_loss DECIMAL(12, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);

-- 3. Daily Stats - Daily trading statistics
CREATE TABLE IF NOT EXISTS daily_stats (
    id SERIAL PRIMARY KEY,
    trade_date DATE UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL DEFAULT '/MNQ',
    starting_capital DECIMAL(12, 2) NOT NULL,
    ending_capital DECIMAL(12, 2) NOT NULL,
    pnl DECIMAL(12, 2) NOT NULL,
    pnl_percent DECIMAL(8, 4),
    trade_count INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    largest_win DECIMAL(12, 2),
    largest_loss DECIMAL(12, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(trade_date);

-- 4. Candles - OHLCV price history
CREATE TABLE IF NOT EXISTS candles (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL DEFAULT '/MNQ',
    timeframe VARCHAR(10) NOT NULL,  -- '5m', '1h', etc.
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(12, 4) NOT NULL,
    high DECIMAL(12, 4) NOT NULL,
    low DECIMAL(12, 4) NOT NULL,
    close DECIMAL(12, 4) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_time ON candles(symbol, timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp DESC);

-- 5. Signals - Trading signal audit log
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL DEFAULT '/MNQ',
    timestamp TIMESTAMP NOT NULL,
    signal_type VARCHAR(20) NOT NULL,  -- 'entry_long', 'exit_long', 'none'
    price DECIMAL(12, 4) NOT NULL,
    stop_loss DECIMAL(12, 4),
    atr DECIMAL(10, 4),
    reason VARCHAR(255),
    executed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type);

-- 6. Positions - Open position tracking (alternative to JSONB)
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL DEFAULT '/MNQ',
    entry_time TIMESTAMP NOT NULL,
    entry_price DECIMAL(12, 4) NOT NULL,
    quantity INTEGER NOT NULL,
    stop_loss DECIMAL(12, 4) NOT NULL,
    entry_atr DECIMAL(10, 4),
    entry_reason VARCHAR(255),
    status VARCHAR(20) NOT NULL DEFAULT 'open',  -- 'open', 'closed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- 7. System Config - Runtime configuration storage
CREATE TABLE IF NOT EXISTS system_config (
    id SERIAL PRIMARY KEY,
    key VARCHAR(100) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    description VARCHAR(255),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. Performance Metrics - Aggregated performance tracking
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    period_type VARCHAR(20) NOT NULL,  -- 'daily', 'weekly', 'monthly'
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL DEFAULT '/MNQ',
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    win_rate DECIMAL(5, 2),
    total_pnl DECIMAL(12, 2) NOT NULL,
    avg_win DECIMAL(12, 2),
    avg_loss DECIMAL(12, 2),
    profit_factor DECIMAL(8, 4),
    max_drawdown DECIMAL(12, 2),
    sharpe_ratio DECIMAL(8, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(period_type, period_start, symbol)
);

CREATE INDEX IF NOT EXISTS idx_perf_metrics_period ON performance_metrics(period_type, period_start);

-- 9. Candle Signals - OHLC bars with UT Bot signal status + Heikin-Ashi OHLC
CREATE TABLE IF NOT EXISTS candle_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL DEFAULT '/MNQ',
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(12, 4) NOT NULL,
    high DECIMAL(12, 4) NOT NULL,
    low DECIMAL(12, 4) NOT NULL,
    close DECIMAL(12, 4) NOT NULL,
    volume BIGINT,
    ut_signal VARCHAR(10) NOT NULL DEFAULT 'NONE',  -- 'BUY', 'SELL', 'NONE'
    ut_trend INTEGER,                                -- 1 (up) or -1 (down)
    ut_trailing_stop DECIMAL(12, 4),
    atr DECIMAL(10, 4),
    ha_open DECIMAL(12, 4),                          -- Heikin-Ashi OHLC
    ha_high DECIMAL(12, 4),
    ha_low DECIMAL(12, 4),
    ha_close DECIMAL(12, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
);

-- Add HA columns to existing production table (idempotent)
ALTER TABLE candle_signals ADD COLUMN IF NOT EXISTS ha_open DECIMAL(12, 4);
ALTER TABLE candle_signals ADD COLUMN IF NOT EXISTS ha_high DECIMAL(12, 4);
ALTER TABLE candle_signals ADD COLUMN IF NOT EXISTS ha_low DECIMAL(12, 4);
ALTER TABLE candle_signals ADD COLUMN IF NOT EXISTS ha_close DECIMAL(12, 4);

CREATE INDEX IF NOT EXISTS idx_candle_signals_time ON candle_signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_candle_signals_signal ON candle_signals(ut_signal);
"""

def create_tables():
    """Create all database tables."""
    print("Connecting to Neon PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)

    try:
        with conn.cursor() as cur:
            print("Creating tables...")
            cur.execute(SCHEMA_SQL)
            conn.commit()
            print("Tables created successfully!")

            # List all tables
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = cur.fetchall()

            print("\nTables in database:")
            for table in tables:
                print(f"  - {table[0]}")

    finally:
        conn.close()
        print("\nConnection closed.")

if __name__ == "__main__":
    create_tables()
