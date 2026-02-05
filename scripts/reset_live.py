#!/usr/bin/env python3
"""Reset live trading state to fresh $30K start.

Truncates all trading data tables while preserving system_config (OAuth tokens)
and candles table schema.

Usage:
    # With Heroku CLI:
    DATABASE_URL=$(heroku config:get DATABASE_URL --app gennyx-qa) python scripts/reset_live.py

    # Or set DATABASE_URL env var directly
"""

import os
import sys

import psycopg2


DATABASE_URL = os.environ.get("DATABASE_URL")

TABLES_TO_TRUNCATE = [
    "trading_state",
    "trades",
    "daily_stats",
    "signals",
    "positions",
    "performance_metrics",
]

# Tables to PRESERVE (not truncated):
# - system_config (contains Schwab OAuth token)
# - candles (schema only, currently empty)
# - candle_signals (new table, preserve if exists)


def reset_trading_data():
    """Truncate all trading data tables."""
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL environment variable not set.")
        print("Usage: DATABASE_URL=$(heroku config:get DATABASE_URL --app gennyx-qa) python scripts/reset_live.py")
        sys.exit(1)

    print(f"Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)

    try:
        with conn.cursor() as cur:
            # Show current state before reset
            print("\n--- Current State ---")
            for table in TABLES_TO_TRUNCATE:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    print(f"  {table}: {count} rows")
                except psycopg2.errors.UndefinedTable:
                    conn.rollback()
                    print(f"  {table}: (table does not exist)")

            # Confirm
            print(f"\nThis will TRUNCATE the following tables:")
            for table in TABLES_TO_TRUNCATE:
                print(f"  - {table}")
            print(f"\nPreserving: system_config, candles, candle_signals")

            response = input("\nType 'RESET' to confirm: ")
            if response != "RESET":
                print("Aborted.")
                return

            # Truncate tables
            print("\nTruncating tables...")
            for table in TABLES_TO_TRUNCATE:
                try:
                    cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
                    print(f"  Truncated: {table}")
                except psycopg2.errors.UndefinedTable:
                    conn.rollback()
                    print(f"  Skipped (not found): {table}")

            conn.commit()

            # Verify
            print("\n--- After Reset ---")
            for table in TABLES_TO_TRUNCATE:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    print(f"  {table}: {count} rows")
                except psycopg2.errors.UndefinedTable:
                    conn.rollback()
                    print(f"  {table}: (table does not exist)")

            print("\nReset complete! Engine will start fresh with initial_capital from config.")

    finally:
        conn.close()
        print("Connection closed.")


if __name__ == "__main__":
    reset_trading_data()
