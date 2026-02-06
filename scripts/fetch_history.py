#!/usr/bin/env python3
"""
Fetch historical /MNQ 5-minute candle data from Schwab API and load into candles table.

Usage:
    # Fetch max history (~9 months)
    DATABASE_URL=... python scripts/fetch_history.py

    # Specific date range
    DATABASE_URL=... python scripts/fetch_history.py --start 2025-01-01 --end 2025-02-01

    # Dry run (no database writes)
    python scripts/fetch_history.py --dry-run
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from schwab import auth

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gennyx.config import load_credentials_from_file, load_token_from_database


CONFIG_DIR = Path(__file__).parent.parent / "config"


def _normalize_timestamp(ts):
    """
    Normalize timestamp to naive UTC for consistent database storage.

    All timestamps are stored as naive UTC to avoid timezone confusion.
    """
    if ts is None:
        return None

    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)

    if isinstance(ts, (int, float)):
        # Schwab returns milliseconds since epoch
        ts = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)

    return ts.replace(tzinfo=None)


def get_schwab_client():
    """
    Initialize Schwab client using available credentials.

    Tries in order:
    1. Local config/schwab_token.json file
    2. SCHWAB_TOKEN environment variable
    3. Database token lookup
    """
    # Get credentials
    file_creds = load_credentials_from_file(CONFIG_DIR)
    api_key = os.environ.get("SCHWAB_API_KEY") or file_creds.get("api_key")
    api_secret = os.environ.get("SCHWAB_API_SECRET") or file_creds.get("api_secret")

    if not api_key or not api_secret:
        raise ValueError(
            "Schwab credentials not found. Set SCHWAB_API_KEY and SCHWAB_API_SECRET "
            "environment variables, or create config/credentials.json"
        )

    # Try local token file first
    local_token_file = CONFIG_DIR / "schwab_token.json"
    if local_token_file.exists():
        try:
            client = auth.client_from_token_file(
                token_path=str(local_token_file),
                api_key=api_key,
                app_secret=api_secret,
            )
            print(f"Using token from {local_token_file}")
            return client
        except Exception as e:
            print(f"Local token invalid: {e}")

    # Try SCHWAB_TOKEN env var
    token_json = os.environ.get("SCHWAB_TOKEN")
    if token_json:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(token_json)
                temp_path = f.name

            client = auth.client_from_token_file(
                token_path=temp_path,
                api_key=api_key,
                app_secret=api_secret,
            )
            print("Using token from SCHWAB_TOKEN environment variable")
            return client
        except Exception as e:
            print(f"Token from env var invalid: {e}")
            Path(temp_path).unlink(missing_ok=True)

    # Try database token
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        token_json = load_token_from_database(database_url)
        if token_json:
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(token_json)
                    temp_path = f.name

                client = auth.client_from_token_file(
                    token_path=temp_path,
                    api_key=api_key,
                    app_secret=api_secret,
                )
                print("Using token from database")
                return client
            except Exception as e:
                print(f"Token from database invalid: {e}")
                Path(temp_path).unlink(missing_ok=True)

    raise ValueError(
        "No valid Schwab token found. Either:\n"
        "  1. Create config/schwab_token.json (run scripts/get_token.py), or\n"
        "  2. Set SCHWAB_TOKEN environment variable with token JSON, or\n"
        "  3. Store token in database"
    )


def resolve_futures_symbol(client, symbol="/MNQ"):
    """
    Resolve generic futures symbol to front-month contract.

    /MNQ -> /MNQH26 (or current front month)
    """
    print(f"Resolving {symbol} to front-month contract...")

    response = client.get_quotes([symbol])
    response.raise_for_status()
    data = response.json()

    if "errors" in data:
        raise ValueError(f"Symbol lookup failed: {data['errors']}")

    # Schwab returns the actual contract symbol
    for resp_symbol in data.keys():
        if resp_symbol.startswith(symbol[:4]):
            print(f"  {symbol} -> {resp_symbol}")
            return resp_symbol

    # If exact match found
    if symbol in data:
        return symbol

    raise ValueError(f"Could not resolve {symbol} to a valid contract")


def fetch_price_history(client, symbol, start, end):
    """
    Fetch 5-minute candle history from Schwab API.

    Args:
        client: Schwab API client
        symbol: Resolved symbol (e.g., /MNQH26)
        start: Start datetime
        end: End datetime

    Returns:
        Raw API response data
    """
    print(f"Fetching {symbol} 5m candles from {start} to {end}...")

    try:
        response = client.get_price_history_every_five_minutes(
            symbol,
            start_datetime=start,
            end_datetime=end,
            need_extended_hours_data=True,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise ValueError(f"API error: {data['error']}")

        candles = data.get("candles", [])
        print(f"  Received {len(candles)} candles")
        return data

    except Exception as e:
        error_msg = str(e)

        # Check for common futures-related errors
        if "not supported" in error_msg.lower() or "invalid" in error_msg.lower():
            print("\n" + "=" * 60)
            print("ERROR: Schwab may not support price history for futures.")
            print("=" * 60)
            print("\nThe get_price_history API typically only works for equities/ETFs.")
            print("For futures historical data, consider:")
            print("  1. Use yfinance with symbol 'MNQ=F' for free data")
            print("  2. Use a futures-specific data provider")
            print("\nExample yfinance usage:")
            print("  import yfinance as yf")
            print("  df = yf.download('MNQ=F', interval='5m', period='60d')")
            print("=" * 60 + "\n")

        raise


def transform_schwab_candles(raw_data, symbol):
    """
    Transform Schwab API response to database format.

    Args:
        raw_data: Raw Schwab API response
        symbol: Symbol for the candles

    Returns:
        List of candle dictionaries ready for database insert
    """
    candles = []
    raw_candles = raw_data.get("candles", [])

    for c in raw_candles:
        # Schwab timestamps are milliseconds since epoch (UTC)
        timestamp = _normalize_timestamp(c.get("datetime"))

        candles.append({
            "symbol": symbol,
            "timeframe": "5m",
            "timestamp": timestamp,
            "open": float(c.get("open", 0)),
            "high": float(c.get("high", 0)),
            "low": float(c.get("low", 0)),
            "close": float(c.get("close", 0)),
            "volume": int(c.get("volume", 0)),
        })

    # Sort by timestamp
    candles.sort(key=lambda x: x["timestamp"])

    return candles


def bulk_insert_candles(candles, db_url):
    """
    Bulk insert/upsert candles into database.

    Uses execute_values for efficient batch insertion.

    Args:
        candles: List of candle dictionaries
        db_url: Database connection URL

    Returns:
        Number of rows affected
    """
    if not candles:
        print("No candles to insert")
        return 0

    print(f"Inserting {len(candles)} candles into database...")

    conn = psycopg2.connect(db_url)

    try:
        with conn.cursor() as cur:
            # Prepare values for execute_values
            values = [
                (
                    c["symbol"],
                    c["timeframe"],
                    c["timestamp"],
                    c["open"],
                    c["high"],
                    c["low"],
                    c["close"],
                    c["volume"],
                )
                for c in candles
            ]

            query = """
                INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (symbol, timeframe, timestamp)
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """

            execute_values(cur, query, values)
            affected = cur.rowcount
            conn.commit()

        print(f"  Inserted/updated {affected} rows")
        return affected

    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


def print_summary(candles):
    """Print summary of fetched data."""
    if not candles:
        print("\nNo data fetched.")
        return

    timestamps = [c["timestamp"] for c in candles]
    min_ts = min(timestamps)
    max_ts = max(timestamps)

    print("\n" + "=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)
    print(f"Total candles: {len(candles)}")
    print(f"Date range:    {min_ts} to {max_ts}")
    print(f"Symbol:        {candles[0]['symbol']}")
    print(f"Timeframe:     {candles[0]['timeframe']}")

    # Sample first and last candle
    print("\nFirst candle:")
    c = candles[0]
    print(f"  {c['timestamp']}: O={c['open']:.2f} H={c['high']:.2f} L={c['low']:.2f} C={c['close']:.2f} V={c['volume']}")

    print("\nLast candle:")
    c = candles[-1]
    print(f"  {c['timestamp']}: O={c['open']:.2f} H={c['high']:.2f} L={c['low']:.2f} C={c['close']:.2f} V={c['volume']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical /MNQ candle data from Schwab API"
    )
    parser.add_argument(
        "--symbol",
        default="/MNQ",
        help="Futures symbol (default: /MNQ)"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD). Default: ~9 months ago"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but don't insert into database"
    )
    args = parser.parse_args()

    # Parse dates
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        # Default: ~9 months ago (Schwab limit for 5-minute data)
        start_date = end_date - timedelta(days=270)

    # Check database URL
    db_url = os.environ.get("DATABASE_URL")
    if not db_url and not args.dry_run:
        print("ERROR: DATABASE_URL environment variable not set.")
        print("Set DATABASE_URL or use --dry-run to test without database.")
        sys.exit(1)

    print("Schwab Historical Data Fetcher")
    print("-" * 40)
    print(f"Symbol:    {args.symbol}")
    print(f"Start:     {start_date.date()}")
    print(f"End:       {end_date.date()}")
    print(f"Dry run:   {args.dry_run}")
    print("-" * 40)

    try:
        # Initialize client
        client = get_schwab_client()

        # Resolve symbol
        resolved_symbol = resolve_futures_symbol(client, args.symbol)

        # Fetch history
        raw_data = fetch_price_history(client, resolved_symbol, start_date, end_date)

        # Transform data (use original symbol /MNQ for consistency)
        candles = transform_schwab_candles(raw_data, args.symbol)

        # Print summary
        print_summary(candles)

        # Insert into database
        if args.dry_run:
            print("\n[DRY RUN] Skipping database insert.")
        elif candles:
            bulk_insert_candles(candles, db_url)

            # Verify
            print("\nVerification query:")
            print(f"  SELECT COUNT(*), MIN(timestamp), MAX(timestamp)")
            print(f"  FROM candles WHERE symbol='{args.symbol}'")

        print("\nDone!")

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
