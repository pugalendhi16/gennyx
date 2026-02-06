#!/usr/bin/env python3
"""Generate UT Bot buy/sell signals from candles table data."""

import os
import sys

import pandas as pd
import psycopg2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gennyx.indicators.ut_bot import ut_bot_alert

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_Xxz6nJLTpeB3@ep-blue-smoke-ah94a4rs-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require",
)


def fetch_candles(
    symbol: str = "/MNQ",
    timeframe: str = "5m",
    limit: int = 200,
) -> pd.DataFrame:
    """Fetch candles from database."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = %s AND timeframe = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(symbol, timeframe, limit))

        if df.empty:
            print(f"No candles found for {symbol} {timeframe}")
            return pd.DataFrame()

        # Set timestamp as index and sort ascending
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Convert to float
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        return df
    finally:
        conn.close()


def generate_signals(
    df: pd.DataFrame,
    sensitivity: float = 1.2,
    atr_period: int = 10,
    use_heikin_ashi: bool = False,
    confirmation_bars: int = 0,
) -> pd.DataFrame:
    """Apply UT Bot indicator and generate signals."""
    if df.empty:
        return df

    # Apply UT Bot
    ut_result = ut_bot_alert(
        df,
        sensitivity=sensitivity,
        atr_period=atr_period,
        use_heikin_ashi=use_heikin_ashi,
        confirmation_bars=confirmation_bars,
    )

    # Merge results
    result = df.copy()
    result["ut_trailing_stop"] = ut_result["ut_trailing_stop"]
    result["ut_trend"] = ut_result["ut_trend"]
    result["ut_buy_signal"] = ut_result["ut_buy_signal"]
    result["ut_sell_signal"] = ut_result["ut_sell_signal"]

    # Add signal column
    result["signal"] = "NONE"
    result.loc[result["ut_buy_signal"], "signal"] = "BUY"
    result.loc[result["ut_sell_signal"], "signal"] = "SELL"

    return result


def print_signals(df: pd.DataFrame, show_all: bool = False):
    """Print the buy/sell signals."""
    if df.empty:
        print("No data to display")
        return

    # Filter to only signals if not showing all
    if not show_all:
        signals_df = df[df["signal"] != "NONE"].copy()
    else:
        signals_df = df.copy()

    if signals_df.empty:
        print("No buy/sell signals found in the data")
        return

    print("\n" + "=" * 80)
    print("UT BOT SIGNALS")
    print("=" * 80)

    for idx, row in signals_df.iterrows():
        timestamp = idx.strftime("%Y-%m-%d %H:%M")
        signal = row["signal"]
        close = row["close"]
        trailing_stop = row["ut_trailing_stop"]
        trend = "UP" if row["ut_trend"] == 1 else "DOWN"

        if signal == "BUY":
            marker = "[BUY] "
        elif signal == "SELL":
            marker = "[SELL]"
        else:
            marker = "      "

        print(f"{marker} {timestamp}  Close: {close:,.2f}  Stop: {trailing_stop:,.2f}  Trend: {trend}")

    # Summary
    buy_count = (df["signal"] == "BUY").sum()
    sell_count = (df["signal"] == "SELL").sum()

    print("\n" + "-" * 80)
    print(f"SUMMARY: {buy_count} BUY signals, {sell_count} SELL signals")
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    print(f"Total candles: {len(df)}")
    print("-" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate UT Bot signals from candles data")
    parser.add_argument("--symbol", default="/MNQ", help="Trading symbol (default: /MNQ)")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe (default: 5m)")
    parser.add_argument("--limit", type=int, default=200, help="Number of candles to fetch (default: 200)")
    parser.add_argument("--sensitivity", type=float, default=1.2, help="UT Bot sensitivity (default: 1.2)")
    parser.add_argument("--atr-period", type=int, default=10, help="ATR period (default: 10)")
    parser.add_argument("--heikin-ashi", action="store_true", help="Use Heikin-Ashi source")
    parser.add_argument("--confirm-bars", type=int, default=0, help="Confirmation bars (1 = TOS-style, 0 = original)")
    parser.add_argument("--all", action="store_true", help="Show all candles, not just signals")
    parser.add_argument("--csv", help="Export results to CSV file")

    args = parser.parse_args()

    print(f"Fetching {args.limit} candles for {args.symbol} {args.timeframe}...")
    df = fetch_candles(args.symbol, args.timeframe, args.limit)

    if df.empty:
        return

    confirm_str = f", confirm={args.confirm_bars}" if args.confirm_bars > 0 else ""
    print(f"Generating UT Bot signals (sensitivity={args.sensitivity}, ATR={args.atr_period}{confirm_str})...")
    result = generate_signals(
        df,
        sensitivity=args.sensitivity,
        atr_period=args.atr_period,
        use_heikin_ashi=args.heikin_ashi,
        confirmation_bars=args.confirm_bars,
    )

    # Print signals
    print_signals(result, show_all=args.all)

    # Export to CSV if requested
    if args.csv:
        result.to_csv(args.csv)
        print(f"\nResults exported to: {args.csv}")


if __name__ == "__main__":
    main()
