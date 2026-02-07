"""
GenNyx Backtester — /MNQ Strategy (Database-backed)

Runs three sessions matching the proven strategy:
  - RTH (9:30-16:00): Full MTF filtered + regular candles
  - Overnight (16:00-9:30): Simple UT Bot + Heikin-Ashi candles
  - Combined: RTH + Overnight merged

Data source: PostgreSQL candles table (fetched via Schwab API)

Usage:
    python scripts/backtest.py                    # Use database (default)
    python scripts/backtest.py --days 60          # Last 60 days
    python scripts/backtest.py --start 2025-09-01 # From specific date
"""

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gennyx.indicators import (
    ut_bot_alert,
    supertrend,
    calculate_adx,
    ema_stack,
    bollinger_squeeze,
)
from gennyx.indicators.ut_bot import calculate_atr
from gennyx.strategy.filters import HTFFilter, TrendFilter
from gennyx.strategy.position import PositionSizer

# ---------------------------------------------------------------------------
# Configuration (matches proven Futures project config/settings.py)
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    "symbol": "/MNQ",
    "days": 60,
    # UT Bot
    "ut_sensitivity": 3.5,
    "ut_atr_period": 10,
    "ut_confirmation_bars": 1,  # TOS-style: 1 bar confirmation
    # Supertrend
    "st_atr_period": 8,
    "st_multiplier": 2.5,
    # ADX
    "adx_period": 14,
    "adx_threshold": 25,
    "adx_min_trend": 20,
    # EMA
    "ema_periods": (9, 20, 50),
    # Bollinger Bands
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_squeeze_percentile": 25,
    # Risk / position sizing (MNQ contract specs)
    "initial_capital": 30000.0,
    "risk_per_trade": 0.03,
    "hard_stop_atr_mult": 2.0,
    "point_value": 2.0,
    "commission": 2.50,
    "slippage": 0.25,
    "margin_per_contract": 2100.0,
    "margin_buffer": 0.80,
    # Warmup
    "warmup_bars": 100,
    # Hour filter (skip entries during these hours ET)
    "blocked_hours": [1, 5, 7, 8, 9, 21, 22, 23],
}

# Session-specific overrides
SESSION_CONFIGS = {
    "rth": {
        "trading_start": "09:30",
        "trading_end": "16:00",
        "use_heikin_ashi": True,
        "use_ha_atr": True,   # RTH: HA candles + HA ATR
        "simple_mode": False,
    },
    "overnight": {
        "trading_start": "16:00",
        "trading_end": "09:30",
        "use_heikin_ashi": True,
        "use_ha_atr": False,  # Overnight: HA candles + Raw ATR
        "simple_mode": True,
    },
}

ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    entry_reason: str
    exit_reason: str
    session: str = ""


# ---------------------------------------------------------------------------
# Trading hours helpers
# ---------------------------------------------------------------------------
def is_trading_hours(ts: pd.Timestamp, start_str: str, end_str: str) -> bool:
    """Check if timestamp is within trading hours (supports overnight)."""
    local = ts.astimezone(ET).time() if ts.tz else ts.time()
    start = datetime.strptime(start_str, "%H:%M").time()
    end = datetime.strptime(end_str, "%H:%M").time()
    if start > end:  # overnight
        return local >= start or local < end
    return start <= local < end


def is_near_close(ts: pd.Timestamp, end_str: str, start_str: str,
                  minutes_before: int = 5) -> bool:
    """Check if timestamp is within N minutes of session close."""
    local = ts.astimezone(ET) if ts.tz else ts
    end = datetime.strptime(end_str, "%H:%M").time()
    close_dt = local.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    # For overnight sessions, if we're after start, close is next day
    start = datetime.strptime(start_str, "%H:%M").time()
    if start > end and local.time() >= start:
        close_dt = close_dt + timedelta(days=1)
    delta = (close_dt - local).total_seconds() / 60
    return 0 <= delta <= minutes_before


# ---------------------------------------------------------------------------
# Data fetching (from PostgreSQL candles table)
# ---------------------------------------------------------------------------
def fetch_data(symbol: str, days: int = None, start_date: str = None,
               end_date: str = None, db_url: str = None):
    """
    Fetch 5m OHLCV data from database and aggregate to 1h.

    Args:
        symbol: Trading symbol (e.g., '/MNQ')
        days: Number of days to fetch (from today backwards)
        start_date: Start date string 'YYYY-MM-DD' (overrides days)
        end_date: End date string 'YYYY-MM-DD' (default: today)
        db_url: Database URL (default: DATABASE_URL env var)

    Returns:
        (df_5m, df_1h) DataFrames with ET-localized timestamps
    """
    db_url = db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError(
            "DATABASE_URL environment variable not set.\n"
            "Set it or pass db_url parameter."
        )

    # Determine date range
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.now(timezone.utc).replace(tzinfo=None)

    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    elif days:
        start_dt = end_dt - timedelta(days=days)
    else:
        start_dt = end_dt - timedelta(days=60)

    print(f"Fetching {symbol} data from database ...")
    print(f"  Date range: {start_dt.date()} to {end_dt.date()}")

    conn = psycopg2.connect(db_url)
    try:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = %s
              AND timeframe = '5m'
              AND timestamp >= %s
              AND timestamp <= %s
            ORDER BY timestamp
        """
        with conn.cursor() as cur:
            cur.execute(query, (symbol, start_dt, end_dt))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(
            f"No candles found for {symbol} in date range.\n"
            f"Run scripts/fetch_history.py to populate the candles table."
        )

    # Build DataFrame
    df_5m = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"])

    # Convert naive UTC to ET
    df_5m["timestamp"] = df_5m["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    df_5m.set_index("timestamp", inplace=True)

    # Convert Decimal to float
    for col in ["open", "high", "low", "close"]:
        df_5m[col] = df_5m[col].astype(float)
    df_5m["volume"] = df_5m["volume"].astype(int)

    # Drop duplicates (keep last)
    df_5m = df_5m[~df_5m.index.duplicated(keep="last")]

    # Aggregate 5m to 1h
    df_1h = df_5m.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    print(f"  Loaded {len(df_5m):,} 5m bars, aggregated to {len(df_1h):,} 1h bars")

    return df_5m, df_1h


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute all indicators on a DataFrame (vectorised)."""
    result = df.copy()

    ut = ut_bot_alert(result, sensitivity=cfg["ut_sensitivity"],
                      atr_period=cfg["ut_atr_period"],
                      use_heikin_ashi=cfg.get("use_heikin_ashi", False),
                      use_ha_atr=cfg.get("use_ha_atr", None),
                      confirmation_bars=cfg.get("ut_confirmation_bars", 0))
    result = pd.concat([result, ut], axis=1)

    st = supertrend(result, atr_period=cfg["st_atr_period"],
                    multiplier=cfg["st_multiplier"])
    result = pd.concat([result, st], axis=1)

    adx = calculate_adx(result, period=cfg["adx_period"])
    result = pd.concat([result, adx], axis=1)

    emas = ema_stack(result, periods=cfg["ema_periods"])
    result = pd.concat([result, emas], axis=1)

    bb = bollinger_squeeze(result, period=cfg["bb_period"],
                           std_dev=cfg["bb_std"],
                           squeeze_percentile=cfg["bb_squeeze_percentile"])
    result = pd.concat([result, bb], axis=1)

    result["atr"] = calculate_atr(result["high"], result["low"],
                                  result["close"], period=cfg["ut_atr_period"])
    return result


def align_htf(primary_df: pd.DataFrame, htf_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill HTF indicators onto the primary timeframe index."""
    return htf_df.reindex(primary_df.index, method="ffill")


# ---------------------------------------------------------------------------
# Entry / exit logic (matches proven Futures project signals.py)
# ---------------------------------------------------------------------------
def check_entry(row, idx, cfg, htf_filter, trend_filter) -> tuple:
    """Check entry conditions based on session mode."""
    start = cfg["trading_start"]
    end = cfg["trading_end"]

    # Must be within trading hours
    if not is_trading_hours(idx, start, end):
        return False, ""

    # Skip blocked hours
    blocked_hours = cfg.get("blocked_hours", [])
    if blocked_hours:
        local_hour = idx.astimezone(ET).hour if idx.tz else idx.hour
        if local_hour in blocked_hours:
            return False, ""

    if cfg["simple_mode"]:
        # Simple: UT Bot buy signal only
        if row.get("ut_buy_signal", False):
            return True, "Long entry: UT Bot buy signal"
        return False, ""
    else:
        # Full MTF filtered
        # 1. HTF bias
        if htf_filter is not None and not htf_filter.is_bullish(idx):
            return False, ""
        # 2. UT Bot buy signal
        if not row.get("ut_buy_signal", False):
            return False, ""
        # 3. Trend filter
        adx_val = row.get("adx", float("nan"))
        ema_aligned = row.get("ema_bullish_aligned", False)
        in_squeeze = row.get("bb_squeeze", False)
        if not trend_filter.is_valid_for_entry(adx_val, ema_aligned, in_squeeze):
            return False, ""
        return True, "Long entry: UT Bot signal, bullish HTF bias"


def check_exit(row, idx, entry_price, entry_atr, cfg) -> tuple:
    """
    Check exit conditions (matches proven Futures project signals.py order):
    1. End of session (5 min before close)
    2. Hard stop (entry - 2*ATR)
    3. Close below UT trailing stop (skipped if no_trailing_stop=True)
    4. UT Bot sell signal
    """
    price = row["close"]
    start = cfg["trading_start"]
    end = cfg["trading_end"]

    # 1. End of session
    if is_near_close(idx, end, start, minutes_before=5):
        return True, "End of day exit"

    # 2. Hard stop
    hard_stop = entry_price - (cfg["hard_stop_atr_mult"] * entry_atr)
    if price <= hard_stop:
        return True, f"Hard stop hit at {hard_stop:.2f}"

    # 3. Close below UT trailing stop (optional)
    if not cfg.get("no_trailing_stop", False):
        ut_stop = row.get("ut_trailing_stop", 0)
        if ut_stop > 0 and price < ut_stop:
            return True, f"Close below UT Bot trailing stop ({ut_stop:.2f})"

    # 4. UT Bot sell signal
    if row.get("ut_sell_signal", False):
        return True, "UT Bot sell signal"

    return False, ""


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(df_5m: pd.DataFrame, htf_aligned, cfg: dict,
                 session_label: str = ""):
    """
    Walk through 5m bars, check entry/exit, track trades & equity.
    Uses session-specific entry/exit logic.
    """
    capital = cfg["initial_capital"]
    position = None
    trades: list[Trade] = []
    equity_curve: list[tuple] = []

    sizer = PositionSizer(
        initial_capital=cfg["initial_capital"],
        risk_per_trade=cfg["risk_per_trade"],
        point_value=cfg["point_value"],
        commission=cfg["commission"],
        slippage_points=cfg["slippage"],
        margin_per_contract=cfg["margin_per_contract"],
        margin_buffer=cfg["margin_buffer"],
    )

    trend_filter = TrendFilter(
        adx_threshold=cfg["adx_threshold"],
        adx_min_trend=cfg["adx_min_trend"],
    )

    htf_filter = HTFFilter(htf_aligned) if htf_aligned is not None else None
    warmup = cfg["warmup_bars"]

    for i in range(warmup, len(df_5m)):
        row = df_5m.iloc[i]
        idx = df_5m.index[i]
        price = row["close"]

        # Skip bars where ATR is NaN
        if pd.isna(row.get("atr", float("nan"))):
            equity_curve.append((idx, capital))
            continue

        # --- EXIT ---
        if position is not None:
            # Check max risk hard stop first (before regular exit logic)
            unrealized = sizer.calculate_pnl(
                position["entry_price"], price, position["quantity"])
            max_risk = capital * cfg["risk_per_trade"]
            if not cfg.get("no_max_risk_stop") and unrealized <= -max_risk:
                should_exit = True
                exit_reason = f"Max risk stop: loss ${abs(unrealized):.2f} >= ${max_risk:.2f}"
            else:
                should_exit, exit_reason = check_exit(
                    row, idx, position["entry_price"],
                    position["entry_atr"], cfg)

            if should_exit:
                pnl = sizer.calculate_pnl(position["entry_price"], price,
                                           position["quantity"])
                capital += pnl
                trades.append(Trade(
                    entry_time=position["entry_time"],
                    exit_time=idx,
                    entry_price=position["entry_price"],
                    exit_price=price,
                    quantity=position["quantity"],
                    pnl=pnl,
                    entry_reason=position["entry_reason"],
                    exit_reason=exit_reason,
                    session=session_label,
                ))
                position = None

        # --- ENTRY ---
        if position is None:
            should_enter, entry_reason = check_entry(
                row, idx, cfg, htf_filter, trend_filter)

            if should_enter:
                atr_val = row.get("atr", 0)
                stop_loss = price - (cfg["hard_stop_atr_mult"] * atr_val) if atr_val > 0 else price - 10
                qty = sizer.calculate_size(capital, price, stop_loss)
                if qty > 0:
                    position = {
                        "entry_time": idx,
                        "entry_price": price,
                        "quantity": qty,
                        "entry_atr": atr_val,
                        "entry_reason": entry_reason,
                    }

        # Track equity
        if position is not None:
            unrealized = sizer.calculate_pnl(
                position["entry_price"], price, position["quantity"])
            equity_curve.append((idx, capital + unrealized))
        else:
            equity_curve.append((idx, capital))

    # Force-close open position at end
    if position is not None:
        final_price = df_5m.iloc[-1]["close"]
        pnl = sizer.calculate_pnl(position["entry_price"], final_price,
                                   position["quantity"])
        capital += pnl
        trades.append(Trade(
            entry_time=position["entry_time"],
            exit_time=df_5m.index[-1],
            entry_price=position["entry_price"],
            exit_price=final_price,
            quantity=position["quantity"],
            pnl=pnl,
            entry_reason=position["entry_reason"],
            exit_reason="End of backtest data",
            session=session_label,
        ))

    return trades, equity_curve


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def max_consecutive(values, predicate):
    best = current = 0
    for v in values:
        if predicate(v):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def calc_metrics(trades: list[Trade], equity_curve: list[tuple],
                 initial_capital: float) -> dict:
    if not trades:
        return {k: 0 for k in [
            "total_trades", "winning_trades", "win_rate", "total_pnl",
            "total_return", "profit_factor", "avg_trade", "avg_winner",
            "avg_loser", "largest_win", "largest_loss", "max_dd", "max_dd_pct",
            "sharpe", "avg_hold_hrs", "max_consec_wins", "max_consec_losses",
        ]}

    pnls = [t.pnl for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0

    # Max drawdown
    peak = initial_capital
    max_dd = max_dd_pct = 0.0
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak * 100 if peak > 0 else 0

    # Sharpe
    sharpe = 0.0
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve, columns=["ts", "equity"])
        eq_df["date"] = eq_df["ts"].dt.date
        daily_eq = eq_df.groupby("date")["equity"].last()
        daily_ret = daily_eq.pct_change().dropna()
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)

    hold_hrs = [(t.exit_time - t.entry_time).total_seconds() / 3600
                for t in trades]

    return {
        "total_trades": len(trades),
        "winning_trades": len(winners),
        "win_rate": len(winners) / len(trades) * 100 if trades else 0,
        "total_pnl": total_pnl,
        "total_return": total_pnl / initial_capital * 100,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_trade": total_pnl / len(trades),
        "avg_winner": sum(winners) / len(winners) if winners else 0,
        "avg_loser": sum(losers) / len(losers) if losers else 0,
        "largest_win": max(winners) if winners else 0,
        "largest_loss": min(losers) if losers else 0,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "sharpe": sharpe,
        "avg_hold_hrs": sum(hold_hrs) / len(hold_hrs) if hold_hrs else 0,
        "max_consec_wins": max_consecutive(pnls, lambda x: x > 0),
        "max_consec_losses": max_consecutive(pnls, lambda x: x <= 0),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def fmt(val, kind):
    if kind == "$":
        return f"${val:+,.2f}"
    if kind == "%":
        return f"{val:+.2f}%"
    if kind == "pct":
        return f"{val:.1f}%"
    if kind == "f2":
        return f"{val:.2f}"
    if kind == "hrs":
        return f"{val:.1f}h"
    if kind == "dd":
        return f"${val:,.2f}"
    if kind == "int":
        return f"{int(val)}"
    return str(val)


def print_session_results(metrics: dict, label: str):
    """Print results for a single session."""
    rows = [
        ("Total Trades",       "int",  "total_trades"),
        ("Winning Trades",     "int",  "winning_trades"),
        ("Win Rate",            "pct",  "win_rate"),
        ("Total P&L",           "$",    "total_pnl"),
        ("Total Return",        "%",    "total_return"),
        ("Profit Factor",       "f2",   "profit_factor"),
        ("Avg Trade P&L",       "$",    "avg_trade"),
        ("Avg Winner",          "$",    "avg_winner"),
        ("Avg Loser",           "$",    "avg_loser"),
        ("Largest Win",         "$",    "largest_win"),
        ("Largest Loss",        "$",    "largest_loss"),
        ("Max Drawdown",        "dd",   "max_dd"),
        ("Max Drawdown %",      "pct",  "max_dd_pct"),
        ("Sharpe Ratio",        "f2",   "sharpe"),
        ("Avg Holding Time",    "hrs",  "avg_hold_hrs"),
        ("Max Consec. Wins",    "int",  "max_consec_wins"),
        ("Max Consec. Losses",  "int",  "max_consec_losses"),
    ]
    print(f"\n  {label}")
    print("  " + "-" * 50)
    for name, kind, key in rows:
        print(f"  {name:<26} {fmt(metrics[key], kind):>22}")


def print_comparison_table(rth_m: dict, ovn_m: dict, both_m: dict):
    """Print 3-column comparison table."""
    rows = [
        ("Total Trades",       "int",  "total_trades"),
        ("Win Rate",            "pct",  "win_rate"),
        ("Total P&L",           "$",    "total_pnl"),
        ("Total Return",        "%",    "total_return"),
        ("Profit Factor",       "f2",   "profit_factor"),
        ("Avg Trade P&L",       "$",    "avg_trade"),
        ("Avg Winner",          "$",    "avg_winner"),
        ("Avg Loser",           "$",    "avg_loser"),
        ("Largest Win",         "$",    "largest_win"),
        ("Largest Loss",        "$",    "largest_loss"),
        ("Max Drawdown",        "dd",   "max_dd"),
        ("Max Drawdown %",      "pct",  "max_dd_pct"),
        ("Sharpe Ratio",        "f2",   "sharpe"),
        ("Avg Holding Time",    "hrs",  "avg_hold_hrs"),
    ]

    w = 84
    sep = "=" * w
    print(f"\n{sep}")
    print(f"{'BACKTEST RESULTS — BY SESSION':^{w}}")
    print(sep)
    print(f"{'Metric':<22} {'RTH (9:30-16:00)':>20} {'Overnight (16-9:30)':>20} {'Combined':>20}")
    print("-" * w)
    for label, kind, key in rows:
        rv = fmt(rth_m[key], kind)
        ov = fmt(ovn_m[key], kind)
        bv = fmt(both_m[key], kind)
        print(f"{label:<22} {rv:>20} {ov:>20} {bv:>20}")
    print(sep)


def print_monthly_breakdown(all_trades: list[Trade], initial_capital: float):
    """Break down results by month per session."""
    if not all_trades:
        return

    df = pd.DataFrame([{
        "session": t.session,
        "entry_time": t.entry_time,
        "pnl": t.pnl,
    } for t in all_trades])
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")

    print(f"\n{'MONTHLY BREAKDOWN':^84}")
    print("=" * 84)

    for session in ["rth", "overnight", "both"]:
        if session == "both":
            sdf = df.copy()
        else:
            sdf = df[df["session"] == session]

        if sdf.empty:
            continue

        label = {"rth": "RTH (9:30 AM - 4 PM)",
                 "overnight": "OVERNIGHT (4 PM - 9:30 AM)",
                 "both": "COMBINED (ALL SESSIONS)"}[session]

        print(f"\n  {label}")
        print(f"  {'Period':<15} {'Trades':>8} {'Wins':>8} {'Win Rate':>10} {'P&L':>14} {'Return':>10}")
        print("  " + "-" * 68)

        total_trades = total_wins = 0
        total_pnl = 0.0

        for month, group in sorted(sdf.groupby("month")):
            t = len(group)
            w = len(group[group["pnl"] > 0])
            pnl = group["pnl"].sum()
            wr = w / t if t > 0 else 0
            total_trades += t
            total_wins += w
            total_pnl += pnl
            print(f"  {month.strftime('%b %Y'):<15} {t:>8} {w:>8} {wr:>9.1%} ${pnl:>12,.2f} {pnl/initial_capital:>9.1%}")

        print("  " + "-" * 68)
        wr = total_wins / total_trades if total_trades > 0 else 0
        print(f"  {'60-DAY TOTAL':<15} {total_trades:>8} {total_wins:>8} {wr:>9.1%} "
              f"${total_pnl:>12,.2f} {total_pnl/initial_capital:>9.1%}")


def print_trades(trades: list[Trade], label: str, max_show: int = 20):
    print(f"\n--- {label} -- Last {min(len(trades), max_show)} of {len(trades)} Trades ---")
    if not trades:
        print("  No trades.")
        return
    print(f"  {'Entry Time':<22} {'Exit Time':<22} {'Entry':>10} {'Exit':>10} "
          f"{'Qty':>4} {'P&L':>12} {'Exit Reason'}")
    print("  " + "-" * 110)
    for t in trades[-max_show:]:
        etime = t.entry_time.strftime("%Y-%m-%d %H:%M")
        xtime = t.exit_time.strftime("%Y-%m-%d %H:%M")
        print(f"  {etime:<22} {xtime:<22} {t.entry_price:>10.2f} {t.exit_price:>10.2f} "
              f"{t.quantity:>4} {t.pnl:>+12.2f} {t.exit_reason}")


def plot_equity(rth_curve, ovn_curve, combined_curve, initial_capital):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed -- skipping equity curve plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    curves = [
        (rth_curve, "RTH", "royalblue"),
        (ovn_curve, "Overnight", "darkorange"),
        (combined_curve, "Combined", "forestgreen"),
    ]

    for curve, label, color in curves:
        if not curve:
            continue
        ts, eq = zip(*curve)
        axes[0].plot(ts, eq, label=label, color=color, linewidth=0.8)

    axes[0].axhline(y=initial_capital, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_title("GenNyx Backtest -- /MNQ (RTH / Overnight / Combined)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    for curve, label, color in curves:
        if not curve:
            continue
        ts, equities = zip(*curve)
        peak = equities[0]
        dd = []
        for eq in equities:
            if eq > peak:
                peak = eq
            dd.append((peak - eq) / peak * 100 if peak > 0 else 0)
        axes[1].fill_between(ts, dd, alpha=0.2, color=color)
        axes[1].plot(ts, dd, color=color, linewidth=0.6, label=label)

    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent.parent / "backtest_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nEquity curve saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(days: int = None, start_date: str = None, end_date: str = None,
         sensitivity: float = None, no_trailing_stop: bool = False,
         blocked_hours: list = None):
    cfg = BASE_CONFIG.copy()
    if days:
        cfg["days"] = days
    if sensitivity:
        cfg["ut_sensitivity"] = sensitivity
    cfg["no_trailing_stop"] = no_trailing_stop
    if blocked_hours:
        cfg["blocked_hours"] = blocked_hours
    cap = cfg["initial_capital"]

    print("=" * 84)
    print(f"{'GenNyx Backtester -- /MNQ (Database-Backed)':^84}")
    print("=" * 84)
    print(f"  Symbol:           {cfg['symbol']}")
    print(f"  Initial Capital:  ${cap:,.2f}")
    print(f"  Risk Per Trade:   {cfg['risk_per_trade']:.0%}")
    print(f"  Point Value:      ${cfg['point_value']}")
    print(f"  UT Bot:           sens={cfg['ut_sensitivity']}, ATR={cfg['ut_atr_period']}, confirm={cfg['ut_confirmation_bars']}")
    exit_mode = "UT SELL signal only" if cfg.get("no_trailing_stop") else "Trailing stop + UT SELL"
    print(f"  Exit Mode:        {exit_mode}")
    blocked = cfg.get("blocked_hours", [])
    if blocked:
        print(f"  Blocked Hours:    {sorted(blocked)} ET")
    print(f"  RTH:              9:30-16:00 ET  | Full MTF filtered, regular candles")
    print(f"  Overnight:        16:00-09:30 ET | Simple UT Bot, Heikin-Ashi candles")
    print()

    # 1. Fetch data from database
    t0 = time.time()
    df_5m_raw, df_1h_raw = fetch_data(
        cfg["symbol"],
        days=cfg["days"],
        start_date=start_date,
        end_date=end_date,
    )
    print(f"  5m bars:  {len(df_5m_raw):,}  ({df_5m_raw.index[0].strftime('%Y-%m-%d')} to "
          f"{df_5m_raw.index[-1].strftime('%Y-%m-%d')})")
    print(f"  1h bars:  {len(df_1h_raw):,}  ({df_1h_raw.index[0].strftime('%Y-%m-%d')} to "
          f"{df_1h_raw.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Fetch time: {time.time() - t0:.1f}s")

    if len(df_5m_raw) < cfg["warmup_bars"] + 50:
        print(f"ERROR: Not enough 5m data. Need {cfg['warmup_bars'] + 50}, got {len(df_5m_raw)}.")
        sys.exit(1)

    # 2. Compute indicators (overnight only - RTH disabled due to poor performance)
    all_trades: list[Trade] = []
    rth_trades, rth_curve = [], []

    print("\n--- RTH Session: DISABLED (underperforming) ---")

    # --- Overnight: Heikin-Ashi candles, simple UT Bot ---
    print("\n--- Overnight Session (Simple UT Bot + Heikin-Ashi) ---")
    ovn_cfg = {**cfg, **SESSION_CONFIGS["overnight"]}

    print("  Computing indicators (Heikin-Ashi) ...", end=" ", flush=True)
    t0 = time.time()
    df_5m_ovn = add_indicators(df_5m_raw, ovn_cfg)
    df_1h_ovn = add_indicators(df_1h_raw, ovn_cfg)
    htf_ovn = align_htf(df_5m_ovn, df_1h_ovn)
    print(f"done ({time.time() - t0:.1f}s)")

    print("  Running backtest ...", end=" ", flush=True)
    t0 = time.time()
    ovn_trades, ovn_curve = run_backtest(df_5m_ovn, htf_ovn, ovn_cfg, "overnight")
    print(f"done ({time.time() - t0:.1f}s) -- {len(ovn_trades)} trades")
    all_trades.extend(ovn_trades)

    # --- Metrics ---
    ovn_m = calc_metrics(ovn_trades, ovn_curve, cap)

    # 3. Output
    print_session_results(ovn_m, "Overnight Only (sens=%.1f)" % cfg["ut_sensitivity"])
    print_monthly_breakdown(all_trades, cap)
    print_trades(ovn_trades, "Overnight Trades")

    # 4. Equity curve
    plot_equity(rth_curve, ovn_curve, ovn_curve, cap)

    # 6. Export trades CSV
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    trades_df = pd.DataFrame([{
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "quantity": t.quantity,
        "pnl": t.pnl,
        "entry_reason": t.entry_reason,
        "exit_reason": t.exit_reason,
        "session": t.session,
    } for t in all_trades])
    trades_file = results_dir / "trades_combined_60d.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"\nTrades exported to {trades_file}")

    print("\nBacktest complete.")


def sweep_sensitivity(days: int = None, start_date: str = None, end_date: str = None):
    """Run backtest across multiple UT Bot sensitivity values (overnight only)."""
    sensitivities = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    cfg = BASE_CONFIG.copy()
    cap = cfg["initial_capital"]

    print("=" * 90)
    print(f"{'UT Bot Sensitivity Sweep -- Overnight Only (Database-Backed)':^90}")
    print("=" * 90)

    # Fetch data once from database
    df_5m_raw, df_1h_raw = fetch_data(
        cfg["symbol"],
        days=days or cfg["days"],
        start_date=start_date,
        end_date=end_date,
    )
    print(f"  5m bars: {len(df_5m_raw):,}  |  1h bars: {len(df_1h_raw):,}\n")

    results = []
    for sens in sensitivities:
        cfg_run = {**cfg, **SESSION_CONFIGS["overnight"]}
        cfg_run["ut_sensitivity"] = sens

        df_5m = add_indicators(df_5m_raw, cfg_run)
        df_1h = add_indicators(df_1h_raw, cfg_run)
        htf = align_htf(df_5m, df_1h)

        trades, curve = run_backtest(df_5m, htf, cfg_run, "overnight")
        m = calc_metrics(trades, curve, cap)
        m["sensitivity"] = sens
        results.append(m)
        print(f"  sens={sens:.1f}  trades={m['total_trades']:>4}  "
              f"wr={m['win_rate']:>5.1f}%  pnl=${m['total_pnl']:>+10,.2f}  "
              f"pf={m['profit_factor']:>5.2f}  dd={m['max_dd_pct']:>5.1f}%  "
              f"sharpe={m['sharpe']:>+5.2f}")

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"{'Sens':>6} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'PnL':>12} "
          f"{'PF':>6} {'AvgWin':>9} {'AvgLoss':>9} {'MaxDD%':>7} {'Sharpe':>7}")
    print("-" * 90)
    for m in results:
        print(f"{m['sensitivity']:>6.1f} {m['total_trades']:>7} {m['winning_trades']:>6} "
              f"{m['win_rate']:>6.1f}% ${m['total_pnl']:>+10,.2f} "
              f"{m['profit_factor']:>6.2f} ${m['avg_winner']:>+8,.2f} ${m['avg_loser']:>+8,.2f} "
              f"{m['max_dd_pct']:>6.1f}% {m['sharpe']:>+7.2f}")
    print("=" * 90)


def compare_atr_modes(days: int = None, start_date: str = None, end_date: str = None):
    """Compare HA ATR vs Raw ATR backtest results side-by-side."""
    cfg = BASE_CONFIG.copy()
    cap = cfg["initial_capital"]

    w = 90
    print("=" * w)
    print(f"{'HA ATR vs Raw ATR Comparison -- Overnight Session (Database-Backed)':^{w}}")
    print("=" * w)
    print(f"  Symbol:           {cfg['symbol']}")
    print(f"  Initial Capital:  ${cap:,.2f}")
    print(f"  UT Sensitivity:   {cfg['ut_sensitivity']}")
    print(f"  ATR Period:       {cfg['ut_atr_period']}")
    print()

    # Fetch data once from database
    df_5m_raw, df_1h_raw = fetch_data(
        cfg["symbol"],
        days=days or cfg["days"],
        start_date=start_date,
        end_date=end_date,
    )
    print(f"  5m bars: {len(df_5m_raw):,}  |  1h bars: {len(df_1h_raw):,}\n")

    modes = [
        ("HA ATR (TOS match)", True),
        ("Raw ATR (original)", False),
    ]

    results = {}
    all_trades = {}
    all_curves = {}

    for label, ha_atr in modes:
        ovn_cfg = {**cfg, **SESSION_CONFIGS["overnight"], "use_ha_atr": ha_atr}

        print(f"  Running: {label} ...", end=" ", flush=True)
        t0 = time.time()
        df_5m = add_indicators(df_5m_raw, ovn_cfg)
        df_1h = add_indicators(df_1h_raw, ovn_cfg)
        htf = align_htf(df_5m, df_1h)
        trades, curve = run_backtest(df_5m, htf, ovn_cfg, "overnight")
        m = calc_metrics(trades, curve, cap)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) -- {len(trades)} trades")

        results[label] = m
        all_trades[label] = trades
        all_curves[label] = curve

    # Comparison table
    labels = list(results.keys())
    m1, m2 = results[labels[0]], results[labels[1]]

    rows = [
        ("Total Trades",       "int",  "total_trades"),
        ("Winning Trades",     "int",  "winning_trades"),
        ("Win Rate",            "pct",  "win_rate"),
        ("Total P&L",           "$",    "total_pnl"),
        ("Total Return",        "%",    "total_return"),
        ("Profit Factor",       "f2",   "profit_factor"),
        ("Avg Trade P&L",       "$",    "avg_trade"),
        ("Avg Winner",          "$",    "avg_winner"),
        ("Avg Loser",           "$",    "avg_loser"),
        ("Largest Win",         "$",    "largest_win"),
        ("Largest Loss",        "$",    "largest_loss"),
        ("Max Drawdown",        "dd",   "max_dd"),
        ("Max Drawdown %",      "pct",  "max_dd_pct"),
        ("Sharpe Ratio",        "f2",   "sharpe"),
        ("Avg Holding Time",    "hrs",  "avg_hold_hrs"),
        ("Max Consec. Wins",    "int",  "max_consec_wins"),
        ("Max Consec. Losses",  "int",  "max_consec_losses"),
    ]

    print(f"\n{'=' * w}")
    print(f"{'COMPARISON: HA ATR vs Raw ATR':^{w}}")
    print(f"{'=' * w}")
    print(f"{'Metric':<24} {'HA ATR (TOS match)':>22} {'Raw ATR (original)':>22} {'Difference':>20}")
    print("-" * w)

    for name, kind, key in rows:
        v1 = m1[key]
        v2 = m2[key]
        s1 = fmt(v1, kind)
        s2 = fmt(v2, kind)
        # Compute difference
        diff = v1 - v2
        if kind == "$":
            sdiff = f"${diff:+,.2f}"
        elif kind in ("%", "pct"):
            sdiff = f"{diff:+.1f}%"
        elif kind == "f2":
            sdiff = f"{diff:+.2f}"
        elif kind == "hrs":
            sdiff = f"{diff:+.1f}h"
        elif kind == "dd":
            sdiff = f"${diff:+,.2f}"
        elif kind == "int":
            sdiff = f"{diff:+.0f}"
        else:
            sdiff = f"{diff:+.2f}"
        print(f"{name:<24} {s1:>22} {s2:>22} {sdiff:>20}")

    print("=" * w)

    # Monthly breakdown per mode
    for label in labels:
        trades = all_trades[label]
        if not trades:
            continue
        df = pd.DataFrame([{
            "session": t.session,
            "entry_time": t.entry_time,
            "pnl": t.pnl,
        } for t in trades])
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["month"] = df["entry_time"].dt.to_period("M")

        print(f"\n  {label} -- Monthly")
        print(f"  {'Period':<15} {'Trades':>8} {'Wins':>8} {'Win Rate':>10} {'P&L':>14} {'Return':>10}")
        print("  " + "-" * 68)
        for month, group in sorted(df.groupby("month")):
            t = len(group)
            wins = len(group[group["pnl"] > 0])
            pnl = group["pnl"].sum()
            wr = wins / t if t > 0 else 0
            print(f"  {month.strftime('%b %Y'):<15} {t:>8} {wins:>8} {wr:>9.1%} ${pnl:>12,.2f} {pnl/cap:>9.1%}")

    # Equity curve with both modes
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        colors = {"HA ATR (TOS match)": "royalblue", "Raw ATR (original)": "darkorange"}
        for label in labels:
            curve = all_curves[label]
            if not curve:
                continue
            ts, eq = zip(*curve)
            axes[0].plot(ts, eq, label=label, color=colors[label], linewidth=0.8)

        axes[0].axhline(y=cap, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("Equity ($)")
        axes[0].set_title("HA ATR vs Raw ATR -- Overnight Session (UT Bot Heikin-Ashi)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        for label in labels:
            curve = all_curves[label]
            if not curve:
                continue
            ts, equities = zip(*curve)
            peak = equities[0]
            dd = []
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd.append((peak - eq) / peak * 100 if peak > 0 else 0)
            axes[1].fill_between(ts, dd, alpha=0.15, color=colors[label])
            axes[1].plot(ts, dd, color=colors[label], linewidth=0.6, label=label)

        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_xlabel("Date")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)
        axes[1].invert_yaxis()

        plt.tight_layout()
        out_path = Path(__file__).resolve().parent.parent / "backtest_atr_compare.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nComparison chart saved to {out_path}")
        plt.close()
    except ImportError:
        print("\nmatplotlib not installed -- skipping chart.")

    print("\nComparison complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GenNyx Backtester")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "sweep", "compare"],
                        help="Command to run (default: run)")
    parser.add_argument("--days", type=int, default=None,
                        help="Number of days to backtest (default: 60)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--sensitivity", type=float, default=None,
                        help="UT Bot sensitivity (default: 3.5)")
    parser.add_argument("--no-trailing-stop", action="store_true",
                        help="Exit only on UT SELL signal (disable trailing stop exit)")
    parser.add_argument("--blocked-hours", type=str, default=None,
                        help="Comma-separated hours to skip entries (e.g., '1,5,7,8,9,21,22,23')")
    args = parser.parse_args()

    # Parse blocked hours
    blocked_hours = None
    if args.blocked_hours:
        blocked_hours = [int(h.strip()) for h in args.blocked_hours.split(",")]

    if args.command == "sweep":
        sweep_sensitivity(days=args.days, start_date=args.start, end_date=args.end)
    elif args.command == "compare":
        compare_atr_modes(days=args.days, start_date=args.start, end_date=args.end)
    else:
        main(days=args.days, start_date=args.start, end_date=args.end,
             sensitivity=args.sensitivity, no_trailing_stop=args.no_trailing_stop,
             blocked_hours=blocked_hours)
