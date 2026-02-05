"""UT Bot Alert indicator implementation (QuantNomad version)."""

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10
) -> pd.Series:
    """Calculate Average True Range manually as fallback."""
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def ut_bot_alert(
    df: pd.DataFrame,
    sensitivity: float = 1.2,
    atr_period: int = 10,
    use_heikin_ashi: bool = False,
) -> pd.DataFrame:
    """
    UT Bot Alert indicator (QuantNomad TradingView version).

    This indicator uses ATR-based trailing stops to generate buy/sell signals.
    A buy signal is generated when price crosses above the trailing stop.
    A sell signal is generated when price crosses below the trailing stop.

    Args:
        df: DataFrame with OHLCV data
        sensitivity: ATR multiplier for trailing stop distance (key parameter)
        atr_period: Period for ATR calculation
        use_heikin_ashi: Whether to use Heikin-Ashi source for calculations

    Returns:
        DataFrame with columns:
        - ut_trailing_stop: The trailing stop level
        - ut_buy_signal: Boolean buy signal
        - ut_sell_signal: Boolean sell signal
        - ut_trend: 1 for uptrend, -1 for downtrend
    """
    result = pd.DataFrame(index=df.index)

    # Use Heikin-Ashi if specified (matches TOS: ATR also uses HA OHLC)
    if use_heikin_ashi:
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha_open = (df["open"].shift(1) + df["close"].shift(1)) / 2
        ha_open.iloc[0] = df["open"].iloc[0]
        ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
        src = ha_close
        atr_high, atr_low, atr_close = ha_high, ha_low, ha_close
    else:
        src = df["close"].copy()
        atr_high, atr_low, atr_close = df["high"], df["low"], df["close"]

    # Calculate ATR
    if ta is not None:
        atr = ta.atr(atr_high, atr_low, atr_close, length=atr_period)
    else:
        atr = calculate_atr(atr_high, atr_low, atr_close, period=atr_period)

    # ATR trailing stop distance
    n_loss = sensitivity * atr

    # Initialize arrays
    n = len(df)
    trailing_stop = np.zeros(n)
    trend = np.zeros(n)

    # Calculate trailing stop iteratively
    for i in range(1, n):
        # Previous values
        prev_stop = trailing_stop[i - 1]
        prev_close = src.iloc[i - 1]
        curr_close = src.iloc[i]
        curr_nloss = n_loss.iloc[i]

        if pd.isna(curr_nloss):
            trailing_stop[i] = prev_stop
            trend[i] = trend[i - 1]
            continue

        # Trailing stop logic
        if curr_close > prev_stop:
            # Price above stop - uptrend
            trailing_stop[i] = max(prev_stop, curr_close - curr_nloss)
            trend[i] = 1
        else:
            # Price below stop - downtrend
            trailing_stop[i] = min(prev_stop, curr_close + curr_nloss)
            trend[i] = -1

    result["ut_trailing_stop"] = trailing_stop
    result["ut_trend"] = trend

    # Generate signals on trend change
    trend_series = pd.Series(trend, index=df.index)
    prev_trend = trend_series.shift(1)

    # Buy signal: trend changes from -1 to 1 (or 0 to 1)
    result["ut_buy_signal"] = (trend_series == 1) & (prev_trend != 1)

    # Sell signal: trend changes from 1 to -1 (or 0 to -1)
    result["ut_sell_signal"] = (trend_series == -1) & (prev_trend != -1)

    return result
