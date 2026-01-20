"""Supertrend indicator implementation."""

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


def supertrend(
    df: pd.DataFrame, atr_period: int = 8, multiplier: float = 2.5
) -> pd.DataFrame:
    """
    Supertrend indicator.

    Supertrend is a trend-following indicator that uses ATR to determine
    support/resistance levels. It flips between upper and lower bands
    based on price action.

    Args:
        df: DataFrame with OHLCV data
        atr_period: Period for ATR calculation
        multiplier: ATR multiplier for band distance

    Returns:
        DataFrame with columns:
        - st_line: The Supertrend line
        - st_direction: 1 for uptrend (bullish), -1 for downtrend (bearish)
        - st_upper_band: Upper band
        - st_lower_band: Lower band
    """
    result = pd.DataFrame(index=df.index)

    # Calculate ATR
    if ta is not None:
        atr = ta.atr(df["high"], df["low"], df["close"], length=atr_period)
    else:
        atr = calculate_atr(df["high"], df["low"], df["close"], period=atr_period)

    # Calculate basic bands
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # Initialize arrays
    n = len(df)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend_line = np.zeros(n)
    direction = np.zeros(n)

    # First values
    upper_band[0] = basic_upper.iloc[0] if not pd.isna(basic_upper.iloc[0]) else 0
    lower_band[0] = basic_lower.iloc[0] if not pd.isna(basic_lower.iloc[0]) else 0
    supertrend_line[0] = upper_band[0]
    direction[0] = -1  # Start bearish

    # Calculate Supertrend iteratively
    for i in range(1, n):
        curr_close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]
        curr_basic_upper = basic_upper.iloc[i]
        curr_basic_lower = basic_lower.iloc[i]

        if pd.isna(curr_basic_upper) or pd.isna(curr_basic_lower):
            upper_band[i] = upper_band[i - 1]
            lower_band[i] = lower_band[i - 1]
            supertrend_line[i] = supertrend_line[i - 1]
            direction[i] = direction[i - 1]
            continue

        # Final upper band
        if curr_basic_upper < upper_band[i - 1] or prev_close > upper_band[i - 1]:
            upper_band[i] = curr_basic_upper
        else:
            upper_band[i] = upper_band[i - 1]

        # Final lower band
        if curr_basic_lower > lower_band[i - 1] or prev_close < lower_band[i - 1]:
            lower_band[i] = curr_basic_lower
        else:
            lower_band[i] = lower_band[i - 1]

        # Determine direction and supertrend line
        if direction[i - 1] == -1:  # Was bearish
            if curr_close > upper_band[i - 1]:
                direction[i] = 1  # Flip to bullish
                supertrend_line[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend_line[i] = upper_band[i]
        else:  # Was bullish
            if curr_close < lower_band[i - 1]:
                direction[i] = -1  # Flip to bearish
                supertrend_line[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend_line[i] = lower_band[i]

    result["st_line"] = supertrend_line
    result["st_direction"] = direction
    result["st_upper_band"] = upper_band
    result["st_lower_band"] = lower_band

    return result
