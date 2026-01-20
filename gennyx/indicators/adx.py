"""ADX (Average Directional Index) indicator implementation."""

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate ADX (Average Directional Index).

    ADX measures trend strength regardless of direction:
    - ADX < 20: Weak trend / ranging market
    - ADX 20-25: Developing trend
    - ADX 25-50: Strong trend
    - ADX > 50: Very strong trend

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period for ADX calculation

    Returns:
        DataFrame with columns:
        - adx: The ADX value
        - plus_di: +DI (Plus Directional Indicator)
        - minus_di: -DI (Minus Directional Indicator)
    """
    result = pd.DataFrame(index=df.index)

    if ta is not None:
        # Use pandas_ta for ADX calculation
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=period)
        if adx_df is not None and not adx_df.empty:
            result["adx"] = adx_df[f"ADX_{period}"]
            result["plus_di"] = adx_df[f"DMP_{period}"]
            result["minus_di"] = adx_df[f"DMN_{period}"]
            return result

    # Manual calculation as fallback
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Calculate True Range
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Smoothed values using Wilder's smoothing
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, min_periods=period).mean()

    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr

    # Calculate DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

    result["adx"] = adx
    result["plus_di"] = plus_di
    result["minus_di"] = minus_di

    return result
