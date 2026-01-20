"""EMA (Exponential Moving Average) stack implementation."""

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


def ema_stack(
    df: pd.DataFrame, periods: tuple = (9, 20, 50)
) -> pd.DataFrame:
    """
    Calculate EMA stack for multiple periods.

    The EMA stack is used to determine trend alignment:
    - Bullish alignment: Close > EMA9 > EMA20 > EMA50
    - Bearish alignment: Close < EMA9 < EMA20 < EMA50

    Args:
        df: DataFrame with OHLCV data
        periods: Tuple of EMA periods (fast, medium, slow)

    Returns:
        DataFrame with columns:
        - ema_{period}: EMA values for each period
        - ema_bullish_aligned: Boolean for bullish alignment
        - ema_bearish_aligned: Boolean for bearish alignment
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]

    # Calculate EMAs
    emas = []
    for period in periods:
        if ta is not None:
            ema = ta.ema(close, length=period)
            # pandas_ta returns None if not enough data, fall back to manual
            if ema is None:
                ema = close.ewm(span=period, adjust=False).mean()
        else:
            ema = close.ewm(span=period, adjust=False).mean()

        result[f"ema_{period}"] = ema
        emas.append(ema)

    # Check alignment (assuming periods are ordered fast to slow)
    if len(periods) >= 3:
        ema_fast = result[f"ema_{periods[0]}"]
        ema_mid = result[f"ema_{periods[1]}"]
        ema_slow = result[f"ema_{periods[2]}"]

        # Bullish: Close > EMA_fast > EMA_mid > EMA_slow
        # Use fillna(False) to handle NaN values from insufficient data
        result["ema_bullish_aligned"] = (
            (close > ema_fast) & (ema_fast > ema_mid) & (ema_mid > ema_slow)
        ).fillna(False)

        # Bearish: Close < EMA_fast < EMA_mid < EMA_slow
        result["ema_bearish_aligned"] = (
            (close < ema_fast) & (ema_fast < ema_mid) & (ema_mid < ema_slow)
        ).fillna(False)
    else:
        result["ema_bullish_aligned"] = False
        result["ema_bearish_aligned"] = False

    return result
