"""Bollinger Bands squeeze detection implementation."""

import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
except ImportError:
    ta = None


def bollinger_squeeze(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, squeeze_percentile: int = 25
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands and detect squeeze conditions.

    A Bollinger Band squeeze indicates low volatility and often precedes
    a significant price move. Squeeze is detected when bandwidth is in
    the lowest percentile of its historical range.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period for BB calculation
        std_dev: Number of standard deviations for bands
        squeeze_percentile: Percentile threshold for squeeze detection

    Returns:
        DataFrame with columns:
        - bb_upper: Upper Bollinger Band
        - bb_middle: Middle band (SMA)
        - bb_lower: Lower Bollinger Band
        - bb_bandwidth: Bandwidth (upper - lower) / middle
        - bb_percent_b: %B indicator (where price is within bands)
        - bb_squeeze: Boolean for squeeze condition
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]

    if ta is not None:
        # Use pandas_ta for Bollinger Bands
        bb = ta.bbands(close, length=period, std=std_dev)
        if bb is not None and not bb.empty:
            # Find columns dynamically (pandas_ta naming varies by version)
            cols = bb.columns.tolist()
            upper_col = [c for c in cols if c.startswith("BBU_")][0]
            middle_col = [c for c in cols if c.startswith("BBM_")][0]
            lower_col = [c for c in cols if c.startswith("BBL_")][0]
            bw_col = [c for c in cols if c.startswith("BBB_")][0]
            pct_col = [c for c in cols if c.startswith("BBP_")][0]

            result["bb_upper"] = bb[upper_col]
            result["bb_middle"] = bb[middle_col]
            result["bb_lower"] = bb[lower_col]
            result["bb_bandwidth"] = bb[bw_col]
            result["bb_percent_b"] = bb[pct_col]
        else:
            # Fallback to manual calculation
            result = _calculate_bb_manual(close, period, std_dev)
    else:
        result = _calculate_bb_manual(close, period, std_dev)

    # Calculate squeeze condition
    # Squeeze = bandwidth is in lowest N percentile of rolling window
    lookback = period * 5  # Use 5x period for percentile calculation
    bandwidth = result["bb_bandwidth"]

    # Rolling percentile rank
    def percentile_rank(series, window):
        """Calculate rolling percentile rank."""
        ranks = series.rolling(window).apply(
            lambda x: (x.iloc[-1] > x[:-1]).sum() / (len(x) - 1) * 100
            if len(x) > 1
            else 50,
            raw=False,
        )
        return ranks

    bw_rank = percentile_rank(bandwidth, lookback)
    result["bb_squeeze"] = (bw_rank <= squeeze_percentile).fillna(False)

    return result


def _calculate_bb_manual(close: pd.Series, period: int, std_dev: float) -> pd.DataFrame:
    """Manual Bollinger Bands calculation."""
    result = pd.DataFrame(index=close.index)

    # Middle band = SMA
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    result["bb_middle"] = sma
    result["bb_upper"] = sma + (std_dev * std)
    result["bb_lower"] = sma - (std_dev * std)

    # Bandwidth = (Upper - Lower) / Middle
    result["bb_bandwidth"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]

    # %B = (Price - Lower) / (Upper - Lower)
    result["bb_percent_b"] = (close - result["bb_lower"]) / (
        result["bb_upper"] - result["bb_lower"]
    )

    return result
