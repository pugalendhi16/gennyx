"""HTF bias and trend filter implementations."""

from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd
import pytz


class HTFFilter:
    """Higher timeframe bias filter."""

    def __init__(self, htf_data: pd.DataFrame):
        """
        Initialize HTF filter with pre-calculated HTF indicators.

        Args:
            htf_data: DataFrame with HTF indicator values aligned to primary timeframe
        """
        self.htf_data = htf_data

    def get_bias(self, idx: pd.Timestamp) -> str:
        """
        Get HTF bias at a specific timestamp.

        Bullish bias: UT Bot trailing stop < price OR Supertrend line < price
        Bearish bias: UT Bot trailing stop > price AND Supertrend line > price

        Args:
            idx: Timestamp to check

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if idx not in self.htf_data.index:
            return "neutral"

        row = self.htf_data.loc[idx]

        # Check if we have the required columns
        has_ut = "ut_trailing_stop" in row.index and "close" in row.index
        has_st = "st_line" in row.index and "st_direction" in row.index

        ut_bullish = False
        st_bullish = False

        if has_ut:
            ut_bullish = row["close"] > row["ut_trailing_stop"]

        if has_st:
            st_bullish = row["st_direction"] == 1

        # Bullish if either condition is met
        if ut_bullish or st_bullish:
            return "bullish"

        # Bearish if neither condition is met
        return "bearish"

    def is_bullish(self, idx: pd.Timestamp) -> bool:
        """Check if HTF bias is bullish."""
        return self.get_bias(idx) == "bullish"

    def is_bearish(self, idx: pd.Timestamp) -> bool:
        """Check if HTF bias is bearish."""
        return self.get_bias(idx) == "bearish"


class TrendFilter:
    """Trend strength and ranging market filter."""

    def __init__(
        self,
        adx_threshold: int = 22,
        adx_min_trend: int = 20,
    ):
        """
        Initialize trend filter.

        Args:
            adx_threshold: ADX value above which trend is confirmed
            adx_min_trend: Minimum ADX for non-ranging market
        """
        self.adx_threshold = adx_threshold
        self.adx_min_trend = adx_min_trend

    def has_trend_confirmation(
        self, adx: float, ema_aligned: bool
    ) -> bool:
        """
        Check if trend confirmation exists.

        Trend confirmed if: ADX > threshold OR EMA alignment

        Args:
            adx: Current ADX value
            ema_aligned: Whether EMAs are aligned (bullish or bearish)

        Returns:
            True if trend is confirmed
        """
        if pd.isna(adx):
            return ema_aligned

        return adx > self.adx_threshold or ema_aligned

    def is_ranging_market(self, adx: float, in_squeeze: bool) -> bool:
        """
        Check if market is ranging (filter out).

        Ranging market: ADX < min_trend OR in Bollinger Band squeeze

        Args:
            adx: Current ADX value
            in_squeeze: Whether Bollinger Bands are in squeeze

        Returns:
            True if market is ranging
        """
        if pd.isna(adx):
            return in_squeeze

        return adx < self.adx_min_trend or in_squeeze

    def is_valid_for_entry(
        self, adx: float, ema_aligned: bool, in_squeeze: bool
    ) -> bool:
        """
        Check if conditions are valid for entry.

        Valid entry requires:
        - Trend confirmation (ADX > threshold OR EMA aligned)
        - NOT a ranging market (ADX >= min_trend AND NOT in squeeze)

        Args:
            adx: Current ADX value
            ema_aligned: Whether EMAs are aligned
            in_squeeze: Whether Bollinger Bands are in squeeze

        Returns:
            True if conditions are valid for entry
        """
        has_trend = self.has_trend_confirmation(adx, ema_aligned)
        is_ranging = self.is_ranging_market(adx, in_squeeze)

        return has_trend and not is_ranging


class TradingHoursFilter:
    """Filter for trading hours (supports regular and overnight sessions)."""

    def __init__(
        self,
        start_time: str = "09:30",
        end_time: str = "16:00",
        timezone: str = "America/New_York",
    ):
        """
        Initialize trading hours filter.

        Args:
            start_time: Trading start time (HH:MM)
            end_time: Trading end time (HH:MM)
            timezone: Timezone for trading hours
        """
        self.start = datetime.strptime(start_time, "%H:%M").time()
        self.end = datetime.strptime(end_time, "%H:%M").time()
        self.tz = pytz.timezone(timezone)
        self.is_overnight = self.start > self.end

    def is_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp is within trading hours.

        Supports both regular hours (start < end) and overnight sessions (start > end).

        Args:
            timestamp: Timestamp to check

        Returns:
            True if within trading hours
        """
        # Convert to target timezone if needed
        if timestamp.tz is not None:
            local_time = timestamp.astimezone(self.tz).time()
        else:
            local_time = timestamp.time()

        if self.is_overnight:
            # Overnight session (e.g., 16:00 to 09:30)
            return local_time >= self.start or local_time < self.end
        else:
            # Regular hours (e.g., 09:30 to 16:00)
            return self.start <= local_time < self.end

    def is_near_close(
        self, timestamp: pd.Timestamp, minutes_before: int = 5
    ) -> bool:
        """
        Check if timestamp is near session close.

        Args:
            timestamp: Timestamp to check
            minutes_before: Minutes before close to consider "near"

        Returns:
            True if near session close
        """
        if timestamp.tz is not None:
            local_time = timestamp.astimezone(self.tz)
        else:
            local_time = timestamp

        # Create close time on same day
        close_dt = local_time.replace(
            hour=self.end.hour, minute=self.end.minute, second=0, microsecond=0
        )

        # For overnight sessions, if current time is after start (e.g., 4 PM),
        # the close is on the next day
        if self.is_overnight and local_time.time() >= self.start:
            close_dt = close_dt + timedelta(days=1)

        # Calculate minutes until close
        delta = close_dt - local_time
        minutes_to_close = delta.total_seconds() / 60

        return 0 <= minutes_to_close <= minutes_before
