"""Build OHLCV candles from live quotes."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import pytz

from .data_feed import Quote


@dataclass
class Candle:
    """Represents a single OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Candle":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
        )


@dataclass
class CandleState:
    """Current building candle state."""

    start_time: datetime
    open: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    close: float = 0.0
    volume: int = 0
    tick_count: int = 0

    def update(self, price: float, volume: int = 0):
        """Update candle with new price."""
        if self.tick_count == 0:
            self.open = price
            self.high = price
            self.low = price
        else:
            self.high = max(self.high, price)
            self.low = min(self.low, price)

        self.close = price
        self.volume += volume
        self.tick_count += 1

    def to_candle(self) -> Candle:
        """Convert to completed candle."""
        return Candle(
            timestamp=self.start_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


class LiveDataBuilder:
    """
    Builds OHLCV candles from streaming quotes.

    Maintains rolling history for indicator calculation and supports
    multiple timeframes (5m primary, 1h HTF).
    """

    def __init__(
        self,
        primary_tf_minutes: int = 5,
        htf_minutes: int = 60,
        max_history: int = 200,
        timezone: str = "America/New_York",
    ):
        """
        Initialize candle builder.

        Args:
            primary_tf_minutes: Primary timeframe in minutes (default 5)
            htf_minutes: Higher timeframe in minutes (default 60)
            max_history: Maximum bars to keep in history
            timezone: Timezone for candle timestamps
        """
        self.primary_tf_minutes = primary_tf_minutes
        self.htf_minutes = htf_minutes
        self.max_history = max_history
        self.tz = pytz.timezone(timezone)

        # Candle history
        self.primary_candles: List[Candle] = []
        self.htf_candles: List[Candle] = []

        # Current building candles
        self._current_primary: Optional[CandleState] = None
        self._current_htf: Optional[CandleState] = None

        # Track last processed quote time
        self._last_quote_time: Optional[datetime] = None

    def _get_candle_start_time(self, timestamp: datetime, tf_minutes: int) -> datetime:
        """Get the start time of the candle for a given timestamp."""
        # Ensure timezone-aware
        if timestamp.tzinfo is None:
            timestamp = self.tz.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(self.tz)

        # Round down to nearest candle boundary
        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
        candle_start_minute = (minutes_since_midnight // tf_minutes) * tf_minutes

        start_hour = candle_start_minute // 60
        start_minute = candle_start_minute % 60

        return timestamp.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )

    def process_quote(self, quote: Quote) -> tuple[Optional[Candle], Optional[Candle]]:
        """
        Process a new quote and update candles.

        Args:
            quote: New quote to process

        Returns:
            Tuple of (completed_primary_candle, completed_htf_candle)
        """
        timestamp = quote.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.tz.localize(timestamp)

        price = quote.last_price
        volume = quote.volume

        completed_primary = None
        completed_htf = None

        # Calculate candle boundaries
        primary_start = self._get_candle_start_time(timestamp, self.primary_tf_minutes)
        htf_start = self._get_candle_start_time(timestamp, self.htf_minutes)

        # Process primary timeframe
        if self._current_primary is None:
            self._current_primary = CandleState(start_time=primary_start)
        elif self._current_primary.start_time != primary_start:
            # Complete the previous candle
            if self._current_primary.tick_count > 0:
                completed_primary = self._current_primary.to_candle()
                self._add_candle(completed_primary, is_htf=False)

            # Start new candle
            self._current_primary = CandleState(start_time=primary_start)

        self._current_primary.update(price, volume)

        # Process HTF
        if self._current_htf is None:
            self._current_htf = CandleState(start_time=htf_start)
        elif self._current_htf.start_time != htf_start:
            # Complete the previous candle
            if self._current_htf.tick_count > 0:
                completed_htf = self._current_htf.to_candle()
                self._add_candle(completed_htf, is_htf=True)

            # Start new candle
            self._current_htf = CandleState(start_time=htf_start)

        self._current_htf.update(price, volume)

        self._last_quote_time = timestamp
        return completed_primary, completed_htf

    def _add_candle(self, candle: Candle, is_htf: bool):
        """Add a completed candle to history."""
        if is_htf:
            self.htf_candles.append(candle)
            if len(self.htf_candles) > self.max_history:
                self.htf_candles = self.htf_candles[-self.max_history:]
        else:
            self.primary_candles.append(candle)
            if len(self.primary_candles) > self.max_history:
                self.primary_candles = self.primary_candles[-self.max_history:]

    def get_primary_dataframe(self, include_current: bool = False) -> pd.DataFrame:
        """Get primary timeframe data as DataFrame."""
        candles = self.primary_candles.copy()
        if include_current and self._current_primary and self._current_primary.tick_count > 0:
            candles.append(self._current_primary.to_candle())

        return self._candles_to_dataframe(candles)

    def get_htf_dataframe(self, include_current: bool = False) -> pd.DataFrame:
        """Get HTF data as DataFrame."""
        candles = self.htf_candles.copy()
        if include_current and self._current_htf and self._current_htf.tick_count > 0:
            candles.append(self._current_htf.to_candle())

        return self._candles_to_dataframe(candles)

    def _candles_to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        """Convert candle list to DataFrame."""
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        data = {
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }

        # Normalize all timestamps to UTC first, then convert to target timezone
        # This avoids pandas issues with mixed timezone-aware datetimes
        utc = pytz.UTC

        normalized_timestamps = []
        for c in candles:
            ts = c.timestamp
            if ts.tzinfo is None:
                # Naive datetime - assume it's in target timezone, convert to UTC
                ts = self.tz.localize(ts).astimezone(utc)
            else:
                # Aware datetime - convert to UTC
                ts = ts.astimezone(utc)
            normalized_timestamps.append(ts)

        # Create index with tz='UTC' to handle tz-aware datetimes
        index = pd.DatetimeIndex(normalized_timestamps, tz='UTC')
        df = pd.DataFrame(data, index=index)

        # Convert to target timezone
        df.index = df.index.tz_convert(self.tz)

        return df

    def has_sufficient_data(self, min_bars: int = 50) -> bool:
        """Check if we have enough data for indicator calculation."""
        return len(self.primary_candles) >= min_bars

    def bootstrap_from_yfinance(self, symbol: str = "MNQ=F", days: int = 2):
        """
        Bootstrap candle history from yfinance.

        Used on first start when no candle history exists.
        """
        import yfinance as yf

        ticker = yf.Ticker(symbol)

        # Fetch 5m data
        df_5m = ticker.history(period=f"{days}d", interval="5m")
        if not df_5m.empty:
            df_5m.columns = [c.lower() for c in df_5m.columns]
            if df_5m.index.tz is None:
                df_5m.index = df_5m.index.tz_localize("UTC")
            df_5m.index = df_5m.index.tz_convert(self.tz)

            for idx, row in df_5m.iterrows():
                candle = Candle(
                    timestamp=idx.to_pydatetime(),
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=int(row.get("volume", 0)),
                )
                self.primary_candles.append(candle)

            if len(self.primary_candles) > self.max_history:
                self.primary_candles = self.primary_candles[-self.max_history:]

        # Fetch 1h data
        df_1h = ticker.history(period=f"{days}d", interval="1h")
        if not df_1h.empty:
            df_1h.columns = [c.lower() for c in df_1h.columns]
            if df_1h.index.tz is None:
                df_1h.index = df_1h.index.tz_localize("UTC")
            df_1h.index = df_1h.index.tz_convert(self.tz)

            for idx, row in df_1h.iterrows():
                candle = Candle(
                    timestamp=idx.to_pydatetime(),
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=int(row.get("volume", 0)),
                )
                self.htf_candles.append(candle)

            if len(self.htf_candles) > self.max_history:
                self.htf_candles = self.htf_candles[-self.max_history:]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "primary_candles": [c.to_dict() for c in self.primary_candles],
            "htf_candles": [c.to_dict() for c in self.htf_candles],
            "current_primary": {
                "start_time": self._current_primary.start_time.isoformat(),
                "open": self._current_primary.open,
                "high": self._current_primary.high,
                "low": self._current_primary.low,
                "close": self._current_primary.close,
                "volume": self._current_primary.volume,
                "tick_count": self._current_primary.tick_count,
            } if self._current_primary else None,
            "current_htf": {
                "start_time": self._current_htf.start_time.isoformat(),
                "open": self._current_htf.open,
                "high": self._current_htf.high,
                "low": self._current_htf.low,
                "close": self._current_htf.close,
                "volume": self._current_htf.volume,
                "tick_count": self._current_htf.tick_count,
            } if self._current_htf else None,
            "last_quote_time": self._last_quote_time.isoformat() if self._last_quote_time else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], timezone: str = "America/New_York") -> "LiveDataBuilder":
        """Restore state from dictionary."""
        builder = cls(timezone=timezone)

        builder.primary_candles = [
            Candle.from_dict(c) for c in data.get("primary_candles", [])
        ]
        builder.htf_candles = [
            Candle.from_dict(c) for c in data.get("htf_candles", [])
        ]

        if data.get("current_primary"):
            cp = data["current_primary"]
            builder._current_primary = CandleState(
                start_time=datetime.fromisoformat(cp["start_time"]),
                open=cp["open"],
                high=cp["high"],
                low=cp["low"],
                close=cp["close"],
                volume=cp["volume"],
                tick_count=cp["tick_count"],
            )

        if data.get("current_htf"):
            ch = data["current_htf"]
            builder._current_htf = CandleState(
                start_time=datetime.fromisoformat(ch["start_time"]),
                open=ch["open"],
                high=ch["high"],
                low=ch["low"],
                close=ch["close"],
                volume=ch["volume"],
                tick_count=ch["tick_count"],
            )

        if data.get("last_quote_time"):
            builder._last_quote_time = datetime.fromisoformat(data["last_quote_time"])

        return builder

    def get_last_price(self) -> Optional[float]:
        """Get the last known price."""
        if self._current_primary and self._current_primary.tick_count > 0:
            return self._current_primary.close
        elif self.primary_candles:
            return self.primary_candles[-1].close
        return None
