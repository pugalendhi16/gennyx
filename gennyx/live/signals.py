"""Live signal generation using existing strategy indicators."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from ..indicators import ut_bot_alert, supertrend, calculate_adx, ema_stack, bollinger_squeeze
from ..strategy.filters import HTFFilter, TrendFilter, TradingHoursFilter


@dataclass
class LiveSignal:
    """Represents a live trading signal."""

    timestamp: datetime
    signal_type: str  # 'entry_long', 'exit_long', 'none'
    price: float
    stop_loss: float = 0.0
    atr: float = 0.0
    reason: str = ""


class LiveSignalGenerator:
    """
    Generates trading signals from live data.

    Reuses existing indicator functions and filter logic.
    Signals are only generated on bar close to match backtest behavior.
    """

    def __init__(self, config):
        """
        Initialize signal generator.

        Args:
            config: Strategy configuration
        """
        self.config = config

        # Filters
        self.trend_filter = TrendFilter(
            adx_threshold=config.adx_threshold,
            adx_min_trend=config.adx_min_trend,
        )
        self.hours_filter = TradingHoursFilter(
            start_time=config.trading_start,
            end_time=config.trading_end,
            timezone=config.timezone,
        )

        # Cached indicator DataFrames
        self._primary_df: Optional[pd.DataFrame] = None
        self._htf_df: Optional[pd.DataFrame] = None
        self._htf_filter: Optional[HTFFilter] = None

    def update_session_config(self):
        """Update hours filter when session config changes (for auto mode)."""
        self.hours_filter = TradingHoursFilter(
            start_time=self.config.trading_start,
            end_time=self.config.trading_end,
            timezone=self.config.timezone,
        )

    def update_data(self, primary_df: pd.DataFrame, htf_df: pd.DataFrame):
        """
        Update data and recalculate indicators.

        Args:
            primary_df: Primary timeframe OHLCV DataFrame
            htf_df: Higher timeframe OHLCV DataFrame
        """
        if primary_df.empty:
            return

        # Add indicators to primary timeframe
        self._primary_df = self._add_indicators(primary_df)

        # Add indicators to HTF and align to primary
        if not htf_df.empty:
            htf_with_indicators = self._add_indicators(htf_df)
            self._htf_df = self._align_htf_to_primary(self._primary_df, htf_with_indicators)
            self._htf_filter = HTFFilter(self._htf_df)
        else:
            self._htf_df = None
            self._htf_filter = None

    def _align_htf_to_primary(self, primary_df: pd.DataFrame, htf_df: pd.DataFrame) -> pd.DataFrame:
        """Align HTF data to primary timeframe using forward fill."""
        return htf_df.reindex(primary_df.index, method="ffill")

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators to a DataFrame."""
        result = df.copy()

        # UT Bot Alert
        ut_bot = ut_bot_alert(
            result,
            sensitivity=self.config.ut_sensitivity,
            atr_period=self.config.ut_atr_period,
            use_heikin_ashi=self.config.use_heikin_ashi,
        )
        result = pd.concat([result, ut_bot], axis=1)

        # Supertrend
        st = supertrend(
            result,
            atr_period=self.config.st_atr_period,
            multiplier=self.config.st_multiplier,
        )
        result = pd.concat([result, st], axis=1)

        # ADX
        adx = calculate_adx(result, period=self.config.adx_period)
        result = pd.concat([result, adx], axis=1)

        # EMA Stack
        emas = ema_stack(result, periods=self.config.ema_periods)
        result = pd.concat([result, emas], axis=1)

        # Bollinger Bands Squeeze
        bb = bollinger_squeeze(
            result,
            period=self.config.bb_period,
            std_dev=self.config.bb_std,
            squeeze_percentile=self.config.bb_squeeze_percentile,
        )
        result = pd.concat([result, bb], axis=1)

        # Add ATR column for stop calculations
        from ..indicators.ut_bot import calculate_atr
        result["atr"] = calculate_atr(
            result["high"], result["low"], result["close"],
            period=self.config.ut_atr_period
        )

        # Compute Heikin-Ashi OHLC (mirrors ut_bot.py HA formula)
        if self.config.use_heikin_ashi:
            ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
            ha_open = (df["open"].shift(1) + df["close"].shift(1)) / 2
            ha_open.iloc[0] = df["open"].iloc[0]
            ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
            ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
            result["ha_open"] = ha_open
            result["ha_high"] = ha_high
            result["ha_low"] = ha_low
            result["ha_close"] = ha_close
        else:
            # Non-HA mode: HA values equal raw values
            result["ha_open"] = df["open"]
            result["ha_high"] = df["high"]
            result["ha_low"] = df["low"]
            result["ha_close"] = df["close"]

        return result

    def check_entry(self) -> LiveSignal:
        """Check for entry signal on the most recent completed bar."""
        if self._primary_df is None or len(self._primary_df) < 2:
            return LiveSignal(
                timestamp=datetime.now(),
                signal_type="none",
                price=0.0,
                reason="Insufficient data",
            )

        idx = self._primary_df.index[-1]
        row = self._primary_df.iloc[-1]
        price = float(row["close"])

        # Check trading hours
        if not self.hours_filter.is_trading_hours(idx):
            return LiveSignal(
                timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                signal_type="none",
                price=price,
                reason="Outside trading hours",
            )

        # Simple mode: just UT Bot signal
        if getattr(self.config, 'simple_mode', False):
            return self._check_simple_entry(idx, row, price)

        # Full mode with filters
        return self._check_filtered_entry(idx, row, price)

    def _check_simple_entry(self, idx, row, price: float) -> LiveSignal:
        """Check simple entry (UT Bot signal only)."""
        ut_buy = row.get("ut_buy_signal", False)

        if not ut_buy:
            return LiveSignal(
                timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                signal_type="none",
                price=float(price),
                reason="No UT Bot buy signal",
            )

        atr = float(row.get("atr", 0))
        stop_loss = float(price - (self.config.hard_stop_atr_mult * atr)) if atr > 0 else float(price - 10)

        return LiveSignal(
            timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
            signal_type="entry_long",
            price=float(price),
            stop_loss=stop_loss,
            atr=atr,
            reason="Long entry: UT Bot buy signal",
        )

    def _check_filtered_entry(self, idx, row, price: float) -> LiveSignal:
        """Check filtered entry (full MTF strategy)."""
        timestamp = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx

        # Check HTF bias
        if self._htf_filter is not None and not self._htf_filter.is_bullish(idx):
            return LiveSignal(
                timestamp=timestamp,
                signal_type="none",
                price=float(price),
                reason="HTF bias not bullish",
            )

        # Check UT Bot buy signal
        ut_buy = row.get("ut_buy_signal", False)
        if not ut_buy:
            return LiveSignal(
                timestamp=timestamp,
                signal_type="none",
                price=float(price),
                reason="No UT Bot buy signal",
            )

        # Check trend confirmation
        adx = row.get("adx", float("nan"))
        ema_aligned = row.get("ema_bullish_aligned", False)
        in_squeeze = row.get("bb_squeeze", False)

        if not self.trend_filter.is_valid_for_entry(adx, ema_aligned, in_squeeze):
            if self.trend_filter.is_ranging_market(adx, in_squeeze):
                return LiveSignal(
                    timestamp=timestamp,
                    signal_type="none",
                    price=float(price),
                    reason="Ranging market detected",
                )
            return LiveSignal(
                timestamp=timestamp,
                signal_type="none",
                price=float(price),
                reason="No trend confirmation",
            )

        atr = float(row.get("atr", 0))
        stop_loss = float(price - (self.config.hard_stop_atr_mult * atr)) if atr > 0 else float(price - 10)

        return LiveSignal(
            timestamp=timestamp,
            signal_type="entry_long",
            price=float(price),
            stop_loss=stop_loss,
            atr=atr,
            reason="Long entry: UT Bot signal, bullish HTF bias",
        )

    def check_exit(
        self,
        entry_price: float,
        entry_atr: float,
        current_price: Optional[float] = None,
    ) -> LiveSignal:
        """Check for exit signal."""
        if self._primary_df is None or len(self._primary_df) < 1:
            return LiveSignal(
                timestamp=datetime.now(),
                signal_type="none",
                price=0.0,
                reason="Insufficient data",
            )

        idx = self._primary_df.index[-1]
        row = self._primary_df.iloc[-1]
        price = float(current_price if current_price is not None else row["close"])
        timestamp = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx

        # Exit: End of trading session
        if self.hours_filter.is_near_close(idx, minutes_before=5):
            return LiveSignal(
                timestamp=timestamp,
                signal_type="exit_long",
                price=price,
                reason="End of trading session",
            )

        # Exit: Hard stop
        hard_stop = entry_price - (self.config.hard_stop_atr_mult * entry_atr)
        if price <= hard_stop:
            return LiveSignal(
                timestamp=timestamp,
                signal_type="exit_long",
                price=price,
                reason=f"Hard stop hit at {hard_stop:.2f}",
            )

        # Exit: Price below UT Bot trailing stop
        ut_stop = float(row.get("ut_trailing_stop", 0))
        if ut_stop > 0 and price < ut_stop:
            return LiveSignal(
                timestamp=timestamp,
                signal_type="exit_long",
                price=price,
                reason=f"Close below UT Bot trailing stop ({ut_stop:.2f})",
            )

        # Exit: UT Bot sell signal
        ut_sell = row.get("ut_sell_signal", False)
        if ut_sell:
            return LiveSignal(
                timestamp=timestamp,
                signal_type="exit_long",
                price=price,
                reason="UT Bot sell signal",
            )

        return LiveSignal(
            timestamp=timestamp,
            signal_type="none",
            price=price,
            reason="",
        )

    def get_current_indicators(self) -> dict:
        """Get current indicator values for display/logging."""
        if self._primary_df is None or len(self._primary_df) < 1:
            return {}

        row = self._primary_df.iloc[-1]
        return {
            "ut_trailing_stop": row.get("ut_trailing_stop"),
            "ut_trend": row.get("ut_trend"),
            "ut_buy_signal": row.get("ut_buy_signal"),
            "ut_sell_signal": row.get("ut_sell_signal"),
            "st_direction": row.get("st_direction"),
            "adx": row.get("adx"),
            "ema_bullish_aligned": row.get("ema_bullish_aligned"),
            "bb_squeeze": row.get("bb_squeeze"),
            "atr": row.get("atr"),
            "ha_open": row.get("ha_open"),
            "ha_high": row.get("ha_high"),
            "ha_low": row.get("ha_low"),
            "ha_close": row.get("ha_close"),
        }
