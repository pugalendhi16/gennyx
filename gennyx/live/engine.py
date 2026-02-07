"""Main trading engine for live paper trading on Heroku."""

import logging
import signal
import sys
import time
from datetime import datetime
from typing import Optional

import pytz

from .data_feed import SchwabDataFeed, Quote
from .candle_builder import LiveDataBuilder
from .signals import LiveSignalGenerator, LiveSignal
from .paper_trader import PaperTradeManager
from .state_manager import StatePersistence


logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """
    Main orchestration engine for live paper trading.

    Polls Schwab API for quotes, builds candles, generates signals,
    and executes paper trades. Designed for Heroku deployment.
    """

    # CME Futures hours: Sunday 6PM ET through Friday 5PM ET
    # Closed: Friday 5PM ET through Sunday 6PM ET
    FUTURES_CLOSE_DAY = 4  # Friday (0=Monday)
    FUTURES_CLOSE_HOUR = 17  # 5PM ET
    FUTURES_OPEN_DAY = 6  # Sunday
    FUTURES_OPEN_HOUR = 18  # 6PM ET

    def __init__(self, config):
        """
        Initialize the live trading engine.

        Args:
            config: Trading configuration
        """
        self.config = config
        self.tz = pytz.timezone(config.timezone)
        self._market_closed_logged = False  # Track if we've logged market closed

        # Initialize components
        self.data_feed = SchwabDataFeed(config)
        self.candle_builder = LiveDataBuilder(
            primary_tf_minutes=self._parse_timeframe(config.primary_tf),
            htf_minutes=self._parse_timeframe(config.htf),
            max_history=200,
            timezone=config.timezone,
        )
        self.signal_generator = LiveSignalGenerator(config)
        self.paper_trader = PaperTradeManager(config)
        self.state_manager = StatePersistence(config.database_url)

        # Engine state
        self._running = False
        self._shutdown_requested = False
        self._last_quote_time: Optional[datetime] = None
        self._last_bar_time: Optional[datetime] = None
        self._current_session: Optional[str] = None

        # Setup signal handlers
        self._setup_signal_handlers()

    def _parse_timeframe(self, tf: str) -> int:
        """Parse timeframe string to minutes."""
        if tf.endswith("m"):
            return int(tf[:-1])
        elif tf.endswith("h"):
            return int(tf[:-1]) * 60
        return int(tf)

    def _get_current_session(self) -> str:
        """Determine current trading session based on ET time."""
        now = datetime.now(self.tz)
        current_minutes = now.hour * 60 + now.minute
        rth_start = 9 * 60 + 30   # 09:30
        rth_end = 16 * 60         # 16:00

        if rth_start <= current_minutes < rth_end:
            return "rth"
        return "overnight"

    def _is_futures_market_closed(self) -> bool:
        """
        Check if CME futures markets are closed.

        CME Equity Index Futures (MNQ, NQ, ES, etc.):
        - Open: Sunday 6:00 PM ET through Friday 5:00 PM ET
        - Closed: Friday 5:00 PM ET through Sunday 6:00 PM ET

        Returns True when markets are closed.
        """
        now = datetime.now(self.tz)  # Already in ET
        weekday = now.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
        hour = now.hour

        # Saturday: always closed
        if weekday == 5:
            return True

        # Friday at or after 5PM ET: closed
        if weekday == self.FUTURES_CLOSE_DAY and hour >= self.FUTURES_CLOSE_HOUR:
            return True

        # Sunday before 6PM ET: closed
        if weekday == self.FUTURES_OPEN_DAY and hour < self.FUTURES_OPEN_HOUR:
            return True

        return False

    def _apply_session_config(self, session: str):
        """Apply session-specific configuration and recalculate indicators."""
        if session == "rth":
            self.config.simple_mode = False
            self.config.use_heikin_ashi = True
            self.config.use_ha_atr = True
            self.config.trading_start = "09:30"
            self.config.trading_end = "16:00"
            logger.info(
                "Session: RTH (Full MTF filtered, HA candles, HA ATR, 09:30-16:00)"
            )
        else:
            self.config.simple_mode = True
            self.config.use_heikin_ashi = True
            self.config.use_ha_atr = False
            self.config.trading_start = "16:00"
            self.config.trading_end = "09:30"
            logger.info(
                "Session: Overnight (Simple UT Bot, HA candles, Raw ATR, 16:00-09:30)"
            )

        self._current_session = session

        # Update signal generator's hours filter for new session times
        self.signal_generator.update_session_config()

        # Recalculate indicators (Heikin-Ashi setting affects UT Bot)
        self._update_signal_generator()

    def _check_session_transition(self, quote: Quote, bar_close_price: float) -> bool:
        """Check if session has changed and handle transition."""
        if self.config.session_type != "auto":
            return False

        new_session = self._get_current_session()
        if new_session == self._current_session:
            return False

        logger.info(
            f"SESSION TRANSITION: {self._current_session} -> {new_session}"
        )

        # Force exit any open position before switching (at candle close price)
        if self.paper_trader.has_position():
            logger.info("Force-closing position for session transition")
            exit_signal = LiveSignal(
                timestamp=datetime.now(self.tz),
                signal_type="exit_long",
                price=bar_close_price,
                reason=f"Session transition: {self._current_session} -> {new_session}",
            )
            self._execute_exit(exit_signal, bar_close_price)

        self._apply_session_config(new_session)
        return True

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def start(self, restore_state: bool = True):
        """Start the live trading engine."""
        logger.info("=" * 60)
        logger.info("GenNyx Live Paper Trading Engine")
        logger.info("=" * 60)
        logger.info(f"Session Type: {self.config.session_type}")
        logger.info(f"Poll Interval: {self.config.poll_interval}s")
        logger.info("=" * 60)

        # Connect to Schwab API
        logger.info("Connecting to Schwab API...")
        if not self.data_feed.connect():
            raise RuntimeError("Failed to connect to Schwab API")
        logger.info("Connected to Schwab API")

        # Restore or bootstrap state
        if restore_state and self.state_manager.has_state():
            self._restore_state()
        else:
            self._bootstrap_data()

        # Initialize session mode (auto switches between RTH and Overnight)
        now_et = datetime.now(self.tz)
        logger.info(f"Current ET time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if self.config.session_type == "auto":
            session = self._get_current_session()
            self._apply_session_config(session)
        else:
            self._apply_session_config(self.config.session_type)
            logger.info(f"Fixed session: {self._current_session}")
            logger.info(f"Trading Hours: {self.config.trading_start} - {self.config.trading_end} ET")
            logger.info(f"Simple Mode: {self.config.simple_mode}")

        self._log_status()

        # Start main loop
        self._running = True
        self._run_loop()

    def _restore_state(self):
        """Restore state from database."""
        logger.info("Restoring state from database...")

        candle_data = self.state_manager.load_candle_history()
        if candle_data:
            self.candle_builder = LiveDataBuilder.from_dict(
                candle_data, timezone=self.config.timezone
            )
            logger.info(
                f"Restored {len(self.candle_builder.primary_candles)} primary candles, "
                f"{len(self.candle_builder.htf_candles)} HTF candles"
            )

        trading_state = self.state_manager.load_trading_state()
        if trading_state:
            self.paper_trader.from_dict(trading_state)
            logger.info(
                f"Restored capital: ${self.paper_trader.capital:,.2f}, "
                f"Position: {'Yes' if self.paper_trader.has_position() else 'No'}"
            )

        self._update_signal_generator()

    def _bootstrap_data(self):
        """Bootstrap candle history from yfinance."""
        logger.info("Bootstrapping candle history from yfinance...")

        self.candle_builder.bootstrap_from_yfinance(
            symbol=self.config.symbol,
            days=2,
        )

        logger.info(
            f"Bootstrapped {len(self.candle_builder.primary_candles)} primary candles, "
            f"{len(self.candle_builder.htf_candles)} HTF candles"
        )

        self._update_signal_generator()

    def _update_signal_generator(self):
        """Update signal generator with current candle data."""
        primary_df = self.candle_builder.get_primary_dataframe()
        htf_df = self.candle_builder.get_htf_dataframe()

        if not primary_df.empty:
            self.signal_generator.update_data(primary_df, htf_df)

    def _run_loop(self):
        """Main trading loop."""
        logger.info(f"Starting main loop (poll interval: {self.config.poll_interval}s)")

        while self._running and not self._shutdown_requested:
            try:
                # Check if futures market is closed (Friday 5PM ET - Sunday 6PM ET)
                if self._is_futures_market_closed():
                    if not self._market_closed_logged:
                        now = datetime.now(self.tz)
                        logger.info(
                            f"Futures market closed (Fri 5PM - Sun 6PM ET). "
                            f"Current: {now.strftime('%A %I:%M %p %Z')}. Sleeping..."
                        )
                        self._market_closed_logged = True
                    time.sleep(300)  # Sleep 5 minutes during market closure
                    continue
                else:
                    if self._market_closed_logged:
                        logger.info("Futures market open. Resuming polling...")
                        self._market_closed_logged = False

                # Fetch quote
                quote = self._fetch_quote()
                if quote is None:
                    time.sleep(self.config.poll_interval)
                    continue

                # Process quote and check for bar close
                completed_primary, completed_htf = self.candle_builder.process_quote(quote)

                # Check max risk hard stop every poll (before bar close logic)
                if self.paper_trader.has_position():
                    unrealized = self.paper_trader.get_unrealized_pnl(quote.last_price)
                    max_risk = self.paper_trader.capital * self.config.risk_per_trade
                    if unrealized <= -max_risk:
                        logger.warning(
                            f"MAX RISK STOP: Unrealized loss ${abs(unrealized):.2f} exceeds "
                            f"max risk ${max_risk:.2f}"
                        )
                        exit_signal = LiveSignal(
                            timestamp=datetime.now(self.tz),
                            signal_type="exit_long",
                            price=quote.last_price,
                            reason=f"Max risk stop: loss ${abs(unrealized):.2f} exceeds ${max_risk:.2f}",
                        )
                        self._execute_exit(exit_signal, quote.last_price)

                # On bar close, update indicators and check signals
                if completed_primary:
                    self._on_bar_close(completed_primary, quote)

                # Periodic state save
                if self.state_manager.should_save(self.config.save_interval):
                    self._save_state()

                time.sleep(self.config.poll_interval)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(self.config.poll_interval)

        self._shutdown()

    def _fetch_quote(self) -> Optional[Quote]:
        """Fetch current quote from Schwab."""
        try:
            quote = self.data_feed.get_quote(self.config.schwab_symbol)
            self._last_quote_time = quote.timestamp
            return quote
        except Exception as e:
            logger.warning(f"Failed to fetch quote: {e}")
            return None

    def _on_bar_close(self, completed_candle, quote: Quote):
        """Handle bar close event. All entry/exit uses candle close price."""
        bar_close_price = completed_candle.close

        logger.info(
            f"Bar close [{self._current_session}]: {completed_candle.timestamp} | "
            f"O:{completed_candle.open:.2f} H:{completed_candle.high:.2f} "
            f"L:{completed_candle.low:.2f} C:{completed_candle.close:.2f}"
        )

        self._last_bar_time = completed_candle.timestamp

        # Check session transition first (auto mode only)
        session_changed = self._check_session_transition(quote, bar_close_price)

        # Update indicators (skip if session change already did it)
        if not session_changed:
            self._update_signal_generator()

        # Save raw candle to candles table
        self.state_manager.save_candle({
            "timestamp": completed_candle.timestamp,
            "open": float(completed_candle.open),
            "high": float(completed_candle.high),
            "low": float(completed_candle.low),
            "close": float(completed_candle.close),
            "volume": completed_candle.volume,
        }, symbol=self.config.schwab_symbol, timeframe=self.config.primary_tf)

        # Save candle + UT Bot signal + HA OHLC to candle_signals DB
        indicators = self.signal_generator.get_current_indicators()
        ut_signal = "BUY" if indicators.get("ut_buy_signal") else (
            "SELL" if indicators.get("ut_sell_signal") else "NONE"
        )
        ut_trend = indicators.get("ut_trend")
        ut_trailing_stop = indicators.get("ut_trailing_stop")
        atr_val = indicators.get("atr")
        ha_open = indicators.get("ha_open")
        ha_high = indicators.get("ha_high")
        ha_low = indicators.get("ha_low")
        ha_close = indicators.get("ha_close")
        self.state_manager.save_candle_signal({
            "timestamp": completed_candle.timestamp,
            "open": float(completed_candle.open),
            "high": float(completed_candle.high),
            "low": float(completed_candle.low),
            "close": float(completed_candle.close),
            "volume": completed_candle.volume,
            "ut_signal": ut_signal,
            "ut_trend": int(ut_trend) if ut_trend is not None else None,
            "ut_trailing_stop": float(ut_trailing_stop) if ut_trailing_stop is not None else None,
            "atr": float(atr_val) if atr_val is not None else None,
            "ha_open": float(ha_open) if ha_open is not None else None,
            "ha_high": float(ha_high) if ha_high is not None else None,
            "ha_low": float(ha_low) if ha_low is not None else None,
            "ha_close": float(ha_close) if ha_close is not None else None,
        }, symbol=self.config.schwab_symbol)

        # Check exit first if in position
        if self.paper_trader.has_position():
            self._check_exit_conditions(bar_close_price)

        # Check entry if not in position
        if not self.paper_trader.has_position():
            signal = self.signal_generator.check_entry()

            if signal.signal_type == "entry_long":
                logger.info(f"ENTRY SIGNAL: {signal.reason}")
                self._execute_entry(signal, bar_close_price)

    def _check_exit_conditions(self, price: float):
        """Check exit conditions using candle close price."""
        if not self.paper_trader.has_position():
            return

        pos = self.paper_trader.current_position
        signal = self.signal_generator.check_exit(
            entry_price=pos.entry_price,
            entry_atr=pos.entry_atr,
            current_price=price,
        )

        if signal.signal_type == "exit_long":
            logger.info(f"EXIT SIGNAL: {signal.reason}")
            self._execute_exit(signal, price)

    def _execute_entry(self, signal, price: float):
        """Execute a paper entry at candle close price."""
        position = self.paper_trader.enter_position(
            price=price,
            stop_loss=signal.stop_loss,
            atr=signal.atr,
            reason=signal.reason,
            timestamp=signal.timestamp,
        )

        if position:
            logger.info(
                f"ENTERED LONG: {position.quantity} contracts @ ${price:.2f} | "
                f"Stop: ${signal.stop_loss:.2f}"
            )

            # Save entry signal to database
            signal_record = {
                "timestamp": signal.timestamp.isoformat() if signal.timestamp else datetime.now().isoformat(),
                "signal_type": "entry_long",
                "price": price,
                "stop_loss": signal.stop_loss,
                "atr": signal.atr,
                "reason": signal.reason,
                "executed": True,
            }
            self.state_manager.save_signal(signal_record, self.config.schwab_symbol)

            self._save_state()

    def _execute_exit(self, signal, price: float):
        """Execute a paper exit at candle close price."""
        # Get position info before exit for trade record
        pos = self.paper_trader.current_position
        entry_atr = pos.entry_atr if pos else None
        stop_loss = pos.stop_loss if pos else None

        trade = self.paper_trader.exit_position(
            price=price,
            reason=signal.reason,
            timestamp=signal.timestamp,
        )

        if trade:
            logger.info(
                f"EXITED LONG: {trade.quantity} contracts @ ${price:.2f} | "
                f"P&L: ${trade.pnl:+.2f} ({trade.pnl_percent:+.2%})"
            )

            # Save trade to database
            trade_record = {
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "pnl": trade.pnl,
                "pnl_percent": trade.pnl_percent,
                "entry_reason": trade.entry_reason,
                "exit_reason": trade.exit_reason,
                "entry_atr": entry_atr,
                "stop_loss": stop_loss,
            }
            self.state_manager.save_trade(trade_record, self.config.schwab_symbol)

            # Save daily stats
            stats = self.paper_trader.get_daily_stats()
            daily_record = {
                "date": stats.date,
                "starting_capital": stats.starting_capital,
                "ending_capital": stats.ending_capital,
                "pnl": stats.pnl,
                "pnl_percent": stats.pnl / stats.starting_capital if stats.starting_capital > 0 else 0,
                "trade_count": stats.trade_count,
                "winning_trades": stats.winning_trades,
                "losing_trades": stats.losing_trades,
                "largest_win": trade.pnl if trade.pnl > 0 else None,
                "largest_loss": trade.pnl if trade.pnl < 0 else None,
            }
            self.state_manager.save_daily_stats(daily_record, self.config.schwab_symbol)

            # Save exit signal to database
            exit_signal_record = {
                "timestamp": signal.timestamp.isoformat() if signal.timestamp else datetime.now().isoformat(),
                "signal_type": "exit_long",
                "price": price,
                "stop_loss": stop_loss,
                "atr": entry_atr,
                "reason": signal.reason,
                "executed": True,
            }
            self.state_manager.save_signal(exit_signal_record, self.config.schwab_symbol)

            logger.info(
                f"Daily: {stats.trade_count} trades, ${stats.pnl:+.2f} P&L, "
                f"Capital: ${self.paper_trader.capital:,.2f}"
            )
            self._save_state()

    def _save_state(self):
        """Save current state to database."""
        trading_state = self.paper_trader.to_dict()
        candle_data = self.candle_builder.to_dict()

        if self.state_manager.save_all(trading_state, candle_data):
            logger.debug("State saved")

        # Check and persist any token refreshes
        self.data_feed.check_and_persist_token()

    def _shutdown(self):
        """Perform graceful shutdown."""
        logger.info("Shutting down...")

        self._running = False
        self._save_state()
        self.state_manager.close()
        self.data_feed.cleanup()

        logger.info("Final state saved")
        self._log_status()
        logger.info("Shutdown complete")

    def _log_status(self):
        """Log current status."""
        logger.info("-" * 60)
        logger.info("Status:")
        logger.info(f"  Session: {self._current_session or 'N/A'} (mode: {self.config.session_type})")
        atr_mode = "HA ATR" if self.config.use_ha_atr else "Raw ATR"
        logger.info(f"  Simple Mode: {self.config.simple_mode} | Heikin-Ashi: {self.config.use_heikin_ashi} | ATR: {atr_mode}")
        logger.info(f"  Capital: ${self.paper_trader.capital:,.2f}")
        logger.info(f"  Position: {'Yes' if self.paper_trader.has_position() else 'No'}")

        if self.paper_trader.has_position():
            pos = self.paper_trader.current_position
            current_price = self.candle_builder.get_last_price() or pos.entry_price
            unrealized = self.paper_trader.get_unrealized_pnl(current_price)
            logger.info(f"    Entry: ${pos.entry_price:.2f}, Unrealized: ${unrealized:+.2f}")

        stats = self.paper_trader.get_daily_stats()
        logger.info(f"  Daily P&L: ${stats.pnl:+.2f} ({stats.trade_count} trades)")

        # Get total stats from database for accurate historical tracking
        db_stats = self.state_manager.get_total_stats(self.config.schwab_symbol)
        if db_stats:
            logger.info(
                f"  Total (DB): {db_stats['total_trades']} trades, "
                f"{db_stats['win_rate']:.0%} win rate, "
                f"${db_stats['total_pnl']:+.2f} P&L"
            )
        else:
            total = self.paper_trader.get_total_stats()
            logger.info(f"  Total: {total['total_trades']} trades, {total['win_rate']:.0%} win rate")

        logger.info(f"  Primary Candles: {len(self.candle_builder.primary_candles)}")
        logger.info("-" * 60)

    def stop(self):
        """Request engine stop."""
        self._shutdown_requested = True
