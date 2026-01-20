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
from .signals import LiveSignalGenerator
from .paper_trader import PaperTradeManager
from .state_manager import StatePersistence


logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """
    Main orchestration engine for live paper trading.

    Polls Schwab API for quotes, builds candles, generates signals,
    and executes paper trades. Designed for Heroku deployment.
    """

    def __init__(self, config):
        """
        Initialize the live trading engine.

        Args:
            config: Trading configuration
        """
        self.config = config
        self.tz = pytz.timezone(config.timezone)

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

        # Setup signal handlers
        self._setup_signal_handlers()

    def _parse_timeframe(self, tf: str) -> int:
        """Parse timeframe string to minutes."""
        if tf.endswith("m"):
            return int(tf[:-1])
        elif tf.endswith("h"):
            return int(tf[:-1]) * 60
        return int(tf)

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
        logger.info(f"Session: {self.config.session_type}")
        logger.info(f"Trading Hours: {self.config.trading_start} - {self.config.trading_end} ET")
        logger.info(f"Simple Mode: {self.config.simple_mode}")
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
                # Fetch quote
                quote = self._fetch_quote()
                if quote is None:
                    time.sleep(self.config.poll_interval)
                    continue

                # Process quote and check for bar close
                completed_primary, completed_htf = self.candle_builder.process_quote(quote)

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
        """Handle bar close event."""
        logger.info(
            f"Bar close: {completed_candle.timestamp} | "
            f"O:{completed_candle.open:.2f} H:{completed_candle.high:.2f} "
            f"L:{completed_candle.low:.2f} C:{completed_candle.close:.2f}"
        )

        self._last_bar_time = completed_candle.timestamp
        self._update_signal_generator()

        # Check exit first if in position
        if self.paper_trader.has_position():
            self._check_exit_conditions(quote)

        # Check entry if not in position
        if not self.paper_trader.has_position():
            signal = self.signal_generator.check_entry()

            if signal.signal_type == "entry_long":
                logger.info(f"ENTRY SIGNAL: {signal.reason}")
                self._execute_entry(signal, quote)

    def _check_exit_conditions(self, quote: Quote):
        """Check exit conditions."""
        if not self.paper_trader.has_position():
            return

        pos = self.paper_trader.current_position
        signal = self.signal_generator.check_exit(
            entry_price=pos.entry_price,
            entry_atr=pos.entry_atr,
            current_price=quote.last_price,
        )

        if signal.signal_type == "exit_long":
            logger.info(f"EXIT SIGNAL: {signal.reason}")
            self._execute_exit(signal, quote)

    def _execute_entry(self, signal, quote: Quote):
        """Execute a paper entry."""
        position = self.paper_trader.enter_position(
            price=quote.last_price,
            stop_loss=signal.stop_loss,
            atr=signal.atr,
            reason=signal.reason,
            timestamp=signal.timestamp,
        )

        if position:
            logger.info(
                f"ENTERED LONG: {position.quantity} contracts @ ${quote.last_price:.2f} | "
                f"Stop: ${signal.stop_loss:.2f}"
            )

            # Save entry signal to database
            signal_record = {
                "timestamp": signal.timestamp.isoformat() if signal.timestamp else datetime.now().isoformat(),
                "signal_type": "entry_long",
                "price": quote.last_price,
                "stop_loss": signal.stop_loss,
                "atr": signal.atr,
                "reason": signal.reason,
                "executed": True,
            }
            self.state_manager.save_signal(signal_record, self.config.schwab_symbol)

            self._save_state()

    def _execute_exit(self, signal, quote: Quote):
        """Execute a paper exit."""
        # Get position info before exit for trade record
        pos = self.paper_trader.current_position
        entry_atr = pos.entry_atr if pos else None
        stop_loss = pos.stop_loss if pos else None

        trade = self.paper_trader.exit_position(
            price=quote.last_price,
            reason=signal.reason,
            timestamp=signal.timestamp,
        )

        if trade:
            logger.info(
                f"EXITED LONG: {trade.quantity} contracts @ ${quote.last_price:.2f} | "
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
                "price": quote.last_price,
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
