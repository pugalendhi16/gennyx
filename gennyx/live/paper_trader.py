"""Paper trading manager for tracking positions and P&L."""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from ..strategy.position import PositionSizer


@dataclass
class PaperPosition:
    """Represents an open paper position."""

    entry_time: datetime
    entry_price: float
    quantity: int
    stop_loss: float
    entry_atr: float
    entry_reason: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "entry_atr": self.entry_atr,
            "entry_reason": self.entry_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperPosition":
        """Create from dictionary."""
        return cls(
            entry_time=datetime.fromisoformat(data["entry_time"]),
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            stop_loss=data["stop_loss"],
            entry_atr=data["entry_atr"],
            entry_reason=data.get("entry_reason", ""),
        )


@dataclass
class PaperTrade:
    """Represents a completed paper trade."""

    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    entry_reason: str
    exit_reason: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
        }


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: date
    starting_capital: float
    ending_capital: float
    pnl: float
    trade_count: int
    winning_trades: int
    losing_trades: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "starting_capital": self.starting_capital,
            "ending_capital": self.ending_capital,
            "pnl": self.pnl,
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
        }


class PaperTradeManager:
    """
    Manages paper trading positions and P&L tracking.
    """

    def __init__(self, config):
        """
        Initialize paper trade manager.

        Args:
            config: Strategy configuration
        """
        self.config = config

        # Position sizer for P&L calculations
        self.position_sizer = PositionSizer(
            initial_capital=config.initial_capital,
            risk_per_trade=config.risk_per_trade,
            point_value=config.point_value,
            commission=config.commission_per_trade,
            slippage_points=config.slippage_points,
            margin_per_contract=getattr(config, 'margin_per_contract', 2100.0),
            use_intraday_margin=getattr(config, 'use_intraday_margin', False),
            margin_buffer=getattr(config, 'margin_buffer', 0.80),
        )

        # State
        self.capital = config.initial_capital
        self.current_position: Optional[PaperPosition] = None
        self.trades: List[PaperTrade] = []

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.daily_winning_trades = 0
        self.daily_losing_trades = 0
        self.daily_start_capital = config.initial_capital
        self.current_date: Optional[date] = None

    def enter_position(
        self,
        price: float,
        stop_loss: float,
        atr: float,
        reason: str = "",
        timestamp: Optional[datetime] = None,
    ) -> Optional[PaperPosition]:
        """Enter a new paper position."""
        if self.current_position is not None:
            return None

        timestamp = timestamp or datetime.now()

        quantity = self.position_sizer.calculate_size(
            self.capital, price, stop_loss
        )

        if quantity <= 0:
            return None

        self.current_position = PaperPosition(
            entry_time=timestamp,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            entry_atr=atr,
            entry_reason=reason,
        )

        self._check_day_change(timestamp.date())

        return self.current_position

    def exit_position(
        self,
        price: float,
        reason: str = "",
        timestamp: Optional[datetime] = None,
    ) -> Optional[PaperTrade]:
        """Exit current paper position."""
        if self.current_position is None:
            return None

        timestamp = timestamp or datetime.now()
        pos = self.current_position

        pnl = self.position_sizer.calculate_pnl(
            pos.entry_price, price, pos.quantity
        )
        pnl_percent = pnl / self.capital if self.capital > 0 else 0

        trade = PaperTrade(
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_percent=pnl_percent,
            entry_reason=pos.entry_reason,
            exit_reason=reason,
        )

        self.capital += pnl

        self._check_day_change(timestamp.date())
        self.daily_pnl += pnl
        self.daily_trade_count += 1
        if pnl >= 0:
            self.daily_winning_trades += 1
        else:
            self.daily_losing_trades += 1

        self.trades.append(trade)
        self.current_position = None

        return trade

    def _check_day_change(self, today: date):
        """Check if day changed and reset daily stats if needed."""
        if self.current_date is None:
            self.current_date = today
            self.daily_start_capital = self.capital
        elif self.current_date != today:
            self.current_date = today
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.daily_winning_trades = 0
            self.daily_losing_trades = 0
            self.daily_start_capital = self.capital

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for current position."""
        if self.current_position is None:
            return 0.0

        return self.position_sizer.calculate_pnl(
            self.current_position.entry_price,
            current_price,
            self.current_position.quantity,
        )

    def get_daily_stats(self) -> DailyStats:
        """Get current daily statistics."""
        return DailyStats(
            date=self.current_date or date.today(),
            starting_capital=self.daily_start_capital,
            ending_capital=self.capital,
            pnl=self.daily_pnl,
            trade_count=self.daily_trade_count,
            winning_trades=self.daily_winning_trades,
            losing_trades=self.daily_losing_trades,
        )

    def get_total_stats(self) -> dict:
        """Get total trading statistics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        winning = [t for t in self.trades if t.pnl >= 0]
        losing = [t for t in self.trades if t.pnl < 0]
        total_pnl = sum(t.pnl for t in self.trades)

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self.trades) if self.trades else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(self.trades) if self.trades else 0.0,
            "largest_win": max((t.pnl for t in winning), default=0.0),
            "largest_loss": min((t.pnl for t in losing), default=0.0),
        }

    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.current_position is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "capital": self.capital,
            "current_position": self.current_position.to_dict() if self.current_position else None,
            "trades": [t.to_dict() for t in self.trades[-100:]],  # Keep last 100 trades
            "daily_pnl": self.daily_pnl,
            "daily_trade_count": self.daily_trade_count,
            "daily_winning_trades": self.daily_winning_trades,
            "daily_losing_trades": self.daily_losing_trades,
            "daily_start_capital": self.daily_start_capital,
            "current_date": self.current_date.isoformat() if self.current_date else None,
        }

    def from_dict(self, data: Dict[str, Any]):
        """Restore state from dictionary."""
        self.capital = data.get("capital", self.config.initial_capital)

        if data.get("current_position"):
            self.current_position = PaperPosition.from_dict(data["current_position"])
        else:
            self.current_position = None

        self.daily_pnl = data.get("daily_pnl", 0.0)
        self.daily_trade_count = data.get("daily_trade_count", 0)
        self.daily_winning_trades = data.get("daily_winning_trades", 0)
        self.daily_losing_trades = data.get("daily_losing_trades", 0)
        self.daily_start_capital = data.get("daily_start_capital", self.capital)

        if data.get("current_date"):
            self.current_date = date.fromisoformat(data["current_date"])
