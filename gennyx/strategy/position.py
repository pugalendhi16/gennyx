"""Position sizing and risk management."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Position:
    """Represents an open position."""

    entry_time: pd.Timestamp
    entry_price: float
    quantity: int
    stop_loss: float
    hard_stop: float
    entry_reason: str = ""


class PositionSizer:
    """Calculates position size based on risk parameters and margin requirements."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_per_trade: float = 0.01,
        point_value: float = 2.0,
        commission: float = 2.50,
        slippage_points: float = 0.25,
        margin_per_contract: float = 2100.0,  # MNQ overnight margin (Robinhood/Schwab)
        use_intraday_margin: bool = False,  # If True, use 25% of overnight margin
        margin_buffer: float = 0.80,  # Use only 80% of capital for margin (safety buffer)
    ):
        """
        Initialize position sizer.

        Args:
            initial_capital: Starting capital
            risk_per_trade: Fraction of capital to risk per trade (0.01 = 1%)
            point_value: Dollar value per point for the contract
            commission: Commission per side
            slippage_points: Expected slippage in points
            margin_per_contract: Margin required per contract (overnight)
            use_intraday_margin: Use reduced intraday margin (25% of overnight)
            margin_buffer: Fraction of capital available for margin (0.80 = 80%)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.point_value = point_value
        self.commission = commission
        self.slippage_points = slippage_points
        self.margin_per_contract = margin_per_contract
        self.use_intraday_margin = use_intraday_margin
        self.margin_buffer = margin_buffer

    def get_effective_margin(self) -> float:
        """Get the effective margin per contract based on settings."""
        if self.use_intraday_margin:
            return self.margin_per_contract * 0.25
        return self.margin_per_contract

    def calculate_max_contracts_by_margin(self, capital: float) -> int:
        """Calculate maximum contracts allowed by margin requirement."""
        available_for_margin = capital * self.margin_buffer
        effective_margin = self.get_effective_margin()
        return int(available_for_margin / effective_margin)

    def calculate_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        max_contracts: int = 10,
    ) -> int:
        """
        Calculate position size based on risk AND margin constraints.

        Args:
            capital: Current capital
            entry_price: Entry price
            stop_loss: Stop loss price
            max_contracts: Maximum contracts to trade

        Returns:
            Number of contracts to trade
        """
        if capital <= 0 or entry_price <= stop_loss:
            return 0

        # Risk amount in dollars
        risk_amount = capital * self.risk_per_trade

        # Points to stop loss (accounting for slippage)
        points_to_stop = entry_price - stop_loss + self.slippage_points

        # Dollar risk per contract
        dollar_risk_per_contract = points_to_stop * self.point_value + (2 * self.commission)

        if dollar_risk_per_contract <= 0:
            return 0

        # Calculate contracts based on risk
        contracts_by_risk = int(risk_amount / dollar_risk_per_contract)

        # Calculate contracts based on margin
        contracts_by_margin = self.calculate_max_contracts_by_margin(capital)

        # Use the minimum of risk-based, margin-based, and max contracts
        contracts = min(contracts_by_risk, contracts_by_margin, max_contracts)

        # Ensure at least 1 contract if we have enough margin
        if contracts < 1 and contracts_by_margin >= 1:
            return 1
        elif contracts < 1:
            return 0  # Not enough margin for even 1 contract

        return contracts

    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        include_costs: bool = True,
    ) -> float:
        """
        Calculate P&L for a trade.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Number of contracts
            include_costs: Whether to include commission and slippage

        Returns:
            Net P&L in dollars
        """
        # Points gained/lost
        points = exit_price - entry_price

        # Gross P&L
        gross_pnl = points * self.point_value * quantity

        if include_costs:
            # Subtract commission (both sides) and slippage
            costs = (2 * self.commission * quantity) + (
                self.slippage_points * self.point_value * quantity
            )
            return gross_pnl - costs

        return gross_pnl


class RiskManager:
    """Manages trading risk and exposure."""

    def __init__(
        self,
        max_daily_loss: float = 0.03,  # 3% max daily loss
        max_consecutive_losses: int = 3,
        max_positions: int = 1,
    ):
        """
        Initialize risk manager.

        Args:
            max_daily_loss: Maximum daily loss as fraction of capital
            max_consecutive_losses: Stop trading after N consecutive losses
            max_positions: Maximum concurrent positions
        """
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.max_positions = max_positions

        # State tracking
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.open_positions = 0

    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0.0

    def record_trade(self, pnl: float):
        """
        Record a completed trade.

        Args:
            pnl: Trade P&L
        """
        self.daily_pnl += pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def can_trade(self, capital: float) -> tuple:
        """
        Check if trading is allowed.

        Args:
            capital: Current capital

        Returns:
            Tuple of (allowed, reason)
        """
        # Check daily loss limit
        if capital > 0:
            daily_loss_pct = -self.daily_pnl / capital
            if daily_loss_pct >= self.max_daily_loss:
                return False, f"Daily loss limit reached ({daily_loss_pct:.1%})"

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Too many consecutive losses ({self.consecutive_losses})"

        # Check max positions
        if self.open_positions >= self.max_positions:
            return False, "Maximum positions reached"

        return True, ""

    def open_position(self):
        """Record opening a position."""
        self.open_positions += 1

    def close_position(self):
        """Record closing a position."""
        self.open_positions = max(0, self.open_positions - 1)
