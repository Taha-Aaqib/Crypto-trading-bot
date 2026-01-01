"""
Risk Manager
Handles position sizing, risk limits, and circuit breakers
"""

from typing import Dict
from datetime import datetime, timedelta
import pandas as pd
from src.utils.logger import get_logger
from src.utils.helpers import calculate_position_size

logger = get_logger()


class RiskManager:
    """Manage trading risk and position sizing"""

    def __init__(self, config: Dict, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.risk_config = config['risk']
        self.trading_config = config['trading']

        # Risk parameters
        self.max_risk_per_trade = self.risk_config['max_risk_per_trade']
        self.max_daily_loss = self.risk_config['max_daily_loss']
        self.max_open_positions = self.trading_config['max_open_positions']

        # Portfolio
        self.initial_capital = config.get(
            'backtest', {}).get('initial_capital', 10000)
        self.current_capital = self.initial_capital

        logger.info(
            f"Risk Manager initialized - Max risk/trade: {self.max_risk_per_trade*100}%")

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management rules

        Args:
            entry_price: Entry price for trade
            stop_loss: Stop loss price

        Returns:
            Position size in quote currency (e.g., USDT)
        """
        # Get current portfolio value
        portfolio_value = self.get_portfolio_value()

        # Calculate position size
        position_size = calculate_position_size(
            portfolio_value,
            self.max_risk_per_trade,
            entry_price,
            stop_loss
        )

        # Ensure position doesn't exceed configured max
        max_position = self.trading_config.get('position_size_usd', 100)
        position_size = min(position_size, max_position)

        logger.info(
            f"Calculated position size: {position_size:.2f} (Entry: {entry_price}, SL: {stop_loss})")

        return position_size

    def can_open_new_trade(self) -> bool:
        """
        Check if a new trade can be opened based on risk rules

        Returns:
            True if new trade allowed, False otherwise
        """
        # Check maximum open positions
        open_trades = self.db_manager.get_open_trades()
        if len(open_trades) >= self.max_open_positions:
            logger.warning(
                f"Max open positions ({self.max_open_positions}) reached")
            return False

        # Check daily loss limit
        if self.has_exceeded_daily_loss():
            logger.warning("Daily loss limit exceeded - trading paused")
            return False

        # Circuit breaker check
        if self.risk_config.get('use_circuit_breaker') and self.is_circuit_breaker_triggered():
            logger.warning("Circuit breaker triggered - trading paused")
            return False

        return True

    def has_exceeded_daily_loss(self) -> bool:
        """Check if daily loss limit has been exceeded"""
        today = datetime.now().date()
        trades_df = self.db_manager.get_trade_history(limit=100)

        if trades_df.empty:
            return False

        # Convert exit_time to datetime if it's a string
        if not pd.api.types.is_datetime64_any_dtype(trades_df['exit_time']):
            trades_df['exit_time'] = pd.to_datetime(
                trades_df['exit_time'], errors='coerce')

        # Filter today's closed trades
        today_trades = trades_df[
            (trades_df['exit_time'].notna()) &
            (trades_df['exit_time'].dt.date == today)
        ]

        if today_trades.empty:
            return False

        daily_pnl = today_trades['pnl'].sum()
        max_loss = self.get_portfolio_value() * self.max_daily_loss

        if daily_pnl < -max_loss:
            logger.warning(
                f"Daily loss limit exceeded: ${daily_pnl:.2f} / ${-max_loss:.2f}")
            return True

        return False

    def is_circuit_breaker_triggered(self) -> bool:
        """
        Circuit breaker: Pause trading after consecutive losses

        Returns:
            True if circuit breaker triggered, False otherwise
        """
        trades_df = self.db_manager.get_trade_history(limit=10)

        if len(trades_df) < 5:
            return False

        # Check last 5 trades
        last_5_trades = trades_df.head(5)

        # If all last 5 trades are losses, trigger circuit breaker
        all_losses = all(last_5_trades['pnl'] < 0)

        if all_losses:
            logger.error("🔴 CIRCUIT BREAKER: 5 consecutive losses detected!")
            return True

        return False

    def get_portfolio_value(self) -> float:
        """
        Calculate current portfolio value

        Returns:
            Current portfolio value
        """
        # Get all closed trades
        trades_df = self.db_manager.get_trade_history(limit=1000)

        if trades_df.empty:
            return self.initial_capital

        # Calculate total PnL
        closed_trades = trades_df[trades_df['pnl'].notna()]
        total_pnl = closed_trades['pnl'].sum(
        ) if not closed_trades.empty else 0

        # Current value = initial capital + total PnL
        current_value = self.initial_capital + total_pnl

        return current_value

    def validate_trade(self, signal: Dict) -> tuple:
        """
        Validate if trade meets all risk criteria

        Args:
            signal: Trade signal dictionary

        Returns:
            Tuple (is_valid: bool, reason: str)
        """
        # Check if we can open new trade
        if not self.can_open_new_trade():
            return (False, "Risk limits prevent new trade")

        # Check risk-reward ratio
        entry = signal['entry_price']
        sl = signal['stop_loss']
        tp = signal['take_profit']

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr_ratio = reward / risk if risk > 0 else 0

        min_rr = self.risk_config['take_profit_rr_ratio']
        if rr_ratio < min_rr:
            return (False, f"RR ratio {rr_ratio:.2f} below minimum {min_rr}")

        return (True, "Trade validated")
