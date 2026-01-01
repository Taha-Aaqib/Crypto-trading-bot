"""
Backtesting Engine
Simulates historical trading to evaluate strategy performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from src.utils.logger import get_logger

logger = get_logger()


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None
    order_id: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Position:
    """Represents an open trading position"""
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    entry_timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    realized_pnl: float = 0.0
    commission: float = 0.0


@dataclass
class BacktestState:
    """Tracks backtesting state"""
    initial_capital: float
    current_capital: float
    equity: float
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_positions: List[Position] = field(default_factory=list)
    pending_orders: List[Order] = field(default_factory=list)
    filled_orders: List[Order] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    trades_won: int = 0
    trades_lost: int = 0
    total_trades: int = 0


class BacktestEngine:
    """
    Event-driven backtesting engine for strategy evaluation

    Features:
    - Historical trade simulation
    - Position management with stop-loss/take-profit
    - Commission and slippage modeling
    - Multi-symbol support
    - Detailed trade logging
    """

    def __init__(self, config: Dict):
        """
        Initialize backtesting engine

        Args:
            config: Configuration dictionary containing:
                - initial_capital: Starting capital
                - commission: Commission rate (e.g., 0.001 = 0.1%)
                - slippage: Slippage rate (e.g., 0.0005 = 0.05%)
                - max_open_positions: Maximum concurrent positions
        """
        self.config = config
        backtest_config = config.get('backtest', {})

        self.initial_capital = backtest_config.get('initial_capital', 10000)
        self.commission_rate = backtest_config.get('commission', 0.001)
        self.slippage_rate = backtest_config.get('slippage', 0.0005)
        self.max_positions = config.get(
            'trading', {}).get('max_open_positions', 3)

        # Initialize state
        self.state = BacktestState(
            initial_capital=self.initial_capital,
            current_capital=self.initial_capital,
            equity=self.initial_capital
        )

        self.order_counter = 0

        logger.info(f"Backtest engine initialized - Capital: ${self.initial_capital:,.2f}, "
                    f"Commission: {self.commission_rate*100:.2f}%, Slippage: {self.slippage_rate*100:.3f}%")

    def run(
        self,
        strategy,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Run backtest on historical data

        Args:
            strategy: Strategy instance with generate_signal() method
            data: Dictionary of {symbol: DataFrame} with OHLCV data
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Backtest")
        logger.info("=" * 60)

        # Filter data by date range
        filtered_data = self._filter_date_range(data, start_date, end_date)

        if not filtered_data:
            logger.error("No data available for backtesting")
            return {}

        # Get common timestamps across all symbols
        timestamps = self._get_common_timestamps(filtered_data)

        logger.info(
            f"Backtesting {len(filtered_data)} symbols over {len(timestamps)} time periods")
        logger.info(f"Date range: {timestamps[0]} to {timestamps[-1]}")

        # Main backtesting loop
        for i, current_time in enumerate(timestamps):
            # Get current market data for all symbols
            current_data = {
                symbol: df.loc[df.index <= current_time].tail(
                    500)  # Last 500 candles for analysis
                for symbol, df in filtered_data.items()
            }

            # Check stop-loss and take-profit for open positions
            self._check_exits(current_data, current_time)

            # Generate signals from strategy
            for symbol in filtered_data.keys():
                if symbol not in current_data or len(current_data[symbol]) < 100:
                    continue

                # Get current price
                current_candle = current_data[symbol].iloc[-1]
                current_price = current_candle['close']

                # Skip if we already have a position in this symbol
                if symbol in self.state.positions:
                    continue

                # Check if we can open new positions
                if len(self.state.positions) >= self.max_positions:
                    continue

                # Generate trading signal
                try:
                    signal = strategy.generate_signal(
                        symbol=symbol,
                        df=current_data[symbol]
                    )

                    if signal and signal.get('action') in ['buy', 'sell']:
                        self._execute_signal(
                            symbol, signal, current_price, current_time)

                except Exception as e:
                    logger.error(f"Error generating signal for {symbol}: {e}")
                    continue

            # Update equity curve
            self._update_equity(current_data, current_time)

            # Log progress every 10%
            if (i + 1) % max(1, len(timestamps) // 10) == 0:
                progress = (i + 1) / len(timestamps) * 100
                logger.info(f"Progress: {progress:.1f}% - Equity: ${self.state.equity:,.2f} - "
                            f"Open positions: {len(self.state.positions)}")

        # Close remaining positions at the end
        self._close_all_positions(filtered_data, timestamps[-1])

        logger.info("=" * 60)
        logger.info("Backtest Completed")
        logger.info("=" * 60)

        return self._generate_results()

    def _execute_signal(
        self,
        symbol: str,
        signal: Dict,
        price: float,
        timestamp: datetime
    ):
        """Execute a trading signal"""
        action = signal.get('action')
        side = OrderSide.BUY if action == 'buy' else OrderSide.SELL

        # Get position sizing from signal
        position_size_usd = signal.get('position_size_usd', 100)
        quantity = position_size_usd / price

        # Apply slippage
        if side == OrderSide.BUY:
            fill_price = price * (1 + self.slippage_rate)
        else:
            fill_price = price * (1 - self.slippage_rate)

        # Calculate commission
        trade_value = quantity * fill_price
        commission = trade_value * self.commission_rate

        # Check if we have enough capital
        required_capital = trade_value + commission
        if required_capital > self.state.current_capital:
            logger.warning(f"Insufficient capital for {symbol} {action} - Required: ${required_capital:.2f}, "
                           f"Available: ${self.state.current_capital:.2f}")
            return

        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            quantity=quantity,
            entry_timestamp=timestamp,
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit'),
            commission=commission
        )

        # Update capital
        self.state.current_capital -= (trade_value + commission)
        self.state.positions[symbol] = position
        self.state.total_trades += 1

        logger.info(f"[{timestamp}] OPEN {side.value.upper()} {symbol} @ ${fill_price:.2f} "
                    f"(qty: {quantity:.6f}, SL: ${signal.get('stop_loss', 0):.2f}, "
                    f"TP: ${signal.get('take_profit', 0):.2f})")

    def _check_exits(self, current_data: Dict[str, pd.DataFrame], timestamp: datetime):
        """Check stop-loss and take-profit for open positions"""
        positions_to_close = []

        for symbol, position in self.state.positions.items():
            if symbol not in current_data or len(current_data[symbol]) == 0:
                continue

            current_candle = current_data[symbol].iloc[-1]
            high = current_candle['high']
            low = current_candle['low']
            close = current_candle['close']

            exit_price = None
            exit_reason = None

            if position.side == OrderSide.BUY:
                # Check stop-loss
                if position.stop_loss and low <= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "Stop Loss"
                # Check take-profit
                elif position.take_profit and high >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "Take Profit"
            else:  # SELL position
                # Check stop-loss
                if position.stop_loss and high >= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "Stop Loss"
                # Check take-profit
                elif position.take_profit and low <= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "Take Profit"

            if exit_price:
                self._close_position(position, exit_price,
                                     timestamp, exit_reason)
                positions_to_close.append(symbol)

        # Remove closed positions
        for symbol in positions_to_close:
            del self.state.positions[symbol]

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        timestamp: datetime,
        reason: str = "Manual"
    ):
        """Close a position and calculate P&L"""
        # Apply slippage on exit
        if position.side == OrderSide.BUY:
            fill_exit_price = exit_price * (1 - self.slippage_rate)
        else:
            fill_exit_price = exit_price * (1 + self.slippage_rate)

        # Calculate P&L
        if position.side == OrderSide.BUY:
            pnl = (fill_exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - fill_exit_price) * position.quantity

        # Calculate exit commission
        exit_value = position.quantity * fill_exit_price
        exit_commission = exit_value * self.commission_rate

        # Net P&L after commissions
        net_pnl = pnl - position.commission - exit_commission

        # Update position
        position.exit_price = fill_exit_price
        position.exit_timestamp = timestamp
        position.realized_pnl = net_pnl
        position.commission += exit_commission

        # Update capital
        self.state.current_capital += exit_value - exit_commission

        # Track wins/losses
        if net_pnl > 0:
            self.state.trades_won += 1
        else:
            self.state.trades_lost += 1

        # Add to closed positions
        self.state.closed_positions.append(position)

        logger.info(f"[{timestamp}] CLOSE {position.side.value.upper()} {position.symbol} @ ${fill_exit_price:.2f} "
                    f"({reason}) - P&L: ${net_pnl:.2f}")

    def _close_all_positions(self, data: Dict[str, pd.DataFrame], timestamp: datetime):
        """Close all remaining positions at market price"""
        for symbol, position in list(self.state.positions.items()):
            if symbol in data and len(data[symbol]) > 0:
                exit_price = data[symbol].iloc[-1]['close']
                self._close_position(position, exit_price,
                                     timestamp, "End of Backtest")

        self.state.positions.clear()

    def _update_equity(self, current_data: Dict[str, pd.DataFrame], timestamp: datetime):
        """Update current equity value"""
        # Start with current capital
        equity = self.state.current_capital

        # Add unrealized P&L from open positions
        for symbol, position in self.state.positions.items():
            if symbol in current_data and len(current_data[symbol]) > 0:
                current_price = current_data[symbol].iloc[-1]['close']

                if position.side == OrderSide.BUY:
                    unrealized_pnl = (
                        current_price - position.entry_price) * position.quantity
                else:
                    unrealized_pnl = (position.entry_price -
                                      current_price) * position.quantity

                equity += unrealized_pnl

        self.state.equity = equity
        self.state.equity_curve.append((timestamp, equity))

    def _filter_date_range(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Dict[str, pd.DataFrame]:
        """Filter data by date range"""
        filtered = {}

        for symbol, df in data.items():
            df_filtered = df.copy()

            if start_date:
                df_filtered = df_filtered[df_filtered.index >= pd.to_datetime(
                    start_date)]
            if end_date:
                df_filtered = df_filtered[df_filtered.index <= pd.to_datetime(
                    end_date)]

            if len(df_filtered) > 0:
                filtered[symbol] = df_filtered

        return filtered

    def _get_common_timestamps(self, data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get common timestamps across all symbols"""
        if not data:
            return []

        # Use the first symbol's timestamps as base
        first_symbol = list(data.keys())[0]
        return sorted(data[first_symbol].index.tolist())

    def _generate_results(self) -> Dict:
        """Generate backtest results summary"""
        results = {
            'initial_capital': self.state.initial_capital,
            'final_capital': self.state.current_capital,
            'final_equity': self.state.equity,
            'total_return': ((self.state.equity - self.state.initial_capital) / self.state.initial_capital) * 100,
            'total_trades': self.state.total_trades,
            'trades_won': self.state.trades_won,
            'trades_lost': self.state.trades_lost,
            'win_rate': (self.state.trades_won / self.state.total_trades * 100) if self.state.total_trades > 0 else 0,
            'closed_positions': self.state.closed_positions,
            'equity_curve': self.state.equity_curve
        }

        logger.info(f"\nBacktest Results:")
        logger.info(f"  Initial Capital: ${results['initial_capital']:,.2f}")
        logger.info(f"  Final Equity: ${results['final_equity']:,.2f}")
        logger.info(f"  Total Return: {results['total_return']:.2f}%")
        logger.info(f"  Total Trades: {results['total_trades']}")
        logger.info(f"  Win Rate: {results['win_rate']:.2f}%")

        return results
