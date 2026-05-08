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

    def __init__(self, config: Dict, db_manager, exchange=None):
        self.config = config
        self.db_manager = db_manager
        self.exchange = exchange  # Set later by TradingBot after OrderExecutor init
        self.risk_config = config['risk']
        self.trading_config = config['trading']

        # Risk parameters
        self.max_risk_per_trade = self.risk_config['max_risk_per_trade']
        self.max_daily_loss = self.risk_config['max_daily_loss']
        self.max_open_positions = self.trading_config['max_open_positions']

        # Portfolio — config value used as fallback when exchange balance isn't available
        self.initial_capital = config.get(
            'backtest', {}).get('initial_capital', 1000)
        self.current_capital = self.initial_capital
        self._cached_balance = None
        self._balance_cache_time = None

        logger.info(
            f"Risk Manager initialized - Max risk/trade: {self.max_risk_per_trade*100}%")

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size (margin) in USD.
        For futures, the exchange applies leverage on top of this margin.

        Returns USD margin amount (e.g. $20) — NOT base currency.
        """
        portfolio_value = self.get_portfolio_value()
        available_balance = self.get_available_balance()

        # Helper returns position size in BASE currency (e.g. BTC)
        position_size_base = calculate_position_size(
            portfolio_value,
            self.max_risk_per_trade,
            entry_price,
            stop_loss
        )

        # Convert base → USD (this is the margin amount)
        position_size_usd = position_size_base * entry_price

        # Cap at configured maximum margin per trade
        max_position = self.trading_config.get('position_size_usd', 100)
        position_size_usd = min(position_size_usd, max_position)

        # Never use more than 50% of portfolio on a single trade
        position_size_usd = min(position_size_usd, portfolio_value * 0.5)

        # Never exceed available balance (leaves $1 buffer for fees)
        if available_balance > 1:
            position_size_usd = min(position_size_usd, available_balance - 1)
        else:
            logger.warning(f"Insufficient available balance: ${available_balance:.2f}")
            return 0

        # Futures: ensure leveraged risk on SL hit stays within max_risk_per_trade
        market_type = self.trading_config.get('market_type', 'spot')
        if market_type == 'futures':
            leverage = self.trading_config.get('leverage', 1)
            price_risk = abs(entry_price - stop_loss)
            if price_risk > 0 and leverage > 1:
                leveraged_base = (position_size_usd * leverage) / entry_price
                max_loss_on_sl = price_risk * leveraged_base
                max_allowed_loss = portfolio_value * self.max_risk_per_trade
                if max_loss_on_sl > max_allowed_loss:
                    position_size_usd = (max_allowed_loss * entry_price) / (price_risk * leverage)
                    logger.info(f"Futures risk limit: margin reduced to ${position_size_usd:.2f}")

        # Final minimum check
        if position_size_usd < 5:  # Binance minimum notional
            logger.warning(f"Position size ${position_size_usd:.2f} too small (min $5)")
            return 0

        logger.info(
            f"Position size: ${position_size_usd:.2f} USD margin "
            f"(Portfolio: ${portfolio_value:.2f}, Available: ${available_balance:.2f})")

        return position_size_usd

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

        # Check available balance (need at least $5 for Binance minimum)
        available = self.get_available_balance()
        if available < 5:
            logger.warning(f"Insufficient available balance: ${available:.2f}")
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

        daily_pnl = today_trades['pnl'].dropna().sum()
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
        trades_df = self.db_manager.get_trade_history(limit=20)

        if trades_df.empty:
            return False

        # Only consider CLOSED trades (open trades have NaN pnl and would bypass the check)
        closed_trades = trades_df[trades_df['pnl'].notna()]

        if len(closed_trades) < 5:
            return False

        # Check last 5 closed trades
        last_5_trades = closed_trades.head(5)

        # If all last 5 closed trades are losses, trigger circuit breaker
        all_losses = all(last_5_trades['pnl'] < 0)

        if all_losses:
            logger.error("🔴 CIRCUIT BREAKER: 5 consecutive losses detected!")
            return True

        return False

    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value.
        Live mode: fetches actual wallet balance from exchange.
        Paper mode: uses initial_capital + closed trade P&L.
        """
        mode = self.trading_config.get('mode', 'paper')

        # Live mode: use real exchange balance
        if mode == 'live' and self.exchange:
            try:
                balance = self._fetch_exchange_balance()
                if balance is not None:
                    return balance
            except Exception as e:
                logger.debug(f"Could not fetch live balance, using DB fallback: {e}")

        # Paper mode or fallback: initial_capital + closed P&L
        trades_df = self.db_manager.get_trade_history(limit=1000)

        if trades_df.empty:
            return self.initial_capital

        closed_trades = trades_df[trades_df['pnl'].notna()]
        total_pnl = closed_trades['pnl'].sum() if not closed_trades.empty else 0

        return self.initial_capital + total_pnl

    def get_available_balance(self) -> float:
        """
        Get available balance (total - margin locked in open trades).
        Live mode: fetches free balance from exchange.
        Paper mode: portfolio value minus open trade margins.
        """
        mode = self.trading_config.get('mode', 'paper')

        # Live mode: use exchange's free balance (already deducts open margins)
        if mode == 'live' and self.exchange:
            try:
                balance = self._fetch_exchange_balance(free_only=True)
                if balance is not None:
                    return balance
            except Exception as e:
                logger.debug(f"Could not fetch free balance: {e}")

        # Paper mode or fallback: portfolio - open trade margins
        portfolio = self.get_portfolio_value()
        open_trades = self.db_manager.get_open_trades()
        locked_margin = sum(t.quantity for t in open_trades)  # quantity = USD margin

        available = portfolio - locked_margin
        return max(available, 0)

    def _fetch_exchange_balance(self, free_only: bool = False) -> float:
        """
        Fetch USDT balance from exchange. Caches for 30 seconds to avoid rate limits.
        """
        now = datetime.now()

        # Use cache if fresh (< 30 seconds old)
        if (self._cached_balance is not None and self._balance_cache_time
                and (now - self._balance_cache_time).total_seconds() < 30):
            bal = self._cached_balance
            return bal.get('free', 0) if free_only else bal.get('total', 0)

        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get('USDT', {})
            self._cached_balance = {
                'total': float(usdt.get('total', 0) or 0),
                'free': float(usdt.get('free', 0) or 0),
                'used': float(usdt.get('used', 0) or 0),
            }
            self._balance_cache_time = now

            logger.debug(f"Exchange balance: total=${self._cached_balance['total']:.2f}, "
                         f"free=${self._cached_balance['free']:.2f}, "
                         f"used=${self._cached_balance['used']:.2f}")

            return self._cached_balance['free'] if free_only else self._cached_balance['total']
        except Exception as e:
            logger.error(f"Failed to fetch exchange balance: {e}")
            return None

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
