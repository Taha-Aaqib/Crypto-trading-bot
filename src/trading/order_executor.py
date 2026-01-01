"""
Order Executor
Executes trades via exchange API (paper or live mode)
"""

import ccxt
from typing import Dict, Optional
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger()


class OrderExecutor:
    """Execute trading orders on exchange"""

    def __init__(self, config: Dict, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.exchange_config = config['exchange']
        self.trading_config = config['trading']

        self.mode = self.trading_config['mode']  # 'paper' or 'live'
        self.exchange = self._initialize_exchange()

        logger.info(f"Order Executor initialized - Mode: {self.mode.upper()}")

    def _initialize_exchange(self):
        """Initialize exchange connection"""
        exchange_name = self.exchange_config['name']

        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'apiKey': self.exchange_config['api_key'],
            'secret': self.exchange_config['api_secret'],
            'enableRateLimit': True
        })

        # Set sandbox mode for paper trading
        if self.mode == 'paper' or self.exchange_config.get('testnet'):
            exchange.set_sandbox_mode(True)
            logger.info(f"Using {exchange_name} in PAPER/TESTNET mode")

        return exchange

    def execute_trade(self, signal: Dict) -> Optional[int]:
        """
        Execute a trade based on signal

        Args:
            signal: Trading signal dictionary

        Returns:
            Trade ID if successful, None otherwise
        """
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            entry_price = signal['entry_price']
            quantity = signal['quantity']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']

            logger.info(f"Executing {direction.upper()} trade for {symbol}")
            logger.info(
                f"Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}, Qty: {quantity}")

            # Paper trading mode
            if self.mode == 'paper':
                trade_id = self._execute_paper_trade(signal)
            else:
                # Live trading mode
                trade_id = self._execute_live_trade(signal)

            return trade_id

        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return None

    def _execute_paper_trade(self, signal: Dict) -> int:
        """Execute trade in paper trading mode (simulated)"""
        # Save to database as open trade
        trade_data = {
            'symbol': signal['symbol'],
            'side': 'buy' if signal['direction'] == 'long' else 'sell',
            'entry_price': signal['entry_price'],
            'quantity': signal['quantity'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'entry_time': datetime.now(),
            'strategy': 'SMC',
            'timeframe': '15m',
            'notes': signal.get('reason', '')
        }

        trade_id = self.db_manager.save_trade(trade_data)

        logger.info(f"[SUCCESS] PAPER TRADE executed - ID: {trade_id}")

        return trade_id

    def _execute_live_trade(self, signal: Dict) -> Optional[int]:
        """Execute real trade on exchange with stop-loss and take-profit"""
        try:
            symbol = signal['symbol']
            side = 'buy' if signal['direction'] == 'long' else 'sell'
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']

            # Calculate position size (quantity in base currency)
            quantity_usd = signal['quantity']
            # Convert to base currency (BTC, ETH, etc.)
            amount = quantity_usd / entry_price

            # Place main market order
            logger.info(f"Placing {side} order for {amount} {symbol}")
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )

            actual_price = order.get(
                'average', order.get('price', entry_price))
            logger.info(f"Order filled at {actual_price}: {order}")

            # Place stop-loss order
            sl_order_id = None
            tp_order_id = None

            try:
                sl_side = 'sell' if side == 'buy' else 'buy'  # Opposite of entry
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop_loss',
                    side=sl_side,
                    amount=amount,
                    price=stop_loss,
                    params={'stopPrice': stop_loss}
                )
                sl_order_id = sl_order.get('id')
                logger.info(
                    f"Stop-loss order placed at {stop_loss}: {sl_order_id}")
            except Exception as e:
                logger.error(f"Failed to place stop-loss order: {e}")

            # Place take-profit order
            try:
                tp_side = 'sell' if side == 'buy' else 'buy'  # Opposite of entry
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=tp_side,
                    amount=amount,
                    price=take_profit
                )
                tp_order_id = tp_order.get('id')
                logger.info(
                    f"Take-profit order placed at {take_profit}: {tp_order_id}")
            except Exception as e:
                logger.error(f"Failed to place take-profit order: {e}")

            # Save to database
            trade_data = {
                'symbol': signal['symbol'],
                'side': side,
                'entry_price': actual_price,
                'quantity': quantity_usd,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'strategy': 'AI-Enhanced SMC',
                'timeframe': '15m',
                'notes': f"Order: {order.get('id')}, SL: {sl_order_id}, TP: {tp_order_id}"
            }

            trade_id = self.db_manager.save_trade(trade_data)

            logger.info(f"[SUCCESS] LIVE TRADE executed - ID: {trade_id}")
            logger.info(
                f"   Entry: {actual_price}, SL: {stop_loss}, TP: {take_profit}")

            return trade_id

        except Exception as e:
            logger.error(f"Error in live trade execution: {e}", exc_info=True)
            return None

    def close_trade(self, trade_id: int, exit_price: float, reason: str):
        """
        Close an open trade (manual close or emergency exit)

        Args:
            trade_id: Database trade ID
            exit_price: Exit price
            reason: Reason for closing (manual, emergency, etc.)
        """
        try:
            logger.info(
                f"Closing trade {trade_id} at {exit_price} - Reason: {reason}")

            # Get trade details from database
            trade = self.db_manager.get_trade(trade_id)

            if not trade:
                logger.error(f"Trade {trade_id} not found")
                return

            # If live mode, close position on exchange
            if self.mode == 'live':
                try:
                    symbol = trade.symbol
                    # Opposite side
                    side = 'sell' if trade.side == 'buy' else 'buy'
                    amount = trade.quantity / trade.entry_price

                    # Place market order to close
                    close_order = self.exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=side,
                        amount=amount
                    )

                    logger.info(f"Position closed on exchange: {close_order}")

                    # Cancel any open SL/TP orders
                    # (This requires order IDs - would need to be stored in trade notes)

                except Exception as e:
                    logger.error(f"Error closing position on exchange: {e}")

            # Update database
            self.db_manager.close_trade(trade_id, exit_price, datetime.now())

            logger.info(f"[SUCCESS] Trade {trade_id} closed successfully")

        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}", exc_info=True)
