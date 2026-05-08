"""
Order Executor
Executes trades via exchange API (paper or live mode)
"""

import ccxt
import threading
from typing import Dict, Optional
from datetime import datetime, timedelta
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
        self.market_type = self.trading_config.get('market_type', 'spot')  # 'spot' or 'futures'
        self.leverage = self.trading_config.get('leverage', 1)
        self.margin_mode = self.trading_config.get('margin_mode', 'isolated')
        self.slippage_rate = config.get('backtest', {}).get('slippage', 0.0005)  # 0.05% default
        self.commission_rate = config.get('backtest', {}).get('commission', 0.001)  # 0.1% default
        self.exchange = self._initialize_exchange()
        self._close_lock = threading.Lock()  # Prevent concurrent close_trade calls

        logger.info(f"Order Executor initialized - Mode: {self.mode.upper()}, Market: {self.market_type.upper()}, Leverage: {self.leverage}x")

    def _initialize_exchange(self):
        """Initialize exchange connection"""
        exchange_name = self.exchange_config['name']

        exchange_class = getattr(ccxt, exchange_name)

        # Set defaultType based on market_type config
        default_type = 'future' if self.market_type == 'futures' else 'spot'

        exchange = exchange_class({
            'apiKey': self.exchange_config['api_key'],
            'secret': self.exchange_config['api_secret'],
            'enableRateLimit': True,
            'options': {
                'defaultType': default_type
            }
        })

        # If testnet is enabled, ensure defaultType matches market_type
        # (data_fetcher forces 'future' for testnet — match that here)
        if self.exchange_config.get('testnet') and self.market_type == 'futures':
            exchange.options['defaultType'] = 'future'

        # Set sandbox mode for paper trading
        if self.mode == 'paper' or self.exchange_config.get('testnet'):
            exchange.set_sandbox_mode(True)
            logger.info(f"Using {exchange_name} in PAPER/TESTNET mode")

        logger.info(f"Exchange defaultType: {exchange.options.get('defaultType', default_type)}")

        return exchange

    def calculate_liquidation_price(self, entry_price: float, direction: str, leverage: int = None) -> Optional[float]:
        """
        Calculate estimated liquidation price for an isolated margin futures position.
        Binance isolated margin liquidation ≈ entry ± entry/leverage (simplified).
        Maintenance margin rate varies by tier but ~0.4% for small positions.
        """
        lev = leverage or self.leverage
        if lev <= 1:
            return None  # no liquidation on 1x or spot

        maint_margin_rate = 0.004  # 0.4% maintenance margin (Binance tier 1)

        if direction == 'long':
            liq_price = entry_price * (1 - (1 / lev) + maint_margin_rate)
        else:  # short
            liq_price = entry_price * (1 + (1 / lev) - maint_margin_rate)

        return round(liq_price, 2)

    def fetch_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current funding rate for a futures symbol.
        Returns dict with 'rate' (e.g. 0.0001 = 0.01%), 'next_funding_time', etc.
        """
        if self.market_type != 'futures':
            return None
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            rate = funding.get('fundingRate', 0)
            next_time = funding.get('fundingDatetime', None)
            logger.debug(f"Funding rate for {symbol}: {rate*100:.4f}% | Next: {next_time}")
            return {'rate': rate, 'next_funding_time': next_time, 'raw': funding}
        except Exception as e:
            logger.debug(f"Could not fetch funding rate for {symbol}: {e}")
            return None

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
        """Execute trade in paper trading mode (simulated with realistic slippage)"""
        # Check if split entry is enabled
        split_entry_cfg = self.config.get('risk', {}).get('split_entry', {})
        use_split_entry = split_entry_cfg.get('enabled', False)

        # Apply slippage to entry price (market order fills worse than expected)
        entry_price = signal['entry_price']
        if signal['direction'] == 'long':
            entry_price *= (1 + self.slippage_rate)  # Buy slightly higher
        else:
            entry_price *= (1 - self.slippage_rate)  # Sell slightly lower

        logger.info(f"  Paper slippage: {signal['entry_price']:.2f} → {entry_price:.2f} "
                    f"({self.slippage_rate*100:.3f}%)")

        total_quantity = signal['quantity']

        if use_split_entry:
            # SPLIT ENTRY: 40% immediate, 60% limit order near SL
            immediate_frac = split_entry_cfg.get('immediate_fraction', 0.40)
            limit_frac = split_entry_cfg.get('limit_fraction', 0.60)
            limit_distance = split_entry_cfg.get('limit_distance_from_sl', 0.30)
            expiry_hours = split_entry_cfg.get('limit_order_expiry_hours', 4)

            immediate_qty = total_quantity * immediate_frac
            limit_qty = total_quantity * limit_frac

            # Calculate limit order price (30% of the way from entry toward SL)
            limit_price = self._calculate_limit_order_price(
                entry_price, signal['stop_loss'], signal['direction'], limit_distance
            )

            logger.info(f"  [SPLIT ENTRY] Immediate: ${immediate_qty:.2f} @ ${entry_price:.2f} (40%)")
            logger.info(f"  [SPLIT ENTRY] Limit order: ${limit_qty:.2f} @ ${limit_price:.2f} (60%) - near SL")

            # Save trade with immediate portion only
            trade_data = {
                'symbol': signal['symbol'],
                'side': 'buy' if signal['direction'] == 'long' else 'sell',
                'entry_price': entry_price,
                'quantity': immediate_qty,  # Only 40% executed now
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'entry_time': datetime.now(),
                'strategy': 'SMC-Split',
                'timeframe': '15m',
                'notes': signal.get('reason', '') + f' | Split entry: {immediate_frac*100:.0f}% filled, {limit_frac*100:.0f}% pending @ ${limit_price:.2f}'
            }

            trade_id = self.db_manager.save_trade(trade_data)

            # Mark trade as split entry
            self.db_manager.update_trade(trade_id, {
                'is_split_entry': True,
                'split_entry_filled': False
            })

            # Create pending limit order for the remaining 60%
            limit_order_data = {
                'trade_id': trade_id,
                'symbol': signal['symbol'],
                'side': 'buy' if signal['direction'] == 'long' else 'sell',
                'limit_price': limit_price,
                'quantity': limit_qty,
                'created_time': datetime.now(),
                'expiry_time': datetime.now() + timedelta(hours=expiry_hours)
            }
            self.db_manager.save_pending_limit_order(limit_order_data)

            logger.info(f"[SUCCESS] PAPER TRADE (SPLIT) executed - ID: {trade_id}")
            logger.info(f"  Immediate: ${immediate_qty:.2f} @ ${entry_price:.2f}")
            logger.info(f"  Pending: ${limit_qty:.2f} @ ${limit_price:.2f} (expires in {expiry_hours}h)")

        else:
            # NORMAL ENTRY: 100% at market
            trade_data = {
                'symbol': signal['symbol'],
                'side': 'buy' if signal['direction'] == 'long' else 'sell',
                'entry_price': entry_price,
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

    def _calculate_limit_order_price(self, entry_price: float, stop_loss: float, direction: str, distance_fraction: float) -> float:
        """
        Calculate limit order price for split entry.
        Places order at distance_fraction of the way from entry toward SL.

        Example: entry=100, SL=90, distance=0.30 → limit price = 100 - (10 * 0.30) = 97
        This gives a better average entry if price retraces before going your way.
        """
        sl_distance = abs(entry_price - stop_loss)
        offset = sl_distance * distance_fraction

        if direction == 'long':
            # For long: limit order below entry (toward SL which is below)
            limit_price = entry_price - offset
        else:
            # For short: limit order above entry (toward SL which is above)
            limit_price = entry_price + offset

        return round(limit_price, 2)

    def fill_limit_order(self, order_id: int, fill_price: float):
        """
        Fill a pending limit order and update the parent trade.
        Called when price reaches the limit order price.
        """
        with self._close_lock:
            self._fill_limit_order_impl(order_id, fill_price)

    def _fill_limit_order_impl(self, order_id: int, fill_price: float):
        """Internal implementation of limit order fill"""
        try:
            # Get the pending order
            pending_orders = self.db_manager.get_pending_limit_orders()
            order = None
            for o in pending_orders:
                if o.id == order_id:
                    order = o
                    break

            if not order:
                logger.warning(f"Limit order {order_id} not found or already filled")
                return

            # Get the parent trade
            trade = self.db_manager.get_trade(order.trade_id)
            if not trade:
                logger.error(f"Parent trade {order.trade_id} not found for limit order {order_id}")
                return

            if trade.status != 'open':
                # Trade was closed before limit order filled - cancel the limit order
                self.db_manager.cancel_pending_limit_order(order_id, 'trade_closed')
                logger.info(f"Limit order {order_id} cancelled - parent trade already closed")
                return

            # Apply slippage to fill price
            if self.mode == 'paper':
                if order.side == 'buy':
                    fill_price *= (1 + self.slippage_rate)
                else:
                    fill_price *= (1 - self.slippage_rate)

            # Calculate new weighted average entry price
            old_qty = trade.quantity
            new_qty = order.quantity
            total_qty = old_qty + new_qty
            old_entry = trade.entry_price

            avg_entry = ((old_entry * old_qty) + (fill_price * new_qty)) / total_qty

            # Update the trade
            self.db_manager.update_trade(order.trade_id, {
                'quantity': total_qty,
                'entry_price': avg_entry,  # Update to weighted average
                'avg_entry_price': avg_entry,
                'split_entry_filled': True,
                'notes': (trade.notes or '') + f' | Limit filled @ ${fill_price:.2f}, avg entry: ${avg_entry:.2f}'
            })

            # Mark order as filled
            self.db_manager.fill_pending_limit_order(order_id, fill_price)

            logger.info(f"[SPLIT ENTRY FILLED] Trade #{order.trade_id}: Limit order filled @ ${fill_price:.2f}")
            logger.info(f"  Old entry: ${old_entry:.2f} x ${old_qty:.2f}")
            logger.info(f"  Limit fill: ${fill_price:.2f} x ${new_qty:.2f}")
            logger.info(f"  New avg entry: ${avg_entry:.2f}, Total position: ${total_qty:.2f}")

        except Exception as e:
            logger.error(f"Error filling limit order {order_id}: {e}", exc_info=True)

    def _setup_futures_market(self, symbol: str):
        """Set leverage, margin mode, and position mode for a futures symbol"""
        try:
            # Set position mode to One-Way (required — Hedge Mode uses different params)
            try:
                self.exchange.set_position_mode(False, symbol)  # False = One-Way Mode
                logger.debug(f"Position mode set to One-Way for {symbol}")
            except Exception as e:
                # Already set or not supported — safe to ignore
                logger.debug(f"Position mode note: {e}")

            # Set margin mode (isolated or cross)
            try:
                self.exchange.set_margin_mode(self.margin_mode, symbol)
                logger.info(f"Margin mode set to {self.margin_mode} for {symbol}")
            except Exception as e:
                # Some exchanges throw if already set to the same mode
                logger.debug(f"Margin mode note: {e}")

            # Set leverage
            self.exchange.set_leverage(self.leverage, symbol)
            logger.info(f"Leverage set to {self.leverage}x for {symbol}")

            # Log funding rate
            funding = self.fetch_funding_rate(symbol)
            if funding:
                logger.info(f"Current funding rate for {symbol}: {funding['rate']*100:.4f}%")

        except Exception as e:
            logger.error(f"Failed to setup futures market for {symbol}: {e}")

    def _execute_live_trade(self, signal: Dict) -> Optional[int]:
        """Execute real trade on exchange with stop-loss and take-profit"""
        try:
            symbol = signal['symbol']
            side = 'buy' if signal['direction'] == 'long' else 'sell'
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']

            # Setup futures market if in futures mode
            if self.market_type == 'futures':
                self._setup_futures_market(symbol)

            # Calculate position size (quantity in base currency)
            quantity_usd = signal['quantity']
            # In futures mode, leverage amplifies the position
            if self.market_type == 'futures':
                effective_usd = quantity_usd * self.leverage
            else:
                effective_usd = quantity_usd
            # Convert to base currency (BTC, ETH, etc.)
            amount = effective_usd / entry_price

            # Place main market order
            logger.info(f"Placing {side} order for {amount} {symbol} ({'futures ' + str(self.leverage) + 'x' if self.market_type == 'futures' else 'spot'})")
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )

            actual_price = order.get(
                'average', order.get('price', entry_price))
            logger.info(f"Order filled at {actual_price}: {order}")

            # Place stop-loss order (CRITICAL for risk management)
            sl_order_id = None
            tp_order_id = None
            sl_side = 'sell' if side == 'buy' else 'buy'  # Opposite of entry

            sl_placed = False
            for sl_attempt in range(1, 4):  # 3 retries
                try:
                    if self.market_type == 'futures':
                        # Futures: use STOP_MARKET (no limit price, just trigger)
                        sl_order = self.exchange.create_order(
                            symbol=symbol,
                            type='STOP_MARKET',
                            side=sl_side,
                            amount=amount,
                            params={
                                'stopPrice': stop_loss,
                                'closePosition': False,
                                'workingType': 'MARK_PRICE'  # Use mark price to avoid wicks
                            }
                        )
                    else:
                        # Spot: regular stop-loss
                        sl_order = self.exchange.create_order(
                            symbol=symbol,
                            type='stop_loss',
                            side=sl_side,
                            amount=amount,
                            price=stop_loss,
                            params={'stopPrice': stop_loss}
                        )

                    sl_order_id = sl_order.get('id')
                    sl_placed = True
                    logger.info(
                        f"Stop-loss order placed at {stop_loss}: {sl_order_id}")
                    break
                except Exception as e:
                    logger.error(f"SL placement attempt {sl_attempt}/3 failed: {e}")
                    if sl_attempt < 3:
                        import time as _time
                        _time.sleep(0.5)

            # CRITICAL: If SL order failed after 3 retries, close the position immediately
            # A live position without stop-loss protection is unacceptable
            if not sl_placed:
                logger.error("CRITICAL: Stop-loss order FAILED after 3 attempts — closing position to avoid unprotected exposure")
                try:
                    close_params = {'reduceOnly': True} if self.market_type == 'futures' else {}
                    self.exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=sl_side,
                        amount=amount,
                        params=close_params
                    )
                    logger.info("Emergency close successful — position closed without SL")
                except Exception as close_err:
                    logger.error(f"CRITICAL: Emergency close also failed: {close_err} — MANUAL INTERVENTION NEEDED")
                return None  # Do not save trade to DB — position was aborted

            # Place take-profit order
            try:
                tp_side = 'sell' if side == 'buy' else 'buy'  # Opposite of entry

                if self.market_type == 'futures':
                    # Futures: use TAKE_PROFIT_MARKET
                    tp_order = self.exchange.create_order(
                        symbol=symbol,
                        type='TAKE_PROFIT_MARKET',
                        side=tp_side,
                        amount=amount,
                        params={
                            'stopPrice': take_profit,
                            'closePosition': False,
                            'workingType': 'MARK_PRICE'
                        }
                    )
                else:
                    # Spot: regular limit order
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
            direction = signal['direction']
            liq_price = None
            if self.market_type == 'futures':
                liq_price = self.calculate_liquidation_price(actual_price, direction, self.leverage)

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
                'notes': f"Order: {order.get('id')}, SL: {sl_order_id}, TP: {tp_order_id}" +
                         (f", Liq: {liq_price}, Lev: {self.leverage}x" if self.market_type == 'futures' else '')
            }

            trade_id = self.db_manager.save_trade(trade_data)

            logger.info(f"[SUCCESS] LIVE TRADE executed - ID: {trade_id}")
            logger.info(
                f"   Entry: {actual_price}, SL: {stop_loss}, TP: {take_profit}")
            if liq_price:
                logger.info(f"   Liquidation price: {liq_price} ({self.leverage}x {self.margin_mode})")

            return trade_id

        except Exception as e:
            logger.error(f"Error in live trade execution: {e}", exc_info=True)
            return None

    def close_trade(self, trade_id: int, exit_price: float, reason: str):
        """
        Close an open trade (manual close or emergency exit)
        Thread-safe: uses a lock to prevent concurrent closes of the same trade.

        Args:
            trade_id: Database trade ID
            exit_price: Exit price
            reason: Reason for closing (manual, emergency, etc.)
        """
        with self._close_lock:
            self._close_trade_impl(trade_id, exit_price, reason)

    def partial_close_trade(self, trade_id: int, current_price: float, close_fraction: float):
        """
        Partially close a trade (e.g. 30% at 1:1 R:R).
        Thread-safe: uses same lock as close_trade.

        - Calculates PnL on the closed portion
        - Reduces trade quantity in DB
        - Moves SL to breakeven (entry price)
        - Sets partial_tp_taken = True
        - In live mode: reduces position on exchange

        Args:
            trade_id: Database trade ID
            current_price: Current market price
            close_fraction: Fraction to close (e.g. 0.30 for 30%)
        """
        with self._close_lock:
            self._partial_close_impl(trade_id, current_price, close_fraction)

    def _partial_close_impl(self, trade_id: int, current_price: float, close_fraction: float):
        """Internal partial close implementation (must be called under _close_lock)"""
        try:
            trade = self.db_manager.get_trade(trade_id)
            if not trade or trade.status != 'open':
                return

            if getattr(trade, 'partial_tp_taken', False):
                logger.debug(f"Trade {trade_id} already had partial TP taken, skipping")
                return

            close_qty = trade.quantity * close_fraction  # USD amount to close
            remaining_qty = trade.quantity - close_qty

            # Apply slippage in paper mode
            exit_price = current_price
            if self.mode == 'paper':
                if trade.side == 'buy':
                    exit_price *= (1 - self.slippage_rate)
                else:
                    exit_price *= (1 + self.slippage_rate)

            # Calculate PnL on closed portion
            config = self.config
            market_type = config.get('trading', {}).get('market_type', 'spot')
            leverage = config.get('trading', {}).get('leverage', 1)
            commission_rate = config.get('backtest', {}).get('commission', 0.001)

            effective_leverage = leverage if market_type == 'futures' else 1
            closed_base = (close_qty * effective_leverage) / trade.entry_price

            if trade.side == 'buy':
                partial_pnl = (exit_price - trade.entry_price) * closed_base
            else:
                partial_pnl = (trade.entry_price - exit_price) * closed_base

            # Deduct commission on closed portion
            entry_commission = trade.entry_price * closed_base * commission_rate
            exit_commission = exit_price * closed_base * commission_rate
            partial_pnl -= (entry_commission + exit_commission)

            # Live mode: reduce position on exchange
            if self.mode == 'live':
                try:
                    symbol = trade.symbol
                    side = 'sell' if trade.side == 'buy' else 'buy'
                    close_amount_base = (close_qty * (leverage if market_type == 'futures' else 1)) / trade.entry_price

                    params = {'reduceOnly': True} if market_type == 'futures' else {}
                    close_order = self.exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=side,
                        amount=close_amount_base,
                        params=params
                    )
                    actual_exit = close_order.get('average', close_order.get('price', exit_price))
                    if actual_exit:
                        exit_price = actual_exit
                    logger.info(f"Partial close executed on exchange at {exit_price}")

                    # Cancel existing SL/TP orders (we'll need new ones for reduced size)
                    try:
                        open_orders = self.exchange.fetch_open_orders(symbol)
                        for order in open_orders:
                            try:
                                self.exchange.cancel_order(order['id'], symbol)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Place new SL at breakeven for remaining position
                    remaining_base = (remaining_qty * (leverage if market_type == 'futures' else 1)) / trade.entry_price
                    sl_side = 'sell' if trade.side == 'buy' else 'buy'
                    try:
                        if market_type == 'futures':
                            self.exchange.create_order(
                                symbol=symbol,
                                type='STOP_MARKET',
                                side=sl_side,
                                amount=remaining_base,
                                params={'stopPrice': trade.entry_price, 'workingType': 'MARK_PRICE'}
                            )
                        logger.info(f"New SL at breakeven ({trade.entry_price}) for remaining position")
                    except Exception as e:
                        logger.error(f"Failed to place new breakeven SL: {e}")

                    # Place new TP for remaining position
                    try:
                        if market_type == 'futures':
                            self.exchange.create_order(
                                symbol=symbol,
                                type='TAKE_PROFIT_MARKET',
                                side=sl_side,
                                amount=remaining_base,
                                params={'stopPrice': trade.take_profit, 'workingType': 'MARK_PRICE'}
                            )
                        logger.info(f"New TP at {trade.take_profit} for remaining position")
                    except Exception as e:
                        logger.error(f"Failed to place new TP: {e}")

                except Exception as e:
                    logger.error(f"Failed to partially close on exchange: {e}")
                    return

            # Update trade in DB: reduce quantity, move SL to breakeven, mark partial TP
            update_data = {
                'quantity': remaining_qty,
                'stop_loss': trade.entry_price,  # Move SL to breakeven
                'partial_tp_taken': True,
                'partial_tp_price': exit_price,  # Record the exact partial TP fill price
                'partial_pnl': (getattr(trade, 'partial_pnl', 0) or 0) + partial_pnl,
                'notes': (trade.notes or '') + f" | Partial TP: closed {close_fraction*100:.0f}% at {exit_price:.2f}, PnL ${partial_pnl:.2f}"
            }
            self.db_manager.update_trade(trade_id, update_data)

            logger.info(
                f"[PARTIAL TP] Trade #{trade_id}: closed {close_fraction*100:.0f}% at ${exit_price:.2f} "
                f"(PnL: ${partial_pnl:.2f}), SL moved to breakeven ${trade.entry_price:.2f}, "
                f"remaining ${remaining_qty:.2f} rides to TP ${trade.take_profit:.2f}")

        except Exception as e:
            logger.error(f"Error in partial close for trade {trade_id}: {e}", exc_info=True)

    def _close_trade_impl(self, trade_id: int, exit_price: float, reason: str):
        """Internal close implementation (must be called under _close_lock)"""
        try:
            logger.info(
                f"Closing trade {trade_id} at {exit_price} - Reason: {reason}")

            # Get trade details from database
            trade = self.db_manager.get_trade(trade_id)

            if not trade:
                logger.error(f"Trade {trade_id} not found")
                return

            if trade.status == 'closed':
                logger.info(f"Trade {trade_id} already closed, skipping")
                return

            # Paper mode: apply realistic slippage on exit
            if self.mode == 'paper':
                original_exit = exit_price
                if trade.side == 'buy':  # Long position closing = selling
                    exit_price *= (1 - self.slippage_rate)
                else:  # Short position closing = buying
                    exit_price *= (1 + self.slippage_rate)
                if abs(exit_price - original_exit) > 0.001:
                    logger.info(f"  Paper exit slippage: {original_exit:.2f} → {exit_price:.2f}")

            # If live mode, close position on exchange
            if self.mode == 'live':
                try:
                    symbol = trade.symbol

                    # Check if position still exists on exchange (may have been liquidated or SL/TP filled)
                    position_exists = True
                    if self.market_type == 'futures':
                        position_exists = self._check_position_exists(symbol, trade.side)

                    if not position_exists:
                        logger.info(f"Position for trade {trade_id} no longer exists on exchange "
                                    f"(likely filled by exchange SL/TP or liquidated)")
                        # Still update DB with the exit price
                        self.db_manager.close_trade(trade_id, exit_price, datetime.now())
                        logger.info(f"[SUCCESS] Trade {trade_id} synced as closed")
                        return

                    # Opposite side
                    side = 'sell' if trade.side == 'buy' else 'buy'

                    # Calculate correct close amount in base currency
                    # trade.quantity is in USD, need to convert to base
                    if self.market_type == 'futures':
                        amount = (trade.quantity * self.leverage) / trade.entry_price
                    else:
                        amount = trade.quantity / trade.entry_price

                    # Cancel any open SL/TP orders for this symbol FIRST
                    try:
                        open_orders = self.exchange.fetch_open_orders(symbol)
                        for order in open_orders:
                            try:
                                self.exchange.cancel_order(order['id'], symbol)
                                logger.info(f"Cancelled pending order {order['id']} for {symbol}")
                            except Exception:
                                pass
                    except Exception as e:
                        logger.debug(f"Could not cancel open orders: {e}")

                    # Place market order to close — with retry
                    max_retries = 3
                    for attempt in range(1, max_retries + 1):
                        try:
                            if self.market_type == 'futures':
                                close_order = self.exchange.create_order(
                                    symbol=symbol,
                                    type='market',
                                    side=side,
                                    amount=amount,
                                    params={'reduceOnly': True}
                                )
                            else:
                                close_order = self.exchange.create_order(
                                    symbol=symbol,
                                    type='market',
                                    side=side,
                                    amount=amount
                                )

                            # Use actual fill price if available
                            actual_exit = close_order.get('average', close_order.get('price', exit_price))
                            if actual_exit:
                                exit_price = actual_exit

                            logger.info(f"Position closed on exchange at {exit_price}: {close_order}")
                            break  # Success
                        except Exception as e:
                            logger.error(f"Close attempt {attempt}/{max_retries} failed: {e}")
                            if attempt < max_retries:
                                import time
                                time.sleep(1)
                            else:
                                logger.error(f"CRITICAL: Failed to close position after {max_retries} attempts! "
                                             f"Manual intervention needed for {symbol}")
                                # Do NOT update DB — position is still open on exchange
                                return

                except Exception as e:
                    logger.error(f"Error closing position on exchange: {e}")

            # Update database
            self.db_manager.close_trade(trade_id, exit_price, datetime.now())

            logger.info(f"[SUCCESS] Trade {trade_id} closed successfully")

        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}", exc_info=True)

    def _check_position_exists(self, symbol: str, side: str) -> bool:
        """
        Check if a position still exists on the exchange.
        Used to detect if Binance filled SL/TP or liquidated the position.
        """
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                pos_side = pos.get('side', '').lower()
                pos_amount = abs(float(pos.get('contracts', 0) or pos.get('contractSize', 0) or 0))
                pos_notional = abs(float(pos.get('notional', 0) or 0))

                # Check if there's an actual open position
                if (pos_amount > 0 or pos_notional > 0):
                    # Match direction: our 'buy' = exchange 'long', our 'sell' = exchange 'short'
                    expected_side = 'long' if side == 'buy' else 'short'
                    if pos_side == expected_side:
                        return True
            return False
        except Exception as e:
            logger.debug(f"Could not check position for {symbol}: {e}")
            return True  # Assume exists if we can't check (safer)
