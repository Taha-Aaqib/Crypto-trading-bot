"""
AI-Enhanced Smart Money Concept-Based Crypto Trading Bot
Main entry point for the trading bot

FYP - Fall 2025
Authors: Muhammad Taha Aaqib, Omer Farooq, Asadullah Bin Hassan
Advisor: Muhammad Asim Butt
"""

from auto_retrain import AutoRetrainer
from src.models.ensemble_model import EnsembleDecisionModel
from src.trading.order_executor import OrderExecutor
from src.trading.risk_manager import RiskManager
from src.trading.strategy import TradingStrategy
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_fetcher import DataFetcher
from src.utils.db_manager import DatabaseManager
import sys
import time
import threading
from datetime import datetime
import schedule
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.helpers import load_config, create_directories

# Load environment variables from .env file
load_dotenv()

logger = get_logger()


class TradingBot:
    """Main trading bot orchestrator"""

    def __init__(self, config_path='config/config.yaml'):
        logger.info("=" * 60)
        logger.info(
            "AI-Enhanced Smart Money Concept Trading Bot - Initializing")
        logger.info("=" * 60)

        # Load configuration
        self.config = load_config(config_path)

        # Create necessary directories
        create_directories()

        # Initialize components
        self.db_manager = DatabaseManager(self.config)
        self.data_fetcher = DataFetcher(self.config)
        self.data_preprocessor = DataPreprocessor(self.config)
        self.strategy = TradingStrategy(self.config)
        self.order_executor = OrderExecutor(self.config, self.db_manager)
        self.risk_manager = RiskManager(self.config, self.db_manager)

        # Give risk manager access to exchange for live balance checks
        self.risk_manager.exchange = self.order_executor.exchange

        # Initialize ensemble model (if enabled)
        self.ensemble = None
        if self.config.get('ensemble', {}).get('enabled', False):
            logger.info("Initializing ensemble model...")
            self.ensemble = EnsembleDecisionModel(self.config)

        # Bot state
        self.is_running = False
        self.symbols = self.config['trading']['symbols']
        self.timeframes = self.config['timeframes']

        # Initialize auto-retrainer
        self.auto_retrainer = AutoRetrainer(config_path)

        # Check if model needs retraining on startup
        logger.info("Checking if model needs retraining...")
        retrain_result = self.auto_retrainer.retrain_if_needed()
        if retrain_result['retrained']:
            logger.info(f"Model retrained: {retrain_result['reason']}")

        logger.info("Trading bot initialized successfully!")
        logger.info(f"Mode: {self.config['trading']['mode'].upper()}")
        logger.info(f"Market: {self.config['trading'].get('market_type', 'spot').upper()}")
        if self.config['trading'].get('market_type') == 'futures':
            logger.info(f"Leverage: {self.config['trading'].get('leverage', 1)}x | Margin: {self.config['trading'].get('margin_mode', 'isolated').upper()}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Timeframes: {self.timeframes}")

    def fetch_multi_timeframe_data(self, symbol: str):
        """Fetch data for all required timeframes (1D, 4H, 1H, 15M)"""
        logger.info(f"Fetching multi-timeframe data for {symbol}")

        try:
            # Fetch 1D data (for bias)
            df_1d = self.data_fetcher.fetch_ohlcv(
                symbol,
                timeframe=self.timeframes['bias'],
                limit=100
            )

            # Fetch 4H data (for structure)
            df_4h = self.data_fetcher.fetch_ohlcv(
                symbol,
                timeframe=self.timeframes['structure'],
                limit=200
            )

            # Fetch 1H data (for pattern refinement - used by ML)
            df_1h = self.data_fetcher.fetch_ohlcv(
                symbol,
                timeframe='1h',
                limit=300
            )

            # Fetch 15M data (for entry)
            df_15m = self.data_fetcher.fetch_ohlcv(
                symbol,
                timeframe=self.timeframes['entry'],
                limit=500
            )

            # Preprocess data
            df_1d = self.data_preprocessor.process_pipeline(df_1d)
            df_4h = self.data_preprocessor.process_pipeline(df_4h)
            df_1h = self.data_preprocessor.process_pipeline(df_1h)
            df_15m = self.data_preprocessor.process_pipeline(df_15m)

            return df_1d, df_4h, df_1h, df_15m

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None, None, None, None

    def _get_validated_live_entry_price(self, symbol: str, reference_price: float):
        """
        Validate signal reference price (typically 15m close) against live ticker.
        Returns live price if deviation is acceptable, else None.
        """
        try:
            ticker = self.data_fetcher.fetch_ticker(symbol)
            live_price = ticker.get('last') if ticker else None

            if live_price is None:
                logger.warning(
                    f"[PRICE GUARD] No live ticker for {symbol}; skipping entry to avoid bad fill")
                return None

            max_dev_pct = self.config.get('trading', {}).get('max_price_deviation_pct', 0.005)
            deviation_pct = abs(live_price - reference_price) / reference_price if reference_price > 0 else 0

            if deviation_pct > max_dev_pct:
                logger.error(
                    f"[PRICE GUARD] {symbol} rejected: signal/ref=${reference_price:.2f}, "
                    f"live=${live_price:.2f}, deviation={deviation_pct*100:.2f}% "
                    f"> {max_dev_pct*100:.2f}%")
                return None

            return live_price
        except Exception as e:
            logger.error(f"[PRICE GUARD] Error validating live price for {symbol}: {e}")
            return None

    def analyze_and_trade(self):
        """Main trading logic - analyze markets and execute trades"""
        logger.info("Running trading analysis...")

        # User-requested safety mode: when 2 trades are already open,
        # skip signal generation and rely on the 2s price monitor thread.
        open_trades_now = self.db_manager.get_open_trades()
        if len(open_trades_now) >= 2:
            logger.info(
                "Monitor-only mode: 2 open trades detected; skipping new-trade analysis this cycle")
            return

        for symbol in self.symbols:
            try:
                # Fetch multi-timeframe data first (needed for managing trades)
                df_1d, df_4h, df_1h, df_15m = self.fetch_multi_timeframe_data(symbol)

                if df_1d is None or df_4h is None or df_1h is None or df_15m is None:
                    logger.warning(f"Skipping {symbol} - insufficient data")
                    continue

                # ALWAYS manage existing trades first (even if can't open new ones)
                self.manage_open_trades(symbol, df_15m)

                # Check if we can take new trades
                if not self.risk_manager.can_open_new_trade():
                    logger.info("Risk limit reached - no new trades")
                    continue

                # Skip if there's already an open trade for this symbol
                open_trades = self.db_manager.get_open_trades()
                has_open = any(t.symbol == symbol for t in open_trades)
                if has_open:
                    logger.info(f"Skipping {symbol} - already has an open trade")
                    continue

                # Perform multi-timeframe analysis
                mtf_analysis = self.strategy.analyze_multi_timeframe(
                    df_1d, df_4h, df_15m)

                # Generate trading signal using ensemble (if enabled) or basic strategy
                if self.ensemble:
                    # Update ensemble's MTF cache for ML predictions
                    # This enables full 1D + 4H + 1H + 15M pattern recognition
                    self.ensemble.update_mtf_cache(symbol, df_1d, df_4h, df_1h, df_15m)

                    # Use ensemble decision (SMC 40% + TA 25% + ML 25% + Sentiment 10%)
                    logger.info(f"Using ensemble model for {symbol}")
                    ensemble_result = self.ensemble.analyze_signal(
                        df_15m, symbol)

                    # Convert ensemble result to signal format
                    # Note: ensemble returns 'long'/'short'/'hold' signals
                    if ensemble_result['signal'] == 'long' and ensemble_result['confidence'] >= self.ensemble.min_confidence:
                        signal = {
                            'symbol': symbol,
                            'direction': 'long',
                            'entry_price': df_15m.iloc[-1]['close'],
                            'timestamp': datetime.now(),
                            'reason': f"Ensemble LONG (confidence: {ensemble_result['confidence']:.2f}, agreement: {ensemble_result['agreement']:.2f})",
                            'ensemble_details': ensemble_result
                        }
                    elif ensemble_result['signal'] == 'short' and ensemble_result['confidence'] >= self.ensemble.min_confidence:
                        signal = {
                            'symbol': symbol,
                            'direction': 'short',
                            'entry_price': df_15m.iloc[-1]['close'],
                            'timestamp': datetime.now(),
                            'reason': f"Ensemble SHORT (confidence: {ensemble_result['confidence']:.2f}, agreement: {ensemble_result['agreement']:.2f})",
                            'ensemble_details': ensemble_result
                        }
                    else:
                        signal = None
                        if ensemble_result['signal'] == 'hold':
                            logger.info(
                                f"Ensemble signal: HOLD - market is neutral, no clear direction")
                        else:
                            logger.info(
                                f"Ensemble signal: {ensemble_result['signal']} (confidence: {ensemble_result['confidence']:.2f}) - below threshold {self.ensemble.min_confidence}")
                else:
                    # Use basic multi-timeframe SMC strategy
                    signal = self.strategy.generate_signal(
                        symbol, mtf_analysis)

                # Note: Ensemble already includes sentiment as a weighted component (10%),
                # so no additional hard sentiment filter is applied here.
                # (strategy.generate_signal applies its own sentiment filter for the non-ensemble path)

                if signal:
                    # Safety: reject entries when candle close and live ticker diverge too much.
                    # If valid, use live ticker as entry for more realistic paper/live behavior.
                    validated_live_entry = self._get_validated_live_entry_price(
                        symbol, signal['entry_price'])
                    if validated_live_entry is None:
                        logger.warning(
                            f"Skipping {symbol} trade due to price inconsistency between signal and live ticker")
                        continue

                    if abs(validated_live_entry - signal['entry_price']) > 1e-9:
                        logger.info(
                            f"Using live entry for {symbol}: {signal['entry_price']:.2f} -> {validated_live_entry:.2f}")
                    signal['entry_price'] = validated_live_entry

                    # Spot mode: block short signals (spot = buy only)
                    market_type = self.config['trading'].get('market_type', 'spot')
                    if market_type == 'spot' and signal.get('direction') == 'short':
                        logger.info(f"Skipping SHORT signal for {symbol} — spot mode only supports LONG")
                        signal = None

                if signal:
                    # Calculate stop loss using strategy's unified method (swing points + ATR + % fallback)
                    if 'stop_loss' not in signal:
                        entry_price = signal['entry_price']
                        df_15m_analyzed = mtf_analysis.get('df_15m_analyzed', df_15m)

                        atr = df_15m_analyzed.iloc[-1].get('atr', 0)

                        signal['stop_loss'] = self.strategy._calculate_stop_loss(
                            entry_price,
                            signal['direction'],
                            atr,
                            df_smc=df_15m_analyzed
                        )

                    if 'take_profit' not in signal:
                        signal['take_profit'] = self.strategy._calculate_take_profit(
                            signal['entry_price'],
                            signal['stop_loss'],
                            signal['direction']
                        )

                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        signal['entry_price'],
                        signal['stop_loss']
                    )

                    if position_size > 0:
                        signal['quantity'] = position_size

                        # Final safety: validate R:R ratio before committing
                        is_valid, reason = self.risk_manager.validate_trade(signal)
                        if not is_valid:
                            logger.warning(f"Trade rejected by risk validation: {reason}")
                            continue

                        # Execute trade
                        trade_id = self.order_executor.execute_trade(signal)

                        if trade_id:
                            logger.info(
                                f"[SUCCESS] Trade executed successfully: {trade_id}")
                    else:
                        logger.warning(
                            "Position size too small - skipping trade")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)

    def manage_open_trades(self, symbol: str, df_15m):
        """Monitor and manage open trades"""
        open_trades = self.db_manager.get_open_trades()

        for trade in open_trades:
            if trade.symbol != symbol:
                continue

            try:
                current_price = df_15m.iloc[-1]['close']

                # Convert trade to dict and add direction property
                trade_dict = trade.__dict__.copy()
                # Add the property manually
                trade_dict['direction'] = trade.direction

                # Debug logging
                logger.info(
                    f"Checking trade {trade.id}: {trade.symbol} {trade.side} @ ${trade.entry_price:.2f}")
                sl_str = f"${trade.stop_loss:.2f}" if trade.stop_loss is not None else "N/A"
                tp_str = f"${trade.take_profit:.2f}" if trade.take_profit is not None else "N/A"
                logger.info(
                    f"  Current: ${current_price:.2f}, SL: {sl_str}, TP: {tp_str}")

                # Check if trade should be exited
                should_exit, reason = self.strategy.should_exit_trade(
                    trade_dict,
                    current_price,
                    df_15m
                )

                if should_exit:
                    logger.info(
                        f"Trade {trade.id} should exit - Reason: {reason}")
                    self.order_executor.close_trade(
                        trade.id, current_price, reason)
                    logger.info(f"Trade {trade.id} closed - Reason: {reason}")
                else:
                    logger.info(f"  Trade {trade.id} holding (no exit signal)")

            except Exception as e:
                logger.error(
                    f"Error managing trade {trade.id}: {e}", exc_info=True)

    def _price_monitor_loop(self):
        """
        Background thread: checks live price against SL/TP every 2 seconds.
        Also monitors pending limit orders for split entry fills.
        Only does simple price comparison — no SMC/counter-trend logic
        (that still runs in the 15-min manage_open_trades cycle).
        """
        logger.info("[Price Monitor] Started — checking SL/TP every 2s")

        while self.is_running:
            try:
                # Expire old limit orders periodically
                self.db_manager.expire_old_limit_orders()

                open_trades = self.db_manager.get_open_trades()
                pending_limit_orders = self.db_manager.get_pending_limit_orders()

                if not open_trades and not pending_limit_orders:
                    time.sleep(2)
                    continue

                # Group by symbol to minimize API calls
                symbols = set(t.symbol for t in open_trades)
                symbols.update(o.symbol for o in pending_limit_orders)

                for symbol in symbols:
                    try:
                        ticker = self.data_fetcher.fetch_ticker(symbol)
                        if not ticker or 'last' not in ticker:
                            continue

                        live_price = ticker['last']

                        # --- CHECK PENDING LIMIT ORDERS (Split Entry) ---
                        for order in pending_limit_orders:
                            if order.symbol != symbol:
                                continue

                            # Check if price crossed the limit order price
                            should_fill = False
                            if order.side == 'buy':
                                # Limit buy fills when price drops to or below limit
                                if live_price <= order.limit_price:
                                    should_fill = True
                            else:  # sell
                                # Limit sell fills when price rises to or above limit
                                if live_price >= order.limit_price:
                                    should_fill = True

                            if should_fill:
                                logger.info(
                                    f"[SPLIT ENTRY] Limit order #{order.id} triggered @ ${live_price:.2f} "
                                    f"(limit: ${order.limit_price:.2f})")
                                self.order_executor.fill_limit_order(order.id, live_price)

                        # --- CHECK OPEN TRADES ---
                        for trade in open_trades:
                            if trade.symbol != symbol:
                                continue

                            # Guard: skip trades with missing SL/TP
                            if trade.stop_loss is None or trade.take_profit is None:
                                logger.debug(f"[Price Monitor] Trade #{trade.id} has no SL/TP, skipping price check")
                                continue

                            # --- Partial Take Profit Check ---
                            # Close 30% at 1:1 R:R, move SL to breakeven
                            partial_tp_cfg = self.config.get('risk', {}).get('partial_tp', {})
                            if partial_tp_cfg.get('enabled', False) and not getattr(trade, 'partial_tp_taken', False):
                                rr_trigger = partial_tp_cfg.get('rr_trigger', 1.0)
                                close_fraction = partial_tp_cfg.get('close_fraction', 0.30)
                                risk_distance = abs(trade.entry_price - trade.stop_loss)
                                partial_tp_price = (
                                    trade.entry_price + risk_distance * rr_trigger
                                    if trade.direction == 'long'
                                    else trade.entry_price - risk_distance * rr_trigger
                                )

                                if trade.direction == 'long' and live_price >= partial_tp_price:
                                    logger.info(
                                        f"[Price Monitor] PARTIAL TP triggered for trade #{trade.id} "
                                        f"at ${live_price:.2f} (1:1 level: ${partial_tp_price:.2f})")
                                    self.order_executor.partial_close_trade(
                                        trade.id, live_price, close_fraction)
                                    continue  # Skip full SL/TP check this cycle
                                elif trade.direction == 'short' and live_price <= partial_tp_price:
                                    logger.info(
                                        f"[Price Monitor] PARTIAL TP triggered for trade #{trade.id} "
                                        f"at ${live_price:.2f} (1:1 level: ${partial_tp_price:.2f})")
                                    self.order_executor.partial_close_trade(
                                        trade.id, live_price, close_fraction)
                                    continue  # Skip full SL/TP check this cycle

                            # --- Trailing Stop Check ---
                            trailing_cfg = self.config.get('risk', {}).get('trailing_stop', {})
                            if trailing_cfg.get('enabled', False):
                                activation_rr = trailing_cfg.get('activation_rr', 1.0)
                                trail_atr_mult = trailing_cfg.get('trail_atr_multiplier', 0.5)
                                risk_dist = abs(trade.entry_price - trade.stop_loss)
                                
                                activation_price = (
                                    trade.entry_price + risk_dist * activation_rr
                                    if trade.direction == 'long'
                                    else trade.entry_price - risk_dist * activation_rr
                                )
                                
                                # Get current ATR for trailing distance
                                trail_dist = risk_dist * trail_atr_mult  # fallback: fraction of risk
                                
                                activated = (
                                    (trade.direction == 'long' and live_price >= activation_price) or
                                    (trade.direction == 'short' and live_price <= activation_price)
                                )
                                
                                if activated:
                                    if trade.direction == 'long':
                                        new_sl = live_price - trail_dist
                                        if new_sl > trade.stop_loss:
                                            old_sl = trade.stop_loss
                                            trade.stop_loss = new_sl
                                            logger.info(
                                                f"[Trailing] Trade #{trade.id} SL moved "
                                                f"${old_sl:.2f} → ${new_sl:.2f} (price: ${live_price:.2f})")
                                    else:  # short
                                        new_sl = live_price + trail_dist
                                        if new_sl < trade.stop_loss:
                                            old_sl = trade.stop_loss
                                            trade.stop_loss = new_sl
                                            logger.info(
                                                f"[Trailing] Trade #{trade.id} SL moved "
                                                f"${old_sl:.2f} → ${new_sl:.2f} (price: ${live_price:.2f})")

                            hit = False
                            reason = ''

                            if trade.direction == 'long':
                                if live_price <= trade.stop_loss:
                                    hit, reason = True, 'stop_loss'
                                elif live_price >= trade.take_profit:
                                    hit, reason = True, 'take_profit'
                            else:  # short
                                if live_price >= trade.stop_loss:
                                    hit, reason = True, 'stop_loss'
                                elif live_price <= trade.take_profit:
                                    hit, reason = True, 'take_profit'

                            # Futures: check liquidation proximity (close at 90% to liq)
                            if not hit and self.config['trading'].get('market_type') == 'futures':
                                liq_price = self.order_executor.calculate_liquidation_price(
                                    trade.entry_price,
                                    trade.direction,
                                    self.config['trading'].get('leverage', 1)
                                )
                                if liq_price:
                                    # Close when price is 90% of the way to liquidation
                                    if trade.direction == 'long':
                                        danger_zone = trade.entry_price - (trade.entry_price - liq_price) * 0.90
                                        if live_price <= danger_zone:
                                            hit, reason = True, 'near_liquidation'
                                    else:
                                        danger_zone = trade.entry_price + (liq_price - trade.entry_price) * 0.90
                                        if live_price >= danger_zone:
                                            hit, reason = True, 'near_liquidation'

                            if hit:
                                logger.info(
                                    f"[Price Monitor] {reason.upper()} HIT for trade #{trade.id} "
                                    f"({trade.symbol} {trade.side}) — price ${live_price:.2f}")

                                # Cancel any pending limit order for this trade before closing
                                pending_order = self.db_manager.get_pending_limit_order_for_trade(trade.id)
                                if pending_order:
                                    self.db_manager.cancel_pending_limit_order(
                                        pending_order.id, 'trade_closed')
                                    logger.info(
                                        f"[SPLIT ENTRY] Cancelled pending limit order #{pending_order.id} "
                                        f"(trade closed)")

                                self.order_executor.close_trade(
                                    trade.id, live_price, reason)

                    except Exception as e:
                        logger.debug(f"[Price Monitor] Ticker error for {symbol}: {e}")

                time.sleep(2)

            except Exception as e:
                logger.error(f"[Price Monitor] Error: {e}")
                time.sleep(5)

        logger.info("[Price Monitor] Stopped")

    def run_schedule(self):
        """Set up and run scheduled tasks"""
        # Run analysis every 15 minutes (matching entry timeframe)
        schedule.every(15).minutes.do(self.analyze_and_trade)

        # Daily performance summary
        schedule.every().day.at("00:00").do(self.daily_summary)

        # Automatic model retraining check (every 6 hours)
        schedule.every(6).hours.do(self.check_model_retraining)

        logger.info("Scheduler started - bot will run every 15 minutes")
        logger.info("Automatic model retraining enabled (checks every 6 hours)")
        logger.info("Press Ctrl+C to stop")

        # Start real-time price monitor in background thread
        self.is_running = True
        price_thread = threading.Thread(
            target=self._price_monitor_loop, daemon=True, name="PriceMonitor")
        price_thread.start()

        # Run immediately on start
        self.analyze_and_trade()

        # Keep running
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutdown signal received...")
                self.shutdown()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

    def check_model_retraining(self):
        """Check if model needs retraining"""
        logger.info("Automatic retraining check...")

        try:
            retrain_result = self.auto_retrainer.retrain_if_needed()

            if retrain_result['retrained']:
                logger.info(f"Model retrained successfully!")
                logger.info(f"   Reason: {retrain_result['reason']}")

                # Reload ensemble ML models with new weights
                if self.ensemble:
                    logger.info("Reloading ensemble models...")
                    # Reload all symbol-specific models
                    for symbol, ml_model in self.ensemble.ml_models.items():
                        logger.info(f"  Reloading {symbol} model...")
                        ml_model.load_model()

            else:
                logger.info(
                    f"Model is up-to-date: {retrain_result['reason']}")

        except Exception as e:
            logger.error(f"Error during automatic retraining: {e}")

    def daily_summary(self):
        """Generate daily performance summary"""
        logger.info("=" * 60)
        logger.info("DAILY PERFORMANCE SUMMARY")
        logger.info("=" * 60)

        # Get today's trades
        trades_df = self.db_manager.get_trade_history(limit=100)
        today = datetime.now().date()

        if not trades_df.empty:
            today_trades = trades_df[trades_df['entry_time'].dt.date == today]
            # Only include closed trades (exclude open trades with NaN pnl)
            today_trades = today_trades[today_trades['pnl'].notna()]

            if not today_trades.empty:
                total_trades = len(today_trades)
                winning_trades = len(today_trades[today_trades['pnl'] > 0])
                losing_trades = len(today_trades[today_trades['pnl'] < 0])
                win_rate = (winning_trades / total_trades *
                            100) if total_trades > 0 else 0
                total_pnl = today_trades['pnl'].sum()

                logger.info(f"Total Trades: {total_trades}")
                logger.info(f"Winning Trades: {winning_trades}")
                logger.info(f"Losing Trades: {losing_trades}")
                logger.info(f"Win Rate: {win_rate:.2f}%")
                logger.info(f"Total PnL: ${total_pnl:.2f}")

                # Save metrics to database
                metrics = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'portfolio_value': self.risk_manager.get_portfolio_value()
                }
                self.db_manager.save_performance_metrics(metrics)
            else:
                logger.info("No trades today")

        logger.info("=" * 60)

    def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("Shutting down trading bot...")
        self.is_running = False

        # Close database connection
        self.db_manager.close()

        logger.info("Trading bot stopped successfully")


def main():
    """Main entry point"""
    try:
        # Initialize and run the bot
        bot = TradingBot()
        bot.run_schedule()

    except Exception as e:
        logger.error(f"Critical error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
