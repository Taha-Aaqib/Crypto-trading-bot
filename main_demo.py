"""
DEMO MODE - Main Bot with Relaxed Rules
Trades more frequently for demonstration purposes
Uses looser conditions to show the bot in action within 1-2 hours
"""

import sys
import time
from datetime import datetime
import schedule
from src.utils.logger import get_logger
from src.utils.helpers import load_config, create_directories
from src.utils.db_manager import DatabaseManager
from src.data.data_fetcher import DataFetcher
from src.data.data_preprocessor import DataPreprocessor
from src.trading.strategy import TradingStrategy
from src.trading.risk_manager import RiskManager
from src.trading.order_executor import OrderExecutor
from src.models.trading_model import TradingModel
from src.models.ensemble_model import EnsembleDecisionModel

logger = get_logger()


class DemoTradingBot:
    """Demo trading bot with relaxed rules for faster demonstration"""

    def __init__(self, config_path='config/config.yaml'):
        logger.info("=" * 60)
        logger.info("[DEMO MODE] - AI Trading Bot")
        logger.info("=" * 60)
        logger.info(
            "WARNING: DEMO MODE - Using relaxed rules for faster trading")
        logger.info("   - Only 2/4 conditions needed (vs 4/4 in real mode)")
        logger.info("   - Lower confidence threshold (40% vs 60%)")
        logger.info("   - More frequent signals expected")
        logger.info("=" * 60)

        # Load configuration
        self.config = load_config(config_path)

        # Override config for demo mode
        self.config['ensemble']['min_confidence'] = 0.4  # Lower from 0.6
        self.config['ensemble']['min_signals'] = 1  # Lower from 2

        # Create necessary directories
        create_directories()

        # Initialize components
        self.db_manager = DatabaseManager(self.config)
        self.data_fetcher = DataFetcher(self.config)
        self.data_preprocessor = DataPreprocessor(self.config)
        self.strategy = TradingStrategy(self.config)
        self.risk_manager = RiskManager(self.config, self.db_manager)
        self.order_executor = OrderExecutor(self.config, self.db_manager)

        # Initialize ML model and ensemble (if enabled)
        self.ml_model = None
        self.ensemble = None
        if self.config.get('ml_model', {}).get('enabled', False):
            logger.info("Initializing ML model...")
            self.ml_model = TradingModel(self.config)

        if self.config.get('ensemble', {}).get('enabled', False):
            logger.info("Initializing ensemble model...")
            self.ensemble = EnsembleDecisionModel(self.config)

        # Bot state
        self.is_running = False
        self.symbols = self.config['trading']['symbols']
        self.timeframes = self.config['timeframes']

        logger.info("Demo bot initialized successfully!")
        logger.info(f"Mode: {self.config['trading']['mode'].upper()} (DEMO)")
        logger.info(f"Symbols: {', '.join(self.symbols)}")

    def fetch_multi_timeframe_data(self, symbol: str):
        """Fetch data for all required timeframes"""
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

            # Fetch 15M data (for entry)
            df_15m = self.data_fetcher.fetch_ohlcv(
                symbol,
                timeframe=self.timeframes['entry'],
                limit=500
            )

            # Preprocess data
            df_1d = self.data_preprocessor.process_pipeline(df_1d)
            df_4h = self.data_preprocessor.process_pipeline(df_4h)
            df_15m = self.data_preprocessor.process_pipeline(df_15m)

            return df_1d, df_4h, df_15m

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None, None, None

    def analyze_and_trade(self):
        """Main trading logic with DEMO MODE relaxed rules"""
        logger.info("[DEMO] Running DEMO MODE analysis...")

        for symbol in self.symbols:
            try:
                # Fetch multi-timeframe data first (needed for managing trades)
                df_1d, df_4h, df_15m = self.fetch_multi_timeframe_data(symbol)

                if df_1d is None or df_4h is None or df_15m is None:
                    logger.warning(f"Skipping {symbol} - insufficient data")
                    continue

                # ALWAYS manage existing trades first (even if can't open new ones)
                self.manage_open_trades(symbol, df_15m)

                # Check if we can take new trades
                if not self.risk_manager.can_open_new_trade():
                    logger.info("Risk limit reached - no new trades")
                    continue

                # Perform multi-timeframe analysis
                mtf_analysis = self.strategy.analyze_multi_timeframe(
                    df_1d, df_4h, df_15m)

                # DEMO MODE: Relaxed signal generation
                signal = self.generate_demo_signal(
                    symbol, mtf_analysis, df_15m)

                if signal:
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        signal['entry_price'],
                        signal['stop_loss']
                    )

                    if position_size > 0:
                        signal['quantity'] = position_size

                        # Execute trade
                        trade_id = self.order_executor.execute_trade(signal)

                        if trade_id:
                            logger.info(
                                f"[SUCCESS] DEMO TRADE executed successfully: {trade_id}")
                            logger.info(
                                f"   {signal['direction'].upper()} {symbol} @ ${signal['entry_price']:.2f}")
                    else:
                        logger.warning(
                            "Position size too small - skipping trade")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)

    def generate_demo_signal(self, symbol: str, mtf_analysis: dict, df_15m):
        """
        Generate signal with DEMO MODE relaxed rules
        Only needs 2/4 conditions instead of 4/4
        """
        bias_1d = mtf_analysis['bias_1d']
        structure_4h = mtf_analysis['structure_4h']
        ema_trend = mtf_analysis['ema_trend_15m']
        latest_15m = mtf_analysis['latest_15m']

        # Count bullish conditions
        bullish_conditions = 0
        if bias_1d == 'bullish':
            bullish_conditions += 1
        if structure_4h == 'bullish':
            bullish_conditions += 1
        if ema_trend == 'bullish':
            bullish_conditions += 1
        if latest_15m.get('choch_bullish') or latest_15m.get('bos_bullish') or latest_15m.get('fvg_bullish'):
            bullish_conditions += 1

        # Count bearish conditions
        bearish_conditions = 0
        if bias_1d == 'bearish':
            bearish_conditions += 1
        if structure_4h == 'bearish':
            bearish_conditions += 1
        if ema_trend == 'bearish':
            bearish_conditions += 1
        if latest_15m.get('choch_bearish') or latest_15m.get('bos_bearish') or latest_15m.get('fvg_bearish'):
            bearish_conditions += 1

        logger.info(
            f"[DEMO] {symbol}: Bullish {bullish_conditions}/4, Bearish {bearish_conditions}/4")

        # DEMO MODE: Only need 2/4 conditions
        signal = None
        if bullish_conditions >= 2:
            signal = {
                'symbol': symbol,
                'direction': 'long',
                'entry_price': latest_15m['close'],
                'timestamp': datetime.now(),
                'reason': f'DEMO: {bullish_conditions}/4 bullish conditions (relaxed rules)'
            }
            logger.info(f"[DEMO] LONG signal for {symbol}!")

        elif bearish_conditions >= 2:
            signal = {
                'symbol': symbol,
                'direction': 'short',
                'entry_price': latest_15m['close'],
                'timestamp': datetime.now(),
                'reason': f'DEMO: {bearish_conditions}/4 bearish conditions (relaxed rules)'
            }
            logger.info(f"[DEMO] SHORT signal for {symbol}!")

        if not signal:
            logger.info(f"No signal for {symbol} - even with relaxed rules")
            return None

        # Calculate stop loss and take profit
        atr = latest_15m.get('atr', 0)
        signal['stop_loss'] = self._calculate_stop_loss(
            signal['entry_price'],
            signal['direction'],
            atr
        )
        signal['take_profit'] = self._calculate_take_profit(
            signal['entry_price'],
            signal['stop_loss'],
            signal['direction']
        )

        return signal

    def _calculate_stop_loss(self, entry_price: float, direction: str, atr: float) -> float:
        """Calculate stop loss using ATR"""
        atr_multiplier = self.config['risk']['stop_loss_atr_multiplier']

        if direction == 'long':
            stop_loss = entry_price - (atr * atr_multiplier)
        else:  # short
            stop_loss = entry_price + (atr * atr_multiplier)

        return stop_loss

    def _calculate_take_profit(self, entry_price: float, stop_loss: float, direction: str) -> float:
        """Calculate take profit using risk-reward ratio"""
        rr_ratio = self.config['risk']['take_profit_rr_ratio']
        risk = abs(entry_price - stop_loss)

        if direction == 'long':
            take_profit = entry_price + (risk * rr_ratio)
        else:  # short
            take_profit = entry_price - (risk * rr_ratio)

        return take_profit

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
                    f"[DEMO] Checking trade {trade.id}: {trade.symbol} {trade.side} @ ${trade.entry_price:.2f}")
                logger.info(
                    f"  Current: ${current_price:.2f}, SL: ${trade.stop_loss:.2f}, TP: ${trade.take_profit:.2f}")

                # Check if trade should be exited
                should_exit, reason = self.strategy.should_exit_trade(
                    trade_dict,
                    current_price,
                    df_15m
                )

                if should_exit:
                    logger.info(
                        f"[DEMO] Trade {trade.id} should exit - Reason: {reason}")
                    self.order_executor.close_trade(
                        trade.id, current_price, reason)
                    logger.info(
                        f"[DEMO] Trade {trade.id} closed - Reason: {reason}")
                else:
                    logger.info(f"  Trade {trade.id} holding (no exit signal)")

            except Exception as e:
                logger.error(
                    f"Error managing trade {trade.id}: {e}", exc_info=True)
                logger.error(f"Error managing trade {trade.id}: {e}")

    def run_schedule(self):
        """Set up and run scheduled tasks"""
        # Run analysis every 15 minutes (matching entry timeframe)
        schedule.every(15).minutes.do(self.analyze_and_trade)

        logger.info("[DEMO] DEMO MODE scheduler started")
        logger.info("   Analysis every 15 minutes")
        logger.info("   Expected: More frequent trades than normal mode")
        logger.info("   Press Ctrl+C to stop")

        # Run immediately on start
        self.analyze_and_trade()

        # Keep running
        self.is_running = True
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

    def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("Shutting down demo bot...")
        self.is_running = False

        # Close database connection
        self.db_manager.close()

        logger.info("Demo bot stopped successfully")


def main():
    """Main entry point"""
    print()
    print("=" * 70)
    print("[DEMO MODE] - AI Trading Bot")
    print("=" * 70)
    print()
    print("WARNING: This is DEMO MODE with relaxed trading rules!")
    print()
    print("Differences from normal mode:")
    print("  - Only 2/4 conditions needed (vs 4/4)")
    print("  - Lower confidence threshold (40% vs 60%)")
    print("  - MORE FREQUENT TRADES expected")
    print()
    print("Purpose: Demonstrate the bot working within 1-2 hours")
    print()
    print("Still using PAPER TRADING - No real money at risk!")
    print("=" * 70)
    print()

    input("Press ENTER to start demo mode... ")

    try:
        # Initialize and run the bot
        bot = DemoTradingBot()
        bot.run_schedule()

    except Exception as e:
        logger.error(f"Critical error in demo mode: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
