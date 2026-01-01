"""
AI-Enhanced Smart Money Concept-Based Crypto Trading Bot
Main entry point for the trading bot

FYP - Fall 2025
Authors: Muhammad Taha Aaqib, Omer Farooq, Asadullah Bin Hassan
Advisor: Muhammad Asim Butt
"""

from auto_retrain import AutoRetrainer
from src.models.ensemble_model import EnsembleDecisionModel
from src.models.trading_model import TradingModel
from src.trading.order_executor import OrderExecutor
from src.trading.risk_manager import RiskManager
from src.trading.strategy import TradingStrategy
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_fetcher import DataFetcher
from src.utils.db_manager import DatabaseManager
import sys
import time
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

        # Initialize auto-retrainer
        self.auto_retrainer = AutoRetrainer(config_path)

        # Check if model needs retraining on startup
        logger.info("Checking if model needs retraining...")
        retrain_result = self.auto_retrainer.retrain_if_needed()
        if retrain_result['retrained']:
            logger.info(f"Model retrained: {retrain_result['reason']}")

        logger.info("Trading bot initialized successfully!")
        logger.info(f"Mode: {self.config['trading']['mode'].upper()}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Timeframes: {self.timeframes}")

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
        """Main trading logic - analyze markets and execute trades"""
        logger.info("Running trading analysis...")

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

                # Generate trading signal using ensemble (if enabled) or basic strategy
                if self.ensemble:
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

                if signal:
                    # Calculate stop loss and take profit if not already set
                    if 'stop_loss' not in signal:
                        atr = df_15m.iloc[-1].get('atr', 0)
                        if signal['direction'] == 'long':
                            signal['stop_loss'] = signal['entry_price'] - \
                                (atr * self.config['risk']
                                 ['stop_loss_atr_multiplier'])
                        else:
                            signal['stop_loss'] = signal['entry_price'] + \
                                (atr * self.config['risk']
                                 ['stop_loss_atr_multiplier'])

                    if 'take_profit' not in signal:
                        risk = abs(signal['entry_price'] - signal['stop_loss'])
                        rr_ratio = self.config['risk']['take_profit_rr_ratio']
                        if signal['direction'] == 'long':
                            signal['take_profit'] = signal['entry_price'] + \
                                (risk * rr_ratio)
                        else:
                            signal['take_profit'] = signal['entry_price'] - \
                                (risk * rr_ratio)

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
                        f"Trade {trade.id} should exit - Reason: {reason}")
                    self.order_executor.close_trade(
                        trade.id, current_price, reason)
                    logger.info(f"Trade {trade.id} closed - Reason: {reason}")
                else:
                    logger.info(f"  Trade {trade.id} holding (no exit signal)")

            except Exception as e:
                logger.error(
                    f"Error managing trade {trade.id}: {e}", exc_info=True)

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

    def check_model_retraining(self):
        """Check if model needs retraining"""
        logger.info("🔄 Automatic retraining check...")

        try:
            retrain_result = self.auto_retrainer.retrain_if_needed()

            if retrain_result['retrained']:
                logger.info(f"✅ Model retrained successfully!")
                logger.info(f"   Reason: {retrain_result['reason']}")

                # Reload the ML model with new weights
                if self.ml_model:
                    logger.info("Reloading ML model...")
                    self.ml_model.load_model()

                if self.ensemble:
                    logger.info("Reloading ensemble models...")
                    # Reload all symbol-specific models
                    for symbol, ml_model in self.ensemble.ml_models.items():
                        logger.info(f"  Reloading {symbol} model...")
                        ml_model.load_model()

            else:
                logger.info(
                    f"ℹ️  Model is up-to-date: {retrain_result['reason']}")

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
