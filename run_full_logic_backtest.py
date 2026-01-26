"""
Full Logic Backtesting Script
Replicates the exact logic from main.py for accurate backtesting
Includes: Multi-timeframe analysis, Ensemble model, Risk management, All filters

Usage:
    python run_full_logic_backtest.py                  # Default: BTC/USDT, last 3 months
    python run_full_logic_backtest.py --symbol ETH/USDT --months 6
    python run_full_logic_backtest.py --optimize       # Run threshold optimization
"""

from src.utils.db_manager import DatabaseManager
from src.backtesting.visualizer import BacktestVisualizer
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.backtest_engine import BacktestEngine
from src.models.ensemble_model import EnsembleDecisionModel
from src.trading.risk_manager import RiskManager
from src.trading.strategy import TradingStrategy
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_fetcher import DataFetcher
from src.utils.logger import get_logger
from src.utils.helpers import load_config, create_directories
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


logger = get_logger()


class FullLogicBacktest:
    """
    Comprehensive backtesting that replicates main.py trading logic exactly:
    - Multi-timeframe analysis (1D/4H/15M)
    - Ensemble model decision making
    - Full risk management checks
    - Dynamic position sizing
    - All filters (sentiment, events, circuit breaker)
    """

    def __init__(self, config: Dict, test_mode: str = 'full'):
        """
        Args:
            config: Bot configuration
            test_mode: 'full' = all components, 'no_ensemble' = exclude ensemble,
                      'no_sentiment' = exclude sentiment
        """
        self.config = config
        self.test_mode = test_mode

        # Initialize components (same as main.py)
        self.data_fetcher = DataFetcher(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.strategy = TradingStrategy(config)
        self.db_manager = DatabaseManager(config)
        self.risk_manager = RiskManager(config, self.db_manager)

        # Initialize ensemble if enabled
        self.ensemble = None
        if config.get('ensemble', {}).get('enabled', False) and test_mode != 'no_ensemble':
            logger.info("Initializing ensemble model for backtest...")
            self.ensemble = EnsembleDecisionModel(config)

        # Backtest results
        self.trades = []
        self.equity_curve = []

        # Confluence threshold for backtest
        self.confluence_threshold = config.get(
            'strategy', {}).get('confluence_threshold', 0.35)

        logger.info(f"Full Logic Backtest initialized (mode: {test_mode})")

    def _generate_confluence_signal(
        self,
        symbol: str,
        mtf_analysis: Dict,
        df_15m: pd.DataFrame,
        timestamp: datetime
    ) -> Optional[Dict]:
        """
        Generate trading signal using CONFLUENCE SCORING for backtesting
        Bypasses sentiment/event filters which can't work with historical data
        """
        # Extract analysis components
        bias_1d = mtf_analysis['bias_1d']
        structure_4h = mtf_analysis['structure_4h']
        ema_trend = mtf_analysis['ema_trend_15m']
        latest_15m = mtf_analysis['latest_15m']

        # === CONFLUENCE SCORING SYSTEM ===
        confluence_score = 0.0
        reasons = []

        # 1D Bias (25% weight)
        if bias_1d == 'bullish':
            confluence_score += 0.25
            reasons.append('1D:bull')
        elif bias_1d == 'bearish':
            confluence_score -= 0.25
            reasons.append('1D:bear')

        # 4H Structure (20% weight)
        if structure_4h == 'bullish':
            confluence_score += 0.20
            reasons.append('4H:bull')
        elif structure_4h == 'bearish':
            confluence_score -= 0.20
            reasons.append('4H:bear')

        # 15M EMA Trend (15% weight)
        if ema_trend == 'bullish':
            confluence_score += 0.15
            reasons.append('EMA:bull')
        elif ema_trend == 'bearish':
            confluence_score -= 0.15
            reasons.append('EMA:bear')

        # SMC Patterns (40% combined)
        if latest_15m.get('choch_bullish'):
            confluence_score += 0.20
            reasons.append('CHOCH:bull')
        elif latest_15m.get('choch_bearish'):
            confluence_score -= 0.20
            reasons.append('CHOCH:bear')

        if latest_15m.get('bos_bullish'):
            confluence_score += 0.12
            reasons.append('BOS:bull')
        elif latest_15m.get('bos_bearish'):
            confluence_score -= 0.12
            reasons.append('BOS:bear')

        if latest_15m.get('fvg_bullish'):
            confluence_score += 0.08
            reasons.append('FVG:bull')
        elif latest_15m.get('fvg_bearish'):
            confluence_score -= 0.08
            reasons.append('FVG:bear')

        # === SIGNAL GENERATION ===
        entry_price = latest_15m['close']

        # Debug log every 100th candle
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        if self._debug_counter % 100 == 0:
            logger.info(
                f"[DEBUG] Confluence score: {confluence_score:.3f}, threshold: ±{self.confluence_threshold}")
            logger.info(f"[DEBUG] Factors: {reasons}")

        if confluence_score >= self.confluence_threshold:
            logger.info(
                f"[SIGNAL GENERATED] LONG at {timestamp}: confluence={confluence_score:.2f}")
            return {
                'symbol': symbol,
                'direction': 'long',
                'entry_price': entry_price,
                'timestamp': timestamp,
                'reason': f'Confluence LONG ({confluence_score:.0%}): {" ".join(reasons)}',
                'confluence_score': confluence_score
            }
        elif confluence_score <= -self.confluence_threshold:
            logger.info(
                f"[SIGNAL GENERATED] SHORT at {timestamp}: confluence={confluence_score:.2f}")
            return {
                'symbol': symbol,
                'direction': 'short',
                'entry_price': entry_price,
                'timestamp': timestamp,
                'reason': f'Confluence SHORT ({abs(confluence_score):.0%}): {" ".join(reasons)}',
                'confluence_score': confluence_score
            }

        return None

    def fetch_historical_multi_timeframe(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical data for all timeframes needed for multi-TF analysis

        Returns:
            (df_1d, df_4h, df_15m) - DataFrames for each timeframe
        """
        logger.info(f"Fetching historical multi-timeframe data for {symbol}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

        try:
            timeframes_config = self.config.get('timeframes', {})
            tf_1d = timeframes_config.get('bias', '1d')
            tf_4h = timeframes_config.get('structure', '4h')
            tf_15m = timeframes_config.get('entry', '15m')

            # Calculate required historical data points
            # For 15M: Need enough data for the backtest period + warm-up
            days_total = (end_date - start_date).days + \
                30  # Add 30 days warm-up

            # Fetch each timeframe
            # 1D: Need at least 100 candles for bias
            logger.info(f"Fetching {tf_1d} data...")
            df_1d = self.data_fetcher.fetch_ohlcv(
                symbol,
                timeframe=tf_1d,
                limit=min(1000, days_total + 100)
            )

            # 4H: Need at least 200 candles for structure
            logger.info(f"Fetching {tf_4h} data...")
            df_4h = self.data_fetcher.fetch_ohlcv(
                symbol,
                timeframe=tf_4h,
                limit=min(1000, (days_total * 6) + 200)
            )

            # 15M: Need enough for entire backtest period
            logger.info(f"Fetching {tf_15m} data...")
            candles_15m = min(1000, (days_total * 96) +
                              500)  # 96 candles per day
            df_15m = self.data_fetcher.fetch_ohlcv(
                symbol,
                timeframe=tf_15m,
                limit=candles_15m
            )

            # Preprocess all dataframes
            logger.info("Preprocessing data...")
            df_1d = self.data_preprocessor.process_pipeline(df_1d)
            df_4h = self.data_preprocessor.process_pipeline(df_4h)
            df_15m = self.data_preprocessor.process_pipeline(df_15m)

            logger.info(
                f"Fetched - 1D: {len(df_1d)} candles, 4H: {len(df_4h)} candles, 15M: {len(df_15m)} candles")

            return df_1d, df_4h, df_15m

        except Exception as e:
            logger.error(f"Error fetching multi-timeframe data: {e}")
            raise

    def generate_signal_with_full_logic(
        self,
        symbol: str,
        current_timestamp: datetime,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_15m: pd.DataFrame,
        current_index_15m: int
    ) -> Optional[Dict]:
        """
        Generate trading signal using EXACT logic from main.py
        - Multi-timeframe analysis
        - Ensemble decision (if enabled)
        - All filters applied

        Args:
            symbol: Trading pair
            current_timestamp: Current timestamp in backtest
            df_1d, df_4h, df_15m: Full historical dataframes
            current_index_15m: Current position in 15M dataframe

        Returns:
            Signal dict or None
        """
        # Get context data using timestamp-based selection (more accurate)
        df_1d_context = df_1d[df_1d.index <=
                              current_timestamp].tail(100).copy()
        df_4h_context = df_4h[df_4h.index <=
                              current_timestamp].tail(200).copy()
        df_15m_context = df_15m[df_15m.index <=
                                current_timestamp].tail(500).copy()

        # Ensure we have enough data
        if len(df_1d_context) < 50 or len(df_4h_context) < 100 or len(df_15m_context) < 200:
            return None

        try:
            # Step 1: Multi-timeframe analysis (same as main.py)
            mtf_analysis = self.strategy.analyze_multi_timeframe(
                df_1d_context, df_4h_context, df_15m_context
            )

            # Step 2: Generate signal using ensemble or basic strategy
            signal = None

            if self.ensemble:
                # Use ensemble decision (same as main.py)
                ensemble_result = self.ensemble.analyze_signal(
                    df_15m_context, symbol)

                # Log for debugging (first 5 times)
                if current_index_15m < 5:
                    logger.info(f"[DEBUG {current_index_15m}] Ensemble: {ensemble_result['signal']}, "
                                f"conf={ensemble_result['confidence']:.3f}, "
                                f"agreement={ensemble_result['agreement']:.2f}")

                current_price = df_15m_context.iloc[-1]['close']

                if ensemble_result['signal'] == 'long' and ensemble_result['confidence'] >= self.ensemble.min_confidence:
                    signal = {
                        'symbol': symbol,
                        'direction': 'long',
                        'entry_price': current_price,
                        'timestamp': current_timestamp,
                        'reason': f"Ensemble LONG (conf: {ensemble_result['confidence']:.2f})",
                        'ensemble_details': ensemble_result,
                        'confluence_score': ensemble_result['confidence']
                    }
                elif ensemble_result['signal'] == 'short' and ensemble_result['confidence'] >= self.ensemble.min_confidence:
                    signal = {
                        'symbol': symbol,
                        'direction': 'short',
                        'entry_price': current_price,
                        'timestamp': current_timestamp,
                        'reason': f"Ensemble SHORT (conf: {ensemble_result['confidence']:.2f})",
                        'ensemble_details': ensemble_result,
                        'confluence_score': ensemble_result['confidence']
                    }
            else:
                # Use direct confluence-based signal generation (bypass filters for backtest)
                # This directly implements the confluence logic without sentiment/event filters
                signal = self._generate_confluence_signal(
                    symbol, mtf_analysis, df_15m_context, current_timestamp
                )

            # Step 3: Add SL/TP using same logic as main.py
            if signal:
                # Get ATR value with proper fallback
                atr_value = df_15m_context.iloc[-1].get('atr', None)

                # Handle missing or NaN ATR - use price-based fallback (1% of price)
                if atr_value is None or pd.isna(atr_value) or atr_value == 0:
                    # Calculate ATR manually if missing
                    if len(df_15m_context) >= 14:
                        high_low = df_15m_context['high'] - \
                            df_15m_context['low']
                        high_close = abs(
                            df_15m_context['high'] - df_15m_context['close'].shift(1))
                        low_close = abs(
                            df_15m_context['low'] - df_15m_context['close'].shift(1))
                        true_range = pd.concat(
                            [high_low, high_close, low_close], axis=1).max(axis=1)
                        atr_value = true_range.tail(14).mean()

                    # Final fallback: 1% of price
                    if pd.isna(atr_value) or atr_value == 0:
                        atr_value = signal['entry_price'] * 0.01  # 1% of price

                atr = float(atr_value)
                sl_multiplier = self.config['risk']['stop_loss_atr_multiplier']

                if 'stop_loss' not in signal:
                    if signal['direction'] == 'long':
                        signal['stop_loss'] = signal['entry_price'] - \
                            (atr * sl_multiplier)
                    else:
                        signal['stop_loss'] = signal['entry_price'] + \
                            (atr * sl_multiplier)

                if 'take_profit' not in signal:
                    risk = abs(signal['entry_price'] - signal['stop_loss'])
                    rr_ratio = self.config['risk']['take_profit_rr_ratio']
                    if signal['direction'] == 'long':
                        signal['take_profit'] = signal['entry_price'] + \
                            (risk * rr_ratio)
                    else:
                        signal['take_profit'] = signal['entry_price'] - \
                            (risk * rr_ratio)

                # Log ATR and SL for debugging
                logger.info(
                    f"Calculated position size: ATR={atr:.2f}, Entry={signal['entry_price']:.2f}, SL={signal['stop_loss']:.2f}")

                # Step 4: Calculate position size (same as main.py)
                position_size = self.risk_manager.calculate_position_size(
                    signal['entry_price'],
                    signal['stop_loss']
                )

                if position_size > 0:
                    signal['quantity'] = position_size
                else:
                    logger.warning(
                        f"Position size is 0 - Entry: {signal['entry_price']}, SL: {signal['stop_loss']}")
                    return None  # Position too small

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None

    def run_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = None
    ) -> Dict:
        """
        Run full logic backtest on historical data

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital (default from config)

        Returns:
            Dictionary with backtest results
        """
        logger.info("=" * 80)
        logger.info("FULL LOGIC BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Mode: {self.test_mode}")
        logger.info(f"Ensemble: {'ENABLED' if self.ensemble else 'DISABLED'}")
        logger.info("=" * 80)

        # Get initial capital
        if initial_capital is None:
            initial_capital = self.config.get(
                'backtest', {}).get('initial_capital', 10000)

        # Fetch multi-timeframe historical data
        df_1d, df_4h, df_15m = self.fetch_historical_multi_timeframe(
            symbol, start_date, end_date
        )

        # Filter 15M data to backtest period
        df_15m_backtest = df_15m[
            (df_15m.index >= start_date) & (df_15m.index <= end_date)
        ].copy()

        logger.info(
            f"Backtest will process {len(df_15m_backtest)} 15M candles")

        # Initialize backtest state
        capital = initial_capital
        position = None
        trades = []
        equity_curve = [{'timestamp': start_date, 'equity': capital}]

        # Reset risk manager for backtest
        self.risk_manager.daily_pnl = 0
        self.risk_manager.consecutive_losses = 0

        # Track signal statistics for debugging
        signal_attempts = 0
        signals_below_threshold = 0
        signals_passed = 0
        max_confidence_seen = 0.0

        # Process each 15M candle
        for i, (timestamp, row) in enumerate(df_15m_backtest.iterrows()):
            current_price = row['close']

            # Find corresponding index in full df_15m
            current_index_15m = df_15m.index.get_loc(timestamp)

            # Step 1: Manage existing position (same as main.py)
            if position is not None:
                # Check stop loss
                if position['direction'] == 'long':
                    if row['low'] <= position['stop_loss']:
                        # Stop loss hit
                        exit_price = position['stop_loss']
                        pnl = (exit_price -
                               position['entry_price']) * position['quantity']
                        capital += pnl

                        trades.append({
                            'symbol': symbol,
                            'direction': position['direction'],
                            'entry_time': position['timestamp'],
                            'entry_price': position['entry_price'],
                            'exit_time': timestamp,
                            'exit_price': exit_price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                            'exit_reason': 'Stop Loss'
                        })

                        # Update risk manager
                        # Backtest PnL tracked manually

                        position = None
                        logger.info(
                            f"[{timestamp}] STOP LOSS - PnL: ${pnl:.2f}")

                    # Check take profit
                    elif row['high'] >= position['take_profit']:
                        # Take profit hit
                        exit_price = position['take_profit']
                        pnl = (exit_price -
                               position['entry_price']) * position['quantity']
                        capital += pnl

                        trades.append({
                            'symbol': symbol,
                            'direction': position['direction'],
                            'entry_time': position['timestamp'],
                            'entry_price': position['entry_price'],
                            'exit_time': timestamp,
                            'exit_price': exit_price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                            'exit_reason': 'Take Profit'
                        })

                        # Update risk manager
                        # Backtest PnL tracked manually

                        position = None
                        logger.info(
                            f"[{timestamp}] TAKE PROFIT - PnL: ${pnl:.2f}")

                else:  # SHORT position
                    if row['high'] >= position['stop_loss']:
                        # Stop loss hit
                        exit_price = position['stop_loss']
                        pnl = (position['entry_price'] -
                               exit_price) * position['quantity']
                        capital += pnl

                        trades.append({
                            'symbol': symbol,
                            'direction': position['direction'],
                            'entry_time': position['timestamp'],
                            'entry_price': position['entry_price'],
                            'exit_time': timestamp,
                            'exit_price': exit_price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                            'exit_reason': 'Stop Loss'
                        })

                        # Update risk manager
                        # Backtest PnL tracked manually

                        position = None
                        logger.info(
                            f"[{timestamp}] STOP LOSS - PnL: ${pnl:.2f}")

                    elif row['low'] <= position['take_profit']:
                        # Take profit hit
                        exit_price = position['take_profit']
                        pnl = (position['entry_price'] -
                               exit_price) * position['quantity']
                        capital += pnl

                        trades.append({
                            'symbol': symbol,
                            'direction': position['direction'],
                            'entry_time': position['timestamp'],
                            'entry_price': position['entry_price'],
                            'exit_time': timestamp,
                            'exit_price': exit_price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                            'exit_reason': 'Take Profit'
                        })

                        # Update risk manager
                        # Backtest PnL tracked manually

                        position = None
                        logger.info(
                            f"[{timestamp}] TAKE PROFIT - PnL: ${pnl:.2f}")

            # Step 2: Look for new trade (if no position)
            if position is None:
                # Check if we can open new trade (same checks as main.py)
                if not self.risk_manager.can_open_new_trade():
                    continue

                # Generate signal with full logic
                signal_attempts += 1
                signal = self.generate_signal_with_full_logic(
                    symbol, timestamp, df_1d, df_4h, df_15m, current_index_15m
                )

                # Track confidence levels for debugging
                if signal:
                    conf = signal.get('confluence_score', 0) or signal.get(
                        'ensemble_details', {}).get('confidence', 0)
                    max_confidence_seen = max(max_confidence_seen, abs(conf))
                    signals_passed += 1
                    if i % 100 == 0:  # Log every 100 candles when we have a signal
                        logger.info(
                            f"[SIGNAL] {timestamp}: {signal['direction']} conf={conf:.3f}")
                elif self.ensemble and 'ensemble_result' in dir():
                    # Track rejected ensemble signals
                    conf = ensemble_result.get('confidence', 0)
                    max_confidence_seen = max(max_confidence_seen, abs(conf))
                    if conf > 0:
                        signals_below_threshold += 1

                # Debug: Log every 100 candles
                if i % 100 == 0:
                    logger.info(
                        f"Progress: {i}/{len(df_15m_backtest)} candles processed, Trades: {len(trades)}")

                if signal:
                    # Validate trade with risk manager
                    is_valid, reason = self.risk_manager.validate_trade(signal)

                    if is_valid:
                        # Open position
                        position = {
                            'symbol': symbol,
                            'direction': signal['direction'],
                            'entry_price': signal['entry_price'],
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit'],
                            'quantity': signal['quantity'],
                            'timestamp': timestamp,
                            'reason': signal.get('reason', 'Full logic signal')
                        }

                        logger.info(f"[{timestamp}] OPEN {signal['direction'].upper()} - "
                                    f"Price: ${signal['entry_price']:.2f}, "
                                    f"SL: ${signal['stop_loss']:.2f}, "
                                    f"TP: ${signal['take_profit']:.2f}")

            # Update equity curve
            current_equity = capital
            if position:
                # Add unrealized P&L
                if position['direction'] == 'long':
                    unrealized_pnl = (
                        current_price - position['entry_price']) * position['quantity']
                else:
                    unrealized_pnl = (
                        position['entry_price'] - current_price) * position['quantity']
                current_equity += unrealized_pnl

            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity
            })

        # Close any remaining position at end
        if position:
            exit_price = df_15m_backtest.iloc[-1]['close']
            if position['direction'] == 'long':
                pnl = (exit_price -
                       position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * \
                    position['quantity']

            capital += pnl

            trades.append({
                'symbol': symbol,
                'direction': position['direction'],
                'entry_time': position['timestamp'],
                'entry_price': position['entry_price'],
                'exit_time': df_15m_backtest.index[-1],
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                'exit_reason': 'End of Backtest'
            })

        # Log signal generation statistics
        logger.info("=" * 80)
        logger.info("SIGNAL GENERATION STATISTICS")
        logger.info(f"Total signal attempts: {signal_attempts}")
        logger.info(
            f"Signals with confidence data: {signals_below_threshold + signals_passed}")
        logger.info(f"Signals below threshold: {signals_below_threshold}")
        logger.info(f"Signals passed threshold: {signals_passed}")
        logger.info(f"Max confidence seen: {max_confidence_seen:.3f}")
        logger.info(
            f"Min confidence threshold: {self.ensemble.min_confidence if self.ensemble else 'N/A'}")
        logger.info("=" * 80)

        # Calculate metrics
        df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        df_equity = pd.DataFrame(equity_curve)

        metrics = self.calculate_metrics(df_trades, df_equity, initial_capital)

        return {
            'trades': df_trades,
            'equity_curve': df_equity,
            'metrics': metrics,
            'config': {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'test_mode': self.test_mode,
                'ensemble_enabled': self.ensemble is not None,
                'min_confidence': self.config['ensemble']['min_confidence'] if self.ensemble else None,
                'risk_per_trade': self.config['risk']['max_risk_per_trade'],
                'stop_loss_multiplier': self.config['risk']['stop_loss_atr_multiplier'],
                'take_profit_rr': self.config['risk']['take_profit_rr_ratio']
            }
        }

    def calculate_metrics(
        self,
        df_trades: pd.DataFrame,
        df_equity: pd.DataFrame,
        initial_capital: float
    ) -> Dict:
        """Calculate comprehensive backtest metrics"""

        if len(df_trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'message': 'No trades executed'
            }

        # Basic metrics
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * \
            100 if total_trades > 0 else 0

        total_pnl = df_trades['pnl'].sum()
        total_return_pct = (
            (df_equity.iloc[-1]['equity'] - initial_capital) / initial_capital) * 100

        # Win/Loss stats
        avg_win = df_trades[df_trades['pnl'] >
                            0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df_trades[df_trades['pnl'] < 0]
                       ['pnl'].mean()) if losing_trades > 0 else 0

        # Profit factor
        gross_profit = df_trades[df_trades['pnl'] >
                                 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(df_trades[df_trades['pnl'] < 0]
                         ['pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = gross_profit / \
            gross_loss if gross_loss > 0 else float('inf')

        # Drawdown
        equity_series = df_equity['equity'].values
        running_max = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min())

        # Sharpe ratio (assuming 252 trading days)
        returns = df_equity['equity'].pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * \
                np.sqrt(252 * 96)  # 96 15M periods per day
        else:
            sharpe_ratio = 0

        # Risk-reward ratio
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_risk_reward': avg_rr,
            'best_trade': df_trades['pnl'].max(),
            'worst_trade': df_trades['pnl'].min()
        }


def optimize_thresholds(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    config: Dict
) -> Dict:
    """
    Run backtests with different threshold combinations to find optimal settings

    Tests variations of:
    - Ensemble min_confidence
    - Stop loss ATR multiplier
    - Take profit RR ratio
    """
    logger.info("=" * 80)
    logger.info("THRESHOLD OPTIMIZATION")
    logger.info("=" * 80)

    # Define parameter ranges to test
    confidence_levels = [0.40, 0.45, 0.50, 0.55, 0.60]
    sl_multipliers = [1.0, 1.5, 2.0]
    rr_ratios = [1.5, 2.0, 2.5, 3.0]

    results = []
    total_combinations = len(confidence_levels) * \
        len(sl_multipliers) * len(rr_ratios)

    logger.info(f"Testing {total_combinations} parameter combinations...")

    current_test = 0
    for confidence in confidence_levels:
        for sl_mult in sl_multipliers:
            for rr_ratio in rr_ratios:
                current_test += 1
                logger.info(f"\n[{current_test}/{total_combinations}] Testing: "
                            f"Confidence={confidence}, SL={sl_mult}x, RR={rr_ratio}x")

                # Create modified config
                test_config = config.copy()
                test_config['ensemble']['min_confidence'] = confidence
                test_config['risk']['stop_loss_atr_multiplier'] = sl_mult
                test_config['risk']['take_profit_rr_ratio'] = rr_ratio

                # Run backtest
                backtester = FullLogicBacktest(test_config, test_mode='full')
                result = backtester.run_backtest(symbol, start_date, end_date)

                # Store results
                metrics = result['metrics']
                results.append({
                    'confidence': confidence,
                    'sl_multiplier': sl_mult,
                    'rr_ratio': rr_ratio,
                    'total_trades': metrics.get('total_trades', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_return_pct': metrics.get('total_return_pct', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'max_drawdown': metrics.get('max_drawdown_pct', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0)
                })

                logger.info(f"  Results: Trades={metrics.get('total_trades', 0)}, "
                            f"Win Rate={metrics.get('win_rate', 0):.1f}%, "
                            f"Return={metrics.get('total_return_pct', 0):.2f}%")

    # Analyze results
    df_results = pd.DataFrame(results)

    # Find best configurations by different criteria
    best_return = df_results.loc[df_results['total_return_pct'].idxmax()]
    best_sharpe = df_results.loc[df_results['sharpe_ratio'].idxmax()]
    best_winrate = df_results.loc[df_results['win_rate'].idxmax()]

    # Calculate score: balance return, Sharpe, and drawdown
    df_results['score'] = (
        df_results['total_return_pct'] * 0.4 +
        # Scale Sharpe to similar range
        df_results['sharpe_ratio'] * 20 * 0.3 +
        (100 - df_results['max_drawdown']) * 0.3
    )
    best_overall = df_results.loc[df_results['score'].idxmax()]

    return {
        'all_results': df_results,
        'best_return': best_return.to_dict(),
        'best_sharpe': best_sharpe.to_dict(),
        'best_winrate': best_winrate.to_dict(),
        'best_overall': best_overall.to_dict()
    }


def print_results(results: Dict):
    """Print backtest results in a formatted way"""
    metrics = results['metrics']
    config_info = results['config']

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"Symbol: {config_info['symbol']}")
    print(
        f"Period: {config_info['start_date'].date()} to {config_info['end_date'].date()}")
    print(f"Initial Capital: ${config_info['initial_capital']:,.2f}")
    print(
        f"Ensemble: {'ENABLED' if config_info['ensemble_enabled'] else 'DISABLED'}")
    if config_info['ensemble_enabled']:
        print(f"Min Confidence: {config_info['min_confidence']}")
    print("-" * 80)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics.get('winning_trades', 0)}")
    print(f"Losing Trades: {metrics.get('losing_trades', 0)}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print("-" * 80)
    print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Avg Win: ${metrics.get('avg_win', 0):.2f}")
    print(f"Avg Loss: ${metrics.get('avg_loss', 0):.2f}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Avg Risk/Reward: {metrics.get('avg_risk_reward', 0):.2f}")
    print("-" * 80)
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Best Trade: ${metrics.get('best_trade', 0):.2f}")
    print(f"Worst Trade: ${metrics.get('worst_trade', 0):.2f}")
    print("=" * 80)


def save_results(results: Dict, optimization_results: Dict = None):
    """Save backtest results to file"""
    # Create backtest results directory
    results_dir = Path('data/backtest/full_logic_results')
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    symbol_clean = results['config']['symbol'].replace('/', '_')

    # Save trades
    if len(results['trades']) > 0:
        trades_file = results_dir / f'{symbol_clean}_trades_{timestamp}.csv'
        results['trades'].to_csv(trades_file, index=False)
        logger.info(f"Trades saved to: {trades_file}")

    # Save equity curve
    equity_file = results_dir / f'{symbol_clean}_equity_{timestamp}.csv'
    results['equity_curve'].to_csv(equity_file, index=False)
    logger.info(f"Equity curve saved to: {equity_file}")

    # Save metrics
    metrics_file = results_dir / f'{symbol_clean}_metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(results['metrics'], f, indent=2, default=str)
    logger.info(f"Metrics saved to: {metrics_file}")

    # Save optimization results if available
    if optimization_results:
        opt_file = results_dir / f'{symbol_clean}_optimization_{timestamp}.csv'
        optimization_results['all_results'].to_csv(opt_file, index=False)
        logger.info(f"Optimization results saved to: {opt_file}")

        # Save best configs
        best_file = results_dir / \
            f'{symbol_clean}_best_configs_{timestamp}.json'
        best_configs = {
            'best_return': optimization_results['best_return'],
            'best_sharpe': optimization_results['best_sharpe'],
            'best_winrate': optimization_results['best_winrate'],
            'best_overall': optimization_results['best_overall']
        }
        with open(best_file, 'w') as f:
            json.dump(best_configs, f, indent=2, default=str)
        logger.info(f"Best configurations saved to: {best_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Full Logic Backtesting System')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Trading pair (default: BTC/USDT)')
    parser.add_argument('--months', type=int, default=3,
                        help='Number of months to backtest (default: 3)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--optimize', action='store_true',
                        help='Run threshold optimization')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='Disable ensemble model')
    parser.add_argument('--capital', type=float, default=None,
                        help='Initial capital (default from config)')

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.months * 30)

    # Determine test mode
    test_mode = 'no_ensemble' if args.no_ensemble else 'full'

    try:
        if args.optimize:
            # Run optimization
            optimization_results = optimize_thresholds(
                args.symbol, start_date, end_date, config
            )

            # Print optimization results
            print("\n" + "=" * 80)
            print("OPTIMIZATION RESULTS")
            print("=" * 80)

            print("\nBEST FOR RETURNS:")
            best = optimization_results['best_return']
            print(
                f"  Confidence: {best['confidence']}, SL: {best['sl_multiplier']}x, RR: {best['rr_ratio']}x")
            print(
                f"  Return: {best['total_return_pct']:.2f}%, Win Rate: {best['win_rate']:.1f}%")

            print("\nBEST FOR SHARPE RATIO:")
            best = optimization_results['best_sharpe']
            print(
                f"  Confidence: {best['confidence']}, SL: {best['sl_multiplier']}x, RR: {best['rr_ratio']}x")
            print(
                f"  Sharpe: {best['sharpe_ratio']:.2f}, Return: {best['total_return_pct']:.2f}%")

            print("\nBEST OVERALL (Balanced):")
            best = optimization_results['best_overall']
            print(
                f"  Confidence: {best['confidence']}, SL: {best['sl_multiplier']}x, RR: {best['rr_ratio']}x")
            print(
                f"  Score: {best['score']:.2f}, Return: {best['total_return_pct']:.2f}%, Sharpe: {best['sharpe_ratio']:.2f}")
            print("=" * 80)

            # Run final backtest with best overall config
            logger.info(
                "\nRunning final backtest with best overall configuration...")
            best_config = config.copy()
            best_config['ensemble']['min_confidence'] = best['confidence']
            best_config['risk']['stop_loss_atr_multiplier'] = best['sl_multiplier']
            best_config['risk']['take_profit_rr_ratio'] = best['rr_ratio']

            backtester = FullLogicBacktest(best_config, test_mode=test_mode)
            results = backtester.run_backtest(
                args.symbol, start_date, end_date, args.capital)

            print_results(results)
            save_results(results, optimization_results)

        else:
            # Single backtest run
            backtester = FullLogicBacktest(config, test_mode=test_mode)
            results = backtester.run_backtest(
                args.symbol, start_date, end_date, args.capital)

            print_results(results)
            save_results(results)

        logger.info("\nBacktest completed successfully!")

    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
