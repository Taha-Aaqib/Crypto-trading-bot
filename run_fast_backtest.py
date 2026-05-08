"""
Fast Backtesting Script - Optimized for Speed
Pre-calculates all indicators and uses vectorized operations

Usage:
    python run_fast_backtest.py                        # Default: BTC/USDT, last 3 months
    python run_fast_backtest.py --symbol ETH/USDT --months 6
    python run_fast_backtest.py --start 2025-10-01     # Test specific period
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.helpers import load_config
from src.utils.logger import get_logger
from src.data.data_fetcher import DataFetcher
from src.data.data_preprocessor import DataPreprocessor
from src.indicators.ta_indicators import TechnicalIndicators
from src.indicators.smc_detector import SMCDetector

logger = get_logger()


class FastBacktest:
    """
    Optimized backtesting - Pre-calculates everything upfront for speed
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.ta_indicators = TechnicalIndicators(config)
        self.smc_detector = SMCDetector(config)
        
        # Risk parameters
        self.risk_per_trade = config['risk']['max_risk_per_trade']  # 0.02 = 2%
        self.sl_atr_mult = config['risk']['stop_loss_atr_multiplier']  # 1.5
        self.tp_rr_ratio = config['risk']['take_profit_rr_ratio']  # 2.0
        self.confluence_threshold = config.get('strategy', {}).get('confluence_threshold', 0.35)
        
        logger.info("Fast Backtest initialized")
    
    def _fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        since_ts: int,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data with pagination to get specific date ranges"""
        import time
        
        all_data = []
        current_since = since_ts
        end_ts = int(end_date.timestamp() * 1000)
        
        # Timeframe to milliseconds
        tf_ms = {
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        
        ms_per_candle = tf_ms.get(timeframe, 15 * 60 * 1000)
        
        while current_since < end_ts:
            try:
                ohlcv = self.data_fetcher.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_data.extend(ohlcv)
                
                # Move to next batch
                last_ts = ohlcv[-1][0]
                current_since = last_ts + ms_per_candle
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error fetching {timeframe} data: {e}")
                break
        
        # Convert to DataFrame
        if not all_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp').sort_index()
        
        logger.info(f"Fetched {len(df)} candles for {symbol} on {timeframe}")
        
        return df
    
    def fetch_and_prepare_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch and pre-calculate ALL indicators upfront"""
        
        logger.info(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Calculate the 'since' timestamp for historical data
        # We need data from before start_date for indicator warm-up
        warmup_start = start_date - timedelta(days=60)  # 60 days warm-up for EMAs
        since_ts = int(warmup_start.timestamp() * 1000)  # Convert to milliseconds
        
        # Fetch all timeframes with historical data using 'since' parameter
        logger.info("Fetching 1D data...")
        df_1d = self._fetch_historical_data(symbol, '1d', since_ts, end_date)
        
        logger.info("Fetching 4H data...")
        df_4h = self._fetch_historical_data(symbol, '4h', since_ts, end_date)
        
        logger.info("Fetching 15M data...")
        df_15m = self._fetch_historical_data(symbol, '15m', since_ts, end_date)
        
        # Pre-process all data
        logger.info("Pre-processing data...")
        df_1d = self.data_preprocessor.process_pipeline(df_1d)
        df_4h = self.data_preprocessor.process_pipeline(df_4h)
        df_15m = self.data_preprocessor.process_pipeline(df_15m)
        
        # Pre-calculate ALL indicators upfront (this is the key optimization!)
        logger.info("Pre-calculating indicators (this is the slow part, but only once)...")
        
        # 1D: Calculate bias
        df_1d = self._calculate_bias(df_1d)
        
        # 4H: Calculate structure
        df_4h = self._calculate_structure(df_4h)
        
        # 15M: Calculate entry signals
        df_15m = self._calculate_entry_signals(df_15m)
        
        logger.info(f"Data ready - 1D: {len(df_1d)}, 4H: {len(df_4h)}, 15M: {len(df_15m)} candles")
        
        return df_1d, df_4h, df_15m
    
    def _calculate_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily bias using EMA crossover"""
        df = df.copy()
        
        # EMAs for bias
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Bias: bullish if price > EMA20 > EMA50
        df['bias'] = 'neutral'
        df.loc[(df['close'] > df['ema_20']) & (df['ema_20'] > df['ema_50']), 'bias'] = 'bullish'
        df.loc[(df['close'] < df['ema_20']) & (df['ema_20'] < df['ema_50']), 'bias'] = 'bearish'
        
        return df
    
    def _calculate_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 4H structure using swing highs/lows"""
        df = df.copy()
        
        # Simple structure based on recent highs/lows
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        
        # Structure: bullish if making higher highs
        df['structure'] = 'neutral'
        df.loc[df['high'] >= df['high_20'], 'structure'] = 'bullish'
        df.loc[df['low'] <= df['low_20'], 'structure'] = 'bearish'
        
        return df
    
    def _calculate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-calculate all entry signals for 15M data"""
        df = df.copy()
        
        # Technical indicators
        df = self.ta_indicators.add_all_indicators(df)
        
        # SMC patterns (calculate once for entire dataset)
        logger.info("Calculating SMC patterns...")
        df_with_smc = self.smc_detector.analyze_smc(df)
        
        # Add SMC columns
        df['bullish_choch'] = df_with_smc.get('bullish_choch', pd.Series([False] * len(df), index=df.index))
        df['bearish_choch'] = df_with_smc.get('bearish_choch', pd.Series([False] * len(df), index=df.index))
        df['bullish_bos'] = df_with_smc.get('bullish_bos', pd.Series([False] * len(df), index=df.index))
        df['bearish_bos'] = df_with_smc.get('bearish_bos', pd.Series([False] * len(df), index=df.index))
        df['bullish_fvg'] = df_with_smc.get('bullish_fvg', pd.Series([False] * len(df), index=df.index))
        df['bearish_fvg'] = df_with_smc.get('bearish_fvg', pd.Series([False] * len(df), index=df.index))
        
        # EMA trend
        if 'ema_9' not in df.columns:
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        if 'ema_21' not in df.columns:
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        df['ema_trend'] = 'neutral'
        df.loc[df['ema_9'] > df['ema_21'], 'ema_trend'] = 'bullish'
        df.loc[df['ema_9'] < df['ema_21'], 'ema_trend'] = 'bearish'
        
        # ATR for stop loss
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
        
        # Pre-calculate confluence score for each candle
        logger.info("Calculating confluence scores...")
        df['confluence_score'] = 0.0
        
        return df
    
    def run_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000
    ) -> Dict:
        """Run fast backtest with pre-calculated indicators"""
        
        logger.info("=" * 70)
        logger.info("FAST BACKTEST")
        logger.info("=" * 70)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Capital: ${initial_capital:,.2f}")
        logger.info("=" * 70)
        
        # Fetch and prepare all data
        df_1d, df_4h, df_15m = self.fetch_and_prepare_data(symbol, start_date, end_date)
        
        # Filter to backtest period (handle timezone)
        if df_15m.index.tz is not None:
            start_ts = pd.Timestamp(start_date).tz_localize(df_15m.index.tz)
            end_ts = pd.Timestamp(end_date).tz_localize(df_15m.index.tz)
        else:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
        
        df_15m_bt = df_15m[(df_15m.index >= start_ts) & (df_15m.index <= end_ts)].copy()
        
        # If no data in range, use all available data
        if len(df_15m_bt) == 0:
            logger.warning(f"No data in date range. Using all {len(df_15m)} candles.")
            df_15m_bt = df_15m.copy()
        
        logger.info(f"Processing {len(df_15m_bt)} candles...")
        
        # Backtest state
        capital = initial_capital
        position = None
        trades = []
        equity_curve = [{'timestamp': start_date, 'equity': capital}]
        
        # Process each candle
        for i, (timestamp, row) in enumerate(df_15m_bt.iterrows()):
            current_price = row['close']
            
            # Progress logging (every 200 candles)
            if i % 200 == 0:
                logger.info(f"Progress: {i}/{len(df_15m_bt)} | Trades: {len(trades)} | Capital: ${capital:,.2f}")
            
            # Manage existing position
            if position is not None:
                exit_trade, exit_price, exit_reason = self._check_exit(position, row)
                
                if exit_trade:
                    pnl = self._calculate_pnl(position, exit_price)
                    capital += pnl
                    
                    trades.append({
                        'symbol': symbol,
                        'direction': position['direction'],
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'exit_time': timestamp,
                        'exit_price': exit_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                        'exit_reason': exit_reason
                    })
                    
                    result = "✅ WIN" if pnl > 0 else "❌ LOSS"
                    logger.info(f"[{timestamp}] {exit_reason} {result}: ${pnl:+.2f}")
                    
                    position = None
                    equity_curve.append({'timestamp': timestamp, 'equity': capital})
            
            # Look for new entry (if no position)
            if position is None:
                signal = self._generate_signal_fast(timestamp, row, df_1d, df_4h)
                
                if signal:
                    # Calculate position size
                    atr = row.get('atr', current_price * 0.01)
                    if pd.isna(atr) or atr == 0:
                        atr = current_price * 0.01
                    
                    sl_distance = atr * self.sl_atr_mult
                    risk_amount = capital * self.risk_per_trade
                    quantity = risk_amount / sl_distance
                    
                    if signal == 'long':
                        sl = current_price - sl_distance
                        tp = current_price + (sl_distance * self.tp_rr_ratio)
                    else:
                        sl = current_price + sl_distance
                        tp = current_price - (sl_distance * self.tp_rr_ratio)
                    
                    position = {
                        'direction': signal,
                        'entry_time': timestamp,
                        'entry_price': current_price,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'quantity': quantity
                    }
                    
                    logger.info(f"[{timestamp}] OPEN {signal.upper()} @ ${current_price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f}")
        
        # Close any open position at end
        if position is not None:
            final_price = df_15m_bt.iloc[-1]['close']
            pnl = self._calculate_pnl(position, final_price)
            capital += pnl
            trades.append({
                'symbol': symbol,
                'direction': position['direction'],
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': df_15m_bt.index[-1],
                'exit_price': final_price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                'exit_reason': 'End of Backtest'
            })
        
        # Calculate metrics
        results = self._calculate_metrics(trades, initial_capital, capital, equity_curve)
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _generate_signal_fast(
        self,
        timestamp: datetime,
        row: pd.Series,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame
    ) -> Optional[str]:
        """Fast signal generation using pre-calculated data"""
        
        # Get current bias from 1D
        df_1d_now = df_1d[df_1d.index <= timestamp]
        if len(df_1d_now) == 0:
            return None
        bias_1d = df_1d_now.iloc[-1].get('bias', 'neutral')
        
        # Get current structure from 4H
        df_4h_now = df_4h[df_4h.index <= timestamp]
        if len(df_4h_now) == 0:
            return None
        structure_4h = df_4h_now.iloc[-1].get('structure', 'neutral')
        
        # Get 15M signals from pre-calculated row
        ema_trend = row.get('ema_trend', 'neutral')
        bullish_choch = row.get('bullish_choch', False)
        bearish_choch = row.get('bearish_choch', False)
        bullish_bos = row.get('bullish_bos', False)
        bearish_bos = row.get('bearish_bos', False)
        bullish_fvg = row.get('bullish_fvg', False)
        bearish_fvg = row.get('bearish_fvg', False)
        
        # Calculate confluence score
        # Weights: 1D=25%, 4H=20%, EMA=15%, CHOCH=20%, BOS=12%, FVG=8%
        score = 0.0
        
        # Bullish factors
        if bias_1d == 'bullish':
            score += 0.25
        elif bias_1d == 'bearish':
            score -= 0.25
            
        if structure_4h == 'bullish':
            score += 0.20
        elif structure_4h == 'bearish':
            score -= 0.20
            
        if ema_trend == 'bullish':
            score += 0.15
        elif ema_trend == 'bearish':
            score -= 0.15
        
        # SMC patterns
        if bullish_choch:
            score += 0.20
        if bearish_choch:
            score -= 0.20
            
        if bullish_bos:
            score += 0.12
        if bearish_bos:
            score -= 0.12
            
        if bullish_fvg:
            score += 0.08
        if bearish_fvg:
            score -= 0.08
        
        # Generate signal if confluence threshold met
        if score >= self.confluence_threshold:
            return 'long'
        elif score <= -self.confluence_threshold:
            return 'short'
        
        return None
    
    def _check_exit(self, position: Dict, row: pd.Series) -> Tuple[bool, float, str]:
        """Check if position should be exited"""
        if position['direction'] == 'long':
            if row['low'] <= position['stop_loss']:
                return True, position['stop_loss'], 'Stop Loss'
            if row['high'] >= position['take_profit']:
                return True, position['take_profit'], 'Take Profit'
        else:  # short
            if row['high'] >= position['stop_loss']:
                return True, position['stop_loss'], 'Stop Loss'
            if row['low'] <= position['take_profit']:
                return True, position['take_profit'], 'Take Profit'
        
        return False, 0, ''
    
    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate PnL for a trade"""
        if position['direction'] == 'long':
            return (exit_price - position['entry_price']) * position['quantity']
        else:
            return (position['entry_price'] - exit_price) * position['quantity']
    
    def _calculate_metrics(
        self,
        trades: List[Dict],
        initial_capital: float,
        final_capital: float,
        equity_curve: List[Dict]
    ) -> Dict:
        """Calculate comprehensive backtest metrics"""
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'trades': [],
                'equity_curve': equity_curve
            }
        
        trades_df = pd.DataFrame(trades)
        
        # Basic stats
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL stats
        total_pnl = trades_df['pnl'].sum()
        total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Drawdown
        equity_df = pd.DataFrame(equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].max()
        
        # Sharpe ratio (simplified - annualized)
        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'] / 100
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 4) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'trades': trades,
            'equity_curve': equity_curve
        }
    
    def _print_results(self, results: Dict):
        """Print formatted backtest results"""
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 70)
        
        logger.info(f"📊 Total Trades:     {results['total_trades']}")
        logger.info(f"✅ Winning Trades:   {results['winning_trades']}")
        logger.info(f"❌ Losing Trades:    {results['losing_trades']}")
        logger.info(f"📈 Win Rate:         {results['win_rate']:.1f}%")
        
        logger.info("-" * 70)
        
        logger.info(f"💰 Total PnL:        ${results['total_pnl']:+,.2f}")
        logger.info(f"📊 Total Return:     {results['total_return_pct']:+.2f}%")
        logger.info(f"📉 Max Drawdown:     {results['max_drawdown']:.2f}%")
        
        logger.info("-" * 70)
        
        logger.info(f"💵 Avg Win:          ${results['avg_win']:+,.2f}")
        logger.info(f"💸 Avg Loss:         ${results['avg_loss']:+,.2f}")
        logger.info(f"⚖️ Profit Factor:    {results['profit_factor']:.2f}")
        logger.info(f"📊 Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        
        logger.info("=" * 70)
        
        # Trade list
        if results['trades']:
            logger.info("\n📋 TRADE HISTORY:")
            logger.info("-" * 70)
            for i, trade in enumerate(results['trades'], 1):
                result_emoji = "✅" if trade['pnl'] > 0 else "❌"
                logger.info(
                    f"{i}. {result_emoji} {trade['direction'].upper():5} | "
                    f"{trade['entry_time'].strftime('%m/%d %H:%M')} → {trade['exit_time'].strftime('%m/%d %H:%M')} | "
                    f"${trade['pnl']:+.2f} ({trade['pnl_pct']:+.1f}%) | {trade['exit_reason']}"
                )


def main():
    parser = argparse.ArgumentParser(description='Fast Backtesting')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--months', type=int, default=1, help='Months to backtest')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Determine dates
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=args.months * 30)
    
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    # Run backtest
    backtester = FastBacktest(config)
    results = backtester.run_backtest(args.symbol, start_date, end_date, args.capital)
    
    return results


if __name__ == '__main__':
    main()
