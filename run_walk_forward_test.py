 """
Walk-Forward Backtest Validation
Splits historical data into rolling windows: train on window N, test on window N+1.
This prevents look-ahead bias and gives statistically rigorous performance estimates.

Usage:
    python run_walk_forward_test.py                      # Default: BTC/USDT, 12 months, 3-month windows
    python run_walk_forward_test.py --symbol ETH/USDT
    python run_walk_forward_test.py --months 18 --window 2
"""

import sys
import os
import warnings
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
from src.utils.helpers import load_config, create_directories
from src.data.data_fetcher import DataFetcher
from src.data.data_preprocessor import DataPreprocessor
from src.trading.strategy import TradingStrategy
from src.models.ensemble_model import EnsembleDecisionModel
from src.backtesting.metrics import BacktestMetrics

logger = get_logger()


class WalkForwardValidator:
    """
    Walk-Forward Validation:
    1. Divide data into N windows (e.g. 3-month blocks)
    2. For each window i:
       - Use windows [0..i-1] as training context (strategy adapts)
       - Test on window i (measure real out-of-sample performance)
    3. Aggregate results across all test windows

    This gives a realistic estimate of strategy performance because
    each test window uses only data available BEFORE that window.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.strategy = TradingStrategy(config)
        self.metrics_calc = BacktestMetrics()

        # Ensemble (if enabled)
        self.ensemble = None
        if config.get('ensemble', {}).get('enabled', False):
            self.ensemble = EnsembleDecisionModel(config)

        # Backtest parameters from config
        self.commission_rate = config.get('backtest', {}).get('commission', 0.001)
        self.slippage_rate = config.get('backtest', {}).get('slippage', 0.0005)
        self.initial_capital = config.get('backtest', {}).get('initial_capital', 1000)
        self.leverage = config.get('trading', {}).get('leverage', 1)
        self.market_type = config.get('trading', {}).get('market_type', 'spot')
        self.position_size_usd = config.get('trading', {}).get('position_size_usd', 100)

        # Risk parameters
        self.max_risk_per_trade = config.get('risk', {}).get('max_risk_per_trade', 0.02)
        self.rr_ratio = config.get('risk', {}).get('take_profit_rr_ratio', 2.0)

    def fetch_data(self, symbol: str, months: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch multi-timeframe historical data"""
        logger.info(f"Fetching {months} months of data for {symbol}...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)

        # Calculate limits based on time range
        total_days = (end_date - start_date).days
        limit_1d = min(total_days + 50, 500)
        limit_4h = min(total_days * 6 + 50, 2000)
        limit_15m = min(total_days * 96 + 50, 10000)

        since_ts = int(start_date.timestamp() * 1000)

        df_1d = self.data_fetcher.fetch_ohlcv(symbol, '1d', limit=limit_1d, since=start_date)
        df_4h = self.data_fetcher.fetch_ohlcv(symbol, '4h', limit=limit_4h, since=start_date)
        df_15m = self.data_fetcher.fetch_ohlcv(symbol, '15m', limit=limit_15m, since=start_date)

        # Preprocess
        df_1d = self.data_preprocessor.process_pipeline(df_1d)
        df_4h = self.data_preprocessor.process_pipeline(df_4h)
        df_15m = self.data_preprocessor.process_pipeline(df_15m)

        logger.info(f"Data fetched: 1D={len(df_1d)}, 4H={len(df_4h)}, 15M={len(df_15m)}")

        return df_1d, df_4h, df_15m

    def split_into_windows(
        self,
        df_15m: pd.DataFrame,
        window_months: int
    ) -> List[Tuple[datetime, datetime]]:
        """Split data into rolling windows"""
        if len(df_15m) == 0:
            return []

        start = df_15m.index[0]
        end = df_15m.index[-1]
        window_delta = timedelta(days=window_months * 30)

        windows = []
        current_start = start

        while current_start + window_delta <= end:
            window_end = current_start + window_delta
            windows.append((current_start, window_end))
            current_start = window_end

        # Include remaining data as last window if > 50% of a full window
        remaining = end - current_start
        if remaining > window_delta * 0.5:
            windows.append((current_start, end))

        return windows

    def simulate_window(
        self,
        symbol: str,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_15m: pd.DataFrame,
        window_start: datetime,
        window_end: datetime,
        capital: float
    ) -> Dict:
        """
        Simulate trading on a single window.
        Only uses data up to window_end (no look-ahead).
        """
        # Filter 15M data to this window
        mask_15m = (df_15m.index >= window_start) & (df_15m.index <= window_end)
        window_15m = df_15m[mask_15m]

        if len(window_15m) < 200:
            return {'trades': [], 'final_capital': capital, 'equity_curve': []}

        trades = []
        current_capital = capital
        equity_curve = [(window_start, current_capital)]
        open_trade = None

        # Step through 15M candles
        step_size = 4  # Check every 4 candles (1 hour) for speed
        for i in range(200, len(window_15m), step_size):
            timestamp = window_15m.index[i]
            current_price = window_15m.iloc[i]['close']
            current_high = window_15m.iloc[i]['high']
            current_low = window_15m.iloc[i]['low']

            # Check if open trade hits SL/TP
            if open_trade is not None:
                hit, reason, exit_price = self._check_trade_exit(
                    open_trade, current_high, current_low, current_price
                )
                if hit:
                    pnl = self._calculate_pnl(open_trade, exit_price)
                    pnl -= abs(pnl) * self.commission_rate * 2  # Entry + exit commission
                    current_capital += pnl
                    trades.append({
                        'entry_time': open_trade['entry_time'],
                        'exit_time': timestamp,
                        'direction': open_trade['direction'],
                        'entry_price': open_trade['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'reason': reason
                    })
                    open_trade = None
                    continue

            # Only generate new signals if no open trade
            if open_trade is not None:
                # Update equity with unrealized P&L
                unrealized = self._calculate_pnl(open_trade, current_price)
                equity_curve.append((timestamp, current_capital + unrealized))
                continue

            equity_curve.append((timestamp, current_capital))

            # Minimum capital check
            if current_capital < 10:
                break

            # Get data available up to this point (no look-ahead)
            available_1d = df_1d[df_1d.index <= timestamp]
            available_4h = df_4h[df_4h.index <= timestamp]
            available_15m = window_15m.iloc[:i + 1]

            if len(available_1d) < 20 or len(available_4h) < 50 or len(available_15m) < 200:
                continue

            # Run multi-timeframe analysis (same logic as main.py)
            try:
                mtf_analysis = self.strategy.analyze_multi_timeframe(
                    available_1d.tail(100),
                    available_4h.tail(200),
                    available_15m.tail(500)
                )

                signal = self.strategy.generate_signal(symbol, mtf_analysis)

                if signal and signal.get('direction'):
                    # Block shorts in spot mode
                    if self.market_type == 'spot' and signal['direction'] == 'short':
                        continue

                    # Validate R:R
                    risk = abs(signal['entry_price'] - signal['stop_loss'])
                    reward = abs(signal['take_profit'] - signal['entry_price'])
                    if risk == 0 or (reward / risk) < self.rr_ratio:
                        continue

                    # Apply slippage to entry
                    entry_price = signal['entry_price']
                    if signal['direction'] == 'long':
                        entry_price *= (1 + self.slippage_rate)
                    else:
                        entry_price *= (1 - self.slippage_rate)

                    # Position sizing: cap at position_size_usd or 50% of capital
                    margin = min(self.position_size_usd, current_capital * 0.5)
                    if margin < 5:
                        continue

                    open_trade = {
                        'direction': signal['direction'],
                        'entry_price': entry_price,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'entry_time': timestamp,
                        'margin': margin
                    }

            except Exception as e:
                logger.debug(f"Signal error at {timestamp}: {e}")
                continue

        # Close any remaining open trade at end of window
        if open_trade is not None:
            exit_price = window_15m.iloc[-1]['close']
            pnl = self._calculate_pnl(open_trade, exit_price)
            pnl -= abs(pnl) * self.commission_rate * 2
            current_capital += pnl
            trades.append({
                'entry_time': open_trade['entry_time'],
                'exit_time': window_15m.index[-1],
                'direction': open_trade['direction'],
                'entry_price': open_trade['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'reason': 'window_end'
            })

        return {
            'trades': trades,
            'final_capital': current_capital,
            'equity_curve': equity_curve
        }

    def _check_trade_exit(self, trade: Dict, high: float, low: float, close: float) -> Tuple[bool, str, float]:
        """Check if trade hits SL or TP using high/low (intrabar check)"""
        sl = trade['stop_loss']
        tp = trade['take_profit']

        if trade['direction'] == 'long':
            if low <= sl:
                exit_price = sl * (1 - self.slippage_rate)  # Slippage on exit
                return True, 'stop_loss', exit_price
            if high >= tp:
                exit_price = tp * (1 - self.slippage_rate)
                return True, 'take_profit', exit_price
        else:  # short
            if high >= sl:
                exit_price = sl * (1 + self.slippage_rate)
                return True, 'stop_loss', exit_price
            if low <= tp:
                exit_price = tp * (1 + self.slippage_rate)
                return True, 'take_profit', exit_price

        return False, '', close

    def _calculate_pnl(self, trade: Dict, exit_price: float) -> float:
        """Calculate P&L for a trade, accounting for leverage"""
        margin = trade['margin']
        entry = trade['entry_price']

        if entry == 0:
            return 0

        if trade['direction'] == 'long':
            price_change_pct = (exit_price - entry) / entry
        else:
            price_change_pct = (entry - exit_price) / entry

        # Leverage amplifies P&L
        lev = self.leverage if self.market_type == 'futures' else 1
        pnl = margin * price_change_pct * lev

        return pnl

    def run(self, symbol: str, months: int = 12, window_months: int = 3) -> Dict:
        """
        Run walk-forward validation.

        Args:
            symbol: Trading pair
            months: Total months of history
            window_months: Size of each test window

        Returns:
            Aggregated results across all windows
        """
        logger.info("=" * 70)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info(f"Symbol: {symbol} | History: {months}mo | Window: {window_months}mo")
        logger.info(f"Leverage: {self.leverage}x | Slippage: {self.slippage_rate*100:.3f}%")
        logger.info(f"Commission: {self.commission_rate*100:.2f}% | Confluence: {self.config.get('strategy', {}).get('confluence_threshold', 0.45)}")
        logger.info("=" * 70)

        # Fetch data
        df_1d, df_4h, df_15m = self.fetch_data(symbol, months)

        if len(df_15m) < 500:
            logger.error("Insufficient 15M data for walk-forward test")
            return {}

        # Split into windows
        windows = self.split_into_windows(df_15m, window_months)
        logger.info(f"Created {len(windows)} test windows")

        if len(windows) < 2:
            logger.error("Need at least 2 windows for walk-forward test")
            return {}

        # Walk forward: first window is warm-up, test on remaining
        all_trades = []
        window_results = []
        capital = self.initial_capital

        for idx, (w_start, w_end) in enumerate(windows):
            if idx == 0:
                logger.info(f"Window {idx + 1}/{len(windows)}: {w_start.strftime('%Y-%m-%d')} → "
                            f"{w_end.strftime('%Y-%m-%d')} [WARM-UP / TRAINING CONTEXT]")
                # Run warm-up window to establish context
                result = self.simulate_window(symbol, df_1d, df_4h, df_15m, w_start, w_end, capital)
                capital = result['final_capital']
                logger.info(f"  Warm-up: {len(result['trades'])} trades, capital: ${capital:.2f}")
                window_results.append({
                    'window': idx + 1,
                    'type': 'warmup',
                    'start': w_start.strftime('%Y-%m-%d'),
                    'end': w_end.strftime('%Y-%m-%d'),
                    'trades': len(result['trades']),
                    'capital': capital
                })
                continue

            logger.info(f"Window {idx + 1}/{len(windows)}: {w_start.strftime('%Y-%m-%d')} → "
                        f"{w_end.strftime('%Y-%m-%d')} [OUT-OF-SAMPLE TEST]")

            window_capital_start = capital
            result = self.simulate_window(symbol, df_1d, df_4h, df_15m, w_start, w_end, capital)
            capital = result['final_capital']

            window_pnl = capital - window_capital_start
            window_return = (window_pnl / window_capital_start * 100) if window_capital_start > 0 else 0
            wins = sum(1 for t in result['trades'] if t['pnl'] > 0)
            losses = sum(1 for t in result['trades'] if t['pnl'] <= 0)
            win_rate = (wins / len(result['trades']) * 100) if result['trades'] else 0

            logger.info(f"  Trades: {len(result['trades'])} | W/L: {wins}/{losses} | "
                        f"Win Rate: {win_rate:.1f}% | Return: {window_return:+.2f}% | Capital: ${capital:.2f}")

            all_trades.extend(result['trades'])
            window_results.append({
                'window': idx + 1,
                'type': 'test',
                'start': w_start.strftime('%Y-%m-%d'),
                'end': w_end.strftime('%Y-%m-%d'),
                'trades': len(result['trades']),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'pnl': window_pnl,
                'return_pct': window_return,
                'capital': capital
            })

        # Aggregate results
        test_windows = [w for w in window_results if w['type'] == 'test']
        total_trades = sum(w['trades'] for w in test_windows)
        total_wins = sum(w.get('wins', 0) for w in test_windows)
        total_losses = sum(w.get('losses', 0) for w in test_windows)
        overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        overall_return = ((capital - self.initial_capital) / self.initial_capital * 100)

        # Profit factor
        gross_profit = sum(t['pnl'] for t in all_trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in all_trades if t['pnl'] <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Expectancy
        avg_win = np.mean([t['pnl'] for t in all_trades if t['pnl'] > 0]) if total_wins > 0 else 0
        avg_loss = abs(np.mean([t['pnl'] for t in all_trades if t['pnl'] <= 0])) if total_losses > 0 else 0
        expectancy = (overall_win_rate / 100 * avg_win) - ((1 - overall_win_rate / 100) * avg_loss)

        # Window consistency (how many test windows were profitable)
        profitable_windows = sum(1 for w in test_windows if w.get('pnl', 0) > 0)
        window_consistency = (profitable_windows / len(test_windows) * 100) if test_windows else 0

        # Max drawdown across all trades
        equity_values = [self.initial_capital]
        running_capital = self.initial_capital
        for t in all_trades:
            running_capital += t['pnl']
            equity_values.append(running_capital)
        equity_arr = np.array(equity_values)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max * 100
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("WALK-FORWARD VALIDATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"  Test Windows:           {len(test_windows)}")
        logger.info(f"  Profitable Windows:     {profitable_windows}/{len(test_windows)} ({window_consistency:.0f}%)")
        logger.info(f"  Total Trades:           {total_trades}")
        logger.info(f"  Win Rate:               {overall_win_rate:.1f}%")
        logger.info(f"  Profit Factor:          {profit_factor:.2f}")
        logger.info(f"  Expectancy:             ${expectancy:.2f}/trade")
        logger.info(f"  Overall Return:         {overall_return:+.2f}%")
        logger.info(f"  Max Drawdown:           {max_drawdown:.2f}%")
        logger.info(f"  Avg Win:                ${avg_win:.2f}")
        logger.info(f"  Avg Loss:               ${avg_loss:.2f}")
        logger.info(f"  Initial Capital:        ${self.initial_capital:.2f}")
        logger.info(f"  Final Capital:          ${capital:.2f}")
        logger.info("=" * 70)

        # Viability assessment
        logger.info("")
        if profit_factor > 1.5 and overall_win_rate > 40 and window_consistency > 50:
            logger.info("ASSESSMENT: STRATEGY SHOWS PROMISE — consider extended paper trading")
        elif profit_factor > 1.0 and overall_win_rate > 35:
            logger.info("ASSESSMENT: MARGINAL — needs parameter tuning before live deployment")
        else:
            logger.info("ASSESSMENT: NOT VIABLE — strategy needs fundamental improvements")

        summary = {
            'symbol': symbol,
            'history_months': months,
            'window_months': window_months,
            'leverage': self.leverage,
            'market_type': self.market_type,
            'confluence_threshold': self.config.get('strategy', {}).get('confluence_threshold', 0.45),
            'test_windows': len(test_windows),
            'profitable_windows': profitable_windows,
            'window_consistency_pct': window_consistency,
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': overall_win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'overall_return_pct': overall_return,
            'max_drawdown_pct': max_drawdown,
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'window_results': window_results,
            'all_trades': all_trades,
            'timestamp': datetime.now().isoformat()
        }

        # Save results
        results_dir = Path('data/backtest/walk_forward_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert to JSON-serializable
        save_data = {k: v for k, v in summary.items() if k != 'all_trades'}
        save_data['trade_count'] = len(all_trades)
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")

        return summary


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Backtest Validation')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair')
    parser.add_argument('--months', type=int, default=12, help='Months of history')
    parser.add_argument('--window', type=int, default=3, help='Window size in months')
    args = parser.parse_args()

    config = load_config()
    create_directories()

    validator = WalkForwardValidator(config)
    results = validator.run(args.symbol, args.months, args.window)


if __name__ == '__main__':
    main()
