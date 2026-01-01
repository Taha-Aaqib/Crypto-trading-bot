"""
Simple Backtesting Script for FYP
Tests strategy on historical data to find optimal settings
"""

from src.data.data_fetcher import DataFetcher
from src.utils.helpers import load_config
from src.utils.logger import get_logger
from src.backtesting.visualizer import BacktestVisualizer
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.backtest_engine import BacktestEngine
import pandas as pd
from datetime import datetime, timedelta
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Set WARNING level for backtest (less noise)
logging.getLogger('TradingBot').setLevel(logging.WARNING)
logger = get_logger()


def run_backtest(
    symbol: str = "BTC/USDT",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-27",
    initial_capital: float = 10000
):
    """Run backtest for single symbol"""

    config = load_config()

    # Update config with backtest parameters
    if 'backtest' not in config:
        config['backtest'] = {}

    config['backtest']['initial_capital'] = initial_capital
    config['backtest']['commission'] = 0.001
    config['backtest']['slippage'] = 0.0005

    # Initialize components
    engine = BacktestEngine(config)
    metrics_calc = BacktestMetrics()
    visualizer = BacktestVisualizer()

    # Fetch historical data
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTESTING: {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"{'='*60}\n")

    fetcher = DataFetcher(config)

    # Convert date string to datetime object (data_fetcher expects datetime, not int)
    start_datetime = pd.to_datetime(start_date)

    df = fetcher.fetch_ohlcv(
        symbol=symbol,
        timeframe="15m",
        since=start_datetime,
        limit=10000
    )

    if df is None or len(df) < 100:
        logger.error(f"❌ Not enough data for {symbol}")
        return None

    logger.info(f"✅ Loaded {len(df)} candles")

    # Import strategy
    from src.trading.strategy import TradingStrategy
    strategy = TradingStrategy(config)

    # Run backtest using engine's built-in method
    logger.info("\n⏳ Running backtest simulation...")

    results = engine.run(
        strategy=strategy,
        data={symbol: df},
        start_date=start_date,
        end_date=end_date
    )
    perf_metrics = metrics_calc.calculate_all_metrics(results)

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info("BACKTEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total Trades: {perf_metrics.get('total_trades', 0)}")
    logger.info(f"Win Rate: {perf_metrics.get('win_rate', 0):.1f}%")
    logger.info(f"Total Return: {perf_metrics.get('total_return', 0):.2f}%")
    logger.info(f"Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(
        f"Max Drawdown: {perf_metrics.get('max_drawdown_pct', 0):.2f}%")
    logger.info(f"Profit Factor: {perf_metrics.get('profit_factor', 0):.2f}")
    logger.info(f"{'='*60}\n")

    # Generate visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots = visualizer.generate_all_plots(
        backtest_results=results,
        metrics=perf_metrics,
        save_prefix=f"{symbol.replace('/', '_')}_{timestamp}"
    )

    logger.info(f"✅ Charts saved to: data/backtest/")
    for plot_type, path in plots.items():
        logger.info(f"  - {plot_type}: {path}")

    return perf_metrics


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CRYPTO TRADING BOT - BACKTEST RUNNER")
    print("Testing strategy on historical data for FYP validation")
    print("="*60 + "\n")

    # Test BTC
    print("📊 Testing BTC/USDT...")
    btc_results = run_backtest(
        symbol="BTC/USDT",
        start_date="2024-11-01",
        end_date="2024-12-31",
        initial_capital=10000
    )

    # Test ETH
    print("\n📊 Testing ETH/USDT...")
    eth_results = run_backtest(
        symbol="ETH/USDT",
        start_date="2024-11-01",
        end_date="2024-12-31",
        initial_capital=10000
    )

    # Summary comparison
    if btc_results and eth_results:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"\n{'Metric':<20} {'BTC/USDT':>15} {'ETH/USDT':>15}")
        print("-"*60)
        print(
            f"{'Win Rate':<20} {btc_results.get('win_rate', 0):>14.1f}% {eth_results.get('win_rate', 0):>14.1f}%")
        print(
            f"{'Total Return':<20} {btc_results.get('total_return', 0):>14.2f}% {eth_results.get('total_return', 0):>14.2f}%")
        print(
            f"{'Sharpe Ratio':<20} {btc_results.get('sharpe_ratio', 0):>15.2f} {eth_results.get('sharpe_ratio', 0):>15.2f}")
        print(
            f"{'Profit Factor':<20} {btc_results.get('profit_factor', 0):>15.2f} {eth_results.get('profit_factor', 0):>15.2f}")
        print(
            f"{'Max Drawdown':<20} {btc_results.get('max_drawdown_pct', 0):>14.2f}% {eth_results.get('max_drawdown_pct', 0):>14.2f}%")
        print("="*60 + "\n")

print("✅ Backtest complete! Check data/backtest/ for charts.")
