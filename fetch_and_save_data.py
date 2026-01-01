"""
Fetch data from CCXT and save to CSV files
This script will populate data/raw/ and data/processed/ folders
"""

import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta
from src.data.data_fetcher import DataFetcher
from src.indicators.smc_detector import SMCDetector
from src.indicators.ta_indicators import TechnicalIndicators
from src.utils.logger import get_logger
from src.utils.helpers import create_directories

logger = get_logger()


def load_config(config_path: str = 'config/config.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fetch_and_save_data(
    config: dict,
    symbol: str,
    timeframe: str = '15m',
    days_back: int = 365
):
    """
    Fetch data from exchange and save to CSV files

    Args:
        config: Configuration dictionary
        symbol: Trading symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe (e.g., '15m', '1h', '1d')
        days_back: Days of historical data to fetch
    """
    logger.info("=" * 60)
    logger.info(f"FETCHING DATA FOR {symbol} ({timeframe})")
    logger.info("=" * 60)

    # Create directories if they don't exist
    create_directories()

    # Initialize data fetcher
    data_fetcher = DataFetcher(config)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Fetching from exchange: {config['exchange']['name']}")
    logger.info("")

    # ========== STEP 1: FETCH RAW DATA ==========
    logger.info("Step 1: Fetching raw OHLCV data from CCXT...")

    df_raw = data_fetcher.fetch_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )

    if df_raw is None or len(df_raw) < 50:
        logger.error(
            f"FAILED: Insufficient data for {symbol} (got {len(df_raw) if df_raw is not None else 0} candles)")
        return False

    logger.info(f"SUCCESS: Fetched {len(df_raw)} candles")
    logger.info(f"   First candle: {df_raw.index[0]}")
    logger.info(f"   Last candle:  {df_raw.index[-1]}")
    logger.info("")

    # Save raw data
    logger.info("SAVING: raw data to CSV...")
    data_fetcher.save_data(df_raw, symbol, timeframe, data_type='raw')
    logger.info(
        f"SAVED: data/raw/{symbol.replace('/', '_')}_{timeframe}_raw.csv")
    logger.info("")

    # ========== STEP 2: ADD INDICATORS ==========
    logger.info("Step 2: Adding SMC indicators...")

    df_processed = df_raw.copy()

    # Add SMC indicators
    smc_detector = SMCDetector(config)
    df_processed = smc_detector.analyze_smc(df_processed)

    smc_columns = ['choch_bullish', 'choch_bearish', 'bos_bullish',
                   'bos_bearish', 'fvg_bullish', 'fvg_bearish']
    smc_found = df_processed[smc_columns].any().sum()
    logger.info(f"SUCCESS: SMC indicators added ({smc_found} types detected)")
    logger.info("")

    logger.info("Step 3: Adding Technical Analysis indicators...")

    # Add TA indicators
    ta_indicators = TechnicalIndicators(config)
    df_processed = ta_indicators.add_all_indicators(df_processed)

    ta_columns = ['rsi', 'macd', 'ema_50', 'ema_200', 'atr',
                  'bb_upper', 'bb_lower', 'stochastic']
    ta_added = [col for col in ta_columns if col in df_processed.columns]
    logger.info(f"SUCCESS: TA indicators added ({len(ta_added)} indicators)")
    logger.info(f"   Indicators: {', '.join(ta_added)}")
    logger.info("")

    # Save processed data
    logger.info("SAVING: processed data to CSV...")
    data_fetcher.save_data(df_processed, symbol,
                           timeframe, data_type='processed')
    logger.info(
        f"SAVED: data/processed/{symbol.replace('/', '_')}_{timeframe}_processed.csv")
    logger.info("")

    # ========== SUMMARY ==========
    logger.info("=" * 60)
    logger.info("DATA FETCH SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Symbol:           {symbol}")
    logger.info(f"Timeframe:        {timeframe}")
    logger.info(f"Total candles:    {len(df_processed)}")
    logger.info(f"Raw columns:      {len(df_raw.columns)}")
    logger.info(f"Processed columns: {len(df_processed.columns)}")
    logger.info(f"Date range:       {(end_date - start_date).days} days")
    logger.info("")
    logger.info("Files created:")
    logger.info(
        f"  [OK] data/raw/{symbol.replace('/', '_')}_{timeframe}_raw.csv")
    logger.info(
        f"  [OK] data/processed/{symbol.replace('/', '_')}_{timeframe}_processed.csv")
    logger.info("=" * 60)

    return True


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("CRYPTO DATA FETCHER")
    print("=" * 60)
    print()

    # Load config
    config = load_config()

    # Get trading symbols from config
    symbols = config['trading'].get('symbols', ['BTC/USDT'])
    timeframe = config['trading'].get('timeframe', '15m')

    print(f"Configured symbols: {symbols}")
    print(f"Timeframe: {timeframe}")
    print()

    # Ask user for confirmation
    days_input = input(
        "How many days of historical data? (default: 30): ").strip()
    days_back = int(days_input) if days_input else 30

    print()
    print(f"Will fetch {days_back} days of data for {len(symbols)} symbol(s)")
    confirm = input("Continue? (y/n): ").strip().lower()

    if confirm != 'y':
        print("CANCELLED by user")
        return

    print()

    # Fetch data for each symbol
    success_count = 0
    for symbol in symbols:
        try:
            success = fetch_and_save_data(
                config=config,
                symbol=symbol,
                timeframe=timeframe,
                days_back=days_back
            )
            if success:
                success_count += 1
            print()
        except Exception as e:
            logger.error(f"ERROR: Failed to fetch data for {symbol}: {e}")
            print()

    # Final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Successfully fetched: {success_count}/{len(symbols)} symbols")
    print()
    print("Next steps:")
    print("  1. Check data/raw/ folder for raw OHLCV data")
    print("  2. Check data/processed/ folder for data with indicators")
    print("  3. Run 'python train_model.py' to train ML model")
    print("=" * 60)


if __name__ == '__main__':
    main()
