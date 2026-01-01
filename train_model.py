"""
Model Training Script
Train the ML trading model with historical data
"""

from src.utils.logger import get_logger
from src.indicators.ta_indicators import TechnicalIndicators
from src.indicators.smc_detector import SMCDetector
from src.data.data_fetcher import DataFetcher
from src.models.trading_model import TradingModel
import sys
import os
import yaml
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()


logger = get_logger()


def load_config(config_path: str = 'config/config.yaml'):
    """Load configuration and inject environment variables"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Inject API keys from environment
    config['exchange']['api_key'] = os.getenv(
        'BINANCE_API_KEY', config['exchange'].get('api_key', ''))
    config['exchange']['api_secret'] = os.getenv(
        'BINANCE_API_SECRET', config['exchange'].get('api_secret', ''))

    return config


def prepare_training_data(config: dict, symbol: str, days_back: int = 365):
    """
    Fetch and prepare training data

    Args:
        config: Configuration dictionary
        symbol: Trading symbol
        days_back: Days of historical data

    Returns:
        DataFrame with complete features
    """
    logger.info(f"Preparing training data for {symbol}...")

    # Fetch historical data
    data_fetcher = DataFetcher(config)

    # Calculate start date based on days_back
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Use 1h timeframe for training (faster, more historical data available)
    df = data_fetcher.fetch_historical_data(
        symbol=symbol,
        timeframe='1h',  # Changed from 15m to 1h for 365 days
        start_date=start_date,
        end_date=end_date
    )

    if df is None or len(df) < 100:
        logger.error(f"Insufficient data for {symbol}")
        return None

    logger.info(f"Fetched {len(df)} candles for {symbol}")

    # Add SMC indicators
    smc_detector = SMCDetector(config)
    df = smc_detector.analyze_smc(df)

    # Add technical indicators
    ta_indicators = TechnicalIndicators(config)
    df = ta_indicators.add_all_indicators(df)

    logger.info(f"Added indicators, final dataset: {len(df)} rows")

    return df


def train_model_for_symbol(config: dict, symbol: str, days_back: int = 365):
    """
    Train model for a specific symbol

    Args:
        config: Configuration dictionary
        symbol: Trading symbol
        days_back: Days of historical data

    Returns:
        Training results dictionary
    """
    logger.info(f"=== Training model for {symbol} ===")

    # Prepare data
    df = prepare_training_data(config, symbol, days_back)

    if df is None or len(df) < 1000:
        logger.error(f"Insufficient data to train model for {symbol}")
        return {'success': False, 'error': 'Insufficient data'}

    # Initialize model with symbol parameter (for per-coin model paths)
    model = TradingModel(config, symbol=symbol)

    # Train
    ml_config = config.get('ml_model', {})
    lookahead = ml_config.get('lookahead_periods', 5)

    results = model.train(df, lookahead=lookahead)

    if results['success']:
        logger.info(f"MODEL TRAINED: Model trained successfully for {symbol}")
        logger.info(f"   Train accuracy: {results['train_accuracy']:.3f}")
        logger.info(f"   Test accuracy: {results['test_accuracy']:.3f}")
        logger.info(f"   Training samples: {results['samples_trained']}")
    else:
        logger.error(f"MODEL FAILED: Model training failed for {symbol}")

    return results


def train_all_models(config_path: str = 'config/config.yaml', days_back: int = 365):
    """
    Train models for all configured trading symbols

    Args:
        config_path: Path to configuration file
        days_back: Days of historical data
    """
    logger.info("=" * 60)
    logger.info("ML MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load config
    config = load_config(config_path)

    # Get trading symbols
    symbols = config['trading'].get('symbols', ['BTC/USDT'])

    logger.info(f"Training models for {len(symbols)} symbols: {symbols}")
    logger.info(f"Using {days_back} days of historical data")
    logger.info("")

    results = {}

    for symbol in symbols:
        try:
            result = train_model_for_symbol(config, symbol, days_back)
            results[symbol] = result
            logger.info("")
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            results[symbol] = {'success': False, 'error': str(e)}

    # Summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    successful = sum(1 for r in results.values() if r.get('success'))
    total = len(results)

    logger.info(f"Successful: {successful}/{total}")

    for symbol, result in results.items():
        if result.get('success'):
            logger.info(
                f"  SUCCESS {symbol}: Test Accuracy = {result['test_accuracy']:.3f}"
            )
        else:
            err = result.get('error', 'Unknown error')
            if 'Insufficient data' in str(err) or 'No historical data' in str(err):
                logger.info(
                    f"  FAILED  {symbol}: {err} - check symbol availability or exchange API endpoints")
            else:
                logger.info(f"  FAILED  {symbol}: {err}")

    logger.info("=" * 60)
    logger.info("Training complete!")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train ML trading models')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help='Specific symbol to train (default: all symbols from config)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Days of historical data to use (default: 365)'
    )

    args = parser.parse_args()

    if args.symbol:
        # Train single symbol
        config = load_config(args.config)
        result = train_model_for_symbol(config, args.symbol, args.days)

        if result.get('success'):
            print(f"\nTRAINING SUCCESSFUL")
            print(f"Test accuracy: {result['test_accuracy']:.3f}")
            sys.exit(0)
        else:
            print(f"\nTRAINING FAILED: {result.get('error')}")
            sys.exit(1)
    else:
        # Train all symbols
        results = train_all_models(args.config, args.days)

        successful = sum(1 for r in results.values() if r.get('success'))
        if successful == len(results):
            print(f"\nALL MODELS TRAINED SUCCESSFULLY")
            sys.exit(0)
        else:
            print(
                f"\nWARNING: Some models failed to train ({successful}/{len(results)} successful)")
            sys.exit(1)
