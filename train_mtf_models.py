"""
Multi-Timeframe Pattern Model Training Script

TRAINING FLOW:
1. Fetch historical data for all 4 timeframes (1D, 4H, 1H, 15M)
2. Analyze SMC patterns on each timeframe  
3. Generate multi-TF features and outcome labels
4. Train pattern recognition model
5. Analyze pattern effectiveness by confluence level

USAGE:
    python train_mtf_models.py                    # Train both BTC and ETH
    python train_mtf_models.py --symbol BTC/USDT  # Train specific symbol
    python train_mtf_models.py --days 120         # Custom training period
"""

import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from src.data.data_fetcher import DataFetcher
from src.models.mtf_pattern_labeler import MultiTimeframePatternLabeler
from src.models.pattern_recognition_model import PatternRecognitionModel
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger()


def load_config_file() -> Dict:
    """Load configuration from config file."""
    return load_config()


def fetch_multi_tf_data(
    fetcher: DataFetcher,
    symbol: str,
    days: int = 90
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch historical data for all 4 timeframes.
    
    API limit is 1000 bars per request. Optimized from bottom-up:
    - 15M: Needs most bars (96/day) - use batch fetch
    - 1H: 24/day - single fetch up to ~41 days
    - 4H: 6/day - single fetch up to ~166 days  
    - 1D: 1/day - single fetch up to 1000 days
    """
    logger.info(f"Fetching multi-TF data for {symbol} ({days} days)...")
    
    # Calculate required bars for each timeframe
    # API limits: 1000 bars per request — batch-fetch when needed
    bars_1d = min(days, 1000)           # 1D: plenty of room
    bars_4h = min(days * 6, 1000)       # 4H: 6 per day, max 166 days in one fetch
    
    logger.info(f"  1D: {bars_1d} bars (single fetch)")
    logger.info(f"  4H: {bars_4h} bars (single fetch)")
    
    # Fetch 1D and 4H with single requests (fit within 1000-bar limit)
    df_1d = fetcher.fetch_ohlcv(symbol, timeframe='1d', limit=bars_1d)
    df_4h = fetcher.fetch_ohlcv(symbol, timeframe='4h', limit=bars_4h)
    
    end_date = datetime.now()
    
    # 1H: batch-fetch for full period (single fetch only covers ~41 days)
    if days <= 41:
        bars_1h = days * 24
        logger.info(f"  1H: {bars_1h} bars (single fetch)")
        df_1h = fetcher.fetch_ohlcv(symbol, timeframe='1h', limit=bars_1h)
    else:
        start_date_1h = end_date - timedelta(days=days)
        logger.info(f"  1H: Batch fetch for {days} days (~{days * 24} bars)")
        df_1h = fetcher.fetch_historical_data(symbol, '1h', start_date_1h, end_date)
    
    # 15M: always batch-fetch for full period (single fetch only covers ~10 days)
    if days <= 10:
        bars_15m = days * 96
        logger.info(f"  15M: {bars_15m} bars (single fetch)")
        df_15m = fetcher.fetch_ohlcv(symbol, timeframe='15m', limit=min(bars_15m, 1000))
    else:
        start_date_15m = end_date - timedelta(days=days)
        logger.info(f"  15M: Batch fetch for {days} days (~{days * 96} bars)")
        df_15m = fetcher.fetch_historical_data(symbol, '15m', start_date_15m, end_date)
    
    logger.info(f"Fetched: 1D={len(df_1d)}, 4H={len(df_4h)}, 1H={len(df_1h)}, 15M={len(df_15m)} bars")
    
    # Log date ranges for debugging data overlap
    if len(df_15m) > 0:
        logger.info(f"  15M date range: {df_15m.index[0]} to {df_15m.index[-1]}")
    if len(df_4h) > 0:
        logger.info(f"  4H date range: {df_4h.index[0]} to {df_4h.index[-1]}")
    
    return df_1d, df_4h, df_1h, df_15m


def train_symbol_model(
    config: Dict,
    symbol: str,
    days: int = 90
) -> Dict:
    """
    Train multi-TF pattern model for a specific symbol.
    
    Args:
        config: Application configuration
        symbol: Trading symbol (e.g., 'BTC/USDT')
        days: Number of days of historical data
        
    Returns:
        Training results dictionary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Multi-TF Pattern Model for {symbol}")
    logger.info(f"{'='*60}\n")
    
    # Initialize components
    fetcher = DataFetcher(config)
    labeler = MultiTimeframePatternLabeler(config)
    model = PatternRecognitionModel(config, symbol=symbol)
    
    # 1. Fetch multi-TF data
    try:
        df_1d, df_4h, df_1h, df_15m = fetch_multi_tf_data(fetcher, symbol, days)
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return {'success': False, 'error': str(e)}
    
    # Validate data
    if len(df_15m) < 500:
        logger.error(f"Insufficient 15M data: {len(df_15m)} bars (need 500+)")
        return {'success': False, 'error': 'Insufficient data'}
    
    # 2. Analyze patterns on all timeframes
    try:
        analyzed_data = labeler.analyze_all_timeframes(df_1d, df_4h, df_1h, df_15m)
    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    # 3. Generate multi-TF features and labels
    try:
        features_df, labels = labeler.generate_labels(analyzed_data)
    except Exception as e:
        logger.error(f"Error generating labels: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    if len(features_df) < 5:
        logger.error(f"Insufficient labeled patterns: {len(features_df)} (need 5+)")
        return {'success': False, 'error': 'Insufficient patterns'}
    
    # 4. Analyze pattern effectiveness by MTF confluence
    pattern_stats = labeler.analyze_pattern_effectiveness(features_df, labels)
    logger.info(f"\n=== Pattern Effectiveness Analysis ===")
    for key, value in pattern_stats.items():
        if 'rate' in key:
            logger.info(f"  {key}: {value*100:.1f}%")
        else:
            logger.info(f"  {key}: {value}")
    
    # 5. Train the model
    try:
        results = model.train(features_df, labels, pattern_stats)
    except Exception as e:
        logger.error(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    # 6. Print summary
    if results.get('success'):
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Training Complete for {symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"  Train Accuracy: {results['train_accuracy']*100:.1f}%")
        logger.info(f"  Test Accuracy:  {results['test_accuracy']*100:.1f}%")
        logger.info(f"  Samples:        {results['samples_trained']}")
        logger.info(f"  Win/Loss:       {results['wins']}/{results['losses']}")
        
        # Top features
        logger.info(f"\n  Top 5 Most Important Features:")
        for i, feat in enumerate(results['feature_importance'][:5]):
            logger.info(f"    {i+1}. {feat['feature']}: {feat['importance']:.4f}")
    
    return results


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train Multi-TF Pattern Models')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Specific symbol to train (e.g., BTC/USDT). Default: train all.')
    parser.add_argument('--days', type=int, default=90,
                        help='Days of historical data to use (default: 90)')
    args = parser.parse_args()
    
    config = load_config_file()
    
    # Determine which symbols to train
    if args.symbol:
        symbols = [args.symbol]
    else:
        # Get from config
        symbols = config.get('trading', {}).get('symbols', ['BTC/USDT', 'ETH/USDT'])
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"  MULTI-TIMEFRAME PATTERN MODEL TRAINING")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Training Period: {args.days} days")
    logger.info(f"{'#'*60}\n")
    
    # Train each symbol
    all_results = {}
    for symbol in symbols:
        results = train_symbol_model(config, symbol, args.days)
        all_results[symbol] = results
    
    # Final summary
    logger.info(f"\n\n{'#'*60}")
    logger.info(f"  TRAINING SUMMARY")
    logger.info(f"{'#'*60}")
    
    for symbol, results in all_results.items():
        if results.get('success'):
            logger.info(f"  ✅ {symbol}: Test Accuracy = {results['test_accuracy']*100:.1f}%")
        else:
            logger.info(f"  ❌ {symbol}: {results.get('error', 'Unknown error')}")
    
    logger.info(f"\nModels saved to: models/saved_models/<SYMBOL>/")
    logger.info(f"Ready to use with main.py\n")


if __name__ == '__main__':
    main()
