"""
LSTM Model Training Script with Walk-Forward Validation
Train LSTM models for trading signal prediction

Usage:
    python train_lstm_model.py                          # Train for all symbols
    python train_lstm_model.py --symbol BTC/USDT        # Train for specific symbol
    python train_lstm_model.py --use-kaggle             # Use Kaggle historical data
    python train_lstm_model.py --walk-forward           # Use walk-forward validation
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import yaml
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import get_logger
from src.models.lstm_model import LSTMTradingModel
from src.indicators.ta_indicators import TechnicalIndicators
from src.indicators.smc_detector import SMCDetector
from src.data.data_fetcher import DataFetcher

# Load environment variables
load_dotenv()

logger = get_logger()


def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Inject API keys from environment
    config['exchange']['api_key'] = os.getenv(
        'BINANCE_API_KEY', config['exchange'].get('api_key', ''))
    config['exchange']['api_secret'] = os.getenv(
        'BINANCE_API_SECRET', config['exchange'].get('api_secret', ''))
    
    return config


def prepare_data_with_indicators(
    df: pd.DataFrame, 
    config: Dict
) -> pd.DataFrame:
    """Add SMC and technical indicators to data"""
    if df is None or len(df) < 200:
        return None
    
    # Add ENHANCED SMC indicators (OB, OTE, sweeps needed for LSTM features)
    smc_detector = SMCDetector(config)
    df = smc_detector.analyze_smc_enhanced(df)
    
    # Add technical indicators
    ta_indicators = TechnicalIndicators(config)
    df = ta_indicators.add_all_indicators(df)
    
    return df


def load_training_data(
    config: Dict, 
    symbol: str, 
    use_kaggle: bool = False,
    days_back: int = 365 * 5  # 5 years default
) -> pd.DataFrame:
    """
    Load training data from Kaggle or API
    
    Args:
        config: Configuration dict
        symbol: Trading pair
        use_kaggle: Whether to use Kaggle data
        days_back: Days of data (for API fetch)
    """
    if use_kaggle:
        try:
            from src.data.kaggle_data_loader import KaggleDataLoader
            
            loader = KaggleDataLoader(config)
            df = loader.prepare_training_data(
                symbol=symbol,
                timeframe='1h',
                start_date=datetime(2019, 1, 1),
                end_date=datetime.now()
            )
            
            if not df.empty:
                logger.info(f"Loaded {len(df)} candles from Kaggle data")
                return prepare_data_with_indicators(df, config)
            else:
                logger.warning("No Kaggle data found, falling back to API")
                print(loader.download_instructions())
        except Exception as e:
            logger.warning(f"Kaggle data load failed: {e}, falling back to API")
    
    # Fallback to API fetch
    data_fetcher = DataFetcher(config)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    df = data_fetcher.fetch_historical_data(
        symbol=symbol,
        timeframe='1h',
        start_date=start_date,
        end_date=end_date
    )
    
    if df is None or len(df) < 1000:
        logger.error(f"Insufficient data for {symbol}")
        return None
    
    logger.info(f"Loaded {len(df)} candles from API")
    return prepare_data_with_indicators(df, config)


def walk_forward_train(
    config: Dict,
    symbol: str,
    df: pd.DataFrame,
    train_months: int = 12,
    test_months: int = 3,
    lookahead: int = 5
) -> Dict:
    """
    Walk-forward validation training
    
    Splits data into rolling windows:
    - Train on months [0, train_months]
    - Test on months [train_months, train_months + test_months]
    - Slide forward by test_months and repeat
    
    Args:
        config: Configuration dict
        symbol: Trading pair
        df: Full dataset
        train_months: Months per training window
        test_months: Months per test window
        lookahead: Prediction lookahead
        
    Returns:
        Results dictionary with per-window metrics
    """
    logger.info("=" * 60)
    logger.info(f"WALK-FORWARD VALIDATION FOR {symbol}")
    logger.info(f"Train window: {train_months} months, Test window: {test_months} months")
    logger.info("=" * 60)
    
    # Calculate window sizes in rows (assuming 1h = 24 rows/day)
    rows_per_month = 24 * 30
    train_size = train_months * rows_per_month
    test_size = test_months * rows_per_month
    
    # Track results per window
    window_results = []
    all_train_acc = []
    all_test_acc = []
    
    # Walk forward through data
    start_idx = 0
    window_num = 0
    
    while start_idx + train_size + test_size <= len(df):
        window_num += 1
        train_end = start_idx + train_size
        test_end = train_end + test_size
        
        train_df = df.iloc[start_idx:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        
        logger.info(f"\n--- Window {window_num} ---")
        logger.info(f"Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} rows)")
        logger.info(f"Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} rows)")
        
        # Train model on this window
        model = LSTMTradingModel(config, symbol=f"{symbol}_wf{window_num}")
        
        # Temporarily disable saving to avoid overwriting
        original_model_path = model.model_path
        original_scaler_path = model.scaler_path
        model.model_path = f"models/temp/wf_{window_num}_model.pt"
        model.scaler_path = f"models/temp/wf_{window_num}_scaler.pkl"
        os.makedirs("models/temp", exist_ok=True)
        
        results = model.train(train_df, lookahead=lookahead, validation_split=0.15)
        
        if not results['success']:
            logger.warning(f"Window {window_num} training failed")
            start_idx += test_size
            continue
        
        # Evaluate on out-of-sample test data
        test_predictions = []
        test_actuals = []
        
        # Need full sequence context for each prediction
        combined_df = pd.concat([train_df.iloc[-model.sequence_length:], test_df])
        
        for i in range(model.sequence_length, len(combined_df)):
            context = combined_df.iloc[i - model.sequence_length:i + 1]
            signal, confidence = model.predict(context)
            
            # Get actual label
            if i < len(combined_df) - lookahead:
                future_return = (
                    combined_df.iloc[i + lookahead]['close'] / 
                    combined_df.iloc[i]['close'] - 1
                )
                if future_return > 0.01:
                    actual = 1
                elif future_return < -0.01:
                    actual = -1
                else:
                    actual = 0
                
                test_predictions.append(signal)
                test_actuals.append(actual)
        
        # Calculate test accuracy
        if test_predictions:
            test_accuracy = np.mean(
                np.array(test_predictions) == np.array(test_actuals)
            )
        else:
            test_accuracy = 0
        
        window_result = {
            'window': window_num,
            'train_start': str(train_df.index[0]),
            'train_end': str(train_df.index[-1]),
            'test_start': str(test_df.index[0]),
            'test_end': str(test_df.index[-1]),
            'train_accuracy': results['train_accuracy'],
            'val_accuracy': results['test_accuracy'],
            'oos_accuracy': test_accuracy,  # Out-of-sample
            'overfit_gap': results['train_accuracy'] - test_accuracy
        }
        
        window_results.append(window_result)
        all_train_acc.append(results['train_accuracy'])
        all_test_acc.append(test_accuracy)
        
        logger.info(f"Train Acc: {results['train_accuracy']:.3f}")
        logger.info(f"OOS Test Acc: {test_accuracy:.3f}")
        logger.info(f"Overfit Gap: {results['train_accuracy'] - test_accuracy:.3f}")
        
        # Clean up temp files
        try:
            os.remove(model.model_path)
            os.remove(model.scaler_path)
        except:
            pass
        
        # Slide forward
        start_idx += test_size
    
    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD SUMMARY")
    logger.info("=" * 60)
    
    if all_test_acc:
        avg_train = np.mean(all_train_acc)
        avg_test = np.mean(all_test_acc)
        std_test = np.std(all_test_acc)
        avg_gap = avg_train - avg_test
        
        logger.info(f"Windows tested: {len(window_results)}")
        logger.info(f"Average Train Accuracy: {avg_train:.3f}")
        logger.info(f"Average OOS Test Accuracy: {avg_test:.3f} (+/- {std_test:.3f})")
        logger.info(f"Average Overfit Gap: {avg_gap:.3f}")
        
        if avg_gap > 0.15:
            logger.warning("⚠️  HIGH OVERFITTING DETECTED (gap > 15%)")
        elif avg_gap > 0.10:
            logger.warning("⚠️  Moderate overfitting (gap 10-15%)")
        else:
            logger.info("✅ Overfitting within acceptable range")
        
        return {
            'success': True,
            'symbol': symbol,
            'windows': window_results,
            'avg_train_accuracy': avg_train,
            'avg_test_accuracy': avg_test,
            'std_test_accuracy': std_test,
            'overfit_gap': avg_gap
        }
    
    return {'success': False, 'error': 'No valid windows'}


def train_single_symbol(
    config: Dict,
    symbol: str,
    use_kaggle: bool = False,
    walk_forward: bool = False,
    days_back: int = 365 * 5
) -> Dict:
    """
    Train LSTM model for a single symbol
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TRAINING LSTM FOR {symbol}")
    logger.info(f"{'=' * 60}")
    
    # Load data
    df = load_training_data(config, symbol, use_kaggle, days_back)
    
    if df is None or len(df) < 1000:
        return {'success': False, 'error': 'Insufficient data'}
    
    logger.info(f"Data loaded: {len(df)} rows")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    ml_config = config.get('ml_model', {})
    lookahead = ml_config.get('lookahead_periods', 5)
    
    if walk_forward:
        # Walk-forward validation
        return walk_forward_train(
            config=config,
            symbol=symbol,
            df=df,
            train_months=12,
            test_months=3,
            lookahead=lookahead
        )
    else:
        # Standard training
        model = LSTMTradingModel(config, symbol=symbol)
        results = model.train(df, lookahead=lookahead)
        
        if results['success']:
            logger.info(f"✅ LSTM trained for {symbol}")
            logger.info(f"   Train accuracy: {results['train_accuracy']:.3f}")
            logger.info(f"   Test accuracy: {results['test_accuracy']:.3f}")
            
            # Check for overfitting
            gap = results['train_accuracy'] - results['test_accuracy']
            if gap > 0.15:
                logger.warning(f"⚠️  Overfitting detected (gap: {gap:.3f})")
        
        return results


def train_all_symbols(
    config_path: str = 'config/config.yaml',
    use_kaggle: bool = False,
    walk_forward: bool = False,
    days_back: int = 365 * 5
) -> Dict:
    """
    Train LSTM models for all configured symbols
    """
    logger.info("=" * 60)
    logger.info("LSTM MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    config = load_config(config_path)
    symbols = config['trading'].get('symbols', ['BTC/USDT'])
    
    logger.info(f"Training for {len(symbols)} symbols: {symbols}")
    logger.info(f"Use Kaggle data: {use_kaggle}")
    logger.info(f"Walk-forward validation: {walk_forward}")
    
    results = {}
    
    for symbol in symbols:
        try:
            result = train_single_symbol(
                config, symbol, use_kaggle, walk_forward, days_back
            )
            results[symbol] = result
        except Exception as e:
            logger.error(f"Error training {symbol}: {e}")
            results[symbol] = {'success': False, 'error': str(e)}
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    successful = sum(1 for r in results.values() if r.get('success'))
    
    for symbol, result in results.items():
        if result.get('success'):
            if 'avg_test_accuracy' in result:  # Walk-forward
                logger.info(
                    f"✅ {symbol}: Avg Test Acc = {result['avg_test_accuracy']:.3f} "
                    f"(+/- {result['std_test_accuracy']:.3f})"
                )
            else:  # Standard
                logger.info(
                    f"✅ {symbol}: Test Acc = {result['test_accuracy']:.3f}"
                )
        else:
            logger.info(f"❌ {symbol}: {result.get('error', 'Failed')}")
    
    logger.info(f"\nSuccessful: {successful}/{len(results)}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM trading models')
    parser.add_argument(
        '--config', type=str, default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--symbol', type=str, default=None,
        help='Specific symbol to train (default: all from config)'
    )
    parser.add_argument(
        '--use-kaggle', action='store_true',
        help='Use Kaggle historical data (2019+)'
    )
    parser.add_argument(
        '--walk-forward', action='store_true',
        help='Use walk-forward validation'
    )
    parser.add_argument(
        '--days', type=int, default=365 * 5,
        help='Days of historical data for API fetch (default: 5 years)'
    )
    
    args = parser.parse_args()
    
    if args.symbol:
        config = load_config(args.config)
        result = train_single_symbol(
            config, args.symbol, args.use_kaggle, args.walk_forward, args.days
        )
        
        if result.get('success'):
            print(f"\n✅ LSTM TRAINING SUCCESSFUL")
            sys.exit(0)
        else:
            print(f"\n❌ TRAINING FAILED: {result.get('error')}")
            sys.exit(1)
    else:
        results = train_all_symbols(
            args.config, args.use_kaggle, args.walk_forward, args.days
        )
        
        successful = sum(1 for r in results.values() if r.get('success'))
        if successful == len(results):
            print(f"\n✅ ALL LSTM MODELS TRAINED SUCCESSFULLY")
            sys.exit(0)
        else:
            print(f"\n⚠️  {successful}/{len(results)} models trained")
            sys.exit(1)
