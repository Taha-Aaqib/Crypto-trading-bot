"""
Train Pattern Recognition Models
Backtest SMC patterns and generate high-quality training data
"""

import sys
from src.data.data_fetcher import DataFetcher
from src.data.data_preprocessor import DataPreprocessor
from src.indicators.smc_detector import SMCDetector
from src.indicators.ta_indicators import TechnicalIndicators
from src.models.pattern_recognition_model import PatternRecognitionModel
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger()


def train_pattern_models():
    """
    Train pattern recognition models for BTC and ETH
    Each model learns: "Which SMC pattern combinations are profitable?"
    """
    config = load_config()
    symbols = config.get('trading', {}).get('symbols', ['BTC/USDT', 'ETH/USDT'])
    
    logger.info("="*70)
    logger.info("PATTERN RECOGNITION MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Training pattern recognition models for: {symbols}")
    logger.info("Each model learns which SMC pattern combinations are profitable")
    logger.info("="*70)
    
    for symbol in symbols:
        logger.info(f"\n{'='*70}")
        logger.info(f"Training pattern model for {symbol}")
        logger.info(f"{'='*70}")
        
        try:
            # Fetch historical data
            logger.info(f"Fetching 365 days of historical data for {symbol}...")
            fetcher = DataFetcher(config)
            df = fetcher.fetch_historical_data(
                symbol,
                timeframe='1h',
                days=365
            )
            
            if df is None or len(df) < 100:
                logger.error(f"Failed to fetch data for {symbol}")
                continue
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            
            # Preprocess data
            logger.info("Preprocessing data...")
            preprocessor = DataPreprocessor(config)
            df = preprocessor.preprocess(df)
            logger.info(f"After preprocessing: {len(df)} rows")
            
            # Add SMC patterns
            logger.info("Detecting SMC patterns...")
            smc_detector = SMCDetector(config)
            df = smc_detector.analyze_smc(df)
            
            # Add technical indicators
            logger.info("Adding technical indicators...")
            ta_indicators = TechnicalIndicators(config)
            df = ta_indicators.add_all_indicators(df)
            
            logger.info(f"Data prepared with {len(df)} candles")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Train pattern recognition model
            logger.info(f"\nTraining pattern recognition model for {symbol}...")
            model = PatternRecognitionModel(config, symbol=symbol)
            results = model.train(df)
            
            if results['success']:
                logger.info(f"\n✅ SUCCESS: Pattern model trained for {symbol}")
                logger.info(f"   Train accuracy: {results['train_accuracy']:.1%}")
                logger.info(f"   Test accuracy: {results['test_accuracy']:.1%}")
                logger.info(f"   Samples trained on: {results['samples_trained']}")
                
                # Show pattern effectiveness
                if 'pattern_stats' in results:
                    logger.info(f"\n   Pattern Effectiveness Stats:")
                    for pattern, stat in results['pattern_stats'].items():
                        if 'win_rate' in pattern:
                            logger.info(f"      {pattern}: {stat:.1%}")
                
                # Show top important features
                logger.info(f"\n   Top 5 Important Pattern Features:")
                for i, feat in enumerate(results['feature_importance'][:5], 1):
                    logger.info(f"      {i}. {feat['feature']}: {feat['importance']:.3f}")
            else:
                logger.error(f"Failed to train pattern model for {symbol}: {results.get('error')}")
        
        except Exception as e:
            logger.error(f"Error training pattern model for {symbol}: {e}", exc_info=True)
    
    logger.info(f"\n{'='*70}")
    logger.info("Pattern Recognition Model Training Complete")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    train_pattern_models()
