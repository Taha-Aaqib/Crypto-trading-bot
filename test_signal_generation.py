"""
Test Signal Generation - Debug script to understand why no trades are generated
"""

from src.models.ensemble_model import EnsembleDecisionModel
from src.trading.strategy import TradingStrategy
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_fetcher import DataFetcher
from src.utils.logger import get_logger
from src.utils.helpers import load_config
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


logger = get_logger()


def test_signal_generation():
    """Test signal generation to see what's happening"""

    config = load_config()

    # Initialize components
    data_fetcher = DataFetcher(config)
    data_preprocessor = DataPreprocessor(config)
    strategy = TradingStrategy(config)
    ensemble = EnsembleDecisionModel(config)

    symbol = 'BTC/USDT'

    logger.info("=" * 80)
    logger.info("SIGNAL GENERATION TEST")
    logger.info("=" * 80)

    # Fetch latest data
    logger.info(f"Fetching data for {symbol}...")
    df_1d = data_fetcher.fetch_ohlcv(symbol, timeframe='1d', limit=100)
    df_4h = data_fetcher.fetch_ohlcv(symbol, timeframe='4h', limit=200)
    df_15m = data_fetcher.fetch_ohlcv(symbol, timeframe='15m', limit=500)

    # Preprocess
    df_1d = data_preprocessor.process_pipeline(df_1d)
    df_4h = data_preprocessor.process_pipeline(df_4h)
    df_15m = data_preprocessor.process_pipeline(df_15m)

    logger.info(
        f"Data fetched - 1D: {len(df_1d)}, 4H: {len(df_4h)}, 15M: {len(df_15m)}")

    # Test 1: Multi-timeframe analysis
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Multi-Timeframe Analysis")
    logger.info("=" * 80)

    mtf_analysis = strategy.analyze_multi_timeframe(df_1d, df_4h, df_15m)

    logger.info(f"1D Bias: {mtf_analysis['bias_1d']}")
    logger.info(f"4H Structure: {mtf_analysis['structure_4h']}")
    logger.info(f"15M EMA Trend: {mtf_analysis['ema_trend_15m']}")

    latest_15m = mtf_analysis['latest_15m']
    logger.info(f"\n15M SMC Patterns:")
    logger.info(f"  CHOCH Bullish: {latest_15m.get('choch_bullish', False)}")
    logger.info(f"  CHOCH Bearish: {latest_15m.get('choch_bearish', False)}")
    logger.info(f"  BOS Bullish: {latest_15m.get('bos_bullish', False)}")
    logger.info(f"  BOS Bearish: {latest_15m.get('bos_bearish', False)}")
    logger.info(f"  FVG Bullish: {latest_15m.get('fvg_bullish', False)}")
    logger.info(f"  FVG Bearish: {latest_15m.get('fvg_bearish', False)}")

    # Test 2: Strategy signal
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Strategy Signal Generation")
    logger.info("=" * 80)

    signal = strategy.generate_signal(symbol, mtf_analysis)

    if signal:
        logger.info(f"✓ SIGNAL GENERATED: {signal['direction'].upper()}")
        logger.info(f"  Reason: {signal['reason']}")
        logger.info(f"  Entry: ${signal['entry_price']:.2f}")
    else:
        logger.info(
            "✗ NO SIGNAL - Multi-timeframe alignment conditions not met")

        # Show what's missing
        bias_ok = mtf_analysis['bias_1d'] in ['bullish', 'bearish']
        structure_ok = mtf_analysis['structure_4h'] in ['bullish', 'bearish']
        ema_ok = mtf_analysis['ema_trend_15m'] in ['bullish', 'bearish']
        smc_bullish = (latest_15m.get('choch_bullish') or latest_15m.get(
            'bos_bullish')) and latest_15m.get('fvg_bullish')
        smc_bearish = (latest_15m.get('choch_bearish') or latest_15m.get(
            'bos_bearish')) and latest_15m.get('fvg_bearish')
        smc_ok = smc_bullish or smc_bearish

        logger.info(f"\nConditions Check:")
        logger.info(
            f"  1D Bias defined: {bias_ok} ({mtf_analysis['bias_1d']})")
        logger.info(
            f"  4H Structure defined: {structure_ok} ({mtf_analysis['structure_4h']})")
        logger.info(
            f"  15M EMA trend defined: {ema_ok} ({mtf_analysis['ema_trend_15m']})")
        logger.info(f"  15M SMC patterns: {smc_ok}")
        logger.info(f"    - Bullish setup: {smc_bullish}")
        logger.info(f"    - Bearish setup: {smc_bearish}")

    # Test 3: Ensemble analysis
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Ensemble Model Analysis")
    logger.info("=" * 80)

    ensemble_result = ensemble.analyze_signal(df_15m, symbol)

    logger.info(f"Signal: {ensemble_result['signal'].upper()}")
    logger.info(f"Confidence: {ensemble_result['confidence']:.2%}")
    logger.info(f"Ensemble Score: {ensemble_result['ensemble_score']:.3f}")
    logger.info(f"Agreement: {ensemble_result['agreement']:.0%}")
    logger.info(f"Min Confidence Threshold: {ensemble.min_confidence:.2%}")

    logger.info(f"\nComponent Breakdown:")
    for component, data in ensemble_result['components'].items():
        logger.info(
            f"  {component.upper()}: {data['signal']} (score: {data['score']:.3f})")

    if ensemble_result['sentiment_confirms']:
        logger.info(f"\n✓ Sentiment CONFIRMS the signal")
    elif ensemble_result['sentiment_contradicts']:
        logger.info(f"\n⚠ Sentiment CONTRADICTS the signal")

    if ensemble_result['confidence'] >= ensemble.min_confidence:
        logger.info(
            f"\n✓ ENSEMBLE SIGNAL VALID - Confidence {ensemble_result['confidence']:.2%} >= {ensemble.min_confidence:.2%}")
    else:
        logger.info(
            f"\n✗ ENSEMBLE SIGNAL BELOW THRESHOLD - Confidence {ensemble_result['confidence']:.2%} < {ensemble.min_confidence:.2%}")

    # Test 4: Check historical data for signals
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Scanning Last 100 Candles for Signals")
    logger.info("=" * 80)

    signal_count = 0
    for i in range(max(0, len(df_15m) - 100), len(df_15m)):
        df_subset = df_15m.iloc[:i+1].copy()
        if len(df_subset) < 200:
            continue

        ensemble_result = ensemble.analyze_signal(df_subset, symbol)

        if ensemble_result['confidence'] >= ensemble.min_confidence:
            signal_count += 1
            logger.info(
                f"  Candle {i}: {ensemble_result['signal'].upper()} - Confidence: {ensemble_result['confidence']:.2%}")

    logger.info(f"\nFound {signal_count} valid signals in last 100 candles")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    test_signal_generation()
