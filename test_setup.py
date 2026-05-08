"""
Test script to verify all modules are working correctly
Run this after setup to ensure everything is configured properly
"""

import sys
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def test_imports():
    """Test if all required modules can be imported"""
    print_header("Testing Module Imports")
    
    modules = [
        ('pandas', 'Data processing'),
        ('numpy', 'Numerical computing'),
        ('ccxt', 'Exchange connectivity'),
        ('yaml', 'Configuration'),
        ('sqlalchemy', 'Database'),
        ('textblob', 'Sentiment analysis'),
        ('vaderSentiment.vaderSentiment', 'VADER sentiment'),
        ('tweepy', 'Twitter API'),
        ('streamlit', 'Dashboard'),
    ]
    
    failed = []
    for module, description in modules:
        try:
            __import__(module)
            print(f"✅ {module:30s} - {description}")
        except ImportError as e:
            print(f"❌ {module:30s} - {description} (FAILED)")
            failed.append(module)
    
    if failed:
        print(f"\n⚠️ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_config():
    """Test configuration loading"""
    print_header("Testing Configuration")
    
    try:
        from src.utils.helpers import load_config
        config = load_config()
        
        print("✅ Configuration file loaded")
        print(f"   Exchange: {config['exchange']['name']}")
        print(f"   Mode: {config['trading']['mode']}")
        print(f"   Symbols: {', '.join(config['trading']['symbols'])}")
        print(f"   Timeframes: {config['timeframes']}")
        
        # Check if API keys are configured
        api_key = config['exchange']['api_key']
        if 'YOUR_' in api_key:
            print("\n⚠️ WARNING: API keys not configured yet!")
            print("   Edit config/config.yaml and add your API keys")
            return False
        else:
            print("✅ API keys configured")
            return True
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_utils():
    """Test utility modules"""
    print_header("Testing Utility Modules")
    
    try:
        from src.utils.logger import get_logger
        from src.utils.helpers import calculate_position_size, timeframe_to_minutes
        
        # Test logger
        logger = get_logger()
        logger.info("Logger test message")
        print("✅ Logger working")
        
        # Test helpers
        position_size = calculate_position_size(10000, 0.02, 50000, 49000)
        print(f"✅ Position size calculation: ${position_size:.2f}")
        
        minutes = timeframe_to_minutes('15m')
        print(f"✅ Timeframe conversion: 15m = {minutes} minutes")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        return False

def test_data_fetcher():
    """Test data fetching"""
    print_header("Testing Data Fetcher")
    
    try:
        from src.data.data_fetcher import DataFetcher
        from src.utils.helpers import load_config
        
        config = load_config()
        fetcher = DataFetcher(config)
        
        print("✅ Data fetcher initialized")
        print("   Attempting to fetch BTC/USDT data...")
        
        df = fetcher.fetch_ohlcv('BTC/USDT', '15m', limit=10)
        
        if not df.empty:
            print(f"✅ Successfully fetched {len(df)} candles")
            print(f"   Latest close: ${df.iloc[-1]['close']:.2f}")
            print(f"   Timestamp: {df.index[-1]}")
            return True
        else:
            print("❌ No data fetched - check API keys and connection")
            return False
            
    except Exception as e:
        print(f"❌ Data fetcher test failed: {e}")
        print("   This is normal if API keys are not configured yet")
        return False

def test_smc_detector():
    """Test SMC detector"""
    print_header("Testing SMC Detector")
    
    try:
        from src.indicators.smc_detector import SMCDetector
        from src.utils.helpers import load_config
        import pandas as pd
        import numpy as np
        
        config = load_config()
        smc = SMCDetector(config)
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        sample_df = pd.DataFrame({
            'open': np.random.uniform(45000, 50000, 100),
            'high': np.random.uniform(45000, 50000, 100),
            'low': np.random.uniform(45000, 50000, 100),
            'close': np.random.uniform(45000, 50000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Ensure high is highest and low is lowest
        sample_df['high'] = sample_df[['open', 'high', 'low', 'close']].max(axis=1)
        sample_df['low'] = sample_df[['open', 'high', 'low', 'close']].min(axis=1)
        
        print("✅ SMC detector initialized")
        print("   Analyzing sample data...")
        
        df_smc = smc.analyze_smc(sample_df)
        
        print(f"✅ SMC analysis complete")
        print(f"   Columns added: {list(set(df_smc.columns) - set(sample_df.columns))}")
        
        return True
        
    except Exception as e:
        print(f"❌ SMC detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database():
    """Test database connection"""
    print_header("Testing Database")
    
    try:
        from src.utils.db_manager import DatabaseManager
        from src.utils.helpers import load_config
        
        config = load_config()
        db = DatabaseManager(config)
        
        print("✅ Database initialized")
        print(f"   Type: {config['database']['type']}")
        print(f"   Path: {config['database']['path']}")
        
        # Test saving a dummy signal
        signal_data = {
            'symbol': 'BTC/USDT',
            'signal_type': 'test',
            'direction': 'long',
            'strength': 0.8,
            'details': 'Test signal from verification script'
        }
        
        db.save_signal(signal_data)
        print("✅ Signal saved to database")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  SMART MONEY CONCEPT TRADING BOT - SETUP VERIFICATION")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {
        'Imports': test_imports(),
        'Configuration': test_config(),
        'Utilities': test_utils(),
        'Database': test_database(),
        'SMC Detector': test_smc_detector(),
        'Data Fetcher': test_data_fetcher(),
    }
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("  🎉 ALL TESTS PASSED! You're ready to start trading!")
    elif passed >= total * 0.8:
        print("  ⚠️ Most tests passed. Review failed tests above.")
    else:
        print("  ❌ Several tests failed. Review setup instructions.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
