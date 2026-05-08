"""
Quick Setup & Test Script
Validates installation and trains initial model
"""

import sys
import os


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def check_dependencies():
    """Check if all required packages are installed"""
    print_header("CHECKING DEPENDENCIES")

    required_packages = [
        'pandas', 'numpy', 'ccxt', 'tweepy', 'textblob',
        'vaderSentiment', 'transformers', 'torch', 'sklearn',
        'streamlit', 'plotly', 'sqlalchemy', 'yaml', 'joblib'
    ]

    missing = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies installed!")
    return True


def check_config():
    """Check if config file exists"""
    print_header("CHECKING CONFIGURATION")

    if not os.path.exists('config/config.yaml'):
        print("❌ config/config.yaml not found")
        return False

    print("✅ Configuration file found")

    # Check if API keys are configured
    try:
        import yaml
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Check exchange config
        exchange_key = config.get('exchange', {}).get('api_key', '')
        if 'YOUR_' in exchange_key or not exchange_key:
            print("⚠️  Exchange API keys not configured (OK for paper trading)")
        else:
            print("✅ Exchange API keys configured")

        # Check Twitter config
        twitter_token = config.get('twitter', {}).get('bearer_token', '')
        if 'YOUR_' in twitter_token or not twitter_token:
            print("⚠️  Twitter API not configured (sentiment will be limited)")
        else:
            print("✅ Twitter API configured")

    except Exception as e:
        print(f"⚠️  Could not validate config: {e}")

    return True


def setup_directories():
    """Create necessary directories"""
    print_header("SETTING UP DIRECTORIES")

    dirs = [
        'data/backtest',
        'data/processed',
        'data/raw',
        'logs',
        'models/saved_models'
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ {dir_path}")

    print("\n✅ Directories created!")
    return True


def test_imports():
    """Test if custom modules can be imported"""
    print_header("TESTING CUSTOM MODULES")

    modules = [
        'src.models.lstm_model',
        'src.models.ensemble_model',
        'src.sentiment.finbert_analyzer',
        'src.indicators.smc_detector',
        'src.indicators.ta_indicators',
        'src.trading.order_executor'
    ]

    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module} - Error: {e}")
            return False

    print("\n✅ All custom modules can be imported!")
    return True


def download_finbert():
    """Download FinBERT model (first time only)"""
    print_header("DOWNLOADING FINBERT MODEL (First Time Only)")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "ProsusAI/finbert"
        print(f"Downloading {model_name}...")
        print("This may take a few minutes (one-time download, ~500MB)")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        print("✅ FinBERT model downloaded successfully!")
        return True

    except Exception as e:
        print(f"⚠️  Could not download FinBERT: {e}")
        print("You can still use VADER sentiment (set use_finbert: false in config)")
        return False


def run_quick_test():
    """Run a quick test of core functionality"""
    print_header("RUNNING QUICK TEST")

    try:
        import yaml
        from src.models.ensemble_model import EnsembleDecisionModel

        # Load config
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Test ensemble model initialization
        model = EnsembleDecisionModel(config)
        print("✅ Ensemble model initialized")
        print(f"   - LSTM available: {model.use_lstm}")
        print(f"   - Weights: SMC={model.weights['smc']}, TA={model.weights['ta']}, ML={model.weights['ml']}")

        # Test indicators
        import pandas as pd
        import numpy as np
        from src.indicators.ta_indicators import TechnicalIndicators

        # Create dummy data
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'volume': np.random.randint(1000, 10000, 100)
        })
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)

        ti = TechnicalIndicators(config)
        df_ind = ti.add_all_indicators(df)
        print("✅ Technical indicators work (EMA, RSI, ATR, Divergence)")

        print("\n✅ Quick test passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all setup steps"""
    print_header("AI-ENHANCED TRADING BOT - SETUP")

    print("This script will:")
    print("1. Check dependencies")
    print("2. Verify configuration")
    print("3. Create directories")
    print("4. Test modules")
    print("5. Download AI models")
    print("6. Run quick test")

    input("\nPress Enter to continue...")

    # Run checks
    steps = [
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
        ("Directories", setup_directories),
        ("Custom Modules", test_imports),
        ("FinBERT Model", download_finbert),
        ("Quick Test", run_quick_test)
    ]

    results = []

    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))
        except Exception as e:
            print(f"\n❌ Error in {step_name}: {e}")
            results.append((step_name, False))

    # Summary
    print_header("SETUP SUMMARY")

    for step_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")

    successful = sum(1 for _, s in results if s)
    total = len(results)

    print(f"\nCompleted: {successful}/{total} steps")

    if successful == total:
        print("\n" + "=" * 60)
        print("  🎉 SETUP COMPLETE! YOU'RE READY TO GO!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Configure API keys in config/config.yaml (optional for paper trading)")
        print("2. Train ML model: python train_model.py")
        print("3. Run bot: python main.py")
        print("4. View dashboard: python dashboard/app.py")
        print("\nFor detailed instructions, see AI_IMPLEMENTATION_GUIDE.md")
    else:
        print("\n⚠️  Some steps failed. Please fix errors above and run again.")
        print("Most features will still work, but some functionality may be limited.")

    return successful == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
