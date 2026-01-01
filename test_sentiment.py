"""
Test script for 4-source sentiment analysis system
Tests each source independently and combined output
"""
import yaml
from src.utils.logger import get_logger
from src.sentiment.hybrid_sentiment import HybridSentimentAnalyzer
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


logger = get_logger()


def test_sentiment_sources():
    """Test each sentiment source independently"""

    # Load environment variables
    load_dotenv()

    # Load config
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Initialize analyzer
    print("\n" + "="*60)
    print("Testing 4-Source Sentiment Analysis System")
    print("="*60 + "\n")

    analyzer = HybridSentimentAnalyzer(config)

    # Test symbols
    symbols = ['BTC/USDT', 'ETH/USDT']

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Testing {symbol}")
        print(f"{'='*60}\n")

        # Get sentiment
        result = analyzer.get_sentiment(symbol)

        # Display results
        print(f"Final Sentiment Score: {result['score']:.4f}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"\nBreakdown by Source:")
        print(f"  CoinGecko   (35%): {result['breakdown']['coingecko']}")
        print(f"  RSS Feeds   (35%): {result['breakdown']['rss']}")
        print(f"  Reddit      (20%): {result['breakdown']['reddit']}")
        print(f"  Fear & Greed(10%): {result['breakdown']['fear_greed']}")

        # Classification
        if result['score'] > 0.3:
            classification = "BULLISH ✅"
        elif result['score'] < -0.3:
            classification = "BEARISH ❌"
        else:
            classification = "NEUTRAL ⚪"

        print(f"\nClassification: {classification}")


def check_api_credentials():
    """Check if all required API credentials are configured"""
    print("\n" + "="*60)
    print("Checking API Credentials")
    print("="*60 + "\n")

    load_dotenv()

    credentials = {
        'COINGECKO_API_KEY': os.getenv('COINGECKO_API_KEY'),
        'REDDIT_CLIENT_ID': os.getenv('REDDIT_CLIENT_ID'),
        'REDDIT_CLIENT_SECRET': os.getenv('REDDIT_CLIENT_SECRET'),
    }

    all_configured = True

    for key, value in credentials.items():
        if value and value not in ['your_key_here', 'your_coingecko_api_key_here', 'your_reddit_client_id_here', 'your_reddit_client_secret_here']:
            print(f"✅ {key}: Configured")
        else:
            print(f"❌ {key}: NOT configured")
            all_configured = False

    # Fear & Greed doesn't need API key
    print(f"✅ Fear & Greed Index: No API key needed")

    # RSS doesn't need API key
    print(f"✅ RSS Feeds: No API key needed")

    if not all_configured:
        print(
            "\n⚠️  Some API keys are missing. The system will work with available sources.")
        print("\nTo add missing keys, update your .env file:")
        print("  COINGECKO_API_KEY=your_key_from_coingecko.com")
        print("  REDDIT_CLIENT_ID=your_reddit_app_id")
        print("  REDDIT_CLIENT_SECRET=your_reddit_app_secret")
    else:
        print("\n✅ All API credentials configured!")

    return all_configured


if __name__ == "__main__":
    try:
        # Check credentials first
        check_api_credentials()

        # Run sentiment tests
        test_sentiment_sources()

        print("\n" + "="*60)
        print("✅ Sentiment system test completed!")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
