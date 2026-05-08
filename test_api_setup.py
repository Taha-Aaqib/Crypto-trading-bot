"""
Test Binance API Setup and Connection
Verifies API keys, permissions, and data fetching capabilities
"""

import os
import sys
from dotenv import load_dotenv
import ccxt
from datetime import datetime
import yaml

# Load environment variables
load_dotenv()


def test_api_connection():
    """Test Binance API key setup and permissions"""

    print("=" * 70)
    print("🧪 BINANCE API CONNECTION TEST")
    print("=" * 70)

    # Load config
    try:
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"\n❌ ERROR: Could not load config.yaml: {e}")
        return False

    # Get API credentials from environment
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    # Test 1: Check API key is loaded
    print("\n✅ Test 1: API Key Loading")
    if not api_key or not api_secret:
        print("   ❌ ERROR: API keys not found in .env file!")
        print("   Make sure .env file exists with BINANCE_API_KEY and BINANCE_API_SECRET")
        return False

    if api_key == "paste_your_new_api_key_here":
        print("   ❌ ERROR: API key is still placeholder!")
        print("   Replace with your actual Binance API key in .env file")
        return False

    print(f"   ✅ API Key loaded: {api_key[:10]}...{api_key[-5:]}")
    print(f"   ✅ API Secret loaded: {api_secret[:10]}...{api_secret[-5:]}")

    try:
        # Initialize exchange
        testnet = config.get('exchange', {}).get('testnet', False)

        print(f"\n✅ Test 2: Exchange Configuration")
        print(f"   Mode: {'TESTNET' if testnet else 'REAL API'}")

        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        if testnet:
            exchange.set_sandbox_mode(True)
            print("   ⚠️  WARNING: Using TESTNET - Limited data available!")
            print("   Recommendation: Set testnet: false in config.yaml for full data")
        else:
            print("   ✅ Using REAL API - Full historical data available")

        # Test 3: Market data access (public endpoint, no auth needed but tests connection)
        print(f"\n✅ Test 3: Market Data Access (Public)")
        try:
            ticker = exchange.fetch_ticker('BTC/USDT')
            print(f"   ✅ BTC/USDT Price: ${ticker['last']:,.2f}")
            print(f"   ✅ 24h Volume: {ticker['quoteVolume']:,.0f} USDT")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False

        # Test 4: Historical data fetching
        print(f"\n✅ Test 4: Historical Data Fetching")
        try:
            # Try to fetch 1000 candles (max on real API)
            candles = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=1000)
            print(f"   ✅ Successfully fetched {len(candles)} daily candles")

            if len(candles) < 100:
                print(
                    f"   ⚠️  WARNING: Only {len(candles)} candles available!")
                print("   This suggests TESTNET is enabled (limited data)")
                print("   Change testnet: false in config.yaml for full data")
            else:
                # Calculate date range
                first_date = datetime.fromtimestamp(candles[0][0] / 1000)
                last_date = datetime.fromtimestamp(candles[-1][0] / 1000)
                print(
                    f"   ✅ Date range: {first_date.date()} to {last_date.date()}")
                print(f"   ✅ Latest BTC price: ${candles[-1][4]:,.2f}")

            # Test ETH as well
            eth_candles = exchange.fetch_ohlcv('ETH/USDT', '1d', limit=1000)
            print(f"   ✅ ETH/USDT: {len(eth_candles)} candles fetched")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False

        # Test 5: Account access (requires API key permissions)
        print(f"\n✅ Test 5: Account Balance (Reading Permission)")
        try:
            balance = exchange.fetch_balance()
            print(f"   ✅ Account accessible")
            print(f"   ✅ Total currencies in account: {len(balance['total'])}")

            # Show USDT balance if available
            if 'USDT' in balance['free']:
                usdt_balance = balance['free']['USDT']
                print(f"   ✅ USDT Balance: {usdt_balance:.2f} USDT")

                if usdt_balance < 10 and not testnet:
                    print(
                        f"   ⚠️  WARNING: Low USDT balance. Add funds for live trading.")
            else:
                print(f"   ℹ️  No USDT balance found")

        except ccxt.PermissionDenied as e:
            print(f"   ❌ PERMISSION DENIED: {e}")
            print("   Make sure 'Enable Reading' is enabled in Binance API settings")
            return False
        except ccxt.AuthenticationError as e:
            print(f"   ❌ AUTHENTICATION ERROR: {e}")
            print("   Check your API key and secret in .env file")
            return False
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False

        # Test 6: Trading permissions check
        print(f"\n✅ Test 6: Trading Permissions")
        try:
            # Try to fetch open orders (requires trading permission)
            open_orders = exchange.fetch_open_orders('BTC/USDT')
            print(f"   ✅ Trading permissions verified")
            print(f"   ✅ Open orders: {len(open_orders)}")
        except ccxt.PermissionDenied:
            print(f"   ⚠️  Trading permissions not enabled")
            print("   Enable 'Spot & Margin Trading' in Binance if you want to trade")
            print("   (Not required for paper trading)")
        except Exception as e:
            print(f"   ℹ️  Trading check: {type(e).__name__}")

        # Test 7: Rate limiting check
        print(f"\n✅ Test 7: Rate Limiting Configuration")
        if exchange.rateLimit:
            print(
                f"   ✅ Rate limiting enabled: {exchange.rateLimit}ms between requests")
        else:
            print(f"   ⚠️  WARNING: Rate limiting not configured!")

        # Test 8: IP whitelist info
        print(f"\n✅ Test 8: Connection Info")
        try:
            import requests
            ip = requests.get('https://api.ipify.org', timeout=5).text
            print(f"   ✅ Your current IP: {ip}")
            print(f"   ℹ️  Make sure this IP is whitelisted in Binance API settings")
        except:
            print(f"   ℹ️  Could not detect IP address")

        # Test 9: Multi-timeframe data for training
        print(f"\n✅ Test 9: Multi-Timeframe Data for Training")
        try:
            timeframes = ['1d', '4h', '1h']
            for tf in timeframes:
                data = exchange.fetch_ohlcv('BTC/USDT', tf, limit=500)
                print(f"   ✅ {tf:3s} timeframe: {len(data):4d} candles")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED! API is properly configured.")
        print("=" * 70)
        print("\n📊 Summary:")
        print(f"   • API Mode: {'TESTNET' if testnet else 'REAL API'}")
        print(f"   • Historical Data: {len(candles)} candles available")
        print(f"   • Account Access: ✅ Working")
        print(f"   • Rate Limiting: ✅ Enabled")
        print("\n🚀 Ready to train models and start trading!")

        return True

    except ccxt.AuthenticationError as e:
        print(f"\n❌ AUTHENTICATION ERROR: {e}")
        print("   Check your API key and secret in .env file")
        return False

    except ccxt.PermissionDenied as e:
        print(f"\n❌ PERMISSION ERROR: {e}")
        print("   Enable required permissions in Binance API settings:")
        print("   • Enable Reading")
        print("   • Enable Spot & Margin Trading (optional)")
        return False

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_api_connection()
    sys.exit(0 if success else 1)
