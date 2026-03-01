"""
Data fetcher module for cryptocurrency market data
Uses CCXT library to fetch OHLCV data from exchanges
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.utils.logger import get_logger
from src.utils.helpers import timeframe_to_minutes
from src.utils.rate_limiter import binance_rate_limiter

logger = get_logger()


class DataFetcher:
    """Fetch cryptocurrency market data from exchanges"""

    def __init__(self, config: Dict):
        self.config = config
        self.exchange_config = config['exchange']
        self.exchange = self._initialize_exchange()

    def _initialize_exchange(self):
        """Initialize exchange connection"""
        exchange_name = self.exchange_config['name']

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)

        # Get trading mode from config
        trading_mode = self.config.get('trading', {}).get('mode', 'paper')

        # Only include API keys if they are provided (for authenticated endpoints)
        api_key = self.exchange_config.get('api_key', '')
        api_secret = self.exchange_config.get('api_secret', '')

        # Determine market type from config (futures or spot)
        market_type = self.config.get('trading', {}).get('market_type', 'spot')
        default_type = 'future' if market_type == 'futures' else 'spot'
        # Override to futures if testnet is enabled (testnet uses futures endpoint)
        if self.exchange_config.get('testnet'):
            default_type = 'future'

        exchange_options = {
            'enableRateLimit': True,
            'options': {
                'defaultType': default_type
            }
        }

        # For paper trading, don't use API keys - use public endpoints only
        # Only add credentials for live trading mode
        if trading_mode == 'live' and api_key and api_secret:
            exchange_options['apiKey'] = api_key
            exchange_options['secret'] = api_secret
            logger.info(
                f"Connected to {exchange_name} LIVE with authentication")
        else:
            logger.info(
                f"Connected to {exchange_name} (public data only - no authentication)")

        exchange = exchange_class(exchange_options)

        # Use testnet if configured
        if self.exchange_config.get('testnet'):
            exchange.set_sandbox_mode(True)

        return exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '15m',
        limit: int = 500,
        since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '15m', '1h', '1d')
            limit: Number of candles to fetch
            since: Start date for historical data

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert since to timestamp if provided
            since_ts = None
            if since:
                since_ts = int(since.timestamp() * 1000)

            # Rate limiting - weight 1 for OHLCV request
            binance_rate_limiter.acquire(tokens=1)

            # Fetch data from exchange
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since_ts
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logger.info(
                f"Fetched {len(df)} candles for {symbol} on {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Fetch historical data for a date range

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_date: Start date
            end_date: End date (default: now)

        Returns:
            DataFrame with historical OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()

        # Calculate timeframe in minutes
        tf_minutes = timeframe_to_minutes(timeframe)

        # Fetch data in chunks
        all_data = []
        current_date = start_date

        while current_date < end_date:
            df = self.fetch_ohlcv(
                symbol, timeframe, limit=1000, since=current_date)

            if df.empty:
                break

            all_data.append(df)

            # Move cursor to AFTER the last fetched candle
            # Skip forward by one candle of the current timeframe
            last_timestamp = df.index[-1].to_pydatetime()
            next_timestamp = last_timestamp + \
                timedelta(minutes=tf_minutes)  # Start after last candle

            # If we got less than 100 candles, we're probably at the end
            if len(df) < 100:
                logger.info(
                    f"Fetched {len(df)} candles, likely reached end of available data")
                break

            current_date = next_timestamp

            # Avoid rate limiting
            self.exchange.sleep(self.exchange.rateLimit)

        if not all_data:
            logger.warning(f"No historical data found for {symbol}")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.sort_index()

        # Filter to date range
        combined_df = combined_df[(combined_df.index >= start_date) & (
            combined_df.index <= end_date)]

        logger.info(
            f"Fetched {len(combined_df)} historical candles for {symbol}")
        return combined_df

    def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker information"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}

    def fetch_balance(self) -> Dict:
        """Fetch account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Fetch order book for liquidity analysis"""
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit=limit)
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {}

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, data_type: str = 'raw'):
        """Save DataFrame to CSV"""
        from src.utils.helpers import get_data_path

        filepath = get_data_path(data_type, symbol, timeframe)
        df.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")

    def load_data(self, symbol: str, timeframe: str, data_type: str = 'raw') -> pd.DataFrame:
        """Load DataFrame from CSV"""
        from src.utils.helpers import get_data_path
        import os

        filepath = get_data_path(data_type, symbol, timeframe)

        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            logger.info(f"Loaded data from {filepath}")
            return df
        else:
            logger.warning(f"Data file not found: {filepath}")
            return pd.DataFrame()
