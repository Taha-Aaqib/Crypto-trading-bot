"""
CoinGecko API Integration for Crypto Sentiment Analysis
Free tier: 10,000 calls/month, 30 calls/min
"""
import requests
from typing import Dict, List, Optional
from datetime import datetime
import time
import os
from src.utils.logger import get_logger

logger = get_logger()


class CoinGeckoAnalyzer:
    """Fetches market sentiment and news from CoinGecko API"""

    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = os.getenv('COINGECKO_API_KEY')
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Rate limit: 30/min = 2 sec between calls

        # Currency mapping (CoinGecko IDs)
        self.currency_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'LTC': 'litecoin',
            'XRP': 'ripple',
            'BCH': 'bitcoin-cash',
            'ADA': 'cardano',
            'DOT': 'polkadot',
            'LINK': 'chainlink',
            'BNB': 'binancecoin',
            'SOL': 'solana',
            'DOGE': 'dogecoin',
            'MATIC': 'matic-network',
            'AVAX': 'avalanche-2',
            'UNI': 'uniswap',
            'ATOM': 'cosmos',
            'XLM': 'stellar',
            'ALGO': 'algorand',
            'VET': 'vechain',
            'ICP': 'internet-computer',
            'FIL': 'filecoin'
        }

    def _rate_limit(self):
        """Enforce rate limiting (30 calls/min = 2 sec between calls)"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def get_coin_data(self, currency: str) -> Optional[Dict]:
        """
        Get comprehensive coin data including sentiment indicators

        Args:
            currency: Currency code (e.g., 'BTC', 'ETH')

        Returns:
            Dict with market data and sentiment indicators
        """
        coin_id = self.currency_map.get(currency)
        if not coin_id:
            logger.warning(f"CoinGecko: Unknown currency {currency}")
            return None

        try:
            self._rate_limit()

            url = f"{self.base_url}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'true',
                'developer_data': 'true',
                'sparkline': 'false'
            }

            # Add API key if available
            headers = {}
            if self.api_key and self.api_key != 'your_coingecko_api_key_here':
                headers['x-cg-demo-api-key'] = self.api_key

            response = requests.get(
                url, params=params, headers=headers, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"CoinGecko API error: {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"CoinGecko error for {currency}: {e}")
            return None

    def get_status_updates(self, currency: str) -> List[str]:
        """
        Get recent status updates/news for a coin

        Args:
            currency: Currency code (e.g., 'BTC')

        Returns:
            List of news descriptions/titles
        """
        coin_id = self.currency_map.get(currency)
        if not coin_id:
            return []

        try:
            self._rate_limit()

            url = f"{self.base_url}/coins/{coin_id}/status_updates"

            # Add API key if available
            headers = {}
            if self.api_key and self.api_key != 'your_coingecko_api_key_here':
                headers['x-cg-demo-api-key'] = self.api_key

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                updates = data.get('status_updates', [])

                # Extract descriptions for FinBERT analysis
                descriptions = []
                for update in updates[:10]:  # Latest 10 updates
                    desc = update.get('description', '')
                    if desc:
                        descriptions.append(desc)

                return descriptions
            else:
                return []

        except Exception as e:
            logger.warning(f"CoinGecko status updates error: {e}")
            return []

    def calculate_sentiment_score(self, currency: str) -> Optional[float]:
        """
        Calculate sentiment score from CoinGecko metrics

        Uses:
        - Community score (social media activity)
        - Developer score (GitHub activity)
        - Market cap rank (popularity)
        - Price change trends

        Args:
            currency: Currency code

        Returns:
            Sentiment score -1 to 1, or None if failed
        """
        data = self.get_coin_data(currency)
        if not data:
            return None

        try:
            # Extract sentiment indicators
            community_score = data.get('community_score', 0)
            developer_score = data.get('developer_score', 0)
            sentiment_votes_up = data.get('sentiment_votes_up_percentage', 50)

            market_data = data.get('market_data', {})
            price_change_24h = market_data.get(
                'price_change_percentage_24h', 0)
            price_change_7d = market_data.get('price_change_percentage_7d', 0)

            # Normalize scores (0-100 to -1 to 1)
            community_norm = (community_score - 50) / 50
            developer_norm = (developer_score - 50) / 50
            sentiment_norm = (sentiment_votes_up - 50) / 50

            # Price momentum (-20% to +20% -> -1 to 1)
            price_24h_norm = max(-1, min(1, price_change_24h / 20))
            price_7d_norm = max(-1, min(1, price_change_7d / 20))

            # Weighted average
            sentiment_score = (
                community_norm * 0.25 +
                developer_norm * 0.15 +
                sentiment_norm * 0.30 +
                price_24h_norm * 0.20 +
                price_7d_norm * 0.10
            )

            logger.info(
                f"CoinGecko sentiment for {currency}: {sentiment_score:.3f}")
            logger.debug(f"  Community: {community_score}, Dev: {developer_score}, "
                         f"Votes: {sentiment_votes_up}%, 24h: {price_change_24h:.1f}%")

            return float(sentiment_score)

        except Exception as e:
            logger.warning(f"CoinGecko sentiment calculation error: {e}")
            return None
