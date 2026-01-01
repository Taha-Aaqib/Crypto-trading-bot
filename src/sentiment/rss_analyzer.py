"""
RSS Feed Analyzer for Crypto News
Fetches headlines from major crypto news outlets and analyzes with FinBERT
"""
import feedparser
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger()


class RSSAnalyzer:
    """
    Fetch and parse RSS feeds from crypto news sources
    Provides headlines for FinBERT sentiment analysis
    """

    def __init__(self, config: Dict):
        self.config = config
        self.feeds = config.get('sentiment', {}).get('rss', {}).get('feeds', [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://decrypt.co/feed",
            "https://bitcoinmagazine.com/.rss/full/"
        ])

        # Cache to avoid re-fetching same headlines
        self._cache = {}
        self._cache_duration = timedelta(minutes=15)

    def get_headlines(self, symbol: str, limit: int = 20) -> List[str]:
        """
        Fetch latest headlines from RSS feeds

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            limit: Max headlines to return

        Returns:
            List of headline strings
        """
        base_currency = symbol.split('/')[0]
        currency_keywords = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'USDT': ['tether', 'usdt', 'stablecoin']
        }
        keywords = currency_keywords.get(
            base_currency, [base_currency.lower()])

        # Check cache
        cache_key = f"{base_currency}_{limit}"
        if cache_key in self._cache:
            cached_time, cached_headlines = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_duration:
                logger.debug(f"Using cached RSS headlines for {base_currency}")
                return cached_headlines

        headlines = []

        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)

                if feed.bozo:  # Feed parsing error
                    logger.warning(f"RSS feed error: {feed_url}")
                    continue

                # Filter entries by keywords
                for entry in feed.entries[:50]:  # Check first 50 entries
                    title = entry.get('title', '').lower()

                    # Check if headline mentions the currency
                    if any(keyword in title for keyword in keywords):
                        headlines.append(entry.get('title', ''))

                    if len(headlines) >= limit:
                        break

                if len(headlines) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed {feed_url}: {e}")
                continue

        # Cache results
        self._cache[cache_key] = (datetime.now(), headlines)

        logger.info(
            f"Fetched {len(headlines)} RSS headlines for {base_currency}")
        return headlines[:limit]

    def get_all_headlines(self, limit: int = 30) -> List[str]:
        """
        Get general crypto market headlines (not symbol-specific)

        Args:
            limit: Max headlines to return

        Returns:
            List of headline strings
        """
        cache_key = f"all_{limit}"
        if cache_key in self._cache:
            cached_time, cached_headlines = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_duration:
                return cached_headlines

        headlines = []

        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)

                if feed.bozo:
                    continue

                for entry in feed.entries:
                    headlines.append(entry.get('title', ''))

                    if len(headlines) >= limit:
                        break

                if len(headlines) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed {feed_url}: {e}")
                continue

        self._cache[cache_key] = (datetime.now(), headlines)
        return headlines[:limit]
