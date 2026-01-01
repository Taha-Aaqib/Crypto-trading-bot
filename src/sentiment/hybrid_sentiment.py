"""
Hybrid Sentiment Analyzer combining multiple free sources with FinBERT:
- CoinGecko API (35% weight) - market sentiment, news, community data
- RSS Feeds (35% weight) - major news outlets analyzed by FinBERT
- Reddit API (20% weight) - community posts analyzed by FinBERT
- Fear & Greed Index (10% weight) - market-wide sentiment indicator
"""
import requests
import praw
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import os
from src.utils.logger import get_logger
from src.sentiment.finbert_analyzer import FinBERTAnalyzer
from src.sentiment.rss_analyzer import RSSAnalyzer
from src.sentiment.coingecko_analyzer import CoinGeckoAnalyzer

logger = get_logger()


class HybridSentimentAnalyzer:
    """
    Combines 4 free sentiment sources with FinBERT analysis

    Sources:
    1. CoinGecko (35%): Market sentiment + news → FinBERT analysis
    2. RSS Feeds (35%): CoinDesk, CoinTelegraph, etc. → FinBERT analysis
    3. Reddit (20%): Community posts → FinBERT analysis
    4. Fear & Greed Index (10%): Direct market sentiment score
    """

    def __init__(self, config: Dict):
        self.config = config

        # Initialize FinBERT for text analysis
        self.finbert = FinBERTAnalyzer(config)

        # Initialize RSS analyzer
        self.rss_analyzer = RSSAnalyzer(config)

        # Initialize CoinGecko analyzer (no API key needed!)
        self.coingecko = CoinGeckoAnalyzer()

        # Reddit API (free)
        self.reddit = None
        if os.getenv('REDDIT_CLIENT_ID'):
            try:
                self.reddit = praw.Reddit(
                    client_id=os.getenv('REDDIT_CLIENT_ID'),
                    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                    user_agent='SmartTradingBot/1.0'
                )
                logger.info("Reddit API initialized successfully")
            except Exception as e:
                logger.warning(f"Reddit API setup failed: {e}")

        # Weights for each source
        self.weights = {
            'coingecko': 0.35,
            'rss': 0.35,
            'reddit': 0.20,
            'fear_greed': 0.10
        }

    def get_sentiment(self, symbol: str) -> Dict:
        """
        Get combined sentiment score for a symbol

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Dict with sentiment score (-1 to 1) and breakdown
        """
        # Extract base currency
        base = symbol.split('/')[0]

        # Get sentiment from each source
        coingecko_score = self._get_coingecko_sentiment(base)
        rss_score = self._get_rss_sentiment(base)
        reddit_score = self._get_reddit_sentiment(base)
        fear_greed_score = self._get_fear_greed_index()

        # Weighted average
        total_weight = 0
        weighted_score = 0

        if coingecko_score is not None:
            weighted_score += coingecko_score * self.weights['coingecko']
            total_weight += self.weights['coingecko']

        if rss_score is not None:
            weighted_score += rss_score * self.weights['rss']
            total_weight += self.weights['rss']

        if reddit_score is not None:
            weighted_score += reddit_score * self.weights['reddit']
            total_weight += self.weights['reddit']

        if fear_greed_score is not None:
            weighted_score += fear_greed_score * self.weights['fear_greed']
            total_weight += self.weights['fear_greed']

        # Normalize by total weight used
        final_score = weighted_score / total_weight if total_weight > 0 else 0

        return {
            'score': final_score,
            'breakdown': {
                'coingecko': coingecko_score,
                'rss': rss_score,
                'reddit': reddit_score,
                'fear_greed': fear_greed_score
            },
            'timestamp': datetime.now()
        }

    def _get_coingecko_sentiment(self, currency: str) -> Optional[float]:
        """
        Get sentiment from CoinGecko API + analyze news with FinBERT

        Returns:
            Sentiment score -1 to 1, or None if failed
        """
        try:
            # Get market sentiment metrics
            sentiment_score = self.coingecko.calculate_sentiment_score(
                currency)

            if sentiment_score is None:
                return None

            # Get status updates/news for FinBERT analysis
            news_items = self.coingecko.get_status_updates(currency)

            if news_items:
                # Analyze news with FinBERT
                finbert_result = self.finbert.analyze_batch(news_items)
                finbert_score = finbert_result.get('compound', 0)

                # Combine market metrics (70%) + news sentiment (30%)
                combined_score = sentiment_score * 0.7 + finbert_score * 0.3
                logger.info(f"CoinGecko sentiment for {currency}: {combined_score:.3f} "
                            f"(metrics: {sentiment_score:.3f}, news: {finbert_score:.3f})")
                return combined_score
            else:
                # No news available, use market metrics only
                logger.info(
                    f"CoinGecko sentiment for {currency}: {sentiment_score:.3f} (metrics only)")
                return sentiment_score

        except Exception as e:
            logger.warning(f"CoinGecko sentiment error: {e}")
            return None

    def _get_rss_sentiment(self, currency: str) -> Optional[float]:
        """
        Get sentiment from RSS feeds using FinBERT

        Returns:
            Sentiment score -1 to 1, or None if failed
        """
        try:
            # Fetch headlines from RSS feeds
            headlines = self.rss_analyzer.get_headlines(
                f"{currency}/USDT", limit=20)

            if not headlines:
                return None

            # Analyze with FinBERT
            sentiment_result = self.finbert.get_aggregate_sentiment(headlines)
            score = sentiment_result.get('sentiment_score', 0)

            logger.info(f"RSS sentiment for {currency}: {score:.3f}")
            return score

        except Exception as e:
            logger.warning(f"RSS sentiment error: {e}")

        return None

    def _get_reddit_sentiment(self, currency: str) -> Optional[float]:
        """
        Get sentiment from Reddit using FinBERT

        Returns:
            Sentiment score -1 to 1, or None if failed
        """
        if not self.reddit:
            return None

        try:
            # Map currency to subreddit
            subreddit_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'USDT': 'CryptoCurrency'
            }

            subreddit_name = subreddit_map.get(currency, 'CryptoCurrency')
            subreddit = self.reddit.subreddit(subreddit_name)

            # Collect post titles mentioning the currency
            titles = []

            for post in subreddit.hot(limit=100):
                if currency.lower() in post.title.lower():
                    titles.append(post.title)

                if len(titles) >= 20:
                    break

            if not titles:
                return None

            # Analyze with FinBERT
            sentiment_result = self.finbert.get_aggregate_sentiment(titles)
            score = sentiment_result.get('sentiment_score', 0)

            logger.info(f"Reddit sentiment for {currency}: {score:.3f}")
            return score

        except Exception as e:
            logger.warning(f"Reddit API error: {e}")

        return None

    def _get_fear_greed_index(self) -> Optional[float]:
        """
        Get crypto Fear & Greed Index from Alternative.me

        Returns:
            Sentiment score -1 to 1, or None if failed
        """
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                value = int(data['data'][0]['value'])

                # Convert 0-100 scale to -1 to 1
                # 0 = Extreme Fear = -1
                # 50 = Neutral = 0
                # 100 = Extreme Greed = 1
                score = (value - 50) / 50

                logger.info(f"Fear & Greed Index: {value} -> {score:.3f}")
                return score

        except Exception as e:
            logger.warning(f"Fear & Greed Index error: {e}")

        return None
