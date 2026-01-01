"""
Twitter Sentiment Analysis
Scrapes tweets and analyzes sentiment using FinBERT (AI-enhanced) or VADER (fallback)
"""

import tweepy
import pandas as pd
from textblob import TextBlob

try:
    # preferred import path
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    try:
        # alternate import path (some installations expose this)
        from vaderSentiment import SentimentIntensityAnalyzer
    except Exception:
        # Fallback: lightweight analyzer that maps TextBlob polarity to a VADER-like compound score
        class SentimentIntensityAnalyzer:
            def polarity_scores(self, text: str):
                polarity = TextBlob(text).sentiment.polarity
                # ensure value in [-1, 1] and return as 'compound' for compatibility
                return {'compound': max(-1.0, min(1.0, polarity))}

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger()


class TwitterSentimentAnalyzer:
    """Analyze cryptocurrency sentiment from Twitter with AI enhancement"""

    def __init__(self, config: Dict, use_finbert: bool = True):
        self.config = config
        self.twitter_config = config.get('twitter', {})
        self.sentiment_config = config.get('sentiment', {})
        self.use_finbert = use_finbert

        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()

        # Initialize FinBERT (lazy loading)
        self.finbert = None
        if self.use_finbert:
            try:
                from src.sentiment.finbert_analyzer import FinBERTAnalyzer
                self.finbert = FinBERTAnalyzer(config)
                logger.info("Using FinBERT for enhanced sentiment analysis")
            except Exception as e:
                logger.warning(
                    f"Could not load FinBERT, falling back to VADER: {e}")
                self.use_finbert = False

        # Initialize Twitter API
        self.api = self._initialize_twitter_api()

    def _initialize_twitter_api(self):
        """Initialize Twitter API v2 client"""
        try:
            # Twitter API v2 authentication
            client = tweepy.Client(
                bearer_token=self.twitter_config.get('bearer_token'),
                consumer_key=self.twitter_config.get('api_key'),
                consumer_secret=self.twitter_config.get('api_secret'),
                access_token=self.twitter_config.get('access_token'),
                access_token_secret=self.twitter_config.get(
                    'access_token_secret')
            )

            logger.info("Twitter API initialized successfully")
            return client
        except Exception as e:
            logger.warning(f"Failed to initialize Twitter API: {e}")
            return None

    def fetch_tweets(
        self,
        keywords: List[str] = None,
        max_results: int = 100,
        hours_back: int = 24
    ) -> List[Dict]:
        """
        Fetch recent tweets about cryptocurrency

        Args:
            keywords: List of keywords to search for
            max_results: Maximum number of tweets to fetch
            hours_back: How many hours back to search

        Returns:
            List of tweet dictionaries
        """
        if not self.api:
            logger.warning("Twitter API not initialized")
            return []

        if keywords is None:
            keywords = self.sentiment_config.get(
                'twitter_keywords', ['BTC', 'Bitcoin'])

        tweets = []

        try:
            # Build search query
            query = ' OR '.join(keywords) + ' -is:retweet lang:en'

            # Calculate start time
            start_time = datetime.utcnow() - timedelta(hours=hours_back)

            # Search tweets
            response = self.api.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # API limit per request
                start_time=start_time,
                tweet_fields=['created_at', 'public_metrics', 'text']
            )

            if response.data:
                for tweet in response.data:
                    tweets.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'metrics': tweet.public_metrics if hasattr(tweet, 'public_metrics') else {}
                    })

                logger.info(
                    f"Fetched {len(tweets)} tweets for keywords: {keywords}")
            else:
                logger.warning(f"No tweets found for keywords: {keywords}")

        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")

        return tweets

    def analyze_sentiment_textblob(self, text: str) -> float:
        """
        Analyze sentiment using TextBlob

        Returns:
            Sentiment polarity (-1 to 1)
        """
        try:
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return 0.0

    def analyze_sentiment_vader(self, text: str) -> float:
        """
        Analyze sentiment using VADER

        Returns:
            Compound sentiment score (-1 to 1)
        """
        try:
            scores = self.vader.polarity_scores(text)
            return scores['compound']
        except Exception as e:
            logger.error(f"Error in VADER analysis: {e}")
            return 0.0

    def analyze_tweet(self, tweet_text: str) -> Dict:
        """
        Analyze single tweet with multiple methods

        Returns:
            Dictionary with sentiment scores
        """
        textblob_score = self.analyze_sentiment_textblob(tweet_text)
        vader_score = self.analyze_sentiment_vader(tweet_text)

        # Average the scores
        average_score = (textblob_score + vader_score) / 2

        return {
            'textblob': textblob_score,
            'vader': vader_score,
            'average': average_score,
            'classification': self._classify_sentiment(average_score)
        }

    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment as positive, negative, or neutral"""
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def get_market_sentiment(
        self,
        symbol: str = 'BTC',
        hours_back: int = 24
    ) -> Dict:
        """
        Get overall market sentiment for a cryptocurrency using FinBERT or VADER

        Args:
            symbol: Cryptocurrency symbol
            hours_back: Time window for analysis

        Returns:
            Dictionary with sentiment metrics
        """
        # Fetch tweets
        keywords = [symbol, symbol.upper()]
        if symbol == 'BTC':
            keywords.extend(['Bitcoin', 'bitcoin'])
        elif symbol == 'ETH':
            keywords.extend(['Ethereum', 'ethereum'])

        tweets = self.fetch_tweets(
            keywords=keywords,
            max_results=self.twitter_config.get('max_tweets', 100),
            hours_back=hours_back
        )

        if not tweets:
            logger.warning(f"No tweets found for {symbol}")
            return {
                'sentiment_score': 0.0,
                'classification': 'neutral',
                'tweet_count': 0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'method': 'none'
            }

        # Use FinBERT if available, otherwise VADER
        if self.use_finbert and self.finbert:
            try:
                # Extract tweet texts
                tweet_texts = [tweet['text'] for tweet in tweets]

                # Get aggregate sentiment from FinBERT
                result = self.finbert.get_aggregate_sentiment(tweet_texts)
                result['tweet_count'] = len(tweets)
                result['method'] = 'finbert'

                logger.info(
                    f"Market sentiment for {symbol} (FinBERT): "
                    f"{result['classification']} ({result['sentiment_score']:.3f})"
                )

                return result

            except Exception as e:
                logger.error(
                    f"FinBERT analysis failed, falling back to VADER: {e}")
                # Fall through to VADER analysis

        # Fallback: Use VADER/TextBlob
        sentiments = []
        for tweet in tweets:
            sentiment = self.analyze_tweet(tweet['text'])
            sentiments.append(sentiment['average'])

        # Calculate metrics
        avg_sentiment = sum(sentiments) / len(sentiments)
        positive_count = sum(1 for s in sentiments if s > 0.05)
        negative_count = sum(1 for s in sentiments if s < -0.05)

        result = {
            'sentiment_score': avg_sentiment,
            'classification': self._classify_sentiment(avg_sentiment),
            'tweet_count': len(tweets),
            'positive_ratio': positive_count / len(tweets),
            'negative_ratio': negative_count / len(tweets),
            'timestamp': datetime.now(),
            'method': 'vader'
        }

        logger.info(
            f"Market sentiment for {symbol} (VADER): {result['classification']} ({avg_sentiment:.3f})")

        return result

    def should_trade_based_on_sentiment(self, symbol: str, direction: str) -> bool:
        """
        Check if sentiment supports the trade direction

        Args:
            symbol: Cryptocurrency symbol
            direction: 'long' or 'short'

        Returns:
            True if sentiment supports trade, False otherwise
        """
        if not self.sentiment_config.get('enabled', True):
            return True

        sentiment = self.get_market_sentiment(symbol)
        sentiment_score = sentiment['sentiment_score']
        min_score = self.sentiment_config.get('min_sentiment_score', 0.3)

        if direction == 'long':
            # For long trades, sentiment should be positive
            return sentiment_score >= min_score
        elif direction == 'short':
            # For short trades, sentiment should be negative
            return sentiment_score <= -min_score

        return True
