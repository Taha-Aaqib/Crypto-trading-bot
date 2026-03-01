"""
Sentiment Filter
Combines sentiment from multiple sources with FinBERT for enhanced accuracy
"""

from typing import Dict
from datetime import datetime
from src.sentiment.hybrid_sentiment import HybridSentimentAnalyzer
from src.utils.logger import get_logger

logger = get_logger()


class SentimentFilter:
    """Filter trades based on 3-source sentiment analysis with FinBERT"""

    def __init__(self, config: Dict):
        self.config = config
        self.sentiment_config = config.get('sentiment', {})
        self.enabled = self.sentiment_config.get('enabled', True)
        self.min_sentiment_score = self.sentiment_config.get(
            'min_sentiment_score', 0.3)

        # Initialize hybrid sentiment analyzer (4 sources)
        self.hybrid_analyzer = HybridSentimentAnalyzer(config)

        logger.info(
            "Sentiment filter initialized with 3 sources (RSS, CoinGecko, Fear & Greed)")

    def get_combined_sentiment(self, symbol: str) -> Dict:
        """
        Get combined sentiment from all 3 sources (RSS, CoinGecko, Fear & Greed)

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            Dictionary with combined sentiment metrics
        """
        if not self.enabled:
            return {
                'sentiment_score': 0.0,
                'classification': 'neutral',
                'sources': {},
                'timestamp': datetime.now()
            }

        # Get sentiment from HybridAnalyzer (4 sources)
        sentiment_data = self.hybrid_analyzer.get_sentiment(symbol)

        combined_score = sentiment_data.get('score', 0.0)
        breakdown = sentiment_data.get('breakdown', {})

        result = {
            'sentiment_score': combined_score,
            'classification': self._classify_sentiment(combined_score),
            'breakdown': breakdown,
            'sources_available': sentiment_data.get('sources_available', 0),
            'sources_total': sentiment_data.get('sources_total', 3),
            'timestamp': sentiment_data.get('timestamp', datetime.now())
        }

        logger.info(
            f"Combined sentiment for {symbol}: {result['classification']} ({combined_score:.3f})")

        return result

    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment as positive, negative, or neutral"""
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def filter_trade(self, symbol: str, direction: str) -> bool:
        """
        Filter trade based on sentiment — SOFT FILTER.

        Only BLOCKS a trade when sentiment STRONGLY contradicts the direction.
        Weak or neutral sentiment should NOT block high-confluence trades.

        Args:
            symbol: Cryptocurrency symbol
            direction: 'long' or 'short'

        Returns:
            True if trade should proceed, False only on strong contradiction
        """
        if not self.enabled:
            return True

        sentiment = self.get_combined_sentiment(symbol)
        sentiment_score = sentiment['sentiment_score']
        sources_available = sentiment.get('sources_available', 0)

        # If few sources available, don't trust sentiment enough to block
        if sources_available < 2:
            logger.info(f"Sentiment filter: only {sources_available} sources — allowing trade (low confidence)")
            return True

        # Strong contradiction threshold (stricter than the old min_sentiment_score)
        # Only block when sentiment is clearly opposing the trade direction
        strong_threshold = max(self.min_sentiment_score, 0.3)

        if direction == 'long':
            # Only block long if sentiment is STRONGLY negative
            if sentiment_score <= -strong_threshold:
                logger.info(
                    f"Sentiment filter BLOCKED LONG for {symbol}: score={sentiment_score:.3f} "
                    f"(strongly negative, threshold: -{strong_threshold})")
                return False
            return True

        elif direction == 'short':
            # Only block short if sentiment is STRONGLY positive
            if sentiment_score >= strong_threshold:
                logger.info(
                    f"Sentiment filter BLOCKED SHORT for {symbol}: score={sentiment_score:.3f} "
                    f"(strongly positive, threshold: +{strong_threshold})")
                return False
            return True

        return True
