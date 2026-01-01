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
    """Filter trades based on 4-source sentiment analysis with FinBERT"""

    def __init__(self, config: Dict):
        self.config = config
        self.sentiment_config = config.get('sentiment', {})
        self.enabled = self.sentiment_config.get('enabled', True)
        self.min_sentiment_score = self.sentiment_config.get(
            'min_sentiment_score', 0.3)

        # Initialize hybrid sentiment analyzer (4 sources)
        self.hybrid_analyzer = HybridSentimentAnalyzer(config)

        logger.info(
            "Sentiment filter initialized with 4 sources (CryptoPanic, RSS, Reddit, Fear & Greed)")

    def get_combined_sentiment(self, symbol: str) -> Dict:
        """
        Get combined sentiment from all 4 sources

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

        # Combine sentiments (currently only Twitter)
# Get sentiment from HybridAnalyzer (4 sources)
        sentiment_data = self.hybrid_analyzer.get_sentiment(symbol)

        combined_score = sentiment_data.get('score', 0.0)
        breakdown = sentiment_data.get('breakdown', {})

        result = {
            'sentiment_score': combined_score,
            'classification': self._classify_sentiment(combined_score),
            'breakdown': breakdown,
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
        Filter trade based on sentiment

        Args:
            symbol: Cryptocurrency symbol
            direction: 'long' or 'short'

        Returns:
            True if trade should proceed, False otherwise
        """
        if not self.enabled:
            return True

        sentiment = self.get_combined_sentiment(symbol)
        sentiment_score = sentiment['sentiment_score']

        if direction == 'long':
            # For long trades, sentiment should be positive
            allowed = sentiment_score >= self.min_sentiment_score
            if not allowed:
                logger.info(
                    f"Sentiment filter blocked LONG trade for {symbol}: score={sentiment_score:.3f}")
            return allowed

        elif direction == 'short':
            # For short trades, sentiment should be negative
            allowed = sentiment_score <= -self.min_sentiment_score
            if not allowed:
                logger.info(
                    f"Sentiment filter blocked SHORT trade for {symbol}: score={sentiment_score:.3f}")
            return allowed

        return True
