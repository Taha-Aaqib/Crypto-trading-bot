"""
Tests for the sentiment filter system.
Verifies soft-filter logic: only blocks on strong contradiction.
"""
import pytest
import yaml
from unittest.mock import patch, MagicMock
from datetime import datetime


@pytest.fixture
def config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def _mock_sentiment(score: float, sources: int = 3):
    """Create a mock sentiment result."""
    return {
        'score': score,
        'breakdown': {'rss': score, 'coingecko': score, 'fear_greed': score},
        'sources_available': sources,
        'sources_total': 3,
        'timestamp': datetime.now()
    }


class TestSentimentFilter:

    @patch('src.sentiment.sentiment_filter.HybridSentimentAnalyzer')
    def test_allows_long_on_positive_sentiment(self, mock_hybrid_cls, config):
        """Positive sentiment should NOT block a long trade."""
        mock_instance = MagicMock()
        mock_instance.get_sentiment.return_value = _mock_sentiment(0.5, 3)
        mock_hybrid_cls.return_value = mock_instance

        from src.sentiment.sentiment_filter import SentimentFilter
        sf = SentimentFilter(config)
        sf.hybrid_analyzer = mock_instance
        assert sf.filter_trade('BTC/USDT', 'long') is True

    @patch('src.sentiment.sentiment_filter.HybridSentimentAnalyzer')
    def test_blocks_long_on_strong_negative(self, mock_hybrid_cls, config):
        """Strongly negative sentiment SHOULD block a long trade."""
        mock_instance = MagicMock()
        mock_instance.get_sentiment.return_value = _mock_sentiment(-0.4, 3)
        mock_hybrid_cls.return_value = mock_instance

        from src.sentiment.sentiment_filter import SentimentFilter
        sf = SentimentFilter(config)
        sf.hybrid_analyzer = mock_instance
        assert sf.filter_trade('BTC/USDT', 'long') is False

    @patch('src.sentiment.sentiment_filter.HybridSentimentAnalyzer')
    def test_allows_long_on_mild_negative(self, mock_hybrid_cls, config):
        """Mildly negative sentiment should NOT block a long (soft filter)."""
        mock_instance = MagicMock()
        mock_instance.get_sentiment.return_value = _mock_sentiment(-0.15, 3)
        mock_hybrid_cls.return_value = mock_instance

        from src.sentiment.sentiment_filter import SentimentFilter
        sf = SentimentFilter(config)
        sf.hybrid_analyzer = mock_instance
        assert sf.filter_trade('BTC/USDT', 'long') is True

    @patch('src.sentiment.sentiment_filter.HybridSentimentAnalyzer')
    def test_blocks_short_on_strong_positive(self, mock_hybrid_cls, config):
        """Strongly positive sentiment SHOULD block a short trade."""
        mock_instance = MagicMock()
        mock_instance.get_sentiment.return_value = _mock_sentiment(0.4, 3)
        mock_hybrid_cls.return_value = mock_instance

        from src.sentiment.sentiment_filter import SentimentFilter
        sf = SentimentFilter(config)
        sf.hybrid_analyzer = mock_instance
        assert sf.filter_trade('BTC/USDT', 'short') is False

    @patch('src.sentiment.sentiment_filter.HybridSentimentAnalyzer')
    def test_allows_trade_on_low_source_count(self, mock_hybrid_cls, config):
        """If fewer than 2 sources, sentiment confidence is too low to block."""
        mock_instance = MagicMock()
        mock_instance.get_sentiment.return_value = _mock_sentiment(-0.5, 1)
        mock_hybrid_cls.return_value = mock_instance

        from src.sentiment.sentiment_filter import SentimentFilter
        sf = SentimentFilter(config)
        sf.hybrid_analyzer = mock_instance
        # Even strongly negative, only 1 source → should allow
        assert sf.filter_trade('BTC/USDT', 'long') is True

    @patch('src.sentiment.sentiment_filter.HybridSentimentAnalyzer')
    def test_disabled_filter_allows_all(self, mock_hybrid_cls, config):
        """Disabled sentiment filter should never block trades."""
        config['sentiment']['enabled'] = False
        mock_hybrid_cls.return_value = MagicMock()

        from src.sentiment.sentiment_filter import SentimentFilter
        sf = SentimentFilter(config)
        assert sf.filter_trade('BTC/USDT', 'long') is True
        assert sf.filter_trade('BTC/USDT', 'short') is True

    @patch('src.sentiment.sentiment_filter.HybridSentimentAnalyzer')
    def test_neutral_sentiment_allows_all(self, mock_hybrid_cls, config):
        """Neutral sentiment should not block any direction."""
        mock_instance = MagicMock()
        mock_instance.get_sentiment.return_value = _mock_sentiment(0.0, 3)
        mock_hybrid_cls.return_value = mock_instance

        from src.sentiment.sentiment_filter import SentimentFilter
        sf = SentimentFilter(config)
        sf.hybrid_analyzer = mock_instance
        assert sf.filter_trade('BTC/USDT', 'long') is True
        assert sf.filter_trade('BTC/USDT', 'short') is True


class TestSentimentWeights:
    """Verify the sentiment weight configuration is consistent."""

    def test_weights_sum_to_one(self, config):
        """Sentiment source weights must sum to 1.0."""
        sources = config['sentiment']['sources']
        total = sum(s['weight'] for s in sources.values() if s.get('enabled'))
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, not 1.0"

    def test_no_reddit_source(self, config):
        """Reddit should have been removed from config."""
        sources = config['sentiment']['sources']
        assert 'reddit' not in sources, "Reddit should be removed from sentiment sources"

    def test_three_sources_only(self, config):
        """Should have exactly 3 sentiment sources: RSS, CoinGecko, Fear&Greed."""
        sources = config['sentiment']['sources']
        expected = {'rss', 'coingecko', 'fear_greed'}
        assert set(sources.keys()) == expected, f"Expected {expected}, got {set(sources.keys())}"
