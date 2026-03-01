"""
Hybrid Sentiment Analyzer combining multiple free sources with FinBERT:
- RSS Feeds (45% weight) - major news outlets analyzed by FinBERT
- CoinGecko API (35% weight) - market metrics (community, dev, momentum)
- Fear & Greed Index (20% weight) - market-wide sentiment indicator
"""
import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from src.utils.logger import get_logger
from src.sentiment.finbert_analyzer import FinBERTAnalyzer
from src.sentiment.rss_analyzer import RSSAnalyzer
from src.sentiment.coingecko_analyzer import CoinGeckoAnalyzer

logger = get_logger()


class HybridSentimentAnalyzer:
    """
    Combines 3 free sentiment sources with FinBERT analysis

    Sources:
    1. RSS Feeds (45%): CoinDesk, CoinTelegraph, etc. → FinBERT analysis
    2. CoinGecko (35%): Market metrics (community, dev activity, momentum)
    3. Fear & Greed Index (20%): Direct market sentiment score
    """

    def __init__(self, config: Dict):
        self.config = config

        # Initialize FinBERT for text analysis
        self.finbert = FinBERTAnalyzer(config)

        # Initialize RSS analyzer
        self.rss_analyzer = RSSAnalyzer(config)

        # Initialize CoinGecko analyzer (no API key needed!)
        self.coingecko = CoinGeckoAnalyzer()

        # Weights for each source
        self.weights = {
            'rss': 0.45,
            'coingecko': 0.35,
            'fear_greed': 0.20
        }

        # Sentiment cache with staleness tracking
        self._cache = {}           # {symbol: {'score': ..., 'breakdown': ..., 'timestamp': ...}}
        self._cache_ttl = 300      # 5 minutes — sentiment doesn't change second by second
        self._source_failures = {} # {source_name: consecutive_failure_count}

    def get_sentiment(self, symbol: str) -> Dict:
        """
        Get combined sentiment score for a symbol.
        Uses a 5-minute cache to avoid hammering APIs and getting stale re-fetches.
        Tracks source failures for graceful degradation.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Dict with sentiment score (-1 to 1), breakdown, sources_available count, and staleness
        """
        # Check cache first
        now = datetime.now()
        if symbol in self._cache:
            cached = self._cache[symbol]
            age_seconds = (now - cached['timestamp']).total_seconds()
            if age_seconds < self._cache_ttl:
                logger.debug(f"Sentiment cache hit for {symbol} (age: {age_seconds:.0f}s)")
                return cached

        # Extract base currency
        base = symbol.split('/')[0]

        # Get sentiment from each source (with failure tracking)
        rss_score = self._safe_fetch('rss', lambda: self._get_rss_sentiment(base))
        coingecko_score = self._safe_fetch('coingecko', lambda: self._get_coingecko_sentiment(base))
        fear_greed_score = self._safe_fetch('fear_greed', lambda: self._get_fear_greed_index())

        # Weighted average (only from sources that returned data)
        total_weight = 0
        weighted_score = 0
        sources_available = 0

        for source_name, score in [('rss', rss_score), ('coingecko', coingecko_score),
                                    ('fear_greed', fear_greed_score)]:
            if score is not None:
                weighted_score += score * self.weights[source_name]
                total_weight += self.weights[source_name]
                sources_available += 1

        # Normalize by total weight used
        final_score = weighted_score / total_weight if total_weight > 0 else 0

        # Reduce confidence when few sources are available
        # If only 1 source, dampen score toward neutral (less reliable)
        if sources_available == 1:
            final_score *= 0.5  # Half-weight single-source sentiment
            logger.warning(f"Only 1 sentiment source available for {symbol} — dampening score")
        elif sources_available == 0:
            final_score = 0.0
            logger.warning(f"No sentiment sources available for {symbol} — returning neutral")

        result = {
            'score': final_score,
            'breakdown': {
                'rss': rss_score,
                'coingecko': coingecko_score,
                'fear_greed': fear_greed_score
            },
            'sources_available': sources_available,
            'sources_total': 3,
            'timestamp': now
        }

        # Update cache
        self._cache[symbol] = result

        return result

    def _safe_fetch(self, source_name: str, fetch_fn) -> Optional[float]:
        """
        Safely fetch sentiment from a source with failure tracking.
        After 3 consecutive failures, back off (skip) to avoid slowing down the bot.
        """
        consecutive_failures = self._source_failures.get(source_name, 0)

        # After 3 consecutive failures, skip for the next cycle (reset after success)
        if consecutive_failures >= 3:
            logger.debug(f"Skipping {source_name} sentiment (failed {consecutive_failures}x consecutively)")
            return None

        try:
            result = fetch_fn()
            if result is not None:
                self._source_failures[source_name] = 0  # Reset on success
            return result
        except Exception as e:
            self._source_failures[source_name] = consecutive_failures + 1
            logger.warning(f"Sentiment source {source_name} failed ({consecutive_failures + 1}x): {e}")
            return None

    def _get_coingecko_sentiment(self, currency: str) -> Optional[float]:
        """
        Get sentiment from CoinGecko market metrics (community, dev, momentum).
        Uses the working /coins/{id} endpoint — NOT the deprecated status_updates.

        Returns:
            Sentiment score -1 to 1, or None if failed
        """
        try:
            sentiment_score = self.coingecko.calculate_sentiment_score(currency)
            if sentiment_score is not None:
                logger.info(f"CoinGecko sentiment for {currency}: {sentiment_score:.3f}")
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
