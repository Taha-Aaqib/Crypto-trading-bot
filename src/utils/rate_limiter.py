"""
Rate Limiter using Token Bucket Algorithm
Prevents Binance API rate limit violations (1200 weight/minute)
"""
import time
import threading
from typing import Optional


class RateLimiter:
    """
    Token bucket rate limiter for API calls

    Binance limits:
    - 1200 weight per minute
    - Each request has different weight (1-50)
    - IP ban if exceeded
    """

    def __init__(self, max_tokens: int = 1200, refill_rate: float = 20.0):
        """
        Initialize rate limiter

        Args:
            max_tokens: Maximum tokens in bucket (default 1200 for Binance)
            refill_rate: Tokens added per second (1200/60 = 20 tokens/sec)
        """
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on time elapsed
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens before making API call

        Args:
            tokens: Number of tokens needed (API weight)
            timeout: Max time to wait in seconds (None = wait forever)

        Returns:
            True if tokens acquired, False if timeout
        """
        deadline = time.time() + timeout if timeout else None

        while True:
            with self.lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate

            # Check timeout
            if deadline and time.time() + wait_time > deadline:
                return False

            # Wait for tokens to refill
            time.sleep(min(wait_time, 0.1))  # Sleep in small increments

    def get_remaining_tokens(self) -> float:
        """Get current token count"""
        with self.lock:
            self._refill()
            return self.tokens

    def reset(self):
        """Reset bucket to full capacity"""
        with self.lock:
            self.tokens = self.max_tokens
            self.last_refill = time.time()


# Global rate limiter instance for Binance
binance_rate_limiter = RateLimiter(
    max_tokens=1200,  # Binance weight limit
    refill_rate=20.0   # 1200 per minute = 20 per second
)
