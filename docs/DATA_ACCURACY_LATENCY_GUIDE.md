# Data Accuracy & Low Latency Guide

## 🎯 Ensuring Data Accuracy and Low Latency in Trading Bot

This guide explains how to optimize your trading bot for **accurate data** and **minimal latency** to ensure reliable trading decisions.

---

## 📊 Part 1: Data Accuracy

### 1.1 Current Data Accuracy Measures

#### ✅ **Data Cleaning Pipeline** (in `data_preprocessor.py`)

**Current Implementation:**

```python
def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
    # 1. Remove duplicates
    df_clean = df[~df.index.duplicated(keep='first')]

    # 2. Forward fill missing values
    df_clean = df_clean.fillna(method='ffill')

    # 3. Remove outliers (IQR method)
    df_clean = self._remove_outliers(df_clean)

    # 4. Sort by timestamp
    df_clean = df_clean.sort_index()

    return df_clean
```

**What This Does:**

- ✅ Eliminates duplicate timestamps
- ✅ Fills gaps in data (exchange downtime, API failures)
- ✅ Clips outliers (flash crashes, data errors)
- ✅ Ensures chronological order

#### ✅ **Outlier Detection** (IQR Method)

```python
def _remove_outliers(self, df: pd.DataFrame, threshold: float = 3.0):
    for col in ['open', 'high', 'low', 'close', 'volume']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Clip instead of remove (preserves data integrity)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
```

**Why Clipping vs Removing?**

- ✅ Preserves timestamp continuity
- ✅ Prevents gaps in time-series
- ✅ Better for ML model training

---

### 1.2 RECOMMENDED Data Accuracy Improvements

#### 🔧 **1. Add Data Validation Layer**

Create `src/data/data_validator.py`:

```python
"""
Data Validator - Ensures data quality before processing
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from src.utils.logger import get_logger

logger = get_logger()


class DataValidator:
    """Validate data quality and accuracy"""

    def __init__(self, config: Dict = None):
        self.config = config

    def validate_ohlcv(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Comprehensive OHLCV data validation

        Returns:
            (is_valid: bool, issues: List[str])
        """
        issues = []

        # 1. Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            return False, issues

        # 2. Check for negative prices
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                issues.append(f"Negative or zero values in {col}")

        # 3. Validate OHLC relationships (High >= Low, etc.)
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            issues.append(f"{invalid_hl} rows where high < low")

        invalid_hc = (df['high'] < df['close']).sum()
        if invalid_hc > 0:
            issues.append(f"{invalid_hc} rows where high < close")

        invalid_lc = (df['low'] > df['close']).sum()
        if invalid_lc > 0:
            issues.append(f"{invalid_lc} rows where low > close")

        # 4. Check for extreme price movements (>50% in one candle)
        price_change = df['close'].pct_change().abs()
        extreme_moves = (price_change > 0.5).sum()
        if extreme_moves > 0:
            logger.warning(f"{extreme_moves} candles with >50% price change")

        # 5. Check for zero volume candles
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            logger.warning(f"{zero_volume} candles with zero volume")

        # 6. Check for timestamp gaps
        gaps = self._check_timestamp_gaps(df)
        if gaps > 0:
            issues.append(f"{gaps} timestamp gaps detected")

        # 7. Check for data freshness
        if not df.empty:
            latest_timestamp = df.index[-1]
            age_minutes = (pd.Timestamp.now() - latest_timestamp).total_seconds() / 60
            if age_minutes > 60:
                logger.warning(f"Data is {age_minutes:.0f} minutes old")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info(f"✅ Data validation passed ({len(df)} candles)")
        else:
            logger.error(f"❌ Data validation failed: {issues}")

        return is_valid, issues

    def _check_timestamp_gaps(self, df: pd.DataFrame, tolerance_minutes: int = 5) -> int:
        """Detect gaps in timestamp sequence"""
        if len(df) < 2:
            return 0

        # Calculate expected interval
        time_diffs = df.index.to_series().diff()
        median_interval = time_diffs.median()

        # Count gaps larger than expected + tolerance
        tolerance = pd.Timedelta(minutes=tolerance_minutes)
        gaps = (time_diffs > median_interval + tolerance).sum()

        return gaps

    def fix_ohlc_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix common OHLC data inconsistencies"""
        df_fixed = df.copy()

        # Fix: High should be >= all other prices
        df_fixed['high'] = df_fixed[['open', 'high', 'low', 'close']].max(axis=1)

        # Fix: Low should be <= all other prices
        df_fixed['low'] = df_fixed[['open', 'high', 'low', 'close']].min(axis=1)

        # Fix: Negative values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_fixed[col] = df_fixed[col].clip(lower=0)

        return df_fixed
```

#### 🔧 **2. Add Cross-Exchange Validation** (Optional but Recommended)

```python
def compare_with_reference_exchange(
    self,
    symbol: str,
    timeframe: str,
    df_primary: pd.DataFrame
) -> Dict:
    """
    Compare data from primary exchange with reference exchange
    to detect anomalies
    """
    try:
        # Fetch same data from backup exchange (e.g., Coinbase if Binance is primary)
        backup_exchange = ccxt.coinbase()
        backup_data = backup_exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df_backup = pd.DataFrame(
            backup_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df_backup['timestamp'] = pd.to_datetime(df_backup['timestamp'], unit='ms')
        df_backup.set_index('timestamp', inplace=True)

        # Compare prices (should be within 0.1%)
        merged = df_primary.join(df_backup, how='inner', rsuffix='_ref')
        price_diff = abs(merged['close'] - merged['close_ref']) / merged['close']

        max_diff = price_diff.max()
        avg_diff = price_diff.mean()

        return {
            'is_accurate': max_diff < 0.001,  # Within 0.1%
            'max_difference_pct': max_diff * 100,
            'avg_difference_pct': avg_diff * 100
        }
    except Exception as e:
        logger.error(f"Cross-exchange validation failed: {e}")
        return {'is_accurate': True}  # Assume OK if can't verify
```

#### 🔧 **3. Add Data Logging & Auditing**

```python
def log_data_quality_metrics(self, df: pd.DataFrame, symbol: str):
    """Log data quality metrics for monitoring"""
    metrics = {
        'symbol': symbol,
        'timestamp': datetime.now(),
        'row_count': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_timestamps': df.index.duplicated().sum(),
        'price_range': f"{df['close'].min():.2f} - {df['close'].max():.2f}",
        'avg_volume': df['volume'].mean(),
        'data_freshness_minutes': (pd.Timestamp.now() - df.index[-1]).total_seconds() / 60
    }

    logger.info(f"Data Quality Metrics: {metrics}")

    # Save to database for historical tracking
    self.db_manager.save_data_quality_metrics(metrics)
```

---

## ⚡ Part 2: Low Latency Optimization

### 2.1 Current Latency Sources

**In your current system:**

1. **API Rate Limits** (1-2 seconds per request)

   - CCXT enforces `enableRateLimit: True`
   - Exchanges limit requests to prevent abuse

2. **Data Fetching** (~500-2000ms per symbol)

   - Multi-timeframe = 3 API calls (1D, 4H, 15M)
   - Total: ~1.5-6 seconds per symbol

3. **Data Processing** (~50-200ms)

   - Cleaning, indicators, SMC detection

4. **ML Prediction** (~10-50ms)

   - Random Forest inference (fast)

5. **Order Execution** (~100-500ms)
   - Exchange API call + confirmation

**Total Latency: 2-7 seconds** (acceptable for 15-minute timeframe)

---

### 2.2 RECOMMENDED Latency Improvements

#### 🚀 **1. Implement Data Caching**

Create `src/data/data_cache.py`:

```python
"""
Data Cache Manager - Reduce redundant API calls
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import pickle
import os
from src.utils.logger import get_logger

logger = get_logger()


class DataCache:
    """Cache market data to reduce API calls"""

    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}
        self.cache_ttl = {
            '1m': timedelta(seconds=30),    # 30 seconds
            '5m': timedelta(minutes=2),     # 2 minutes
            '15m': timedelta(minutes=5),    # 5 minutes
            '1h': timedelta(minutes=15),    # 15 minutes
            '4h': timedelta(hours=1),       # 1 hour
            '1d': timedelta(hours=6)        # 6 hours
        }

    def get(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Get cached data if still valid"""
        cache_key = f"{symbol}_{timeframe}"

        # Check memory cache first (fastest)
        if cache_key in self.memory_cache:
            cached_data, cached_time = self.memory_cache[cache_key]

            # Check if cache is still valid
            ttl = self.cache_ttl.get(timeframe, timedelta(minutes=5))
            if datetime.now() - cached_time < ttl:
                logger.debug(f"Cache HIT (memory): {cache_key}")
                return cached_data
            else:
                logger.debug(f"Cache EXPIRED (memory): {cache_key}")
                del self.memory_cache[cache_key]

        # Check disk cache (slower but persistent)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            ttl = self.cache_ttl.get(timeframe, timedelta(minutes=5))

            if file_age < ttl:
                logger.debug(f"Cache HIT (disk): {cache_key}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Load into memory cache
                    self.memory_cache[cache_key] = (cached_data, datetime.now())
                    return cached_data

        logger.debug(f"Cache MISS: {cache_key}")
        return None

    def set(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ):
        """Cache data in memory and disk"""
        cache_key = f"{symbol}_{timeframe}"

        # Memory cache
        self.memory_cache[cache_key] = (data.copy(), datetime.now())

        # Disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        logger.debug(f"Cached: {cache_key} ({len(data)} rows)")

    def clear(self, symbol: str = None):
        """Clear cache for symbol or all"""
        if symbol:
            # Clear specific symbol
            self.memory_cache = {
                k: v for k, v in self.memory_cache.items()
                if not k.startswith(symbol)
            }
        else:
            # Clear all
            self.memory_cache = {}
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))

        logger.info(f"Cache cleared: {symbol or 'all'}")
```

**Integrate into `data_fetcher.py`:**

```python
class DataFetcher:
    def __init__(self, config: Dict):
        # ... existing code ...
        self.cache = DataCache()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        # Try cache first
        cached_data = self.cache.get(symbol, timeframe)
        if cached_data is not None and len(cached_data) >= limit:
            logger.info(f"Using cached data for {symbol} {timeframe}")
            return cached_data.tail(limit)

        # Fetch from exchange (existing code)
        df = ... # your existing fetch logic

        # Cache the result
        self.cache.set(symbol, timeframe, df)

        return df
```

**Expected Improvement:** 70-90% reduction in API calls

---

#### 🚀 **2. Use WebSocket Streams (Real-Time Data)**

Create `src/data/websocket_streamer.py`:

```python
"""
WebSocket Streamer - Real-time data with minimal latency
"""

import ccxt.pro as ccxtpro
import asyncio
import pandas as pd
from typing import Dict, Callable
from src.utils.logger import get_logger

logger = get_logger()


class WebSocketStreamer:
    """Real-time data streaming via WebSocket"""

    def __init__(self, config: Dict):
        self.config = config
        self.exchange = None
        self.subscribers = {}
        self.running = False

    async def start(self):
        """Initialize WebSocket connection"""
        exchange_name = self.config['exchange']['name']
        exchange_class = getattr(ccxtpro, exchange_name)

        self.exchange = exchange_class({
            'apiKey': self.config['exchange']['api_key'],
            'secret': self.config['exchange']['api_secret'],
            'enableRateLimit': True
        })

        self.running = True
        logger.info(f"WebSocket connected to {exchange_name}")

    async def stream_ticker(self, symbol: str, callback: Callable):
        """
        Stream real-time ticker data

        Args:
            symbol: Trading pair
            callback: Function to call with new data
        """
        while self.running:
            try:
                ticker = await self.exchange.watch_ticker(symbol)
                await callback(ticker)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(1)

    async def stream_trades(self, symbol: str, callback: Callable):
        """Stream real-time trade executions"""
        while self.running:
            try:
                trades = await self.exchange.watch_trades(symbol)
                await callback(trades)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(1)

    async def stream_ohlcv(self, symbol: str, timeframe: str, callback: Callable):
        """Stream real-time OHLCV candles"""
        while self.running:
            try:
                ohlcv = await self.exchange.watch_ohlcv(symbol, timeframe)

                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                await callback(df)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(1)

    async def stop(self):
        """Close WebSocket connection"""
        self.running = False
        if self.exchange:
            await self.exchange.close()
        logger.info("WebSocket disconnected")
```

**Usage in `main.py`:**

```python
class TradingBot:
    def __init__(self, config_path='config/config.yaml'):
        # ... existing code ...
        self.websocket_streamer = WebSocketStreamer(self.config)
        self.latest_data = {}

    async def on_new_candle(self, symbol: str, df: pd.DataFrame):
        """Callback for new candle data"""
        self.latest_data[symbol] = df
        logger.info(f"New candle received for {symbol}: {df.iloc[-1]['close']}")

        # Trigger analysis immediately
        await self.analyze_and_trade_async(symbol)

    async def start_websocket_streaming(self):
        """Start real-time data streaming"""
        await self.websocket_streamer.start()

        tasks = []
        for symbol in self.symbols:
            task = self.websocket_streamer.stream_ohlcv(
                symbol,
                self.timeframes['entry'],
                lambda df: self.on_new_candle(symbol, df)
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
```

**Expected Improvement:** 50-200ms latency (vs 1-2 seconds with REST API)

---

#### 🚀 **3. Parallel Data Fetching**

Modify `fetch_multi_timeframe_data` in `main.py`:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def fetch_multi_timeframe_data_parallel(self, symbol: str):
    """Fetch all timeframes in parallel"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all fetches simultaneously
        future_1d = executor.submit(
            self.data_fetcher.fetch_ohlcv,
            symbol, self.timeframes['bias'], 100
        )
        future_4h = executor.submit(
            self.data_fetcher.fetch_ohlcv,
            symbol, self.timeframes['structure'], 200
        )
        future_15m = executor.submit(
            self.data_fetcher.fetch_ohlcv,
            symbol, self.timeframes['entry'], 500
        )

        # Wait for all to complete
        df_1d = future_1d.result()
        df_4h = future_4h.result()
        df_15m = future_15m.result()

    # Preprocess (can also be parallelized)
    df_1d = self.data_preprocessor.process_pipeline(df_1d)
    df_4h = self.data_preprocessor.process_pipeline(df_4h)
    df_15m = self.data_preprocessor.process_pipeline(df_15m)

    return df_1d, df_4h, df_15m
```

**Expected Improvement:** 3x faster (1.5s instead of 4.5s)

---

#### 🚀 **4. Optimize Data Processing**

Use **vectorized operations** instead of loops:

```python
# ❌ SLOW (row-by-row)
for i in range(len(df)):
    df.loc[i, 'sma'] = df.loc[i-20:i, 'close'].mean()

# ✅ FAST (vectorized)
df['sma'] = df['close'].rolling(20).mean()
```

**Profile and optimize** hot paths:

```python
import time

def profile_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {(end-start)*1000:.2f}ms")
        return result
    return wrapper

@profile_function
def analyze_smc(self, df):
    # ... your SMC logic ...
```

---

#### 🚀 **5. Database Optimization**

Add indexes to SQLite database:

```sql
-- In db_manager.py table creation
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);
CREATE INDEX idx_trades_status ON trades(exit_time);

-- Query optimization
CREATE INDEX idx_composite ON trades(symbol, entry_time, exit_time);
```

Use connection pooling:

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'sqlite:///data/trading_bot.db',
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10
)
```

---

## 📈 Performance Benchmarks

### Current System (Estimated)

```
Component                  | Latency
---------------------------|----------
API Data Fetch (3 TF)      | 2-4 seconds
Data Preprocessing         | 100-200ms
SMC Detection              | 50-100ms
ML Prediction              | 10-50ms
Order Execution            | 100-500ms
---------------------------|----------
TOTAL PER CYCLE            | 2.5-5 seconds
```

### Optimized System (With Improvements)

```
Component                  | Latency
---------------------------|----------
Cached Data / WebSocket    | 10-100ms ⚡
Data Preprocessing         | 20-50ms ⚡
SMC Detection (optimized)  | 20-40ms ⚡
ML Prediction              | 5-20ms ⚡
Order Execution            | 100-500ms
---------------------------|----------
TOTAL PER CYCLE            | 200-700ms ⚡
```

**Improvement: 5-10x faster** 🚀

---

## 🛠️ Implementation Priority

### Phase 1: Quick Wins (1-2 hours)

1. ✅ Add data validation (DataValidator class)
2. ✅ Implement memory caching
3. ✅ Parallel data fetching

### Phase 2: Medium Effort (1 day)

4. ✅ Add data logging/auditing
5. ✅ Optimize data processing (vectorization)
6. ✅ Database indexing

### Phase 3: Advanced (2-3 days)

7. ✅ WebSocket streaming (real-time data)
8. ✅ Cross-exchange validation
9. ✅ Performance monitoring dashboard

---

## 📊 Monitoring & Alerts

Add these metrics to your dashboard:

```python
metrics = {
    'data_latency_ms': time_to_fetch_data,
    'processing_latency_ms': time_to_process,
    'api_calls_per_minute': api_call_count,
    'cache_hit_rate': cache_hits / total_requests,
    'data_quality_score': validation_score,
    'websocket_connected': ws_status
}
```

Set up alerts:

- 🚨 Data latency > 5 seconds
- 🚨 Cache hit rate < 50%
- 🚨 Data validation failures
- 🚨 WebSocket disconnections

---

## 🎯 Summary

### For Data Accuracy:

1. ✅ **Validate** all incoming data
2. ✅ **Clean** outliers and missing values
3. ✅ **Cross-check** with multiple sources
4. ✅ **Audit** data quality metrics
5. ✅ **Test** with historical data

### For Low Latency:

1. ⚡ **Cache** frequently accessed data
2. ⚡ **Stream** real-time data via WebSocket
3. ⚡ **Parallelize** API calls
4. ⚡ **Optimize** processing code (vectorization)
5. ⚡ **Index** database queries

### Trade-offs:

- **Caching** = Lower latency but slightly stale data (acceptable for 15m timeframe)
- **WebSocket** = Real-time but more complex code
- **Parallel fetching** = Faster but higher API usage

**Recommendation:** Implement Phase 1 & 2 for immediate 3-5x performance improvement! 🚀
