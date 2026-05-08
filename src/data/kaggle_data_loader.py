"""
Kaggle Historical Data Loader
Loads Binance historical data from Kaggle datasets (2017-present)
Provides much longer history than API fetching for ML training

Supported Kaggle datasets:
- https://www.kaggle.com/datasets/jorijnsmit/binance-full-history
- https://www.kaggle.com/datasets/jessevent/all-crypto-currencies

Usage:
    from src.data.kaggle_data_loader import KaggleDataLoader
    
    loader = KaggleDataLoader(config)
    df = loader.load_historical_data('BTC/USDT', '1h')
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger()


class KaggleDataLoader:
    """
    Load and preprocess historical crypto data from Kaggle datasets
    
    Expected data directory structure:
        data/kaggle/
            BTCUSDT-1h.csv       # Or similar naming
            ETHUSDT-1h.csv
            ...
    
    Supported CSV formats:
        1. Standard OHLCV: timestamp, open, high, low, close, volume
        2. Binance format: Open time, Open, High, Low, Close, Volume, Close time, ...
    """
    
    def __init__(self, config: Dict):
        self.config = config
        # Check multiple possible data directories
        self.data_dir = Path(config.get('data', {}).get('kaggle_dir', 'data/kaggle'))
        self.alt_data_dir = Path('kaggle')  # Alternative location at project root
        self.raw_dir = Path('data/raw')
        self.processed_dir = Path('data/processed')
        
        # Create directories if needed
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"KaggleDataLoader initialized, data_dir: {self.data_dir}, alt: {self.alt_data_dir}")
    
    def find_data_file(self, symbol: str, timeframe: str) -> Optional[Path]:
        """
        Find the data file for a given symbol and timeframe
        Supports both CSV and Parquet formats
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '15m')
            
        Returns:
            Path to data file or None
        """
        # Normalize symbol: BTC/USDT -> BTCUSDT and BTC-USDT
        symbol_clean = symbol.replace('/', '').upper()
        symbol_dash = symbol.replace('/', '-').upper()
        
        # Try various naming patterns (CSV and Parquet)
        patterns = [
            # Parquet patterns
            f"{symbol_dash}.parquet",
            f"{symbol_clean}.parquet",
            f"{symbol_clean}-{timeframe}.parquet",
            f"{symbol_dash}-{timeframe}.parquet",
            # CSV patterns
            f"{symbol_clean}-{timeframe}.csv",
            f"{symbol_clean}_{timeframe}.csv", 
            f"{symbol_clean}-{timeframe.upper()}.csv",
            f"{symbol_clean}_{timeframe.upper()}.csv",
            f"{symbol_clean}.csv",
            f"{symbol_dash}.csv",
            f"Binance_{symbol_clean}_{timeframe}.csv",
            f"binance_{symbol_clean}_{timeframe}.csv",
        ]
        
        # Check all directories
        dirs_to_check = [self.data_dir, self.alt_data_dir, self.raw_dir]
        
        for check_dir in dirs_to_check:
            if not check_dir.exists():
                continue
            for pattern in patterns:
                filepath = check_dir / pattern
                if filepath.exists():
                    logger.info(f"Found data file: {filepath}")
                    return filepath
        
        # Try glob search for partial matches
        for check_dir in dirs_to_check:
            if not check_dir.exists():
                continue
            for ext in ['*.parquet', '*.csv']:
                search_pattern = f"*{symbol_clean}*{ext}"
                matches = list(check_dir.glob(search_pattern))
                if not matches:
                    search_pattern = f"*{symbol_dash}*{ext}"
                    matches = list(check_dir.glob(search_pattern))
                if matches:
                    logger.info(f"Found data file via glob: {matches[0]}")
                    return matches[0]
                
        logger.warning(f"No data file found for {symbol} {timeframe}")
        logger.info(f"Searched in: {dirs_to_check}")
        logger.info(f"Expected patterns: {patterns[:5]}...")
        
        return None
    
    def detect_csv_format(self, filepath: Path) -> Dict:
        """
        Auto-detect CSV format (columns, date format, etc.)
        
        Returns:
            Dict with format info: columns mapping, date_column, date_format
        """
        # Read first few rows to detect format
        sample = pd.read_csv(filepath, nrows=5)
        columns = [c.lower().strip() for c in sample.columns]
        
        format_info = {
            'date_column': None,
            'date_format': None,
            'columns': {}
        }
        
        # Detect date column
        date_candidates = ['timestamp', 'time', 'date', 'datetime', 'open time', 'open_time', 'opentime']
        for col in sample.columns:
            if col.lower().strip() in date_candidates:
                format_info['date_column'] = col
                break
        
        # If first column looks like a timestamp (numeric or ISO date)
        if format_info['date_column'] is None:
            first_col = sample.columns[0]
            first_val = sample[first_col].iloc[0]
            
            # Check if it's a Unix timestamp (milliseconds)
            if isinstance(first_val, (int, float)) and first_val > 1e12:
                format_info['date_column'] = first_col
                format_info['date_format'] = 'unix_ms'
            # Check if it's a Unix timestamp (seconds)
            elif isinstance(first_val, (int, float)) and first_val > 1e9:
                format_info['date_column'] = first_col
                format_info['date_format'] = 'unix_s'
            # Assume ISO format if string
            elif isinstance(first_val, str):
                format_info['date_column'] = first_col
        
        # Map OHLCV columns
        ohlcv_mapping = {
            'open': ['open', 'o'],
            'high': ['high', 'h'],
            'low': ['low', 'l'],
            'close': ['close', 'c'],
            'volume': ['volume', 'vol', 'v', 'quote asset volume', 'quote_asset_volume']
        }
        
        for target, candidates in ohlcv_mapping.items():
            for col in sample.columns:
                if col.lower().strip() in candidates:
                    format_info['columns'][target] = col
                    break
        
        logger.debug(f"Detected CSV format: {format_info}")
        return format_info
    
    def load_parquet(self, filepath: Path) -> pd.DataFrame:
        """
        Load and standardize Parquet file to OHLCV format
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: DatetimeIndex named 'timestamp'
        """
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        
        # Standardize column names (lowercase)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Detect and set timestamp index
        date_candidates = ['timestamp', 'time', 'date', 'datetime', 'open_time', 'opentime']
        date_col = None
        
        for col in df.columns:
            if col in date_candidates:
                date_col = col
                break
        
        # If no explicit date column, check if index is datetime
        if date_col is None and isinstance(df.index, pd.DatetimeIndex):
            df.index.name = 'timestamp'
        elif date_col is not None:
            # Parse the date column
            if df[date_col].dtype in ['int64', 'float64']:
                # Unix timestamp
                val = df[date_col].iloc[0]
                if val > 1e12:
                    df['timestamp'] = pd.to_datetime(df[date_col], unit='ms')
                else:
                    df['timestamp'] = pd.to_datetime(df[date_col], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df[date_col])
            df.set_index('timestamp', inplace=True)
        else:
            # Use first column as timestamp
            first_col = df.columns[0]
            if df[first_col].dtype in ['int64', 'float64']:
                val = df[first_col].iloc[0]
                if val > 1e12:
                    df['timestamp'] = pd.to_datetime(df[first_col], unit='ms')
                elif val > 1e9:
                    df['timestamp'] = pd.to_datetime(df[first_col], unit='s')
                else:
                    df['timestamp'] = pd.to_datetime(df[first_col])
            else:
                df['timestamp'] = pd.to_datetime(df[first_col])
            df.set_index('timestamp', inplace=True)
        
        # Map column names to standard OHLCV
        col_mapping = {}
        ohlcv_mapping = {
            'open': ['open', 'o', 'open_price'],
            'high': ['high', 'h', 'high_price'],
            'low': ['low', 'l', 'low_price'],
            'close': ['close', 'c', 'close_price'],
            'volume': ['volume', 'vol', 'v', 'quote_asset_volume', 'quote asset volume']
        }
        
        for target, candidates in ohlcv_mapping.items():
            for col in df.columns:
                if col.lower() in candidates:
                    if col != target:
                        col_mapping[col] = target
                    break
        
        if col_mapping:
            df.rename(columns=col_mapping, inplace=True)
        
        # Select only OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        available = [c for c in required_cols if c in df.columns]
        
        if len(available) < 5:
            # Try to find columns that might have different names
            logger.warning(f"Only found {available}, looking for alternatives...")
            logger.info(f"Available columns: {list(df.columns)}")
        
        df = df[available].copy()
        
        # Convert to numeric
        for col in available:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def load_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load file based on extension (CSV or Parquet)
        """
        if filepath.suffix.lower() == '.parquet':
            return self.load_parquet(filepath)
        else:
            return self.load_csv(filepath)
    
    def load_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load and standardize CSV file to OHLCV format
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: DatetimeIndex named 'timestamp'
        """
        format_info = self.detect_csv_format(filepath)
        
        # Load full file
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        
        # Parse date column
        date_col = format_info['date_column']
        if date_col is None:
            raise ValueError(f"Could not detect date column in {filepath}")
        
        if format_info['date_format'] == 'unix_ms':
            df['timestamp'] = pd.to_datetime(df[date_col], unit='ms')
        elif format_info['date_format'] == 'unix_s':
            df['timestamp'] = pd.to_datetime(df[date_col], unit='s')
        else:
            df['timestamp'] = pd.to_datetime(df[date_col])
        
        df.set_index('timestamp', inplace=True)
        
        # Rename columns to standard OHLCV
        rename_map = {}
        for target, source in format_info['columns'].items():
            if source != target and source in df.columns:
                rename_map[source] = target
        
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        
        # Select only OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df[required_cols].copy()
        
        # Convert to numeric (in case of string data)
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def load_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load historical data for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h')
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with OHLCV data
        """
        filepath = self.find_data_file(symbol, timeframe)
        
        if filepath is None:
            logger.error(f"No data file found for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Use appropriate loader based on file type
        df = self.load_file(filepath)
        
        # Apply date filters
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]
        
        logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
        
        if len(df) > 0:
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data: handle missing values, outliers, gaps
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        original_len = len(df)
        
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Remove rows with zero or negative prices
        df = df[(df['open'] > 0) & (df['high'] > 0) & 
                (df['low'] > 0) & (df['close'] > 0)]
        
        # Fix OHLC consistency (high >= open,close >= low)
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Remove extreme outliers (price changes > 50% in one candle)
        pct_change = df['close'].pct_change().abs()
        df = df[pct_change < 0.5]  # Remove > 50% moves (likely errors)
        
        # Fill missing volume with 0
        df['volume'] = df['volume'].fillna(0)
        
        cleaned_len = len(df)
        if original_len != cleaned_len:
            logger.info(f"Cleaned data: {original_len} -> {cleaned_len} rows")
        
        return df
    
    def resample_timeframe(
        self, 
        df: pd.DataFrame, 
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to a different timeframe
        
        Args:
            df: Source DataFrame (must be lower timeframe)
            target_timeframe: Target timeframe (e.g., '4h', '1d')
            
        Returns:
            Resampled DataFrame
        """
        # Map timeframe strings to pandas offsets
        tf_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1ME'
        }
        
        if target_timeframe not in tf_map:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        
        offset = tf_map[target_timeframe]
        
        resampled = df.resample(offset).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"Resampled to {target_timeframe}: {len(resampled)} candles")
        
        return resampled
    
    def prepare_training_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        clean: bool = True
    ) -> pd.DataFrame:
        """
        Load and prepare data for ML training
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_date: Start date (default: 2019-01-01)
            end_date: End date (default: now)
            clean: Whether to clean data
            
        Returns:
            Cleaned OHLCV DataFrame ready for feature engineering
        """
        if start_date is None:
            start_date = datetime(2019, 1, 1)
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Preparing training data for {symbol} {timeframe}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Load raw data (may be 1-minute)
        df = self.load_historical_data(symbol, timeframe, start_date, end_date)
        
        if df.empty:
            logger.error(f"No data loaded for {symbol}")
            return df
        
        # Detect if we need to resample (check data frequency)
        if len(df) > 1:
            time_diff = (df.index[1] - df.index[0]).total_seconds()
            data_tf_minutes = time_diff / 60
            
            # Map target timeframe to minutes
            tf_minutes_map = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360,
                '8h': 480, '12h': 720, '1d': 1440
            }
            target_minutes = tf_minutes_map.get(timeframe, 60)
            
            # Resample if data is finer than target
            if data_tf_minutes < target_minutes:
                logger.info(f"Data is {data_tf_minutes}m, resampling to {timeframe}")
                df = self.resample_timeframe(df, timeframe)
        
        if clean:
            df = self.clean_data(df)
        
        logger.info(f"Prepared {len(df)} candles for training")
        
        return df
    
    def list_available_data(self) -> List[Dict]:
        """
        List all available data files
        
        Returns:
            List of dicts with file info
        """
        files = []
        
        for csv_file in self.data_dir.glob('*.csv'):
            try:
                # Quick stats without loading full file
                sample = pd.read_csv(csv_file, nrows=1)
                row_count = sum(1 for _ in open(csv_file)) - 1  # -1 for header
                
                files.append({
                    'filename': csv_file.name,
                    'path': str(csv_file),
                    'columns': list(sample.columns),
                    'rows': row_count
                })
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
        
        return files
    
    def download_instructions(self) -> str:
        """
        Return instructions for downloading Kaggle data
        """
        return """
╔══════════════════════════════════════════════════════════════════════════╗
║                     KAGGLE DATA DOWNLOAD INSTRUCTIONS                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. Go to Kaggle and download Binance historical data:                   ║
║     https://www.kaggle.com/datasets/jorijnsmit/binance-full-history      ║
║                                                                          ║
║  2. Extract the CSV files to: data/kaggle/                               ║
║                                                                          ║
║  3. Expected file names:                                                 ║
║     - BTCUSDT-1h.csv                                                     ║
║     - ETHUSDT-1h.csv                                                     ║
║     - (or similar naming conventions)                                    ║
║                                                                          ║
║  4. Required columns (auto-detected):                                    ║
║     - timestamp/time/date (Unix ms or ISO format)                        ║
║     - open, high, low, close, volume                                     ║
║                                                                          ║
║  Alternative datasets:                                                   ║
║  - https://www.kaggle.com/datasets/jessevent/all-crypto-currencies       ║
║  - https://www.kaggle.com/datasets/tencars/392-crypto-currency-pairs     ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    # Test the loader
    import yaml
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    loader = KaggleDataLoader(config)
    
    # List available data
    print("\n=== Available Data Files ===")
    files = loader.list_available_data()
    
    if not files:
        print(loader.download_instructions())
    else:
        for f in files:
            print(f"  {f['filename']}: {f['rows']:,} rows")
        
        # Try loading BTC data
        print("\n=== Loading BTC/USDT 1h ===")
        df = loader.prepare_training_data('BTC/USDT', '1h')
        
        if not df.empty:
            print(f"Loaded {len(df):,} candles")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print(f"\nSample data:")
            print(df.head())
