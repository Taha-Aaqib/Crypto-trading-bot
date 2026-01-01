"""
Data preprocessing module
Cleans and prepares OHLCV data for analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger()


class DataPreprocessor:
    """Preprocess and clean market data"""
    
    def __init__(self, config: Dict = None):
        self.config = config
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data
        - Remove duplicates
        - Handle missing values
        - Remove outliers
        """
        df_clean = df.copy()
        
        # Remove duplicates
        original_len = len(df_clean)
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        if len(df_clean) < original_len:
            logger.info(f"Removed {original_len - len(df_clean)} duplicate rows")
        
        # Forward fill missing values
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            df_clean = df_clean.fillna(method='ffill')
            logger.info(f"Filled {missing_count} missing values")
        
        # Remove outliers using IQR method
        df_clean = self._remove_outliers(df_clean)
        
        # Ensure data is sorted by index
        df_clean = df_clean.sort_index()
        
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        df_no_outliers = df.copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Count outliers
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    # Replace outliers with boundary values instead of removing
                    df_no_outliers[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Clipped {outliers} outliers in {col}")
        
        return df_no_outliers
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic features like returns, ranges, etc."""
        df_features = df.copy()
        
        # Price returns
        df_features['returns'] = df_features['close'].pct_change()
        
        # Log returns
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        
        # High-Low range
        df_features['hl_range'] = df_features['high'] - df_features['low']
        df_features['hl_range_pct'] = (df_features['hl_range'] / df_features['close']) * 100
        
        # Open-Close range
        df_features['oc_range'] = abs(df_features['open'] - df_features['close'])
        
        # Candle body size
        df_features['body_size'] = abs(df_features['close'] - df_features['open'])
        
        # Upper and lower wicks
        df_features['upper_wick'] = df_features['high'] - df_features[['open', 'close']].max(axis=1)
        df_features['lower_wick'] = df_features[['open', 'close']].min(axis=1) - df_features['low']
        
        # Candle direction (1 for bullish, -1 for bearish)
        df_features['direction'] = np.where(df_features['close'] > df_features['open'], 1, -1)
        
        # Volume changes
        df_features['volume_change'] = df_features['volume'].pct_change()
        
        logger.info(f"Added {len(df_features.columns) - len(df.columns)} basic features")
        
        return df_features
    
    def resample_timeframe(
        self,
        df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to different timeframe
        
        Args:
            df: OHLCV DataFrame
            target_timeframe: Target timeframe (e.g., '1h', '4h', '1d')
        
        Returns:
            Resampled DataFrame
        """
        # Define aggregation rules
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        df_resampled = df.resample(target_timeframe).agg(ohlc_dict).dropna()
        
        logger.info(f"Resampled data to {target_timeframe}: {len(df_resampled)} candles")
        
        return df_resampled
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        Normalize data for ML models
        
        Args:
            df: DataFrame to normalize
            columns: Columns to normalize (None = all numeric)
            method: 'minmax' or 'zscore'
        
        Returns:
            Normalized DataFrame
        """
        df_norm = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                if method == 'minmax':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                
                elif method == 'zscore':
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    df_norm[col] = (df[col] - mean_val) / std_val
        
        logger.info(f"Normalized {len(columns)} columns using {method}")
        
        return df_norm
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> tuple:
        """
        Split data into train and test sets (time-based)
        
        Args:
            df: DataFrame to split
            test_size: Proportion of data for testing
        
        Returns:
            Tuple of (train_df, test_df)
        """
        split_index = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
        
        logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
        
        return train_df, test_df
    
    def process_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline
        
        Args:
            df: Raw OHLCV DataFrame
        
        Returns:
            Processed DataFrame ready for analysis
        """
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Add basic features
        df_features = self.add_basic_features(df_clean)
        
        logger.info("Preprocessing pipeline completed")
        
        return df_features

