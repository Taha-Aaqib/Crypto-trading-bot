"""
Technical Analysis Indicators
RSI, EMA, ATR only
"""

import pandas as pd
import numpy as np
from typing import Dict
from src.utils.logger import get_logger

logger = get_logger()


class TechnicalIndicators:
    """Calculate technical indicators for trading"""

    def __init__(self, config: Dict):
        self.config = config
        self.indicator_config = config.get('indicators', {})

    def calculate_ema(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df[column].ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index

        Args:
            df: DataFrame with price data
            period: RSI period (default 14)
            column: Column to calculate RSI on

        Returns:
            RSI values
        """
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        Used for volatility measurement and stop loss placement
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat(
            [high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr


    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add EMA, RSI, ATR, and RSI divergence indicators to DataFrame
        """
        logger.debug("Calculating EMA, RSI, ATR, and divergence indicators")

        df_indicators = df.copy()

        # EMAs (50 and 200 for trend filter)
        df_indicators['ema_50'] = self.calculate_ema(
            df, self.indicator_config.get('ema_fast', 50))
        df_indicators['ema_200'] = self.calculate_ema(
            df, self.indicator_config.get('ema_slow', 200))

        # RSI
        df_indicators['rsi'] = self.calculate_rsi(
            df, self.indicator_config.get('rsi_period', 14))

        # ATR (for stop loss calculation)
        df_indicators['atr'] = self.calculate_atr(
            df, self.indicator_config.get('atr_period', 14))

        # RSI Divergences (valid for 14 candles after detection)
        df_indicators = self.detect_rsi_divergence(
            df_indicators, 
            lookback=50,
            validity_window=self.indicator_config.get('divergence_validity', 14)
        )

        logger.debug("Added EMA, RSI, ATR, and divergence indicators")

        return df_indicators

    def detect_rsi_divergence(self, df: pd.DataFrame, lookback: int = 50, 
                                validity_window: int = 14) -> pd.DataFrame:
        """
        Detect RSI divergences (bullish and bearish)
        
        Bullish divergence: Price makes LL -> LL, RSI makes LL -> HL
        Bearish divergence: Price makes HH -> HH, RSI makes HH -> LH
        
        Divergence is valid for `validity_window` candles after detection.
        
        Args:
            df: DataFrame with 'close', 'low', 'high', and 'rsi' columns
            lookback: How far back to look for swing points
            validity_window: How many candles the divergence signal remains valid (default 14)
            
        Returns:
            DataFrame with divergence columns added
        """
        df_div = df.copy()
        
        # Ensure RSI is calculated
        if 'rsi' not in df_div.columns:
            df_div['rsi'] = self.calculate_rsi(df_div)
        
        # Initialize divergence columns
        df_div['bullish_divergence'] = False
        df_div['bearish_divergence'] = False
        df_div['divergence_strength'] = 0.0
        
        # Find swing lows (for bullish divergence)
        # A swing low is where low is lower than surrounding candles
        swing_window = 5
        df_div['is_swing_low'] = (
            (df_div['low'] <= df_div['low'].shift(1)) & 
            (df_div['low'] <= df_div['low'].shift(2)) &
            (df_div['low'] <= df_div['low'].shift(-1)) & 
            (df_div['low'] <= df_div['low'].shift(-2))
        )
        
        # Find swing highs (for bearish divergence)
        df_div['is_swing_high'] = (
            (df_div['high'] >= df_div['high'].shift(1)) & 
            (df_div['high'] >= df_div['high'].shift(2)) &
            (df_div['high'] >= df_div['high'].shift(-1)) & 
            (df_div['high'] >= df_div['high'].shift(-2))
        )
        
        # Get indices of swing points
        swing_low_indices = df_div.index[df_div['is_swing_low']].tolist()
        swing_high_indices = df_div.index[df_div['is_swing_high']].tolist()
        
        # Track when divergences were detected for validity window
        bullish_div_detected_at = {}  # idx -> detection_idx
        bearish_div_detected_at = {}
        
        # Detect bullish divergence: Price LL, RSI HL
        for i, idx in enumerate(swing_low_indices):
            if i == 0:
                continue
            
            prev_idx = swing_low_indices[i - 1]
            
            # Check if within lookback range
            try:
                idx_pos = df_div.index.get_loc(idx)
                prev_pos = df_div.index.get_loc(prev_idx)
                if idx_pos - prev_pos > lookback:
                    continue
            except (KeyError, TypeError):
                continue
            
            # Price makes lower low
            price_ll = df_div.loc[idx, 'low'] < df_div.loc[prev_idx, 'low']
            
            # RSI makes higher low (divergence)
            rsi_hl = df_div.loc[idx, 'rsi'] > df_div.loc[prev_idx, 'rsi']
            
            if price_ll and rsi_hl:
                # Bullish divergence detected
                bullish_div_detected_at[idx] = idx
                # Calculate strength based on RSI difference
                rsi_diff = df_div.loc[idx, 'rsi'] - df_div.loc[prev_idx, 'rsi']
                df_div.loc[idx, 'divergence_strength'] = min(rsi_diff / 10, 1.0)
        
        # Detect bearish divergence: Price HH, RSI LH
        for i, idx in enumerate(swing_high_indices):
            if i == 0:
                continue
            
            prev_idx = swing_high_indices[i - 1]
            
            # Check if within lookback range
            try:
                idx_pos = df_div.index.get_loc(idx)
                prev_pos = df_div.index.get_loc(prev_idx)
                if idx_pos - prev_pos > lookback:
                    continue
            except (KeyError, TypeError):
                continue
            
            # Price makes higher high
            price_hh = df_div.loc[idx, 'high'] > df_div.loc[prev_idx, 'high']
            
            # RSI makes lower high (divergence)
            rsi_lh = df_div.loc[idx, 'rsi'] < df_div.loc[prev_idx, 'rsi']
            
            if price_hh and rsi_lh:
                # Bearish divergence detected
                bearish_div_detected_at[idx] = idx
                # Calculate strength based on RSI difference
                rsi_diff = df_div.loc[prev_idx, 'rsi'] - df_div.loc[idx, 'rsi']
                df_div.loc[idx, 'divergence_strength'] = min(rsi_diff / 10, 1.0)
        
        # Apply validity window - divergence is valid for N candles after detection
        for detection_idx in bullish_div_detected_at:
            try:
                det_pos = df_div.index.get_loc(detection_idx)
                end_pos = min(det_pos + validity_window + 1, len(df_div))
                df_div.iloc[det_pos:end_pos, df_div.columns.get_loc('bullish_divergence')] = True
            except (KeyError, TypeError):
                continue
        
        for detection_idx in bearish_div_detected_at:
            try:
                det_pos = df_div.index.get_loc(detection_idx)
                end_pos = min(det_pos + validity_window + 1, len(df_div))
                df_div.iloc[det_pos:end_pos, df_div.columns.get_loc('bearish_divergence')] = True
            except (KeyError, TypeError):
                continue
        
        # Clean up temp columns
        df_div.drop(['is_swing_low', 'is_swing_high'], axis=1, inplace=True)
        
        return df_div

    def get_trend_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Get trend signal based on EMA filter (as per proposal)
        Long: price > EMA50 > EMA200
        Short: price < EMA50 < EMA200

        Returns:
            Series with 1 (bullish), -1 (bearish), 0 (neutral)
        """
        df_trend = df.copy()

        if 'ema_50' not in df_trend.columns:
            df_trend['ema_50'] = self.calculate_ema(df_trend, 50)
        if 'ema_200' not in df_trend.columns:
            df_trend['ema_200'] = self.calculate_ema(df_trend, 200)

        trend_signal = pd.Series(0, index=df_trend.index)

        # Bullish: price > EMA50 and EMA50 > EMA200
        bullish_condition = (df_trend['close'] > df_trend['ema_50']) & (
            df_trend['ema_50'] > df_trend['ema_200'])

        # Bearish: price < EMA50 and EMA50 < EMA200
        bearish_condition = (df_trend['close'] < df_trend['ema_50']) & (
            df_trend['ema_50'] < df_trend['ema_200'])

        trend_signal[bullish_condition] = 1
        trend_signal[bearish_condition] = -1

        return trend_signal
