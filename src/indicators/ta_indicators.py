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

        rs = gain / loss
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
        Add EMA, RSI, and ATR indicators to DataFrame
        """
        logger.debug("Calculating EMA, RSI, and ATR indicators")

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

        logger.debug("Added EMA, RSI, and ATR indicators")

        return df_indicators

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
