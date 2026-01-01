"""
Technical Analysis Indicators
RSI, EMA, MACD, ATR, etc.
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

    def calculate_sma(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """Calculate Simple Moving Average"""
        return df[column].rolling(window=period).mean()

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

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = 'close'
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Returns:
            Dictionary with macd, signal, and histogram
        """
        ema_fast = self.calculate_ema(df, fast, column)
        ema_slow = self.calculate_ema(df, slow, column)

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal

        return {
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }

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

    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: int = 2,
        column: str = 'close'
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(df, period, column)
        std = df[column].rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {
            'bb_middle': sma,
            'bb_upper': upper_band,
            'bb_lower': lower_band
        }

    def calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()

        return {
            'stoch_k': k,
            'stoch_d': d
        }

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (trend strength)
        """
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate ATR
        atr = self.calculate_atr(df, period)

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators"""
        # Volume Moving Average
        volume_ma = df['volume'].rolling(window=20).mean()

        # On-Balance Volume (OBV)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # Volume Weighted Average Price (VWAP)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        return {
            'volume_ma': volume_ma,
            'obv': obv,
            'vwap': vwap
        }

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with all indicators
        """
        logger.info("Calculating all technical indicators")

        df_indicators = df.copy()

        # EMAs (50 and 200 for trend filter as per proposal)
        df_indicators['ema_50'] = self.calculate_ema(
            df, self.indicator_config.get('ema_fast', 50))
        df_indicators['ema_200'] = self.calculate_ema(
            df, self.indicator_config.get('ema_slow', 200))

        # RSI
        df_indicators['rsi'] = self.calculate_rsi(
            df, self.indicator_config.get('rsi_period', 14))

        # MACD
        macd_dict = self.calculate_macd(df)
        df_indicators['macd'] = macd_dict['macd']
        df_indicators['macd_signal'] = macd_dict['macd_signal']
        df_indicators['macd_histogram'] = macd_dict['macd_histogram']

        # ATR (for stop loss calculation)
        df_indicators['atr'] = self.calculate_atr(
            df, self.indicator_config.get('atr_period', 14))

        # Bollinger Bands
        bb_dict = self.calculate_bollinger_bands(df)
        df_indicators['bb_upper'] = bb_dict.get('bb_upper', 0)
        df_indicators['bb_middle'] = bb_dict.get('bb_middle', 0)
        df_indicators['bb_lower'] = bb_dict.get('bb_lower', 0)

        # ADX (trend strength)
        df_indicators['adx'] = self.calculate_adx(df)

        # Volume indicators
        volume_dict = self.calculate_volume_indicators(df)
        df_indicators['volume_ma'] = volume_dict['volume_ma']
        df_indicators['obv'] = volume_dict['obv']
        df_indicators['vwap'] = volume_dict['vwap']

        logger.info(
            f"Added {len(df_indicators.columns) - len(df.columns)} technical indicators")

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
