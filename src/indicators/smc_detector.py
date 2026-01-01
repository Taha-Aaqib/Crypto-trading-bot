"""
Smart Money Concept (SMC) Detector
Identifies CHOCH, BOS, FVG, and Liquidity Zones
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger()


class SMCDetector:
    """
    Detect Smart Money Concepts in price data
    - CHOCH: Change of Character
    - BOS: Break of Structure
    - FVG: Fair Value Gap
    - Liquidity Zones
    """

    def __init__(self, config: Dict):
        self.config = config
        self.smc_config = config.get('smc', {})
        self.fvg_threshold = self.smc_config.get('fvg_threshold', 0.001)
        self.liquidity_lookback = self.smc_config.get('liquidity_lookback', 20)

    def detect_swing_points(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Detect swing highs and swing lows

        Args:
            df: OHLCV DataFrame
            window: Number of candles on each side to confirm swing

        Returns:
            DataFrame with swing_high and swing_low columns
        """
        df_swings = df.copy()

        # Initialize columns
        df_swings['swing_high'] = np.nan
        df_swings['swing_low'] = np.nan

        # Detect swing highs
        for i in range(window, len(df) - window):
            high = df.iloc[i]['high']

            # Check if current high is higher than surrounding candles
            is_swing_high = all(
                high > df.iloc[i - j]['high'] and high > df.iloc[i + j]['high']
                for j in range(1, window + 1)
            )

            if is_swing_high:
                df_swings.iloc[i, df_swings.columns.get_loc(
                    'swing_high')] = high

        # Detect swing lows
        for i in range(window, len(df) - window):
            low = df.iloc[i]['low']

            # Check if current low is lower than surrounding candles
            is_swing_low = all(
                low < df.iloc[i - j]['low'] and low < df.iloc[i + j]['low']
                for j in range(1, window + 1)
            )

            if is_swing_low:
                df_swings.iloc[i, df_swings.columns.get_loc('swing_low')] = low

        return df_swings

    def detect_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Determine market structure: bullish or bearish
        Based on higher highs/higher lows or lower highs/lower lows
        """
        df_structure = df.copy()

        # Get swing points
        df_structure = self.detect_swing_points(df_structure)

        # Extract swing highs and lows (drop NaN)
        swing_highs = df_structure['swing_high'].dropna()
        swing_lows = df_structure['swing_low'].dropna()

        # Determine trend
        df_structure['market_structure'] = 'neutral'

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for higher highs and higher lows (bullish)
            recent_highs = swing_highs.iloc[-2:]
            recent_lows = swing_lows.iloc[-2:]

            if recent_highs.iloc[1] > recent_highs.iloc[0] and recent_lows.iloc[1] > recent_lows.iloc[0]:
                df_structure['market_structure'] = 'bullish'
            # Check for lower highs and lower lows (bearish)
            elif recent_highs.iloc[1] < recent_highs.iloc[0] and recent_lows.iloc[1] < recent_lows.iloc[0]:
                df_structure['market_structure'] = 'bearish'

        return df_structure

    def detect_choch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Change of Character (CHOCH)
        CHOCH occurs when price breaks the most recent swing point against the trend
        """
        df_choch = df.copy()
        df_choch = self.detect_swing_points(df_choch)

        df_choch['choch_bullish'] = False
        df_choch['choch_bearish'] = False

        swing_highs = df_choch['swing_high'].dropna()
        swing_lows = df_choch['swing_low'].dropna()

        for i in range(len(df_choch)):
            current_price = df_choch.iloc[i]['close']

            # Bullish CHOCH: Price breaks above recent swing high in downtrend
            if len(swing_highs) > 0:
                recent_high = swing_highs.iloc[-1]
                if current_price > recent_high:
                    df_choch.iloc[i, df_choch.columns.get_loc(
                        'choch_bullish')] = True

            # Bearish CHOCH: Price breaks below recent swing low in uptrend
            if len(swing_lows) > 0:
                recent_low = swing_lows.iloc[-1]
                if current_price < recent_low:
                    df_choch.iloc[i, df_choch.columns.get_loc(
                        'choch_bearish')] = True

        logger.info(
            f"Detected {df_choch['choch_bullish'].sum()} bullish and {df_choch['choch_bearish'].sum()} bearish CHOCH")

        return df_choch

    def detect_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Break of Structure (BOS)
        BOS confirms trend continuation after breaking a swing point in trend direction
        """
        df_bos = df.copy()
        df_bos = self.detect_swing_points(df_bos)
        df_bos = self.detect_market_structure(df_bos)

        df_bos['bos_bullish'] = False
        df_bos['bos_bearish'] = False

        for i in range(1, len(df_bos)):
            current_price = df_bos.iloc[i]['close']
            market_structure = df_bos.iloc[i]['market_structure']

            # Bullish BOS: In uptrend, break above recent swing high
            if market_structure == 'bullish':
                recent_swing_high = df_bos.iloc[:i]['swing_high'].dropna()
                if len(recent_swing_high) > 0 and current_price > recent_swing_high.iloc[-1]:
                    df_bos.iloc[i, df_bos.columns.get_loc(
                        'bos_bullish')] = True

            # Bearish BOS: In downtrend, break below recent swing low
            elif market_structure == 'bearish':
                recent_swing_low = df_bos.iloc[:i]['swing_low'].dropna()
                if len(recent_swing_low) > 0 and current_price < recent_swing_low.iloc[-1]:
                    df_bos.iloc[i, df_bos.columns.get_loc(
                        'bos_bearish')] = True

        logger.info(
            f"Detected {df_bos['bos_bullish'].sum()} bullish and {df_bos['bos_bearish'].sum()} bearish BOS")

        return df_bos

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVG)
        FVG is a gap in price where candle[i-1].low > candle[i+1].high (bullish)
        or candle[i-1].high < candle[i+1].low (bearish)
        """
        df_fvg = df.copy()

        # Initialize columns with proper data types
        df_fvg.loc[:, 'fvg_bullish'] = False
        df_fvg.loc[:, 'fvg_bearish'] = False
        df_fvg.loc[:, 'fvg_high'] = np.nan
        df_fvg.loc[:, 'fvg_low'] = np.nan

        for i in range(1, len(df_fvg) - 1):
            prev_candle = df_fvg.iloc[i - 1]
            curr_candle = df_fvg.iloc[i]
            next_candle = df_fvg.iloc[i + 1]

            # Bullish FVG: Gap up (prev.high < next.low)
            if prev_candle['high'] < next_candle['low']:
                gap_size = (next_candle['low'] -
                            prev_candle['high']) / prev_candle['high']

                if gap_size >= self.fvg_threshold:
                    df_fvg.loc[df_fvg.index[i], 'fvg_bullish'] = True
                    df_fvg.loc[df_fvg.index[i],
                               'fvg_low'] = prev_candle['high']
                    df_fvg.loc[df_fvg.index[i],
                               'fvg_high'] = next_candle['low']

            # Bearish FVG: Gap down (prev.low > next.high)
            elif prev_candle['low'] > next_candle['high']:
                gap_size = (prev_candle['low'] -
                            next_candle['high']) / prev_candle['low']

                if gap_size >= self.fvg_threshold:
                    df_fvg.loc[df_fvg.index[i], 'fvg_bearish'] = True
                    df_fvg.loc[df_fvg.index[i],
                               'fvg_high'] = prev_candle['low']
                    df_fvg.loc[df_fvg.index[i],
                               'fvg_low'] = next_candle['high']

        logger.info(
            f"Detected {df_fvg['fvg_bullish'].sum()} bullish and {df_fvg['fvg_bearish'].sum()} bearish FVG")

        return df_fvg

    def detect_liquidity_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect liquidity zones (areas where stop losses cluster)
        Typically around recent swing highs and lows
        """
        df_liq = df.copy()
        df_liq = self.detect_swing_points(
            df_liq, window=self.liquidity_lookback)

        df_liq['liquidity_high'] = df_liq['swing_high']
        df_liq['liquidity_low'] = df_liq['swing_low']

        # Liquidity zones are potential reversal areas
        df_liq['liquidity_zone'] = False
        df_liq.loc[df_liq['liquidity_high'].notna(
        ) | df_liq['liquidity_low'].notna(), 'liquidity_zone'] = True

        logger.info(
            f"Detected {df_liq['liquidity_zone'].sum()} liquidity zones")

        return df_liq

    def analyze_smc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete SMC analysis pipeline

        Returns:
            DataFrame with all SMC indicators
        """
        logger.info("Starting SMC analysis")

        df_smc = df.copy()

        # Detect all SMC patterns
        df_smc = self.detect_swing_points(df_smc)
        df_smc = self.detect_market_structure(df_smc)
        df_smc = self.detect_choch(df_smc)
        df_smc = self.detect_bos(df_smc)
        df_smc = self.detect_fvg(df_smc)
        df_smc = self.detect_liquidity_zones(df_smc)

        logger.info("SMC analysis completed")

        return df_smc

    def get_smc_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on SMC

        Returns:
            DataFrame with buy/sell signals
        """
        df_signals = df.copy()

        df_signals['smc_signal'] = 0  # 0 = neutral, 1 = long, -1 = short

        # Long signal: Bullish CHOCH or BOS + FVG retest
        long_condition = (
            (df_signals['choch_bullish'] | df_signals['bos_bullish']) &
            df_signals['fvg_bullish']
        )

        # Short signal: Bearish CHOCH or BOS + FVG retest
        short_condition = (
            (df_signals['choch_bearish'] | df_signals['bos_bearish']) &
            df_signals['fvg_bearish']
        )

        df_signals.loc[long_condition, 'smc_signal'] = 1
        df_signals.loc[short_condition, 'smc_signal'] = -1

        logger.info(
            f"Generated {(df_signals['smc_signal'] == 1).sum()} long and {(df_signals['smc_signal'] == -1).sum()} short SMC signals")

        return df_signals
