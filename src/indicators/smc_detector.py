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
        Determine market structure PER CANDLE (rolling).
        At each swing point, re-evaluate based on the last 2 swing highs
        and last 2 swing lows up to that point, then forward-fill.
        This allows BOS to detect both bullish and bearish breaks as
        market structure changes over time.
        """
        df_structure = df.copy()

        # Get swing points
        df_structure = self.detect_swing_points(df_structure)

        # Extract swing highs and lows (drop NaN)
        swing_highs = df_structure['swing_high'].dropna()
        swing_lows = df_structure['swing_low'].dropna()

        # Initialize all as neutral
        df_structure['market_structure'] = 'neutral'

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return df_structure

        # Build list of swing events sorted by index
        high_events = [(idx, 'high', val) for idx, val in zip(swing_highs.index, swing_highs.values)]
        low_events = [(idx, 'low', val) for idx, val in zip(swing_lows.index, swing_lows.values)]
        events = sorted(high_events + low_events, key=lambda x: x[0])

        # Track last 2 highs and lows as we go
        last_highs = []
        last_lows = []
        structure_at_idx = {}

        for idx, kind, val in events:
            if kind == 'high':
                last_highs.append(val)
                if len(last_highs) > 2:
                    last_highs = last_highs[-2:]
            else:
                last_lows.append(val)
                if len(last_lows) > 2:
                    last_lows = last_lows[-2:]

            if len(last_highs) >= 2 and len(last_lows) >= 2:
                hh = last_highs[-1] > last_highs[-2]  # Higher high
                hl = last_lows[-1] > last_lows[-2]    # Higher low

                if hh and hl:
                    structure_at_idx[idx] = 'bullish'
                elif not hh and not hl:
                    structure_at_idx[idx] = 'bearish'
                else:
                    structure_at_idx[idx] = 'neutral'

        # Apply via reindex + forward-fill (efficient, no per-row loop)
        if structure_at_idx:
            structure_series = pd.Series(structure_at_idx)
            structure_series = structure_series.reindex(df_structure.index, method='ffill')
            structure_series = structure_series.fillna('neutral')
            df_structure['market_structure'] = structure_series

        return df_structure

    def detect_choch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Change of Character (CHOCH)
        CHOCH occurs ONLY at the candle that FIRST breaks the most recent swing point
        This is a one-time event, not every candle above/below the level
        """
        df_choch = df.copy()
        df_choch = self.detect_swing_points(df_choch)

        df_choch['choch_bullish'] = False
        df_choch['choch_bearish'] = False

        swing_highs = df_choch['swing_high'].dropna()
        swing_lows = df_choch['swing_low'].dropna()

        # Track if we've already broken each swing level
        last_broken_high = None
        last_broken_low = None

        for i in range(len(df_choch)):
            current_high = df_choch.iloc[i]['high']
            current_low = df_choch.iloc[i]['low']

            # Get swing points BEFORE this candle only
            prior_swing_highs = swing_highs[swing_highs.index < df_choch.index[i]]
            prior_swing_lows = swing_lows[swing_lows.index < df_choch.index[i]]

            # Bullish CHOCH: FIRST break above recent swing high
            if len(prior_swing_highs) > 0:
                recent_high = prior_swing_highs.iloc[-1]
                # Only mark if this is the FIRST break (not already broken)
                if current_high > recent_high and last_broken_high != recent_high:
                    df_choch.iloc[i, df_choch.columns.get_loc('choch_bullish')] = True
                    last_broken_high = recent_high

            # Bearish CHOCH: FIRST break below recent swing low
            if len(prior_swing_lows) > 0:
                recent_low = prior_swing_lows.iloc[-1]
                # Only mark if this is the FIRST break (not already broken)
                if current_low < recent_low and last_broken_low != recent_low:
                    df_choch.iloc[i, df_choch.columns.get_loc('choch_bearish')] = True
                    last_broken_low = recent_low

        logger.debug(
            f"Detected {df_choch['choch_bullish'].sum()} bullish and {df_choch['choch_bearish'].sum()} bearish CHOCH")

        return df_choch

    def detect_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Break of Structure (BOS)
        BOS confirms trend continuation - ONLY marks the FIRST candle that breaks
        """
        df_bos = df.copy()
        df_bos = self.detect_swing_points(df_bos)
        df_bos = self.detect_market_structure(df_bos)

        df_bos['bos_bullish'] = False
        df_bos['bos_bearish'] = False

        # Track broken levels to avoid marking multiple times
        last_broken_high = None
        last_broken_low = None

        for i in range(1, len(df_bos)):
            current_high = df_bos.iloc[i]['high']
            current_low = df_bos.iloc[i]['low']
            market_structure = df_bos.iloc[i]['market_structure']

            # Get swing points BEFORE this candle
            prior_swing_highs = df_bos.iloc[:i]['swing_high'].dropna()
            prior_swing_lows = df_bos.iloc[:i]['swing_low'].dropna()

            # Bullish BOS: In uptrend, FIRST break above recent swing high
            if market_structure == 'bullish' and len(prior_swing_highs) > 0:
                recent_high = prior_swing_highs.iloc[-1]
                if current_high > recent_high and last_broken_high != recent_high:
                    df_bos.iloc[i, df_bos.columns.get_loc('bos_bullish')] = True
                    last_broken_high = recent_high

            # Bearish BOS: In downtrend, FIRST break below recent swing low
            elif market_structure == 'bearish' and len(prior_swing_lows) > 0:
                recent_low = prior_swing_lows.iloc[-1]
                if current_low < recent_low and last_broken_low != recent_low:
                    df_bos.iloc[i, df_bos.columns.get_loc('bos_bearish')] = True
                    last_broken_low = recent_low

        logger.debug(
            f"Detected {df_bos['bos_bullish'].sum()} bullish and {df_bos['bos_bearish'].sum()} bearish BOS")

        return df_bos

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect SIGNIFICANT Fair Value Gaps (FVG)
        Only mark FVG if it's significant (larger gap) and near CHOCH/BOS
        FVG is a gap in price where candle[i-1].low > candle[i+1].high (bullish)
        or candle[i-1].high < candle[i+1].low (bearish)
        """
        df_fvg = df.copy()

        # Need CHOCH/BOS columns first
        if 'choch_bullish' not in df_fvg.columns:
            df_fvg = self.detect_choch(df_fvg)
        if 'bos_bullish' not in df_fvg.columns:
            df_fvg = self.detect_bos(df_fvg)

        # Initialize columns with proper data types
        df_fvg.loc[:, 'fvg_bullish'] = False
        df_fvg.loc[:, 'fvg_bearish'] = False
        df_fvg.loc[:, 'fvg_high'] = np.nan
        df_fvg.loc[:, 'fvg_low'] = np.nan

        # Use higher threshold for significant FVGs (0.3% instead of 0.1%)
        significant_fvg_threshold = self.fvg_threshold * 3  # 0.3%
        
        # Look for CHOCH/BOS within last 10 candles to validate FVG
        lookback_for_structure = 10

        for i in range(1, len(df_fvg) - 1):
            prev_candle = df_fvg.iloc[i - 1]
            curr_candle = df_fvg.iloc[i]
            next_candle = df_fvg.iloc[i + 1]

            # Check if there's a recent CHOCH or BOS nearby (within lookback)
            start_idx = max(0, i - lookback_for_structure)
            end_idx = min(len(df_fvg), i + lookback_for_structure)
            
            has_nearby_bullish_structure = (
                df_fvg.iloc[start_idx:end_idx]['choch_bullish'].any() or
                df_fvg.iloc[start_idx:end_idx]['bos_bullish'].any()
            )
            has_nearby_bearish_structure = (
                df_fvg.iloc[start_idx:end_idx]['choch_bearish'].any() or
                df_fvg.iloc[start_idx:end_idx]['bos_bearish'].any()
            )

            # Bullish FVG: Gap up (prev.high < next.low)
            if prev_candle['high'] < next_candle['low']:
                gap_size = (next_candle['low'] - prev_candle['high']) / prev_candle['high']

                # Only mark if significant AND near bullish structure
                if gap_size >= significant_fvg_threshold and has_nearby_bullish_structure:
                    df_fvg.loc[df_fvg.index[i], 'fvg_bullish'] = True
                    df_fvg.loc[df_fvg.index[i], 'fvg_low'] = prev_candle['high']
                    df_fvg.loc[df_fvg.index[i], 'fvg_high'] = next_candle['low']

            # Bearish FVG: Gap down (prev.low > next.high)
            elif prev_candle['low'] > next_candle['high']:
                gap_size = (prev_candle['low'] - next_candle['high']) / prev_candle['low']

                # Only mark if significant AND near bearish structure
                if gap_size >= significant_fvg_threshold and has_nearby_bearish_structure:
                    df_fvg.loc[df_fvg.index[i], 'fvg_bearish'] = True
                    df_fvg.loc[df_fvg.index[i], 'fvg_high'] = prev_candle['low']
                    df_fvg.loc[df_fvg.index[i], 'fvg_low'] = next_candle['high']

        logger.debug(
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

        logger.debug(
            f"Detected {df_liq['liquidity_zone'].sum()} liquidity zones")

        return df_liq

    def analyze_smc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete SMC analysis pipeline

        Returns:
            DataFrame with all SMC indicators
        """
        logger.debug("Starting SMC analysis")

        df_smc = df.copy()

        # Detect all SMC patterns
        df_smc = self.detect_swing_points(df_smc)
        df_smc = self.detect_market_structure(df_smc)
        df_smc = self.detect_choch(df_smc)
        df_smc = self.detect_bos(df_smc)
        df_smc = self.detect_fvg(df_smc)
        df_smc = self.detect_liquidity_zones(df_smc)

        logger.debug("SMC analysis completed")

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
