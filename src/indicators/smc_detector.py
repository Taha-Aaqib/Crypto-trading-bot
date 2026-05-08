"""
Smart Money Concept (SMC) Detector
Enhanced ICT-style analysis with:
- CHOCH: Change of Character
- BOS: Break of Structure
- FVG: Fair Value Gap (with mitigation tracking)
- Order Blocks (OB)
- Liquidity Zones & Sweeps
- OTE (Optimal Trade Entry) - Fibonacci 0.618-0.786 zones
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger()


class SMCDetector:
    """
    Detect Smart Money Concepts in price data
    
    ICT Concepts:
    - CHOCH: Change of Character (trend reversal signal)
    - BOS: Break of Structure (trend continuation)
    - FVG: Fair Value Gap (imbalance zones)
    - OB: Order Blocks (institutional entry zones)
    - Liquidity: Swing highs/lows where stops cluster
    - OTE: Optimal Trade Entry (0.618-0.786 fib retracement)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.smc_config = config.get('smc', {})
        self.fvg_threshold = self.smc_config.get('fvg_threshold', 0.001)
        self.liquidity_lookback = self.smc_config.get('liquidity_lookback', 20)
        
        # OTE Fibonacci levels
        self.ote_fib_low = 0.618
        self.ote_fib_high = 0.786

    def detect_swing_points_vectorized(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Detect swing highs and lows using vectorized operations (FAST)
        
        Args:
            df: OHLCV DataFrame
            window: Number of candles on each side to confirm swing
            
        Returns:
            DataFrame with swing_high and swing_low columns
        """
        df_swings = df.copy()
        
        # Rolling max/min for window on each side
        high_rolling_max_left = df['high'].shift(1).rolling(window).max()
        high_rolling_max_right = df['high'].shift(-window).rolling(window).max()
        
        low_rolling_min_left = df['low'].shift(1).rolling(window).min()
        low_rolling_min_right = df['low'].shift(-window).rolling(window).min()
        
        # Swing high: current high > max of left AND right windows
        is_swing_high = (df['high'] > high_rolling_max_left) & (df['high'] > high_rolling_max_right)
        
        # Swing low: current low < min of left AND right windows
        is_swing_low = (df['low'] < low_rolling_min_left) & (df['low'] < low_rolling_min_right)
        
        df_swings['swing_high'] = np.where(is_swing_high, df['high'], np.nan)
        df_swings['swing_low'] = np.where(is_swing_low, df['low'], np.nan)
        
        return df_swings

    def detect_swing_points(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Detect swing highs and swing lows
        Uses vectorized version for better performance

        Args:
            df: OHLCV DataFrame
            window: Number of candles on each side to confirm swing

        Returns:
            DataFrame with swing_high and swing_low columns
        """
        # Use fast vectorized version
        return self.detect_swing_points_vectorized(df, window)

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
            if prev_candle['high'] < next_candle['low'] and prev_candle['high'] > 0:
                gap_size = (next_candle['low'] - prev_candle['high']) / prev_candle['high']

                # Only mark if significant AND near bullish structure
                if gap_size >= significant_fvg_threshold and has_nearby_bullish_structure:
                    df_fvg.loc[df_fvg.index[i], 'fvg_bullish'] = True
                    df_fvg.loc[df_fvg.index[i], 'fvg_low'] = prev_candle['high']
                    df_fvg.loc[df_fvg.index[i], 'fvg_high'] = next_candle['low']

            # Bearish FVG: Gap down (prev.low > next.high)
            elif prev_candle['low'] > next_candle['high'] and prev_candle['low'] > 0:
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

    # ==================== NEW ICT FEATURES ====================

    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks (OB) - Institutional entry zones
        
        Bullish OB: Last bearish candle before a bullish move that breaks structure
        Bearish OB: Last bullish candle before a bearish move that breaks structure
        
        Returns:
            DataFrame with order block zones
        """
        df_ob = df.copy()
        
        # Ensure we have structure signals
        if 'bos_bullish' not in df_ob.columns:
            df_ob = self.detect_bos(df_ob)
        if 'choch_bullish' not in df_ob.columns:
            df_ob = self.detect_choch(df_ob)
        
        # Initialize columns
        df_ob['ob_bullish'] = False
        df_ob['ob_bearish'] = False
        df_ob['ob_bullish_high'] = np.nan
        df_ob['ob_bullish_low'] = np.nan
        df_ob['ob_bearish_high'] = np.nan
        df_ob['ob_bearish_low'] = np.nan
        
        # Detect candle direction
        is_bullish_candle = df_ob['close'] > df_ob['open']
        is_bearish_candle = df_ob['close'] < df_ob['open']
        
        # Find bullish structure breaks
        bullish_break = df_ob['bos_bullish'] | df_ob['choch_bullish']
        bearish_break = df_ob['bos_bearish'] | df_ob['choch_bearish']
        
        # Lookback window for finding the order block candle
        ob_lookback = 10
        
        for i in range(ob_lookback, len(df_ob)):
            # Bullish OB: Find last bearish candle before bullish break
            if bullish_break.iloc[i]:
                for j in range(1, ob_lookback + 1):
                    if i - j >= 0 and is_bearish_candle.iloc[i - j]:
                        df_ob.iloc[i - j, df_ob.columns.get_loc('ob_bullish')] = True
                        df_ob.iloc[i - j, df_ob.columns.get_loc('ob_bullish_high')] = df_ob.iloc[i - j]['high']
                        df_ob.iloc[i - j, df_ob.columns.get_loc('ob_bullish_low')] = df_ob.iloc[i - j]['low']
                        break
            
            # Bearish OB: Find last bullish candle before bearish break
            if bearish_break.iloc[i]:
                for j in range(1, ob_lookback + 1):
                    if i - j >= 0 and is_bullish_candle.iloc[i - j]:
                        df_ob.iloc[i - j, df_ob.columns.get_loc('ob_bearish')] = True
                        df_ob.iloc[i - j, df_ob.columns.get_loc('ob_bearish_high')] = df_ob.iloc[i - j]['high']
                        df_ob.iloc[i - j, df_ob.columns.get_loc('ob_bearish_low')] = df_ob.iloc[i - j]['low']
                        break
        
        logger.debug(f"Detected {df_ob['ob_bullish'].sum()} bullish and {df_ob['ob_bearish'].sum()} bearish Order Blocks")
        
        return df_ob
    
    def detect_ote_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect OTE (Optimal Trade Entry) zones using Fibonacci retracements
        
        OTE Zone: 0.618 - 0.786 retracement of the swing
        - For longs: Price in discount (below 0.5), ideally in OTE zone
        - For shorts: Price in premium (above 0.5), ideally in OTE zone
        
        Returns:
            DataFrame with OTE zone information
        """
        df_ote = df.copy()
        
        if 'swing_high' not in df_ote.columns:
            df_ote = self.detect_swing_points(df_ote)
        
        # Initialize columns
        df_ote['in_discount'] = False  # Below 50% (good for longs)
        df_ote['in_premium'] = False   # Above 50% (good for shorts)
        df_ote['in_ote_long'] = False  # In OTE zone for long entry
        df_ote['in_ote_short'] = False # In OTE zone for short entry
        df_ote['equilibrium'] = np.nan # 50% level
        df_ote['ote_low'] = np.nan     # 0.618 level
        df_ote['ote_high'] = np.nan    # 0.786 level
        
        # Get recent swing points for each row
        swing_highs = df_ote['swing_high'].dropna()
        swing_lows = df_ote['swing_low'].dropna()
        
        if len(swing_highs) < 1 or len(swing_lows) < 1:
            return df_ote
        
        # For each row, find the most recent swing high and low
        for i in range(len(df_ote)):
            current_idx = df_ote.index[i]
            current_close = df_ote.iloc[i]['close']
            
            # Get prior swings
            prior_highs = swing_highs[swing_highs.index < current_idx]
            prior_lows = swing_lows[swing_lows.index < current_idx]
            
            if len(prior_highs) == 0 or len(prior_lows) == 0:
                continue
            
            recent_high = prior_highs.iloc[-1]
            recent_low = prior_lows.iloc[-1]
            
            # Skip if high <= low (invalid range)
            if recent_high <= recent_low:
                continue
            
            swing_range = recent_high - recent_low
            
            # Calculate Fibonacci levels
            equilibrium = recent_low + 0.5 * swing_range
            ote_618 = recent_high - self.ote_fib_low * swing_range  # 0.618 retracement from high
            ote_786 = recent_high - self.ote_fib_high * swing_range  # 0.786 retracement from high
            
            df_ote.iloc[i, df_ote.columns.get_loc('equilibrium')] = equilibrium
            df_ote.iloc[i, df_ote.columns.get_loc('ote_low')] = min(ote_618, ote_786)
            df_ote.iloc[i, df_ote.columns.get_loc('ote_high')] = max(ote_618, ote_786)
            
            # Determine zone
            if current_close < equilibrium:
                df_ote.iloc[i, df_ote.columns.get_loc('in_discount')] = True
                # Check if in OTE zone for long (price between 0.618 and 0.786 from low)
                if min(ote_618, ote_786) <= current_close <= max(ote_618, ote_786):
                    df_ote.iloc[i, df_ote.columns.get_loc('in_ote_long')] = True
            else:
                df_ote.iloc[i, df_ote.columns.get_loc('in_premium')] = True
                # For shorts, OTE is from swing low side
                short_ote_618 = recent_low + self.ote_fib_low * swing_range
                short_ote_786 = recent_low + self.ote_fib_high * swing_range
                if min(short_ote_618, short_ote_786) <= current_close <= max(short_ote_618, short_ote_786):
                    df_ote.iloc[i, df_ote.columns.get_loc('in_ote_short')] = True
        
        logger.debug(f"OTE zones: {df_ote['in_ote_long'].sum()} long, {df_ote['in_ote_short'].sum()} short opportunities")
        
        return df_ote
    
    def detect_fvg_mitigation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Track FVG mitigation - when price returns and fills the gap
        
        Mitigated FVGs should not generate new signals (they're "used up")
        
        Returns:
            DataFrame with fvg_mitigated column
        """
        df_mit = df.copy()
        
        if 'fvg_bullish' not in df_mit.columns:
            df_mit = self.detect_fvg(df_mit)
        
        df_mit['fvg_mitigated'] = False
        df_mit['fvg_fresh'] = False  # Fresh = not yet mitigated
        
        # Track active FVGs
        active_bullish_fvgs = []  # List of (index, high, low)
        active_bearish_fvgs = []
        
        for i in range(len(df_mit)):
            current_high = df_mit.iloc[i]['high']
            current_low = df_mit.iloc[i]['low']
            
            # Add new FVGs
            if df_mit.iloc[i]['fvg_bullish']:
                fvg_high = df_mit.iloc[i]['fvg_high']
                fvg_low = df_mit.iloc[i]['fvg_low']
                if pd.notna(fvg_high) and pd.notna(fvg_low):
                    active_bullish_fvgs.append((df_mit.index[i], fvg_high, fvg_low))
            
            if df_mit.iloc[i]['fvg_bearish']:
                fvg_high = df_mit.iloc[i]['fvg_high']
                fvg_low = df_mit.iloc[i]['fvg_low']
                if pd.notna(fvg_high) and pd.notna(fvg_low):
                    active_bearish_fvgs.append((df_mit.index[i], fvg_high, fvg_low))
            
            # Check for mitigation (price fills the gap)
            mitigated_bullish = []
            for fvg_idx, fvg_high, fvg_low in active_bullish_fvgs:
                # Bullish FVG mitigated when price drops into the gap
                if current_low <= fvg_high:  # Price entered the FVG zone
                    df_mit.loc[fvg_idx, 'fvg_mitigated'] = True
                    mitigated_bullish.append((fvg_idx, fvg_high, fvg_low))
            
            mitigated_bearish = []
            for fvg_idx, fvg_high, fvg_low in active_bearish_fvgs:
                # Bearish FVG mitigated when price rises into the gap
                if current_high >= fvg_low:  # Price entered the FVG zone
                    df_mit.loc[fvg_idx, 'fvg_mitigated'] = True
                    mitigated_bearish.append((fvg_idx, fvg_high, fvg_low))
            
            # Remove mitigated FVGs from active list
            for m in mitigated_bullish:
                active_bullish_fvgs.remove(m)
            for m in mitigated_bearish:
                active_bearish_fvgs.remove(m)
        
        # Fresh FVGs are those not yet mitigated
        df_mit['fvg_fresh'] = (
            (df_mit['fvg_bullish'] | df_mit['fvg_bearish']) & 
            ~df_mit['fvg_mitigated']
        )
        
        logger.debug(f"FVG mitigation: {df_mit['fvg_mitigated'].sum()} mitigated, {df_mit['fvg_fresh'].sum()} fresh")
        
        return df_mit
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect liquidity sweeps - when price takes out stops then reverses
        
        Bullish sweep: Price drops below swing low, then closes back above (sweep and reverse)
        Bearish sweep: Price rises above swing high, then closes back below
        
        Returns:
            DataFrame with liquidity sweep signals
        """
        df_sweep = df.copy()
        
        if 'swing_high' not in df_sweep.columns:
            df_sweep = self.detect_swing_points(df_sweep)
        
        df_sweep['sweep_bullish'] = False  # Swept lows, potential reversal up
        df_sweep['sweep_bearish'] = False  # Swept highs, potential reversal down
        
        swing_highs = df_sweep['swing_high'].dropna()
        swing_lows = df_sweep['swing_low'].dropna()
        
        # Lookback for recent swing levels
        sweep_lookback = 20
        
        for i in range(sweep_lookback, len(df_sweep)):
            current_idx = df_sweep.index[i]
            current_high = df_sweep.iloc[i]['high']
            current_low = df_sweep.iloc[i]['low']
            current_close = df_sweep.iloc[i]['close']
            current_open = df_sweep.iloc[i]['open']
            
            # Get recent swing levels (within lookback window)
            lookback_start = df_sweep.index[max(0, i - sweep_lookback)]
            
            recent_swing_highs = swing_highs[
                (swing_highs.index >= lookback_start) & 
                (swing_highs.index < current_idx)
            ]
            recent_swing_lows = swing_lows[
                (swing_lows.index >= lookback_start) & 
                (swing_lows.index < current_idx)
            ]
            
            # Bullish sweep: Low swept below recent swing low, but closed above it
            if len(recent_swing_lows) > 0:
                recent_low_level = recent_swing_lows.iloc[-1]
                # Wick went below, body closed above = sweep
                if current_low < recent_low_level and current_close > recent_low_level:
                    df_sweep.iloc[i, df_sweep.columns.get_loc('sweep_bullish')] = True
            
            # Bearish sweep: High swept above recent swing high, but closed below it
            if len(recent_swing_highs) > 0:
                recent_high_level = recent_swing_highs.iloc[-1]
                # Wick went above, body closed below = sweep
                if current_high > recent_high_level and current_close < recent_high_level:
                    df_sweep.iloc[i, df_sweep.columns.get_loc('sweep_bearish')] = True
        
        logger.debug(f"Liquidity sweeps: {df_sweep['sweep_bullish'].sum()} bullish, {df_sweep['sweep_bearish'].sum()} bearish")
        
        return df_sweep
    
    def analyze_smc_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete ENHANCED SMC analysis pipeline
        Includes all ICT concepts: CHOCH, BOS, FVG, OB, OTE, Sweeps
        
        Returns:
            DataFrame with all SMC indicators
        """
        logger.debug("Starting enhanced SMC analysis")
        
        df_smc = df.copy()
        
        # Core SMC patterns
        df_smc = self.detect_swing_points(df_smc)
        df_smc = self.detect_market_structure(df_smc)
        df_smc = self.detect_choch(df_smc)
        df_smc = self.detect_bos(df_smc)
        df_smc = self.detect_fvg(df_smc)
        df_smc = self.detect_liquidity_zones(df_smc)
        
        # Enhanced ICT features
        df_smc = self.detect_order_blocks(df_smc)
        df_smc = self.detect_ote_zones(df_smc)
        df_smc = self.detect_fvg_mitigation(df_smc)
        df_smc = self.detect_liquidity_sweeps(df_smc)
        
        logger.debug("Enhanced SMC analysis completed")
        
        return df_smc
    
    def get_smc_signals_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ENHANCED trading signals based on SMC + ICT concepts
        
        Long signal criteria:
        - Bullish CHOCH/BOS OR liquidity sweep bullish
        - Price in discount zone (below 50%) OR in OTE long zone
        - Near bullish order block OR fresh bullish FVG
        
        Short signal criteria:
        - Bearish CHOCH/BOS OR liquidity sweep bearish
        - Price in premium zone (above 50%) OR in OTE short zone
        - Near bearish order block OR fresh bearish FVG
        
        Returns:
            DataFrame with buy/sell signals and confidence scores
        """
        df_signals = df.copy()
        
        # Ensure all indicators are present
        required_cols = ['choch_bullish', 'bos_bullish', 'in_discount', 'ob_bullish', 'fvg_fresh']
        missing = [c for c in required_cols if c not in df_signals.columns]
        if missing:
            df_signals = self.analyze_smc_enhanced(df_signals)
        
        df_signals['smc_signal'] = 0
        df_signals['smc_confidence'] = 0.0
        
        # Long signal components (each adds to confidence)
        long_structure = df_signals['choch_bullish'] | df_signals['bos_bullish']
        long_sweep = df_signals.get('sweep_bullish', False)
        long_zone = df_signals.get('in_discount', False) | df_signals.get('in_ote_long', False)
        long_ob = df_signals.get('ob_bullish', False)
        long_fvg = df_signals['fvg_bullish'] & ~df_signals.get('fvg_mitigated', False)
        
        # Short signal components
        short_structure = df_signals['choch_bearish'] | df_signals['bos_bearish']
        short_sweep = df_signals.get('sweep_bearish', False)
        short_zone = df_signals.get('in_premium', False) | df_signals.get('in_ote_short', False)
        short_ob = df_signals.get('ob_bearish', False)
        short_fvg = df_signals['fvg_bearish'] & ~df_signals.get('fvg_mitigated', False)
        
        # Calculate confidence scores
        for i in range(len(df_signals)):
            long_conf = 0
            short_conf = 0
            
            # Structure break is required (40%)
            if long_structure.iloc[i]:
                long_conf += 0.40
            if short_structure.iloc[i]:
                short_conf += 0.40
            
            # Liquidity sweep (25%)
            if isinstance(long_sweep, pd.Series) and long_sweep.iloc[i]:
                long_conf += 0.25
            if isinstance(short_sweep, pd.Series) and short_sweep.iloc[i]:
                short_conf += 0.25
            
            # In correct zone (20%)
            if isinstance(long_zone, pd.Series) and long_zone.iloc[i]:
                long_conf += 0.20
            if isinstance(short_zone, pd.Series) and short_zone.iloc[i]:
                short_conf += 0.20
            
            # Order block (10%)
            if isinstance(long_ob, pd.Series) and long_ob.iloc[i]:
                long_conf += 0.10
            if isinstance(short_ob, pd.Series) and short_ob.iloc[i]:
                short_conf += 0.10
            
            # Fresh FVG (5%)
            if isinstance(long_fvg, pd.Series) and long_fvg.iloc[i]:
                long_conf += 0.05
            if isinstance(short_fvg, pd.Series) and short_fvg.iloc[i]:
                short_conf += 0.05
            
            # Set signal based on confidence
            if long_conf >= 0.40 and long_conf > short_conf:
                df_signals.iloc[i, df_signals.columns.get_loc('smc_signal')] = 1
                df_signals.iloc[i, df_signals.columns.get_loc('smc_confidence')] = long_conf
            elif short_conf >= 0.40 and short_conf > long_conf:
                df_signals.iloc[i, df_signals.columns.get_loc('smc_signal')] = -1
                df_signals.iloc[i, df_signals.columns.get_loc('smc_confidence')] = short_conf
        
        logger.info(
            f"Enhanced SMC signals: {(df_signals['smc_signal'] == 1).sum()} long, "
            f"{(df_signals['smc_signal'] == -1).sum()} short"
        )
        
        return df_signals
