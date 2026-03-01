"""
Multi-Timeframe Pattern Labeler
Generates training labels using TOP-DOWN SMC analysis:

FLOW:
1D (Daily)  → Swing highs/lows, overall bias
    ↓
4H          → BOS, CHOCH, FVG relative to daily structure
    ↓
1H          → More detailed patterns within 4H context
    ↓
15M         → Entry patterns, trade execution, OUTCOME LABELING

ML learns: "When 1D bullish + 4H BOS + 1H FVG + 15M entry = X% win rate"
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
from src.indicators.smc_detector import SMCDetector
from src.indicators.ta_indicators import TechnicalIndicators
from src.utils.logger import get_logger

logger = get_logger()


class MultiTimeframePatternLabeler:
    """
    Generate training labels using multi-timeframe SMC analysis.
    
    TOP-DOWN APPROACH:
    - Daily: Identifies swing structure and overall bias
    - 4H: Identifies BOS, CHOCH, FVG within daily context
    - 1H: Refines patterns within 4H context  
    - 15M: Entry signals and trade outcomes for labeling
    """

    def __init__(self, config: Dict):
        self.config = config
        self.smc_detector = SMCDetector(config)
        self.ta_indicators = TechnicalIndicators(config)
        
        # Risk parameters for labeling
        self.risk_config = config.get('risk', {})
        self.sl_atr_mult = self.risk_config.get('stop_loss_atr_multiplier', 1.5)
        self.tp_rr_ratio = self.risk_config.get('take_profit_rr_ratio', 2.0)
        
        # Timeframe hierarchy
        self.timeframes = {
            'bias': '1d',       # Daily for swing/bias
            'structure': '4h',  # 4H for market structure
            'refinement': '1h', # 1H for pattern refinement
            'entry': '15m'      # 15M for entry signals
        }
        
        logger.info(f"MultiTimeframePatternLabeler initialized")
        logger.info(f"Timeframe hierarchy: {self.timeframes}")

    def analyze_all_timeframes(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze SMC patterns on all timeframes.
        
        Returns:
            Dict with analyzed DataFrames for each timeframe
        """
        logger.info("Analyzing patterns on all timeframes...")
        
        # 1D: Swing structure and bias
        logger.info("Analyzing 1D (Daily) for swing structure...")
        df_1d_analyzed = self.smc_detector.analyze_smc(df_1d)
        df_1d_analyzed = self.ta_indicators.add_all_indicators(df_1d_analyzed)
        df_1d_analyzed['tf'] = '1d'
        
        # Identify swing highs/lows on daily
        df_1d_analyzed = self._mark_swing_zones(df_1d_analyzed, lookback=15)
        
        # 4H: Market structure
        logger.info("Analyzing 4H for market structure...")
        df_4h_analyzed = self.smc_detector.analyze_smc(df_4h)
        df_4h_analyzed = self.ta_indicators.add_all_indicators(df_4h_analyzed)
        df_4h_analyzed['tf'] = '4h'
        
        # 1H: Pattern refinement
        logger.info("Analyzing 1H for pattern refinement...")
        df_1h_analyzed = self.smc_detector.analyze_smc(df_1h)
        df_1h_analyzed = self.ta_indicators.add_all_indicators(df_1h_analyzed)
        df_1h_analyzed['tf'] = '1h'
        
        # 15M: Entry signals
        logger.info("Analyzing 15M for entry signals...")
        df_15m_analyzed = self.smc_detector.analyze_smc(df_15m)
        df_15m_analyzed = self.ta_indicators.add_all_indicators(df_15m_analyzed)
        df_15m_analyzed['tf'] = '15m'
        
        return {
            '1d': df_1d_analyzed,
            '4h': df_4h_analyzed,
            '1h': df_1h_analyzed,
            '15m': df_15m_analyzed
        }

    def _mark_swing_zones(self, df: pd.DataFrame, lookback: int = 15) -> pd.DataFrame:
        """
        Mark swing highs and lows on daily timeframe.
        These define the overall bias zones.
        """
        df = df.copy()
        
        # Find highest high and lowest low in lookback period
        df['swing_high'] = df['high'].rolling(lookback).max()
        df['swing_low'] = df['low'].rolling(lookback).min()
        
        # Current price position relative to swing
        df['price_vs_swing_high'] = (df['close'] - df['swing_high']) / df['swing_high']
        df['price_vs_swing_low'] = (df['close'] - df['swing_low']) / df['swing_low']
        
        # Swing bias
        df['swing_bias'] = 'neutral'
        df.loc[df['close'] > df['swing_high'].shift(1), 'swing_bias'] = 'bullish_breakout'
        df.loc[df['close'] < df['swing_low'].shift(1), 'swing_bias'] = 'bearish_breakout'
        df.loc[(df['close'] > df['swing_low']) & (df['close'] < df['swing_high']), 'swing_bias'] = 'range'
        
        return df

    def create_multi_tf_features(
        self,
        analyzed_data: Dict[str, pd.DataFrame],
        target_timestamp: pd.Timestamp
    ) -> Dict:
        """
        Create features from all timeframes for a specific 15M candle.
        
        Maps higher TF context to the 15M entry point.
        """
        features = {}
        
        # Get data for each timeframe at or before target timestamp
        df_1d = analyzed_data['1d']
        df_4h = analyzed_data['4h']
        df_1h = analyzed_data['1h']
        df_15m = analyzed_data['15m']
        
        # Find the relevant candle on each timeframe
        try:
            # 1D features (most recent daily candle before this 15m)
            d1_mask = df_1d.index <= target_timestamp
            if d1_mask.any():
                d1_row = df_1d[d1_mask].iloc[-1]
                features.update(self._extract_tf_features(d1_row, '1d'))
            else:
                features.update(self._empty_tf_features('1d'))
            
            # 4H features
            h4_mask = df_4h.index <= target_timestamp
            if h4_mask.any():
                h4_row = df_4h[h4_mask].iloc[-1]
                features.update(self._extract_tf_features(h4_row, '4h'))
            else:
                features.update(self._empty_tf_features('4h'))
            
            # 1H features
            h1_mask = df_1h.index <= target_timestamp
            if h1_mask.any():
                h1_row = df_1h[h1_mask].iloc[-1]
                features.update(self._extract_tf_features(h1_row, '1h'))
            else:
                features.update(self._empty_tf_features('1h'))
            
            # 15M features (current candle)
            m15_mask = df_15m.index == target_timestamp
            if m15_mask.any():
                m15_row = df_15m[m15_mask].iloc[-1]
                features.update(self._extract_tf_features(m15_row, '15m'))
            else:
                features.update(self._empty_tf_features('15m'))
            
            # === MULTI-TF CONFLUENCE FEATURES ===
            features['mtf_all_bullish'] = int(
                (features.get('1d_choch_bullish', 0) or features.get('1d_bos_bullish', 0)) and
                (features.get('4h_choch_bullish', 0) or features.get('4h_bos_bullish', 0)) and
                (features.get('1h_choch_bullish', 0) or features.get('1h_fvg_bullish', 0)) and
                (features.get('15m_choch_bullish', 0) or features.get('15m_fvg_bullish', 0))
            )
            
            features['mtf_all_bearish'] = int(
                (features.get('1d_choch_bearish', 0) or features.get('1d_bos_bearish', 0)) and
                (features.get('4h_choch_bearish', 0) or features.get('4h_bos_bearish', 0)) and
                (features.get('1h_choch_bearish', 0) or features.get('1h_fvg_bearish', 0)) and
                (features.get('15m_choch_bearish', 0) or features.get('15m_fvg_bearish', 0))
            )
            
            # Count how many TFs agree on direction
            bullish_tfs = sum([
                1 if features.get('1d_bias', 0) > 0 else 0,
                1 if features.get('4h_structure_bullish', 0) else 0,
                1 if features.get('1h_structure_bullish', 0) else 0,
                1 if features.get('15m_choch_bullish', 0) or features.get('15m_bos_bullish', 0) else 0
            ])
            bearish_tfs = sum([
                1 if features.get('1d_bias', 0) < 0 else 0,
                1 if features.get('4h_structure_bearish', 0) else 0,
                1 if features.get('1h_structure_bearish', 0) else 0,
                1 if features.get('15m_choch_bearish', 0) or features.get('15m_bos_bearish', 0) else 0
            ])
            
            features['mtf_bullish_count'] = bullish_tfs
            features['mtf_bearish_count'] = bearish_tfs
            features['mtf_confluence_strength'] = max(bullish_tfs, bearish_tfs)  # 0-4
            
        except Exception as e:
            logger.error(f"Error creating multi-TF features: {e}")
            features = self._empty_all_features()
        
        return features

    def _extract_tf_features(self, row: pd.Series, tf_prefix: str) -> Dict:
        """Extract features from a single timeframe row."""
        features = {}
        
        # Pattern presence
        features[f'{tf_prefix}_choch_bullish'] = int(row.get('choch_bullish', False))
        features[f'{tf_prefix}_choch_bearish'] = int(row.get('choch_bearish', False))
        features[f'{tf_prefix}_bos_bullish'] = int(row.get('bos_bullish', False))
        features[f'{tf_prefix}_bos_bearish'] = int(row.get('bos_bearish', False))
        features[f'{tf_prefix}_fvg_bullish'] = int(row.get('fvg_bullish', False))
        features[f'{tf_prefix}_fvg_bearish'] = int(row.get('fvg_bearish', False))
        
        # Market structure
        structure = row.get('market_structure', 'neutral')
        features[f'{tf_prefix}_structure_bullish'] = int(structure == 'bullish')
        features[f'{tf_prefix}_structure_bearish'] = int(structure == 'bearish')
        
        # EMA trend
        ema_50 = row.get('ema_50', 0)
        ema_200 = row.get('ema_200', 0)
        features[f'{tf_prefix}_ema_bullish'] = int(ema_50 > ema_200) if ema_50 and ema_200 else 0
        features[f'{tf_prefix}_ema_bearish'] = int(ema_50 < ema_200) if ema_50 and ema_200 else 0
        
        # RSI
        rsi = row.get('rsi', 50)
        features[f'{tf_prefix}_rsi'] = float(rsi) / 100 if rsi else 0.5
        features[f'{tf_prefix}_rsi_bullish'] = int(50 < rsi < 70) if rsi else 0
        features[f'{tf_prefix}_rsi_bearish'] = int(30 < rsi < 50) if rsi else 0
        features[f'{tf_prefix}_rsi_overbought'] = int(rsi > 70) if rsi else 0
        features[f'{tf_prefix}_rsi_oversold'] = int(rsi < 30) if rsi else 0
        
        # Swing bias (only for 1D)
        if tf_prefix == '1d':
            swing_bias = row.get('swing_bias', 'neutral')
            features[f'{tf_prefix}_bias'] = 1 if 'bullish' in swing_bias else (-1 if 'bearish' in swing_bias else 0)
            features[f'{tf_prefix}_in_range'] = int(swing_bias == 'range')
        
        # Pattern confluence on this TF
        pattern_count = (
            features[f'{tf_prefix}_choch_bullish'] + features[f'{tf_prefix}_choch_bearish'] +
            features[f'{tf_prefix}_bos_bullish'] + features[f'{tf_prefix}_bos_bearish'] +
            features[f'{tf_prefix}_fvg_bullish'] + features[f'{tf_prefix}_fvg_bearish']
        )
        features[f'{tf_prefix}_pattern_count'] = pattern_count
        
        return features

    def _empty_tf_features(self, tf_prefix: str) -> Dict:
        """Return empty features for a timeframe."""
        features = {
            f'{tf_prefix}_choch_bullish': 0,
            f'{tf_prefix}_choch_bearish': 0,
            f'{tf_prefix}_bos_bullish': 0,
            f'{tf_prefix}_bos_bearish': 0,
            f'{tf_prefix}_fvg_bullish': 0,
            f'{tf_prefix}_fvg_bearish': 0,
            f'{tf_prefix}_structure_bullish': 0,
            f'{tf_prefix}_structure_bearish': 0,
            f'{tf_prefix}_ema_bullish': 0,
            f'{tf_prefix}_ema_bearish': 0,
            f'{tf_prefix}_rsi': 0.5,
            f'{tf_prefix}_rsi_bullish': 0,
            f'{tf_prefix}_rsi_bearish': 0,
            f'{tf_prefix}_rsi_overbought': 0,
            f'{tf_prefix}_rsi_oversold': 0,
            f'{tf_prefix}_pattern_count': 0,
        }
        # 1D has extra swing bias features
        if tf_prefix == '1d':
            features['1d_bias'] = 0
            features['1d_in_range'] = 0
        return features

    def _empty_all_features(self) -> Dict:
        """Return empty features for all timeframes."""
        features = {}
        for tf in ['1d', '4h', '1h', '15m']:
            features.update(self._empty_tf_features(tf))
        features['mtf_all_bullish'] = 0
        features['mtf_all_bearish'] = 0
        features['mtf_bullish_count'] = 0
        features['mtf_bearish_count'] = 0
        features['mtf_confluence_strength'] = 0
        return features

    def generate_labels(
        self,
        analyzed_data: Dict[str, pd.DataFrame],
        lookforward_15m: int = 40  # 40 x 15m = 10 hours for outcome
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate training data with multi-TF features and outcome labels.
        
        CORRECT FLOW:
        - 1D: Swing direction (bias) - sets allowed trade direction
        - 4H: Pattern trigger (BOS/CHOCH/FVG) - THE SIGNAL SOURCE
        - 1H: Refines/confirms the 4H pattern
        - 15M: Entry timing and outcome labeling
        
        We iterate through 4H candles (where patterns are detected),
        then find the corresponding 15M entry and label based on outcome.
        
        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Generating multi-timeframe training labels (4H pattern → 15M entry)...")
        
        df_1d = analyzed_data['1d']
        df_4h = analyzed_data['4h']
        df_1h = analyzed_data['1h']
        df_15m = analyzed_data['15m']
        
        # Get the time range where 15M data exists (we need this for outcome labels)
        if len(df_15m) == 0:
            logger.error("No 15M data available!")
            return pd.DataFrame(), pd.Series(dtype=int)
        
        m15_start = df_15m.index[0]
        lookforward_hours = (lookforward_15m * 15) / 60  # Convert bars to hours
        m15_end = df_15m.index[-1] - pd.Timedelta(hours=lookforward_hours)  # Leave room for outcome
        
        logger.info(f"  Looking for 4H patterns between {m15_start} and {m15_end}")
        
        all_features = []
        all_labels = []
        all_timestamps = []
        patterns_found = 0
        
        # Iterate through 4H candles (this is where main patterns are detected)
        for i in range(10, len(df_4h) - 1):
            h4_timestamp = df_4h.index[i]
            
            # Skip if this 4H candle is outside our 15M data range
            if h4_timestamp < m15_start or h4_timestamp > m15_end:
                continue
            
            h4_row = df_4h.iloc[i]
            
            # Check if 4H has a pattern (BOS, CHOCH, or FVG)
            has_4h_pattern = (
                h4_row.get('choch_bullish', False) or 
                h4_row.get('choch_bearish', False) or
                h4_row.get('bos_bullish', False) or 
                h4_row.get('bos_bearish', False) or
                h4_row.get('fvg_bullish', False) or 
                h4_row.get('fvg_bearish', False)
            )
            
            if not has_4h_pattern:
                continue
            
            # Determine 4H pattern direction
            is_4h_bullish = (
                h4_row.get('choch_bullish', False) or 
                h4_row.get('bos_bullish', False) or
                h4_row.get('fvg_bullish', False)
            )
            is_4h_bearish = (
                h4_row.get('choch_bearish', False) or 
                h4_row.get('bos_bearish', False) or
                h4_row.get('fvg_bearish', False)
            )
            
            # Find the 1D swing bias at this time
            d1_mask = df_1d.index <= h4_timestamp
            if not d1_mask.any():
                continue
            d1_row = df_1d[d1_mask].iloc[-1]
            
            # Get 1D swing bias
            swing_bias = d1_row.get('swing_bias', 'neutral')
            d1_bullish = 'bullish' in swing_bias or d1_row.get('bos_bullish', False) or d1_row.get('choch_bullish', False)
            d1_bearish = 'bearish' in swing_bias or d1_row.get('bos_bearish', False) or d1_row.get('choch_bearish', False)
            
            # Check 1H confirmation within the 4H candle window
            h1_mask = (df_1h.index > h4_timestamp - pd.Timedelta(hours=4)) & (df_1h.index <= h4_timestamp)
            h1_confirms_bullish = False
            h1_confirms_bearish = False
            if h1_mask.any():
                h1_subset = df_1h[h1_mask]
                h1_confirms_bullish = h1_subset['choch_bullish'].any() or h1_subset['bos_bullish'].any() or h1_subset['fvg_bullish'].any()
                h1_confirms_bearish = h1_subset['choch_bearish'].any() or h1_subset['bos_bearish'].any() or h1_subset['fvg_bearish'].any()
            
            # Find 15M entry point (after 4H pattern forms)
            m15_mask = (df_15m.index > h4_timestamp) & (df_15m.index <= h4_timestamp + pd.Timedelta(hours=4))
            if not m15_mask.any():
                continue
            
            m15_entry_idx = df_15m[m15_mask].index[0]
            m15_entry_row = df_15m.loc[m15_entry_idx]
            entry_price = m15_entry_row['close']
            atr = m15_entry_row.get('atr', entry_price * 0.02)
            
            # Get future 15M bars for outcome labeling
            future_start = df_15m.index.get_loc(m15_entry_idx)
            future_end = min(future_start + lookforward_15m, len(df_15m))
            future_rows = df_15m.iloc[future_start:future_end]
            
            if len(future_rows) < 5:
                continue
            
            # Create multi-TF features at entry time
            features = self.create_multi_tf_features(analyzed_data, m15_entry_idx)
            
            # Add confluence indicators
            features['d1_aligns_with_4h'] = int(
                (d1_bullish and is_4h_bullish) or (d1_bearish and is_4h_bearish)
            )
            features['h1_confirms_4h'] = int(
                (h1_confirms_bullish and is_4h_bullish) or (h1_confirms_bearish and is_4h_bearish)
            )
            features['full_mtf_alignment'] = int(
                features['d1_aligns_with_4h'] and features['h1_confirms_4h']
            )
            
            # Calculate trade outcome
            if is_4h_bullish:
                # Long trade
                sl = entry_price - (atr * self.sl_atr_mult)
                tp = entry_price + (atr * self.sl_atr_mult * self.tp_rr_ratio)
                
                label = 0  # default neutral
                for _, future_row in future_rows.iterrows():
                    if future_row['low'] <= sl:
                        label = -1  # Hit SL = loss
                        break
                    elif future_row['high'] >= tp:
                        label = 1  # Hit TP = win
                        break
                        
            elif is_4h_bearish:
                # Short trade
                sl = entry_price + (atr * self.sl_atr_mult)
                tp = entry_price - (atr * self.sl_atr_mult * self.tp_rr_ratio)
                
                label = 0
                for _, future_row in future_rows.iterrows():
                    if future_row['high'] >= sl:
                        label = -1  # Hit SL = loss
                        break
                    elif future_row['low'] <= tp:
                        label = 1  # Hit TP = win
                        break
            else:
                label = 0
            
            all_features.append(features)
            all_labels.append(label)
            all_timestamps.append(m15_entry_idx)
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features, index=all_timestamps)
        labels = pd.Series(all_labels, index=all_timestamps, name='label')
        
        # Stats
        wins = (labels == 1).sum()
        losses = (labels == -1).sum()
        neutral = (labels == 0).sum()
        total = len(labels)
        
        if total > 0:
            logger.info(f"Generated {total} labeled samples from 4H patterns:")
            logger.info(f"  - Wins: {wins} ({wins/total*100:.1f}%)")
            logger.info(f"  - Losses: {losses} ({losses/total*100:.1f}%)")
            logger.info(f"  - Neutral: {neutral} ({neutral/total*100:.1f}%)")
            
            # MTF alignment stats
            if 'full_mtf_alignment' in features_df.columns:
                aligned = features_df['full_mtf_alignment'].sum()
                logger.info(f"  - Full MTF Alignment: {aligned} ({aligned/total*100:.1f}%)")
        else:
            logger.warning("No labeled samples generated!")
        
        return features_df, labels

    def analyze_pattern_effectiveness(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series
    ) -> Dict:
        """
        Analyze which multi-TF pattern combinations are most effective.
        """
        stats = {}
        
        # Overall win rate
        non_neutral = labels[labels != 0]
        if len(non_neutral) > 0:
            stats['overall_win_rate'] = (non_neutral == 1).sum() / len(non_neutral)
            stats['total_patterns'] = len(labels)
            stats['tradeable_patterns'] = len(non_neutral)
        
        # Win rate by MTF confluence
        for confluence_level in range(1, 5):
            mask = features_df['mtf_confluence_strength'] == confluence_level
            if mask.any():
                subset_labels = labels[mask]
                non_neutral_subset = subset_labels[subset_labels != 0]
                if len(non_neutral_subset) > 0:
                    win_rate = (non_neutral_subset == 1).sum() / len(non_neutral_subset)
                    stats[f'confluence_{confluence_level}_win_rate'] = win_rate
                    stats[f'confluence_{confluence_level}_count'] = len(non_neutral_subset)
        
        # Win rate when all TFs align
        all_bullish_mask = features_df['mtf_all_bullish'] == 1
        if all_bullish_mask.any():
            subset = labels[all_bullish_mask]
            non_neutral_subset = subset[subset != 0]
            if len(non_neutral_subset) > 0:
                stats['all_tf_bullish_win_rate'] = (non_neutral_subset == 1).sum() / len(non_neutral_subset)
                stats['all_tf_bullish_count'] = len(non_neutral_subset)
        
        all_bearish_mask = features_df['mtf_all_bearish'] == 1
        if all_bearish_mask.any():
            subset = labels[all_bearish_mask]
            non_neutral_subset = subset[subset != 0]
            if len(non_neutral_subset) > 0:
                stats['all_tf_bearish_win_rate'] = (non_neutral_subset == 1).sum() / len(non_neutral_subset)
                stats['all_tf_bearish_count'] = len(non_neutral_subset)
        
        return stats
