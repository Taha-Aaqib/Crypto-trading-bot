"""
Main Trading Strategy
Integrates SMC, Technical Indicators, and Sentiment
"""

import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
from src.indicators.smc_detector import SMCDetector
from src.indicators.ta_indicators import TechnicalIndicators
from src.sentiment.sentiment_filter import SentimentFilter
from src.utils.logger import get_logger

logger = get_logger()


class TradingStrategy:
    """
    Main trading strategy combining all components:
    - Multi-timeframe SMC analysis (1D bias, 4H structure, 15M entry)
    - EMA trend filter (50/200)
    - Sentiment filtering
    """

    def __init__(self, config: Dict):
        self.config = config
        self.timeframes = config.get('timeframes', {})

        # Initialize components
        self.smc_detector = SMCDetector(config)
        self.ta_indicators = TechnicalIndicators(config)
        self.sentiment_filter = SentimentFilter(config)

        logger.info("Trading strategy initialized")

    def analyze_multi_timeframe(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_15m: pd.DataFrame
    ) -> Dict:
        """
        Multi-timeframe analysis as per proposal:
        - 1D: Directional bias (CHOCH, BOS, FVG)
        - 4H: Trend structure
        - 15M: Entry signals with EMA filter

        Args:
            df_1d: Daily timeframe data
            df_4h: 4-hour timeframe data
            df_15m: 15-minute timeframe data

        Returns:
            Dictionary with multi-timeframe analysis
        """
        logger.info("Starting multi-timeframe analysis")

        # 1D Timeframe: Directional Bias
        # Look at recent candles (not just the last one) since SMC events are sparse
        df_1d_smc = self.smc_detector.analyze_smc(df_1d)
        latest_1d = df_1d_smc.iloc[-1]

        bias_1d = 'neutral'
        lookback_1d = min(20, len(df_1d_smc))
        recent_1d = df_1d_smc.iloc[-lookback_1d:]
        # Find most recent CHOCH/BOS direction
        bull_choch_1d = recent_1d['choch_bullish'].values.astype(bool)
        bear_choch_1d = recent_1d['choch_bearish'].values.astype(bool)
        bull_bos_1d = recent_1d['bos_bullish'].values.astype(bool)
        bear_bos_1d = recent_1d['bos_bearish'].values.astype(bool)

        last_bull_pos = -1
        last_bear_pos = -1
        if bull_choch_1d.any() or bull_bos_1d.any():
            bull_mask = bull_choch_1d | bull_bos_1d
            last_bull_pos = lookback_1d - 1 - bull_mask[::-1].argmax()
        if bear_choch_1d.any() or bear_bos_1d.any():
            bear_mask = bear_choch_1d | bear_bos_1d
            last_bear_pos = lookback_1d - 1 - bear_mask[::-1].argmax()

        if last_bull_pos > last_bear_pos:
            bias_1d = 'bullish'
        elif last_bear_pos > last_bull_pos:
            bias_1d = 'bearish'

        # 4H Timeframe: Trend Structure
        df_4h_smc = self.smc_detector.analyze_smc(df_4h)
        df_4h_ta = self.ta_indicators.add_all_indicators(df_4h)
        latest_4h = df_4h_ta.iloc[-1]

        structure_4h = latest_4h.get('market_structure', 'neutral')

        # 15M Timeframe: Entry Signals
        df_15m_smc = self.smc_detector.analyze_smc(df_15m)
        df_15m_ta = self.ta_indicators.add_all_indicators(df_15m)
        df_15m_combined = df_15m_smc.join(
            df_15m_ta[['ema_50', 'ema_200', 'atr', 'rsi']], rsuffix='_ta')

        latest_15m = df_15m_combined.iloc[-1]

        # EMA Trend Filter (15M)
        ema_trend = self.ta_indicators.get_trend_signal(df_15m_ta).iloc[-1]

        analysis = {
            'bias_1d': bias_1d,
            'structure_4h': structure_4h,
            'ema_trend_15m': 'bullish' if ema_trend == 1 else ('bearish' if ema_trend == -1 else 'neutral'),
            'latest_1d': latest_1d.to_dict(),
            'latest_4h': latest_4h.to_dict(),
            'latest_15m': latest_15m.to_dict(),
            'df_15m_analyzed': df_15m_combined
        }

        logger.info(
            f"Multi-TF Analysis - 1D Bias: {bias_1d}, 4H Structure: {structure_4h}, 15M EMA: {analysis['ema_trend_15m']}")

        return analysis

    def _generate_signal_from_df(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate signal from single dataframe using CONFLUENCE SCORING
        Used for backtesting - analyzes 15M data with all available indicators
        """
        if len(df) < 200:
            return None

        # Add all indicators
        df_analyzed = self.ta_indicators.add_all_indicators(df)
        df_smc = self.smc_detector.analyze_smc(df_analyzed)
        latest = df_smc.iloc[-1]

        # Get EMA trend
        ema_trend = self.ta_indicators.get_trend_signal(df_analyzed).iloc[-1]

        # === CONFLUENCE SCORING FOR BACKTEST ===
        confluence_score = 0.0
        reasons = []

        # EMA Trend (25% weight - since no multi-TF in backtest, this is key)
        if ema_trend == 1:
            confluence_score += 0.25
            reasons.append('EMA:bull')
        elif ema_trend == -1:
            confluence_score -= 0.25
            reasons.append('EMA:bear')

        # RSI (15% weight) - New logic: RSI as momentum strength
        rsi = latest.get('rsi', 50)
        if rsi > 55:  # Bullish momentum (strength)
            rsi_strength = (rsi - 55) / 45  # Scale 55-100 to 0-1
            confluence_score += 0.15 * rsi_strength
            reasons.append(f'RSI:bull({rsi:.0f})')
        elif rsi < 45:  # Bearish momentum (weakness)
            rsi_weakness = (45 - rsi) / 45  # Scale 0-45 to 1-0
            confluence_score -= 0.15 * rsi_weakness
            reasons.append(f'RSI:bear({rsi:.0f})')

        # --- SMC signals: find most recent event in last 50 candles ---
        smc_lookback = min(50, len(df_smc))
        recent_smc = df_smc.iloc[-smc_lookback:]

        def _last_true(col):
            """Position of last True in recent_smc, or -1"""
            mask = recent_smc[col].values.astype(bool)
            if mask.any():
                return smc_lookback - 1 - int(mask[::-1].argmax())
            return -1

        # CHOCH (25% weight - strongest SMC signal)
        last_bull_choch = _last_true('choch_bullish')
        last_bear_choch = _last_true('choch_bearish')
        if last_bull_choch > last_bear_choch:
            confluence_score += 0.25
            reasons.append('CHOCH:bull')
        elif last_bear_choch > last_bull_choch:
            confluence_score -= 0.25
            reasons.append('CHOCH:bear')

        # BOS (15% weight)
        last_bull_bos = _last_true('bos_bullish')
        last_bear_bos = _last_true('bos_bearish')
        if last_bull_bos > last_bear_bos:
            confluence_score += 0.15
            reasons.append('BOS:bull')
        elif last_bear_bos > last_bull_bos:
            confluence_score -= 0.15
            reasons.append('BOS:bear')

        # FVG (10% weight)
        last_bull_fvg = _last_true('fvg_bullish')
        last_bear_fvg = _last_true('fvg_bearish')
        if last_bull_fvg > last_bear_fvg:
            confluence_score += 0.10
            reasons.append('FVG:bull')
        elif last_bear_fvg > last_bull_fvg:
            confluence_score -= 0.10
            reasons.append('FVG:bear')

        # Market structure (10% weight)
        structure = latest.get('market_structure', 'neutral')
        if structure == 'bullish':
            confluence_score += 0.10
            reasons.append('Structure:bull')
        elif structure == 'bearish':
            confluence_score -= 0.10
            reasons.append('Structure:bear')

        # === SIGNAL GENERATION ===
        confluence_threshold = self.config.get(
            'strategy', {}).get('confluence_threshold', 0.35)

        atr = latest.get('atr', 0)
        entry_price = latest['close']

        if confluence_score >= confluence_threshold:
            # LONG signal — use swing-point SL via _calculate_stop_loss
            sl = self._calculate_stop_loss(entry_price, 'long', atr, df_smc=df_smc)
            tp = self._calculate_take_profit(entry_price, sl, 'long')
            return {
                'action': 'buy',
                'symbol': symbol,
                'direction': 'long',
                'entry_price': entry_price,
                'stop_loss': sl,
                'take_profit': tp,
                'position_size_usd': self.config.get('trading', {}).get('position_size_usd', 100),
                'reason': f'Confluence LONG ({confluence_score:.0%}): {" ".join(reasons)}',
                'confluence_score': confluence_score,
                'timestamp': df.index[-1] if hasattr(df.index, '__iter__') else None
            }

        elif confluence_score <= -confluence_threshold:
            # SHORT signal — use swing-point SL via _calculate_stop_loss
            sl = self._calculate_stop_loss(entry_price, 'short', atr, df_smc=df_smc)
            tp = self._calculate_take_profit(entry_price, sl, 'short')
            return {
                'action': 'sell',
                'symbol': symbol,
                'direction': 'short',
                'entry_price': entry_price,
                'stop_loss': sl,
                'take_profit': tp,
                'position_size_usd': self.config.get('trading', {}).get('position_size_usd', 100),
                'reason': f'Confluence SHORT ({abs(confluence_score):.0%}): {" ".join(reasons)}',
                'confluence_score': confluence_score,
                'timestamp': df.index[-1] if hasattr(df.index, '__iter__') else None
            }

        return None

    def generate_signal(
        self,
        symbol: str,
        mtf_analysis: Dict = None,
        df: pd.DataFrame = None
    ) -> Optional[Dict]:
        """
        Generate trading signal using CONFLUENCE SCORING SYSTEM

        Instead of requiring perfect alignment (which rarely happens),
        we use a weighted scoring approach:
        - Each factor contributes points toward bullish/bearish
        - Trade when total score exceeds threshold
        - Higher confluence = higher confidence

        Args:
            symbol: Trading symbol
            mtf_analysis: Multi-timeframe analysis results (for live trading)
            df: Single dataframe (for backtesting)

        Returns:
            Signal dictionary if valid, None otherwise
        """
        logger.info(f"Generating signal for {symbol}")

        # If df is provided (backtesting mode), perform quick analysis
        if df is not None and mtf_analysis is None:
            return self._generate_signal_from_df(symbol, df)

        # Extract analysis components (live trading mode)
        bias_1d = mtf_analysis['bias_1d']
        structure_4h = mtf_analysis['structure_4h']
        ema_trend = mtf_analysis['ema_trend_15m']
        latest_15m = mtf_analysis['latest_15m']

        # === CONFLUENCE SCORING SYSTEM ===
        # Each factor adds/subtracts from score
        # Positive = bullish, Negative = bearish

        confluence_score = 0.0
        reasons = []

        # 1D Bias (25% weight) - Higher timeframe = more weight
        if bias_1d == 'bullish':
            confluence_score += 0.25
            reasons.append('1D:bull')
        elif bias_1d == 'bearish':
            confluence_score -= 0.25
            reasons.append('1D:bear')

        # 4H Structure (20% weight)
        if structure_4h == 'bullish':
            confluence_score += 0.20
            reasons.append('4H:bull')
        elif structure_4h == 'bearish':
            confluence_score -= 0.20
            reasons.append('4H:bear')

        # 15M EMA Trend (15% weight)
        if ema_trend == 'bullish':
            confluence_score += 0.15
            reasons.append('EMA:bull')
        elif ema_trend == 'bearish':
            confluence_score -= 0.15
            reasons.append('EMA:bear')

        # SMC Patterns on 15M (40% weight combined - these are key entry triggers)
        # Use the FULL analyzed dataframe with lookback, not just the last candle
        # (SMC events are sparse point-events, almost never on the very last candle)
        df_15m_analyzed = mtf_analysis.get('df_15m_analyzed', None)

        if df_15m_analyzed is not None and len(df_15m_analyzed) > 0:
            smc_lookback = min(50, len(df_15m_analyzed))
            recent_smc = df_15m_analyzed.iloc[-smc_lookback:]

            def _last_true_live(col):
                if col not in recent_smc.columns:
                    return -1
                mask = recent_smc[col].values.astype(bool)
                return (smc_lookback - 1 - int(mask[::-1].argmax())) if mask.any() else -1

            # CHOCH (20% - strongest reversal signal)
            last_bull_choch = _last_true_live('choch_bullish')
            last_bear_choch = _last_true_live('choch_bearish')
            if last_bull_choch > last_bear_choch:
                confluence_score += 0.20
                reasons.append('CHOCH:bull')
            elif last_bear_choch > last_bull_choch:
                confluence_score -= 0.20
                reasons.append('CHOCH:bear')

            # BOS (12% - trend continuation)
            last_bull_bos = _last_true_live('bos_bullish')
            last_bear_bos = _last_true_live('bos_bearish')
            if last_bull_bos > last_bear_bos:
                confluence_score += 0.12
                reasons.append('BOS:bull')
            elif last_bear_bos > last_bull_bos:
                confluence_score -= 0.12
                reasons.append('BOS:bear')

            # FVG (8% - entry zone)
            last_bull_fvg = _last_true_live('fvg_bullish')
            last_bear_fvg = _last_true_live('fvg_bearish')
            if last_bull_fvg > last_bear_fvg:
                confluence_score += 0.08
                reasons.append('FVG:bull')
            elif last_bear_fvg > last_bull_fvg:
                confluence_score -= 0.08
                reasons.append('FVG:bear')
            
            # OTE Zone filter (15% bonus - premium entry quality)
            latest_row = recent_smc.iloc[-1]
            in_ote_long = latest_row.get('in_ote_long', False) if 'in_ote_long' in recent_smc.columns else False
            in_ote_short = latest_row.get('in_ote_short', False) if 'in_ote_short' in recent_smc.columns else False
            in_discount = latest_row.get('in_discount', False) if 'in_discount' in recent_smc.columns else False
            in_premium = latest_row.get('in_premium', False) if 'in_premium' in recent_smc.columns else False
            
            if in_ote_long:
                confluence_score += 0.15
                reasons.append('OTE:long')
            elif in_ote_short:
                confluence_score -= 0.15
                reasons.append('OTE:short')
            elif in_discount:
                confluence_score += 0.05
                reasons.append('DISC')
            elif in_premium:
                confluence_score -= 0.05
                reasons.append('PREM')
        else:
            # Fallback: use last candle dict (shouldn't happen normally)
            if latest_15m.get('choch_bullish'):
                confluence_score += 0.20
                reasons.append('CHOCH:bull')
            elif latest_15m.get('choch_bearish'):
                confluence_score -= 0.20
                reasons.append('CHOCH:bear')

            if latest_15m.get('bos_bullish'):
                confluence_score += 0.12
                reasons.append('BOS:bull')
            elif latest_15m.get('bos_bearish'):
                confluence_score -= 0.12
                reasons.append('BOS:bear')

            if latest_15m.get('fvg_bullish'):
                confluence_score += 0.08
                reasons.append('FVG:bull')
            elif latest_15m.get('fvg_bearish'):
                confluence_score -= 0.08
                reasons.append('FVG:bear')

        # === SIGNAL GENERATION ===
        # Threshold: 0.35 = need at least 35% confluence
        # This allows partial alignment while still being selective
        confluence_threshold = self.config.get(
            'strategy', {}).get('confluence_threshold', 0.35)

        signal = None

        if confluence_score >= confluence_threshold:
            # LONG signal
            signal = {
                'symbol': symbol,
                'direction': 'long',
                'entry_price': latest_15m['close'],
                'timestamp': datetime.now(),
                'reason': f'Confluence LONG ({confluence_score:.0%}): {" ".join(reasons)}',
                'confluence_score': confluence_score
            }
            logger.info(
                f"[CONFLUENCE] LONG signal: {confluence_score:.2f} >= {confluence_threshold} [{' '.join(reasons)}]")

        elif confluence_score <= -confluence_threshold:
            # SHORT signal
            signal = {
                'symbol': symbol,
                'direction': 'short',
                'entry_price': latest_15m['close'],
                'timestamp': datetime.now(),
                'reason': f'Confluence SHORT ({abs(confluence_score):.0%}): {" ".join(reasons)}',
                'confluence_score': confluence_score
            }
            logger.info(
                f"[CONFLUENCE] SHORT signal: {confluence_score:.2f} <= -{confluence_threshold} [{' '.join(reasons)}]")

        if not signal:
            logger.debug(
                f"No signal: confluence {confluence_score:.2f} below threshold ±{confluence_threshold}")
            return None

        # Apply sentiment filter
        if not self.sentiment_filter.filter_trade(symbol, signal['direction']):
            logger.info(f"Signal rejected by sentiment filter")
            return None

        # Calculate stop loss and take profit
        # Pass the analyzed 15M dataframe so SL can use swing points (SMC approach)
        atr = latest_15m.get('atr', 0)
        df_15m_analyzed = mtf_analysis.get('df_15m_analyzed', None)
        signal['stop_loss'] = self._calculate_stop_loss(
            signal['entry_price'],
            signal['direction'],
            atr,
            df_smc=df_15m_analyzed
        )
        signal['take_profit'] = self._calculate_take_profit(
            signal['entry_price'],
            signal['stop_loss'],
            signal['direction']
        )

        # Validate that stop loss has meaningful distance from entry price
        price_distance = abs(signal['entry_price'] - signal['stop_loss']) / signal['entry_price']
        min_distance_percent = 0.001  # 0.1% minimum
        
        if price_distance < min_distance_percent:
            logger.warning(
                f"Invalid signal: SL distance too small ({price_distance*100:.4f}% < {min_distance_percent*100}%). "
                f"Entry: {signal['entry_price']}, SL: {signal['stop_loss']}")
            return None

        logger.info(
            f"[SIGNAL] Valid {signal['direction'].upper()} signal for {symbol} at {signal['entry_price']:.2f} (SL: {signal['stop_loss']:.2f})")


        return signal

    def _calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        df_smc: pd.DataFrame = None
    ) -> float:
        """
        Calculate stop loss using structural levels.

        Priority:
          1. Order Block level (strongest — institutional entry zone)
          2. Recent swing high/low from SMC analysis (structural level)
          3. ATR-based stop loss
          4. Percentage-based fallback (always valid)
        """
        atr_multiplier = self.config['risk']['stop_loss_atr_multiplier']
        fallback_sl_percent = self.config['risk'].get('stop_loss_percent', 0.02)

        if pd.isna(atr) or atr <= 0:
            atr = entry_price * fallback_sl_percent / atr_multiplier
            logger.warning(f"ATR invalid, derived fallback ATR={atr:.2f}")

        sl_buffer = atr * 0.3
        stop_loss = None

        if df_smc is not None:
            lookback = min(80, len(df_smc))
            recent = df_smc.iloc[-lookback:]

            # --- 1. Try Order Block based SL (best structural level) ---
            if direction == 'long' and 'ob_bullish_low' in recent.columns:
                ob_lows = recent['ob_bullish_low'].dropna()
                valid_ob = ob_lows[ob_lows < entry_price]
                if len(valid_ob) > 0:
                    ob_level = valid_ob.iloc[-1]  # most recent OB below entry
                    candidate = ob_level - sl_buffer
                    dist_pct = (entry_price - candidate) / entry_price
                    if 0.003 <= dist_pct <= 0.05:
                        stop_loss = candidate
                        logger.info(f"  SL (order block): {ob_level:.2f} - buffer = {stop_loss:.2f} ({dist_pct*100:.2f}%)")
            elif direction == 'short' and 'ob_bearish_high' in recent.columns:
                ob_highs = recent['ob_bearish_high'].dropna()
                valid_ob = ob_highs[ob_highs > entry_price]
                if len(valid_ob) > 0:
                    ob_level = valid_ob.iloc[-1]
                    candidate = ob_level + sl_buffer
                    dist_pct = (candidate - entry_price) / entry_price
                    if 0.003 <= dist_pct <= 0.05:
                        stop_loss = candidate
                        logger.info(f"  SL (order block): {ob_level:.2f} + buffer = {stop_loss:.2f} ({dist_pct*100:.2f}%)")

            # --- 2. Try swing-point based SL ---
            if stop_loss is None and 'swing_high' in recent.columns and 'swing_low' in recent.columns:
                if direction == 'long':
                    swing_lows = recent['swing_low'].dropna()
                    valid = swing_lows[swing_lows < entry_price]
                    if len(valid) > 0:
                        nearest_swing = valid.iloc[-1]
                        candidate = nearest_swing - sl_buffer
                        dist_pct = (entry_price - candidate) / entry_price
                        if 0.003 <= dist_pct <= 0.05:
                            stop_loss = candidate
                            logger.info(f"  SL (swing low): {nearest_swing:.2f} - buffer = {stop_loss:.2f} ({dist_pct*100:.2f}%)")
                else:
                    swing_highs = recent['swing_high'].dropna()
                    valid = swing_highs[swing_highs > entry_price]
                    if len(valid) > 0:
                        nearest_swing = valid.iloc[-1]
                        candidate = nearest_swing + sl_buffer
                        dist_pct = (candidate - entry_price) / entry_price
                        if 0.003 <= dist_pct <= 0.05:
                            stop_loss = candidate
                            logger.info(f"  SL (swing high): {nearest_swing:.2f} + buffer = {stop_loss:.2f} ({dist_pct*100:.2f}%)")

        # --- 2. ATR-based fallback ---
        if stop_loss is None:
            atr_sl = atr * atr_multiplier
            min_sl_distance = entry_price * fallback_sl_percent

            if atr_sl >= min_sl_distance:
                if direction == 'long':
                    stop_loss = entry_price - atr_sl
                else:
                    stop_loss = entry_price + atr_sl
                logger.info(f"  SL (ATR): {stop_loss:.2f} (ATR={atr:.2f} x {atr_multiplier})")

        # --- 3. Percentage-based last resort (always valid) ---
        if stop_loss is None:
            if direction == 'long':
                stop_loss = entry_price * (1 - fallback_sl_percent)
            else:
                stop_loss = entry_price * (1 + fallback_sl_percent)
            logger.warning(f"  SL (% fallback): {stop_loss:.2f} ({fallback_sl_percent*100}%)")

        return stop_loss

    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str
    ) -> float:
        """Calculate take profit using risk-reward ratio"""
        rr_ratio = self.config['risk']['take_profit_rr_ratio']
        risk = abs(entry_price - stop_loss)

        if direction == 'long':
            take_profit = entry_price + (risk * rr_ratio)
        else:  # short
            take_profit = entry_price - (risk * rr_ratio)

        return take_profit

    def should_exit_trade(
        self,
        trade: Dict,
        current_price: float,
        df_15m: pd.DataFrame
    ) -> tuple:
        """
        Determine if an open trade should be exited

        Args:
            trade: Current trade dictionary
            current_price: Current market price
            df_15m: Current 15-minute data

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        # Guard: if SL/TP is missing, skip price-based exit checks but still check counter-trend
        sl = trade.get('stop_loss')
        tp = trade.get('take_profit')
        skip_price_check = (sl is None or tp is None)
        if skip_price_check:
            logger.warning(f"    Trade missing SL ({sl}) or TP ({tp}), skipping price exit check")

        # Check stop loss
        if not skip_price_check and trade['direction'] == 'long':
            logger.info(
                f"    Exit check: direction=long, price={current_price:.2f}, SL={sl:.2f}, TP={tp:.2f}")
            if current_price <= sl:
                logger.info(
                    f"    STOP LOSS HIT! ({current_price:.2f} <= {sl:.2f})")
                return (True, 'stop_loss')
            if current_price >= tp:
                logger.info(
                    f"    TAKE PROFIT HIT! ({current_price:.2f} >= {tp:.2f})")
                return (True, 'take_profit')
        elif not skip_price_check:  # short
            logger.info(
                f"    Exit check: direction=short, price={current_price:.2f}, SL={sl:.2f}, TP={tp:.2f}")
            if current_price >= sl:
                logger.info(
                    f"    STOP LOSS HIT! ({current_price:.2f} >= {sl:.2f})")
                return (True, 'stop_loss')
            if current_price <= tp:
                logger.info(
                    f"    TAKE PROFIT HIT! ({current_price:.2f} <= {tp:.2f})")
                return (True, 'take_profit')

        logger.info(
            f"    Price between SL and TP, checking counter-trend signals...")

        # Check for counter-trend signals (skip if no 15M data available)
        if df_15m is None or len(df_15m) == 0:
            logger.info(f"    No 15M data available, skipping counter-trend check")
            return (False, 'hold')

        df_15m_smc = self.smc_detector.analyze_smc(df_15m)
        latest = df_15m_smc.iloc[-1]

        if trade['direction'] == 'long' and (latest.get('choch_bearish') or latest.get('bos_bearish')):
            logger.info(f"    Counter-trend bearish signal detected")
            return (True, 'counter_trend_signal')
        elif trade['direction'] == 'short' and (latest.get('choch_bullish') or latest.get('bos_bullish')):
            logger.info(f"    Counter-trend bullish signal detected")
            return (True, 'counter_trend_signal')

        logger.info(f"    No exit conditions met, holding trade")

        return (False, '')
