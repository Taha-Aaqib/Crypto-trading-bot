"""
Main Trading Strategy
Integrates SMC, Technical Indicators, Sentiment, and Event Filters
"""

import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
from src.indicators.smc_detector import SMCDetector
from src.indicators.ta_indicators import TechnicalIndicators
from src.sentiment.sentiment_filter import SentimentFilter
from src.events.event_filter import EventFilter
from src.utils.logger import get_logger

logger = get_logger()


class TradingStrategy:
    """
    Main trading strategy combining all components:
    - Multi-timeframe SMC analysis (1D bias, 4H structure, 15M entry)
    - EMA trend filter (50/200)
    - Sentiment filtering
    - Economic event filtering
    """

    def __init__(self, config: Dict):
        self.config = config
        self.timeframes = config.get('timeframes', {})

        # Initialize components
        self.smc_detector = SMCDetector(config)
        self.ta_indicators = TechnicalIndicators(config)
        self.sentiment_filter = SentimentFilter(config)
        self.event_filter = EventFilter(config)

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
        df_1d_smc = self.smc_detector.analyze_smc(df_1d)
        latest_1d = df_1d_smc.iloc[-1]

        bias_1d = 'neutral'
        if latest_1d.get('bos_bullish') or latest_1d.get('choch_bullish'):
            bias_1d = 'bullish'
        elif latest_1d.get('bos_bearish') or latest_1d.get('choch_bearish'):
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

        # RSI (15% weight)
        rsi = latest.get('rsi', 50)
        if rsi < 35:  # Oversold
            confluence_score += 0.15
            reasons.append('RSI:oversold')
        elif rsi > 65:  # Overbought
            confluence_score -= 0.15
            reasons.append('RSI:overbought')

        # CHOCH (25% weight - strongest SMC signal)
        if latest.get('choch_bullish'):
            confluence_score += 0.25
            reasons.append('CHOCH:bull')
        elif latest.get('choch_bearish'):
            confluence_score -= 0.25
            reasons.append('CHOCH:bear')

        # BOS (15% weight)
        if latest.get('bos_bullish'):
            confluence_score += 0.15
            reasons.append('BOS:bull')
        elif latest.get('bos_bearish'):
            confluence_score -= 0.15
            reasons.append('BOS:bear')

        # FVG (10% weight)
        if latest.get('fvg_bullish'):
            confluence_score += 0.10
            reasons.append('FVG:bull')
        elif latest.get('fvg_bearish'):
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
            # LONG signal
            return {
                'action': 'buy',
                'symbol': symbol,
                'direction': 'long',
                'entry_price': entry_price,
                'stop_loss': entry_price - (atr * self.config['risk']['stop_loss_atr_multiplier']),
                'take_profit': entry_price + (atr * self.config['risk']['stop_loss_atr_multiplier'] * self.config['risk']['take_profit_rr_ratio']),
                'position_size_usd': self.config.get('trading', {}).get('position_size_usd', 100),
                'reason': f'Confluence LONG ({confluence_score:.0%}): {" ".join(reasons)}',
                'confluence_score': confluence_score,
                'timestamp': df.index[-1] if hasattr(df.index, '__iter__') else None
            }

        elif confluence_score <= -confluence_threshold:
            # SHORT signal
            return {
                'action': 'sell',
                'symbol': symbol,
                'direction': 'short',
                'entry_price': entry_price,
                'stop_loss': entry_price + (atr * self.config['risk']['stop_loss_atr_multiplier']),
                'take_profit': entry_price - (atr * self.config['risk']['stop_loss_atr_multiplier'] * self.config['risk']['take_profit_rr_ratio']),
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
        # CHOCH (20% - strongest reversal signal)
        if latest_15m.get('choch_bullish'):
            confluence_score += 0.20
            reasons.append('CHOCH:bull')
        elif latest_15m.get('choch_bearish'):
            confluence_score -= 0.20
            reasons.append('CHOCH:bear')

        # BOS (12% - trend continuation)
        if latest_15m.get('bos_bullish'):
            confluence_score += 0.12
            reasons.append('BOS:bull')
        elif latest_15m.get('bos_bearish'):
            confluence_score -= 0.12
            reasons.append('BOS:bear')

        # FVG (8% - entry zone)
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

        # Apply event filter
        if not self.event_filter.is_trading_allowed():
            logger.info(f"Signal rejected by event filter")
            return None

        # Calculate stop loss and take profit
        atr = latest_15m.get('atr', 0)
        signal['stop_loss'] = self._calculate_stop_loss(
            signal['entry_price'],
            signal['direction'],
            atr
        )
        signal['take_profit'] = self._calculate_take_profit(
            signal['entry_price'],
            signal['stop_loss'],
            signal['direction']
        )

        logger.info(
            f"[SIGNAL] Valid {signal['direction'].upper()} signal for {symbol} at {signal['entry_price']}")

        return signal

    def _calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float
    ) -> float:
        """Calculate stop loss using ATR"""
        atr_multiplier = self.config['risk']['stop_loss_atr_multiplier']

        if direction == 'long':
            stop_loss = entry_price - (atr * atr_multiplier)
        else:  # short
            stop_loss = entry_price + (atr * atr_multiplier)

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
        # Check stop loss
        if trade['direction'] == 'long':
            logger.info(
                f"    Exit check: direction=long, price={current_price:.2f}, SL={trade['stop_loss']:.2f}, TP={trade['take_profit']:.2f}")
            if current_price <= trade['stop_loss']:
                logger.info(
                    f"    STOP LOSS HIT! ({current_price:.2f} <= {trade['stop_loss']:.2f})")
                return (True, 'stop_loss')
            if current_price >= trade['take_profit']:
                logger.info(
                    f"    TAKE PROFIT HIT! ({current_price:.2f} >= {trade['take_profit']:.2f})")
                return (True, 'take_profit')
        else:  # short
            logger.info(
                f"    Exit check: direction=short, price={current_price:.2f}, SL={trade['stop_loss']:.2f}, TP={trade['take_profit']:.2f}")
            if current_price >= trade['stop_loss']:
                logger.info(
                    f"    STOP LOSS HIT! ({current_price:.2f} >= {trade['stop_loss']:.2f})")
                return (True, 'stop_loss')
            if current_price <= trade['take_profit']:
                logger.info(
                    f"    TAKE PROFIT HIT! ({current_price:.2f} <= {trade['take_profit']:.2f})")
                return (True, 'take_profit')

        logger.info(
            f"    Price between SL and TP, checking counter-trend signals...")

        # Check for counter-trend signals
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
