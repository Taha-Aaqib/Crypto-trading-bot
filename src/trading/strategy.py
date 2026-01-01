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
        Generate signal from single dataframe (for backtesting)
        Uses simplified logic on 15m timeframe
        """
        if len(df) < 200:
            return None

        # Add indicators
        df_analyzed = self.ta_indicators.add_all_indicators(df)
        df_smc = self.smc_detector.analyze_smc(df_analyzed)
        latest = df_smc.iloc[-1]

        # Get trend from EMA
        ema_trend = self.ta_indicators.get_trend_signal(df_analyzed).iloc[-1]

        # LONG signal: Bullish trend + bullish SMC pattern
        if (ema_trend == 1 and
            (latest.get('choch_bullish') or latest.get('bos_bullish')) and
                latest.get('fvg_bullish')):

            atr = latest.get('atr', 0)
            entry_price = latest['close']

            return {
                'action': 'buy',
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss': entry_price - (atr * 1.5),
                'take_profit': entry_price + (atr * 3),
                'position_size_usd': self.config.get('trading', {}).get('position_size_usd', 100),
                'reason': 'Bullish SMC + EMA trend'
            }

        # SHORT signal: Bearish trend + bearish SMC pattern
        elif (ema_trend == -1 and
              (latest.get('choch_bearish') or latest.get('bos_bearish')) and
              latest.get('fvg_bearish')):

            atr = latest.get('atr', 0)
            entry_price = latest['close']

            return {
                'action': 'sell',
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss': entry_price + (atr * 1.5),
                'take_profit': entry_price - (atr * 3),
                'position_size_usd': self.config.get('trading', {}).get('position_size_usd', 100),
                'reason': 'Bearish SMC + EMA trend'
            }

        return None

    def generate_signal(
        self,
        symbol: str,
        mtf_analysis: Dict = None,
        df: pd.DataFrame = None
    ) -> Optional[Dict]:
        """
        Generate trading signal based on all filters

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

        # Determine signal direction
        signal = None

        # LONG Signal Conditions:
        # 1. 1D bias is bullish
        # 2. 4H structure is bullish
        # 3. 15M EMA trend is bullish (price > EMA50 > EMA200)
        # 4. 15M shows bullish CHOCH/BOS + FVG
        if (bias_1d == 'bullish' and
            structure_4h == 'bullish' and
            ema_trend == 'bullish' and
            (latest_15m.get('choch_bullish') or latest_15m.get('bos_bullish')) and
                latest_15m.get('fvg_bullish')):

            signal = {
                'symbol': symbol,
                'direction': 'long',
                'entry_price': latest_15m['close'],
                'timestamp': datetime.now(),
                'reason': 'Multi-TF bullish alignment with SMC and EMA filter'
            }

        # SHORT Signal Conditions:
        # 1. 1D bias is bearish
        # 2. 4H structure is bearish
        # 3. 15M EMA trend is bearish (price < EMA50 < EMA200)
        # 4. 15M shows bearish CHOCH/BOS + FVG
        elif (bias_1d == 'bearish' and
              structure_4h == 'bearish' and
              ema_trend == 'bearish' and
              (latest_15m.get('choch_bearish') or latest_15m.get('bos_bearish')) and
              latest_15m.get('fvg_bearish')):

            signal = {
                'symbol': symbol,
                'direction': 'short',
                'entry_price': latest_15m['close'],
                'timestamp': datetime.now(),
                'reason': 'Multi-TF bearish alignment with SMC and EMA filter'
            }

        if not signal:
            logger.info(
                f"No signal generated for {symbol} - conditions not met")
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
