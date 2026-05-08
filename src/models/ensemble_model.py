"""
Ensemble Decision Model
Combines SMC, Technical Analysis, ML Predictions, and Sentiment Analysis
for optimal trading decisions with weighted voting

ML OPTIONS:
- LSTM model: Deep learning for temporal patterns (recommended)
- Pattern Recognition: Multi-timeframe pattern matching
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
from src.models.pattern_recognition_model import PatternRecognitionModel
from src.models.mtf_pattern_labeler import MultiTimeframePatternLabeler
from src.indicators.smc_detector import SMCDetector
from src.indicators.ta_indicators import TechnicalIndicators
from src.sentiment.sentiment_filter import SentimentFilter
from src.utils.logger import get_logger

logger = get_logger()

# Try to import LSTM model
try:
    from src.models.lstm_model import LSTMTradingModel
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logger.warning("LSTM model not available, using pattern recognition only")


class EnsembleDecisionModel:
    """
    Ensemble model that combines multiple signal sources:
    - SMC signals (40% weight): CHOCH, BOS, FVG, liquidity zones, Order Blocks, OTE
    - Technical indicators (25% weight): RSI, EMA, MACD
    - ML predictions (25% weight): LSTM (preferred) or Pattern Recognition
    - Sentiment analysis (10% weight): RSS + CoinGecko + Fear&Greed via FinBERT

    Target Accuracy: 65-75% (optimized for speed and reliability)
    """

    def __init__(self, config: Dict):
        self.config = config

        # Weights for ensemble (must sum to 1.0)
        # Load weights from config (with sensible defaults matching config.yaml)
        ensemble_config = config.get('ensemble', {})
        weights_config = ensemble_config.get('weights', {})
        self.weights = {
            'smc': weights_config.get('smc', 0.30),      # SMC core strategy (30%)
            'ta': weights_config.get('ta', 0.30),         # TA with RSI divergence (30%)
            'ml': weights_config.get('ml', 0.30),         # LSTM temporal patterns (30%)
            'sentiment': weights_config.get('sentiment', 0.10)  # Sentiment confirmation (10%)
        }

        # Initialize components
        self.smc_detector = SMCDetector(config)
        self.ta_indicators = TechnicalIndicators(config)
        self.sentiment_filter = SentimentFilter(config)
        
        # ML model configuration
        ml_config = config.get('ml_model', {})
        self.use_lstm = ml_config.get('use_lstm', True) and LSTM_AVAILABLE
        self.ml_models = {}  # Per-symbol ML models (LSTM or Pattern)
        self.lstm_models = {}  # LSTM models specifically
        
        # Multi-TF labeler for creating prediction features (for pattern model fallback)
        self.mtf_labeler = MultiTimeframePatternLabeler(config)
        
        # Cache for multi-TF analyzed data (to avoid re-analyzing)
        self.mtf_cache = {}
        self.mtf_cache_timestamps = {}
        
        # PRE-LOAD ML MODELS FOR ALL SYMBOLS
        symbols_to_load = config.get('trading', {}).get('symbols', ['BTC/USDT', 'ETH/USDT'])
        
        for symbol in symbols_to_load:
            try:
                # Try LSTM first (preferred)
                if self.use_lstm:
                    logger.info(f"Loading LSTM model for {symbol}...")
                    lstm_model = LSTMTradingModel(config, symbol=symbol)
                    if lstm_model.is_trained:
                        self.lstm_models[symbol] = lstm_model
                        logger.info(f"✅ LSTM model for {symbol} loaded")
                    else:
                        logger.warning(f"LSTM model for {symbol} not trained, falling back to pattern model")
                
                # Also load pattern model as fallback
                logger.info(f"Loading Pattern Recognition Model for {symbol}...")
                pattern_model = PatternRecognitionModel(config, symbol=symbol)
                if pattern_model.is_trained:
                    self.ml_models[symbol] = pattern_model
                    logger.info(f"✅ Pattern model for {symbol} loaded")
                else:
                    logger.warning(f"Pattern model for {symbol} not trained")
                    self.ml_models[symbol] = pattern_model
                    
            except Exception as e:
                logger.warning(f"Error loading ML model for {symbol}: {e}")

        # Thresholds from config
        self.min_confidence = ensemble_config.get(
            'min_confidence', 0.45)  # Default 45% (tuned for leveraged futures)
        self.min_signals = ensemble_config.get(
            'min_signals', 2)  # At least 2 components agree

        # SMART SENTIMENT CONFIRMATION SYSTEM
        # When sentiment aligns with SMC/TA, boost confidence (sentiment confirms the trade)
        self.sentiment_confirmation_boost = ensemble_config.get(
            'sentiment_confirmation_boost', 0.15)

        logger.info(
            f"Ensemble model initialized - Weights: {self.weights}, Min Confidence: {self.min_confidence}")

    def analyze_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Generate comprehensive trading signal using all components

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol

        Returns:
            Dictionary with signal, confidence, and component scores
        """
        logger.info(f"Analyzing {symbol} with ensemble model...")

        # 1. SMC Analysis (40%)
        smc_score, smc_signal, smc_details = self._get_smc_score(df)

        # 2. Technical Analysis (25%)
        ta_score, ta_signal, ta_details = self._get_ta_score(df)

        # 3. ML Prediction (25%) - Use symbol-specific model
        ml_score, ml_signal, ml_details = self._get_ml_score(df, symbol)

        # 4. Sentiment Analysis (10%)
        sentiment_score, sentiment_signal, sentiment_details = self._get_sentiment_score(
            symbol)

        # Calculate weighted ensemble score
        ensemble_score = (
            smc_score * self.weights['smc'] +
            ta_score * self.weights['ta'] +
            ml_score * self.weights['ml'] +
            sentiment_score * self.weights['sentiment']
        )
        
        # Log the raw scores for debugging
        logger.info(
            f"  Raw Scores → SMC:{smc_score:.2f} TA:{ta_score:.2f} ML:{ml_score:.2f} S:{sentiment_score:.2f} "
            f"| Weighted: SMC:{smc_score*self.weights['smc']:.2f} TA:{ta_score*self.weights['ta']:.2f} "
            f"ML:{ml_score*self.weights['ml']:.2f} S:{sentiment_score*self.weights['sentiment']:.2f} "
            f"= Total:{ensemble_score:.2f}"
        )

        # Determine final signal
        final_signal = self._determine_signal(ensemble_score)

        # Calculate base confidence
        confidence = abs(ensemble_score)

        # Count agreeing components
        signals = [smc_signal, ta_signal, ml_signal, sentiment_signal]
        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')
        agreement = max(bullish_count, bearish_count) / len(signals)

        # SMART SENTIMENT CONFIRMATION SYSTEM
        # If sentiment agrees with the primary signal (SMC+TA), BOOST confidence
        # This makes sentiment meaningful - it CONFIRMS good trades
        primary_signal = smc_signal if smc_signal == ta_signal else 'neutral'
        sentiment_confirms = (sentiment_signal ==
                              primary_signal and primary_signal != 'neutral')
        sentiment_contradicts = (sentiment_signal != 'neutral' and primary_signal !=
                                 'neutral' and sentiment_signal != primary_signal)

        if sentiment_confirms:
            # Sentiment confirms SMC+TA = High confidence trade
            confidence_boost = self.sentiment_confirmation_boost
            confidence = min(1.0, confidence + confidence_boost)
            logger.info(
                f"  ✓ Sentiment CONFIRMS {primary_signal} signal (+{confidence_boost:.0%} boost)")
        elif sentiment_contradicts:
            # Sentiment warns against trade = Reduce confidence
            confidence_penalty = self.sentiment_confirmation_boost * 0.5
            confidence = max(0, confidence - confidence_penalty)
            logger.info(
                f"  ⚠ Sentiment WARNS against {primary_signal} signal (-{confidence_penalty:.0%} penalty)")

        result = {
            'signal': final_signal,
            'confidence': confidence,
            'ensemble_score': ensemble_score,
            'agreement': agreement,
            'sentiment_confirms': sentiment_confirms,
            'sentiment_contradicts': sentiment_contradicts,
            'components': {
                'smc': {'signal': smc_signal, 'score': smc_score, 'details': smc_details},
                'ta': {'signal': ta_signal, 'score': ta_score, 'details': ta_details},
                'ml': {'signal': ml_signal, 'score': ml_score, 'details': ml_details},
                'sentiment': {'signal': sentiment_signal, 'score': sentiment_score, 'details': sentiment_details}
            },
            'timestamp': datetime.now()
        }

        # Enhanced logging with sentiment confirmation status
        sentiment_status = "✓CONFIRMS" if sentiment_confirms else (
            "⚠WARNS" if sentiment_contradicts else "~neutral")
        
        # Convert signals to numbers: +1 bullish, -1 bearish, 0 neutral
        def sig_to_num(sig):
            if sig == 'bullish':
                return '+1'
            elif sig == 'bearish':
                return '-1'
            else:
                return '0'
        
        logger.info(
            f"{symbol} Ensemble: {final_signal.upper()} "
            f"(confidence: {confidence:.2f}, agreement: {agreement:.2f}) "
            f"[SMC:{sig_to_num(smc_signal)} TA:{sig_to_num(ta_signal)} ML:{sig_to_num(ml_signal)} S:{sig_to_num(sentiment_signal)} {sentiment_status}]"
        )

        return result

    def _get_smc_score(self, df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """
        Get SMC component score (ENHANCED with Order Blocks, OTE, Sweeps).
        
        Finds the MOST RECENT event of each type (CHOCH, BOS, FVG, OB, Sweeps)
        to determine directional bias with recency weighting.
        """
        try:
            # Use enhanced SMC analysis (includes OB, OTE, sweeps)
            df_smc = self.smc_detector.analyze_smc_enhanced(df)
            
            # Use a generous lookback — SMC events are sparse
            lookback = max(50, len(df_smc) // 5)
            lookback = min(lookback, len(df_smc))
            recent = df_smc.iloc[-lookback:]
            total = len(recent)

            score = 0.0
            details = {}

            # --- Helper: find position (from start of `recent`) of last True ---
            def last_true_pos(col_name):
                """Return index-position in `recent` of the last True value, or None."""
                if col_name not in recent.columns:
                    return None
                mask = recent[col_name].values.astype(bool)
                if mask.any():
                    # Search from the end
                    return int(total - 1 - mask[::-1].argmax())
                return None

            def recency_weight(pos):
                """0→1 weight; more recent → closer to 1."""
                if pos is None:
                    return 0.0
                return max(0.5, (pos + 1) / total)  # floor at 0.5

            # ----- CHOCH (strongest SMC signal) -----
            last_bull_choch = last_true_pos('choch_bullish')
            last_bear_choch = last_true_pos('choch_bearish')

            if last_bull_choch is not None and last_bear_choch is not None:
                if last_bull_choch > last_bear_choch:
                    w = recency_weight(last_bull_choch)
                    score += 0.4 * w
                    details['choch'] = f'bullish(pos {last_bull_choch}/{total})'
                else:
                    w = recency_weight(last_bear_choch)
                    score -= 0.4 * w
                    details['choch'] = f'bearish(pos {last_bear_choch}/{total})'
            elif last_bull_choch is not None:
                w = recency_weight(last_bull_choch)
                score += 0.4 * w
                details['choch'] = f'bullish(pos {last_bull_choch}/{total})'
            elif last_bear_choch is not None:
                w = recency_weight(last_bear_choch)
                score -= 0.4 * w
                details['choch'] = f'bearish(pos {last_bear_choch}/{total})'
            else:
                details['choch'] = 'neutral'

            # ----- BOS (trend continuation) -----
            last_bull_bos = last_true_pos('bos_bullish')
            last_bear_bos = last_true_pos('bos_bearish')

            if last_bull_bos is not None and last_bear_bos is not None:
                if last_bull_bos > last_bear_bos:
                    w = recency_weight(last_bull_bos)
                    score += 0.3 * w
                    details['bos'] = f'bullish(pos {last_bull_bos}/{total})'
                else:
                    w = recency_weight(last_bear_bos)
                    score -= 0.3 * w
                    details['bos'] = f'bearish(pos {last_bear_bos}/{total})'
            elif last_bull_bos is not None:
                w = recency_weight(last_bull_bos)
                score += 0.3 * w
                details['bos'] = f'bullish(pos {last_bull_bos}/{total})'
            elif last_bear_bos is not None:
                w = recency_weight(last_bear_bos)
                score -= 0.3 * w
                details['bos'] = f'bearish(pos {last_bear_bos}/{total})'
            else:
                details['bos'] = 'neutral'

            # ----- FVG (fair value gaps) -----
            last_bull_fvg = last_true_pos('fvg_bullish')
            last_bear_fvg = last_true_pos('fvg_bearish')

            if last_bull_fvg is not None and last_bear_fvg is not None:
                if last_bull_fvg > last_bear_fvg:
                    w = recency_weight(last_bull_fvg)
                    score += 0.2 * w
                    details['fvg'] = f'bullish(pos {last_bull_fvg}/{total})'
                else:
                    w = recency_weight(last_bear_fvg)
                    score -= 0.2 * w
                    details['fvg'] = f'bearish(pos {last_bear_fvg}/{total})'
            elif last_bull_fvg is not None:
                w = recency_weight(last_bull_fvg)
                score += 0.2 * w
                details['fvg'] = f'bullish(pos {last_bull_fvg}/{total})'
            elif last_bear_fvg is not None:
                w = recency_weight(last_bear_fvg)
                score -= 0.2 * w
                details['fvg'] = f'bearish(pos {last_bear_fvg}/{total})'
            else:
                details['fvg'] = 'neutral'

            # ----- Order Blocks (NEW) -----
            last_bull_ob = last_true_pos('ob_bullish')
            last_bear_ob = last_true_pos('ob_bearish')
            
            if last_bull_ob is not None and last_bear_ob is not None:
                if last_bull_ob > last_bear_ob:
                    w = recency_weight(last_bull_ob)
                    score += 0.15 * w
                    details['order_block'] = f'bullish(pos {last_bull_ob}/{total})'
                else:
                    w = recency_weight(last_bear_ob)
                    score -= 0.15 * w
                    details['order_block'] = f'bearish(pos {last_bear_ob}/{total})'
            elif last_bull_ob is not None:
                w = recency_weight(last_bull_ob)
                score += 0.15 * w
                details['order_block'] = f'bullish(pos {last_bull_ob}/{total})'
            elif last_bear_ob is not None:
                w = recency_weight(last_bear_ob)
                score -= 0.15 * w
                details['order_block'] = f'bearish(pos {last_bear_ob}/{total})'
            else:
                details['order_block'] = 'neutral'

            # ----- Liquidity Sweeps (NEW) -----
            last_bull_sweep = last_true_pos('sweep_bullish')
            last_bear_sweep = last_true_pos('sweep_bearish')
            
            if last_bull_sweep is not None and last_bear_sweep is not None:
                if last_bull_sweep > last_bear_sweep:
                    w = recency_weight(last_bull_sweep)
                    score += 0.15 * w
                    details['sweep'] = f'bullish(pos {last_bull_sweep}/{total})'
                else:
                    w = recency_weight(last_bear_sweep)
                    score -= 0.15 * w
                    details['sweep'] = f'bearish(pos {last_bear_sweep}/{total})'
            elif last_bull_sweep is not None:
                w = recency_weight(last_bull_sweep)
                score += 0.15 * w
                details['sweep'] = f'bullish(pos {last_bull_sweep}/{total})'
            elif last_bear_sweep is not None:
                w = recency_weight(last_bear_sweep)
                score -= 0.15 * w
                details['sweep'] = f'bearish(pos {last_bear_sweep}/{total})'
            else:
                details['sweep'] = 'neutral'

            # ----- OTE Zone (NEW) -----
            latest = df_smc.iloc[-1]
            in_ote_long = latest.get('in_ote_long', False)
            in_ote_short = latest.get('in_ote_short', False)
            in_discount = latest.get('in_discount', False)
            in_premium = latest.get('in_premium', False)
            
            if in_ote_long:
                score += 0.1
                details['ote'] = 'long_entry_zone'
            elif in_ote_short:
                score -= 0.1
                details['ote'] = 'short_entry_zone'
            elif in_discount:
                score += 0.05  # Slight bullish bias in discount
                details['ote'] = 'discount'
            elif in_premium:
                score -= 0.05  # Slight bearish bias in premium
                details['ote'] = 'premium'
            else:
                details['ote'] = 'neutral'

            # Market structure - use latest candle (now rolling, so it's correct)
            structure = latest.get('market_structure', 'neutral')
            if structure == 'bullish':
                score += 0.1
            elif structure == 'bearish':
                score -= 0.1
            details['structure'] = structure
            
            # Log what was found
            details['lookback'] = lookback
            
            # Debug: Log SMC findings
            logger.info(
                f"  SMC Details → CHOCH:{details['choch']} "
                f"BOS:{details['bos']} "
                f"FVG:{details['fvg']} "
                f"OB:{details.get('order_block', 'n/a')} "
                f"Sweep:{details.get('sweep', 'n/a')} "
                f"OTE:{details.get('ote', 'n/a')} "
                f"Structure:{structure}"
            )

            # Normalize to [-1, 1]
            score = max(-1.0, min(1.0, score))
            signal = 'bullish' if score > 0.15 else (
                'bearish' if score < -0.15 else 'neutral')

            return score, signal, details

        except Exception as e:
            logger.error(f"Error in SMC analysis: {e}")
            return 0.0, 'neutral', {'error': str(e)}

    def _get_ta_score(self, df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Get Technical Analysis component score (includes RSI divergences)"""
        try:
            df_ta = self.ta_indicators.add_all_indicators(df)
            if df_ta.empty:
                return 0.0, 'neutral', {'error': 'empty_dataframe'}
            latest = df_ta.iloc[-1]

            score = 0.0
            details = {}

            # RSI - Momentum based (not just extremes)
            rsi = latest.get('rsi', 50)
            if rsi > 55:
                # Bullish momentum - scales with strength
                rsi_contribution = min(0.3, (rsi - 55) / 45 * 0.3)
                score += rsi_contribution
                details['rsi'] = f'bullish({rsi:.0f})'
            elif rsi < 45:
                # Bearish momentum - scales with weakness
                rsi_contribution = min(0.3, (45 - rsi) / 45 * 0.3)
                score -= rsi_contribution
                details['rsi'] = f'bearish({rsi:.0f})'
            else:
                details['rsi'] = 'neutral'

            # RSI DIVERGENCES (NEW) - Strong reversal signals
            # Valid for 14 candles after detection
            bullish_div = latest.get('bullish_divergence', False)
            bearish_div = latest.get('bearish_divergence', False)
            div_strength = latest.get('divergence_strength', 0.0)
            
            if bullish_div:
                # Bullish divergence = price LL but RSI HL (reversal likely)
                div_contribution = 0.25 + (0.15 * div_strength)  # 0.25 to 0.40
                score += div_contribution
                details['divergence'] = f'bullish(strength:{div_strength:.2f})'
            elif bearish_div:
                # Bearish divergence = price HH but RSI LH (reversal likely)
                div_contribution = 0.25 + (0.15 * div_strength)
                score -= div_contribution
                details['divergence'] = f'bearish(strength:{div_strength:.2f})'
            else:
                details['divergence'] = 'none'

            # EMA trend (main signal)
            trend = self.ta_indicators.get_trend_signal(df_ta).iloc[-1]
            if trend == 1:
                score += 0.5  # Bullish trend - strong weight
                details['ema_trend'] = 'bullish'
            elif trend == -1:
                score -= 0.5  # Bearish trend - strong weight
                details['ema_trend'] = 'bearish'
            else:
                details['ema_trend'] = 'neutral'

            # Price vs EMA200 (additional confirmation)
            price = latest.get('close', 0)
            ema200 = latest.get('ema_200', price)
            if price > ema200:
                score += 0.2
                details['price_vs_ema200'] = 'above'
            elif price < ema200:
                score -= 0.2
                details['price_vs_ema200'] = 'below'

            # Normalize
            score = max(-1.0, min(1.0, score))
            signal = 'bullish' if score > 0.15 else (
                'bearish' if score < -0.15 else 'neutral')

            return score, signal, details

        except Exception as e:
            logger.error(f"Error in TA analysis: {e}")
            return 0.0, 'neutral', {'error': str(e)}

    def _get_ml_score(self, df: pd.DataFrame, symbol: str) -> Tuple[float, str, Dict]:
        """
        Get ML prediction score.
        
        Uses LSTM model if available and trained (preferred for temporal patterns),
        otherwise falls back to Pattern Recognition model.
        """
        try:
            # Try LSTM first (preferred - captures temporal patterns)
            if self.use_lstm and symbol in self.lstm_models:
                lstm_model = self.lstm_models[symbol]
                if lstm_model.is_trained:
                    return self._get_lstm_score(df, symbol, lstm_model)
            
            # Fallback to pattern recognition model
            return self._get_pattern_score(df, symbol)
            
        except Exception as e:
            logger.error(f"Error in ML prediction for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 'neutral', {'error': str(e), 'symbol': symbol}
    
    def _get_lstm_score(self, df: pd.DataFrame, symbol: str, lstm_model) -> Tuple[float, str, Dict]:
        """
        Get LSTM setup quality score.
        
        LSTM predicts P(setup succeeds). Base rate is ~30% (2:1 R:R).
        Score is relative to base rate, not 0.5:
          P(success)=0.50 → score=+0.60 (much above base = great)
          P(success)=0.40 → score=+0.20 (above base = decent)
          P(success)=0.35 → score=0.00  (at base = neutral)
          P(success)=0.30 → score=-0.20 (below base = bad)
        
        Also acts as GATEKEEPER: if P(success) < veto threshold, sets ml_veto=True
        """
        try:
            # Use 1h data from cache — LSTM was trained on 1h candles
            cache_key = f"{symbol}_1h"
            cache_fresh = False
            if cache_key in self.mtf_cache:
                cache_age = (datetime.now() - self.mtf_cache_timestamps.get(cache_key, datetime.min)).total_seconds()
                if cache_age < 7200:  # 2 hours max staleness
                    df_for_lstm = self.mtf_cache[cache_key]
                    cache_fresh = True
                    logger.debug(f"LSTM using 1h data from cache ({len(df_for_lstm)} candles, {cache_age:.0f}s old)")
                else:
                    logger.warning(f"LSTM 1h cache stale ({cache_age:.0f}s), falling back to passed data")
            
            if not cache_fresh:
                df_for_lstm = df
                logger.debug(f"LSTM using passed data (no fresh 1h cache)")
            
            # Analyze data with ENHANCED SMC (OB, OTE, sweeps) and TA indicators
            df_analyzed = self.smc_detector.analyze_smc_enhanced(df_for_lstm)
            df_analyzed = self.ta_indicators.add_all_indicators(df_analyzed)
            
            # Get LSTM prediction: P(success) + confidence
            p_success, confidence = lstm_model.predict(df_analyzed)
            
            # Score relative to base rate (~0.30 with 2:1 R:R)
            # Scale so that base_rate=0, max deviation maps to ±1
            BASE_RATE = 0.35  # slightly above actual 0.30 to be conservative
            score = (p_success - BASE_RATE) / (1.0 - BASE_RATE)  # [0,1] → [-0.54, 1.0]
            score = max(-1.0, min(1.0, score * 2))  # amplify and clip
            
            # Determine signal based on whether setup is better/worse than base rate
            VETO_THRESHOLD = 0.40  # below this = definitely bad setup
            if p_success >= 0.50:
                signal = 'bullish'    # ML says setup is well above average
            elif p_success < VETO_THRESHOLD:
                signal = 'bearish'    # ML says setup is bad (veto)
            else:
                signal = 'neutral'    # ML says setup is around average
            
            # ML veto flag: if model thinks setup will likely fail
            ml_veto = p_success < VETO_THRESHOLD
            
            details = {
                'model_type': 'lstm_setup_quality',
                'p_success': float(p_success),
                'confidence': float(confidence),
                'score': score,
                'ml_veto': ml_veto,
                'symbol': symbol
            }
            
            return score, signal, details
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction for {symbol}: {e}")
            return 0.0, 'neutral', {'error': str(e), 'model_type': 'lstm'}
    
    def _get_pattern_score(self, df: pd.DataFrame, symbol: str) -> Tuple[float, str, Dict]:
        """
        Get Pattern Recognition Model score using Multi-TF features.
        
        Fallback when LSTM is not available.
        """
        try:
            # Get or load pattern recognition model for this symbol
            if symbol not in self.ml_models:
                logger.info(f"Loading pattern model for {symbol}")
                model = PatternRecognitionModel(self.config, symbol=symbol)
                self.ml_models[symbol] = model
            
            pattern_model = self.ml_models[symbol]
            
            if not pattern_model.is_trained:
                logger.debug(f"Pattern model for {symbol} not trained yet")
                return 0.0, 'neutral', {'status': 'not_trained', 'symbol': symbol}
            
            # Check if we have multi-TF analyzed data in cache
            if symbol in self.mtf_cache:
                analyzed_data = self.mtf_cache[symbol]
                latest_ts = df.index[-1] if len(df) > 0 else None
                
                if latest_ts is not None:
                    # Create multi-TF features for prediction
                    features = self.mtf_labeler.create_multi_tf_features(analyzed_data, latest_ts)
                    pattern_score, confidence = pattern_model.predict_from_features(features)
                else:
                    pattern_score, confidence = 0.0, 0.0
            else:
                # No multi-TF cache - use fallback single-TF prediction
                # Create basic features from the available dataframe
                logger.debug(f"No MTF cache for {symbol}, using single-TF fallback")
                
                # Analyze the provided df as 15M (entry timeframe)
                df_analyzed = self.smc_detector.analyze_smc(df)
                df_analyzed = self.ta_indicators.add_all_indicators(df_analyzed)
                
                if len(df_analyzed) > 0:
                    latest_ts = df_analyzed.index[-1]
                    # Create features with only 15M data (other TFs will be empty)
                    analyzed_data = {
                        '1d': pd.DataFrame(),  # Empty - no daily data
                        '4h': pd.DataFrame(),  # Empty - no 4H data
                        '1h': pd.DataFrame(),  # Empty - no 1H data
                        '15m': df_analyzed
                    }
                    features = self.mtf_labeler.create_multi_tf_features(analyzed_data, latest_ts)
                    pattern_score, confidence = pattern_model.predict_from_features(features)
                else:
                    pattern_score, confidence = 0.0, 0.0
            
            # Convert pattern score to ensemble score
            # pattern_score: -1 (poor bearish pattern) to +1 (excellent bullish pattern)
            score = pattern_score * confidence  # Score weighted by confidence
            
            # Determine signal strength
            if pattern_score > 0.6:
                signal = 'bullish'
            elif pattern_score < -0.6:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            details = {
                'pattern_score': float(pattern_score),
                'confidence': float(confidence),
                'status': 'mtf_prediction' if symbol in self.mtf_cache else 'single_tf_fallback',
                'symbol': symbol
            }
            
            logger.debug(
                f"Pattern recognition for {symbol}: score={pattern_score:.2f}, "
                f"confidence={confidence:.2f}, signal={signal}"
            )
            
            return float(score), signal, details
        
        except Exception as e:
            logger.error(f"Error in pattern recognition for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 'neutral', {'error': str(e), 'symbol': symbol}
    
    def update_mtf_cache(self, symbol: str, df_1d: pd.DataFrame, df_4h: pd.DataFrame, 
                         df_1h: pd.DataFrame, df_15m: pd.DataFrame):
        """
        Update the multi-TF analyzed data cache for a symbol.
        
        Call this from the main trading loop when you have data from all timeframes.
        This enables full multi-TF ML predictions and 1h data for LSTM.
        """
        try:
            analyzed_data = self.mtf_labeler.analyze_all_timeframes(df_1d, df_4h, df_1h, df_15m)
            self.mtf_cache[symbol] = analyzed_data
            # Store raw 1h data for LSTM (trained on 1h timeframe)
            self.mtf_cache[f"{symbol}_1h"] = df_1h
            self.mtf_cache_timestamps[f"{symbol}_1h"] = datetime.now()
            logger.debug(f"Updated MTF cache for {symbol}")
        except Exception as e:
            logger.error(f"Error updating MTF cache for {symbol}: {e}")

    def _get_sentiment_score(self, symbol: str) -> Tuple[float, str, Dict]:
        """Get sentiment analysis score"""
        try:
            sentiment = self.sentiment_filter.get_combined_sentiment(symbol)

            score = sentiment['sentiment_score']
            signal = sentiment['classification']

            # Map classification to our signal format
            if signal == 'positive':
                signal = 'bullish'
            elif signal == 'negative':
                signal = 'bearish'

            details = {
                'score': score,
                'classification': sentiment['classification'],
                'sources': sentiment.get('sources', {})
            }

            return score, signal, details

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0, 'neutral', {'error': str(e)}

    def _determine_signal(self, ensemble_score: float) -> str:
        """Determine final signal based on ensemble score"""
        if ensemble_score >= 0.15:
            return 'long'
        elif ensemble_score <= -0.15:
            return 'short'
        else:
            return 'hold'

    def should_trade(self, signal_result: Dict) -> bool:
        """
        Determine if trade should be executed.
        
        Checks:
        1. ML gatekeeper — if LSTM says setup will fail, veto the trade
        2. Minimum confidence threshold
        3. Minimum component agreement (3/4 = 75%)
        4. Must not be hold signal
        """
        # ML GATEKEEPER: If LSTM predicts setup will fail, reject
        ml_details = signal_result.get('components', {}).get('ml', {}).get('details', {})
        ml_veto = ml_details.get('ml_veto', False)
        p_success = ml_details.get('p_success', 0.5)
        
        if ml_veto:
            logger.info(
                f"Trade VETOED by ML: P(success)={p_success:.3f} — setup quality too low")
            return False

        # Check confidence threshold
        if signal_result['confidence'] < self.min_confidence:
            logger.info(
                f"Trade rejected: Low confidence ({signal_result['confidence']:.2f} < {self.min_confidence})")
            return False

        # Check agreement threshold (require 3/4 = 75% by default)
        if signal_result['agreement'] < (self.min_signals / 4):
            logger.info(
                f"Trade rejected: Low agreement ({signal_result['agreement']:.2f}, need {self.min_signals}/4)")
            return False

        # Must not be hold signal
        if signal_result['signal'] == 'hold':
            return False

        return True
