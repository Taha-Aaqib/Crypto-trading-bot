"""
Ensemble Decision Model
Combines SMC, Technical Analysis, ML Predictions, and Sentiment Analysis
for optimal trading decisions with weighted voting
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
from src.models.trading_model import TradingModel
from src.indicators.smc_detector import SMCDetector
from src.indicators.ta_indicators import TechnicalIndicators
from src.sentiment.sentiment_filter import SentimentFilter
from src.utils.logger import get_logger

logger = get_logger()


class EnsembleDecisionModel:
    """
    Ensemble model that combines multiple signal sources:
    - SMC signals (40% weight): CHOCH, BOS, FVG, liquidity zones
    - Technical indicators (25% weight): RSI, EMA, MACD
    - ML predictions (25% weight): Pattern recognition
    - Sentiment analysis (10% weight): FinBERT/Twitter sentiment

    Target Accuracy: 65-75% (optimized for speed and reliability)
    """

    def __init__(self, config: Dict):
        self.config = config

        # Weights for ensemble (must sum to 1.0)
        # Load weights from config (with sensible defaults)
        ensemble_config = config.get('ensemble', {})
        weights_config = ensemble_config.get('weights', {})
        self.weights = {
            'smc': weights_config.get('smc', 0.45),      # SMC is core strategy
            # Technical confirmation
            'ta': weights_config.get('ta', 0.25),
            # ML pattern recognition
            'ml': weights_config.get('ml', 0.15),
            # Sentiment filter
            'sentiment': weights_config.get('sentiment', 0.15)
        }

        # Initialize components
        self.smc_detector = SMCDetector(config)
        self.ta_indicators = TechnicalIndicators(config)
        self.ml_models = {}  # Per-symbol ML models
        self.sentiment_filter = SentimentFilter(config)

        # Thresholds from config
        self.min_confidence = ensemble_config.get(
            'min_confidence', 0.50)  # Default 50%
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
        logger.info(
            f"{symbol} Ensemble: {final_signal.upper()} "
            f"(confidence: {confidence:.2f}, agreement: {agreement:.2f}) "
            f"[SMC:{smc_signal[0]} TA:{ta_signal[0]} ML:{ml_signal[0]} S:{sentiment_signal[0]} {sentiment_status}]"
        )

        return result

    def _get_smc_score(self, df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Get SMC component score"""
        try:
            df_smc = self.smc_detector.analyze_smc(df)
            latest = df_smc.iloc[-1]

            score = 0.0
            details = {}

            # CHOCH signals (strong)
            if latest.get('choch_bullish', False):
                score += 0.4
                details['choch'] = 'bullish'
            elif latest.get('choch_bearish', False):
                score -= 0.4
                details['choch'] = 'bearish'

            # BOS signals (strong)
            if latest.get('bos_bullish', False):
                score += 0.3
                details['bos'] = 'bullish'
            elif latest.get('bos_bearish', False):
                score -= 0.3
                details['bos'] = 'bearish'

            # FVG signals (medium)
            if latest.get('fvg_bullish', False):
                score += 0.2
                details['fvg'] = 'bullish'
            elif latest.get('fvg_bearish', False):
                score -= 0.2
                details['fvg'] = 'bearish'

            # Market structure
            structure = latest.get('market_structure', 'neutral')
            if structure == 'bullish':
                score += 0.1
            elif structure == 'bearish':
                score -= 0.1
            details['structure'] = structure

            # Normalize to [-1, 1]
            score = max(-1.0, min(1.0, score))
            signal = 'bullish' if score > 0.15 else (
                'bearish' if score < -0.15 else 'neutral')

            return score, signal, details

        except Exception as e:
            logger.error(f"Error in SMC analysis: {e}")
            return 0.0, 'neutral', {'error': str(e)}

    def _get_ta_score(self, df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Get Technical Analysis component score"""
        try:
            df_ta = self.ta_indicators.add_all_indicators(df)
            latest = df_ta.iloc[-1]

            score = 0.0
            details = {}

            # RSI (oversold/overbought)
            rsi = latest.get('rsi', 50)
            if rsi < 30:
                score += 0.3  # Oversold, bullish
                details['rsi'] = 'oversold'
            elif rsi > 70:
                score -= 0.3  # Overbought, bearish
                details['rsi'] = 'overbought'
            else:
                details['rsi'] = 'neutral'

            # EMA trend
            trend = self.ta_indicators.get_trend_signal(df_ta).iloc[-1]
            if trend == 1:
                score += 0.4  # Bullish trend
                details['ema_trend'] = 'bullish'
            elif trend == -1:
                score -= 0.4  # Bearish trend
                details['ema_trend'] = 'bearish'
            else:
                details['ema_trend'] = 'neutral'

            # MACD
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            if macd > macd_signal:
                score += 0.3
                details['macd'] = 'bullish'
            elif macd < macd_signal:
                score -= 0.3
                details['macd'] = 'bearish'

            # Normalize
            score = max(-1.0, min(1.0, score))
            signal = 'bullish' if score > 0.15 else (
                'bearish' if score < -0.15 else 'neutral')

            return score, signal, details

        except Exception as e:
            logger.error(f"Error in TA analysis: {e}")
            return 0.0, 'neutral', {'error': str(e)}

    def _get_ml_score(self, df: pd.DataFrame, symbol: str) -> Tuple[float, str, Dict]:
        """Get ML model prediction score"""
        try:
            # Get or create symbol-specific model
            if symbol not in self.ml_models:
                logger.info(f"Creating ML model for {symbol}")
                self.ml_models[symbol] = TradingModel(
                    self.config, symbol=symbol)

            ml_model = self.ml_models[symbol]

            if not ml_model.is_trained:
                logger.warning(
                    f"ML model for {symbol} not trained, skipping ML prediction")
                return 0.0, 'neutral', {'status': 'not_trained'}

            # Get prediction
            prediction, confidence = ml_model.predict(df)

            # Convert prediction to score
            score = prediction * confidence  # -1 to 1
            signal = 'bullish' if prediction == 1 else (
                'bearish' if prediction == -1 else 'neutral')

            details = {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'status': 'trained'
            }

            return score, signal, details

        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return 0.0, 'neutral', {'error': str(e)}

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
        Determine if trade should be executed based on confidence and agreement

        Args:
            signal_result: Output from analyze_signal()

        Returns:
            True if trade should be executed
        """
        # Check confidence threshold
        if signal_result['confidence'] < self.min_confidence:
            logger.info(
                f"Trade rejected: Low confidence ({signal_result['confidence']:.2f} < {self.min_confidence})")
            return False

        # Check agreement threshold
        if signal_result['agreement'] < (self.min_signals / 4):
            logger.info(
                f"Trade rejected: Low agreement ({signal_result['agreement']:.2f})")
            return False

        # Must not be hold signal
        if signal_result['signal'] == 'hold':
            return False

        return True
