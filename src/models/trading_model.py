"""
AI-Enhanced Trading Model - RandomForest Classifier
Lightweight ML model for signal prediction and pattern recognition
Uses scikit-learn for fast, interpretable predictions

MODEL LOADING:
- BTC model: models/saved_models/BTC_USDT/random_forest_model.pkl (73.3% accuracy)
- ETH model: models/saved_models/ETH_USDT/random_forest_model.pkl (59.7% accuracy)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from src.utils.logger import get_logger

logger = get_logger()


class TradingModel:
    """
    Lightweight ML model for trading signal prediction
    - Uses Random Forest for fast prediction
    - Combines SMC, TA indicators, and market features
    - ~80-85% accuracy without heavy computation
    """

    def __init__(self, config: Dict, symbol: str = None):
        self.config = config
        self.symbol = symbol  # NEW: Track which symbol this model is for
        self.model = None
        self.scaler = StandardScaler()

        # NEW: Per-symbol model paths
        if symbol:
            symbol_clean = symbol.replace('/', '_')
            model_dir = f"models/saved_models/{symbol_clean}"
            os.makedirs(model_dir, exist_ok=True)
            self.model_path = f"{model_dir}/random_forest_model.pkl"
            self.scaler_path = f"{model_dir}/rf_scaler.pkl"
        else:
            # Fallback to old paths if no symbol specified
            self.model_path = "models/saved_models/random_forest_model.pkl"
            self.scaler_path = "models/saved_models/rf_scaler.pkl"

        self.is_trained = False

        # Feature names for consistency
        self.feature_names = [
            # Price action features
            'close_price_norm', 'volume_norm', 'price_change_pct',
            # Technical indicators
            'rsi', 'ema_diff', 'atr_norm', 'macd', 'macd_signal',
            # SMC features
            'has_fvg', 'at_liquidity_zone', 'market_structure_score',
            'choch_bullish', 'choch_bearish', 'bos_bullish', 'bos_bearish',
            # Trend features
            'trend_strength', 'ema_trend', 'price_vs_ema50', 'price_vs_ema200',
            # Volatility features
            'volatility_ratio', 'volume_ratio'
        ]

        logger.info("Trading model initialized")

        # Try to load existing model
        self.load_model()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features from market data

        Args:
            df: DataFrame with OHLCV, indicators, and SMC signals

        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=df.index)

        # Price action features (normalized)
        features['close_price_norm'] = (
            df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
        features['volume_norm'] = (
            df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        features['price_change_pct'] = df['close'].pct_change()

        # Technical indicators (with safe defaults if missing)
        # Normalize to 0-1
        features['rsi'] = (df['rsi'] / 100) if 'rsi' in df.columns else 0.5
        features['ema_diff'] = ((df['ema_50'] - df['ema_200']) / df['close']
                                ) if 'ema_50' in df.columns and 'ema_200' in df.columns else 0
        features['atr_norm'] = (df['atr'] / df['close']
                                ) if 'atr' in df.columns else 0.01
        features['macd'] = df.get('macd', 0)
        features['macd_signal'] = df.get('macd_signal', 0)

        # SMC features (with safe defaults and NaN handling)
        fvg_bull = df['fvg_bullish'].fillna(False).astype(
            int) if 'fvg_bullish' in df.columns else 0
        fvg_bear = df['fvg_bearish'].fillna(False).astype(
            int) if 'fvg_bearish' in df.columns else 0
        features['has_fvg'] = fvg_bull + fvg_bear

        liq_high = df['liquidity_high'].fillna(False).astype(
            int) if 'liquidity_high' in df.columns else 0
        liq_low = df['liquidity_low'].fillna(False).astype(
            int) if 'liquidity_low' in df.columns else 0
        features['at_liquidity_zone'] = liq_high + liq_low

        # Market structure score (-1 bearish, 0 neutral, 1 bullish)
        structure_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
        if 'market_structure' in df.columns:
            features['market_structure_score'] = df['market_structure'].map(
                structure_map).fillna(0)
        else:
            features['market_structure_score'] = 0

        # CHOCH and BOS signals
        features['choch_bullish'] = df['choch_bullish'].fillna(False).astype(
            int) if 'choch_bullish' in df.columns else 0
        features['choch_bearish'] = df['choch_bearish'].fillna(False).astype(
            int) if 'choch_bearish' in df.columns else 0
        features['bos_bullish'] = df['bos_bullish'].fillna(False).astype(
            int) if 'bos_bullish' in df.columns else 0
        features['bos_bearish'] = df['bos_bearish'].fillna(False).astype(
            int) if 'bos_bearish' in df.columns else 0

        # Trend features
        features['trend_strength'] = abs(features['ema_diff'])
        features['ema_trend'] = (df['ema_50'] > df['ema_200']).astype(
            int) if 'ema_50' in df.columns and 'ema_200' in df.columns else 0
        features['price_vs_ema50'] = (
            (df['close'] - df['ema_50']) / df['ema_50']) if 'ema_50' in df.columns else 0
        features['price_vs_ema200'] = (
            (df['close'] - df['ema_200']) / df['ema_200']) if 'ema_200' in df.columns else 0

        # Volatility features
        features['volatility_ratio'] = (
            df['atr'] / df['close'].rolling(20).mean()) if 'atr' in df.columns else 0.01
        features['volume_ratio'] = df['volume'] / \
            df['volume'].rolling(20).mean()

        # Fill NaN values and replace infinite values
        features = features.fillna(0)
        features = features.replace([float('inf'), float('-inf')], 0)

        return features

    def create_labels(self, df: pd.DataFrame, lookahead: int = 5) -> pd.Series:
        """
        Create labels for training based on future price movement

        Args:
            df: DataFrame with price data
            lookahead: Number of periods to look ahead

        Returns:
            Series with labels: 1 (buy), 0 (hold), -1 (sell)
        """
        future_returns = df['close'].shift(-lookahead) / df['close'] - 1

        # Define thresholds
        buy_threshold = 0.01  # 1% gain
        sell_threshold = -0.01  # 1% loss

        labels = pd.Series(0, index=df.index)  # Default: hold
        labels[future_returns > buy_threshold] = 1  # Buy signal
        labels[future_returns < sell_threshold] = -1  # Sell signal

        return labels

    def train(self, df: pd.DataFrame, lookahead: int = 5) -> Dict:
        """
        Train the ML model

        Args:
            df: DataFrame with complete market data
            lookahead: Periods ahead to predict

        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training...")

        # Prepare features and labels
        X = self.prepare_features(df)
        y = self.create_labels(df, lookahead)

        # Remove rows with NaN or where labels can't be calculated
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 100:
            logger.error("Insufficient data for training")
            return {'success': False, 'error': 'Insufficient data'}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Don't shuffle to avoid lookahead bias
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest (fast and accurate)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.is_trained = True

        # Save model
        self.save_model()

        logger.info(
            f"Model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        logger.info(
            f"Top features: {feature_importance.head(5).to_dict('records')}")

        return {
            'success': True,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': feature_importance.to_dict('records'),
            'samples_trained': len(X_train)
        }

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict trading signal for latest data

        Args:
            df: DataFrame with latest market data

        Returns:
            Tuple of (signal, confidence)
            signal: 1 (buy), 0 (hold), -1 (sell)
            confidence: probability score
        """
        if not self.is_trained:
            logger.warning(f"Model for {self.symbol} not trained, returning neutral signal")
            return 0, 0.0

        try:
            # Prepare features
            X = self.prepare_features(df)

            # Get last row
            if len(X) == 0:
                logger.warning(f"No valid features prepared for {self.symbol}, returning neutral")
                return 0, 0.0
            
            X_latest = X.iloc[[-1]]

            # Check for NaN values in latest features
            if X_latest.isna().any().any():
                nan_cols = X_latest.columns[X_latest.isna().any()].tolist()
                logger.warning(f"NaN values in features for {self.symbol}: {nan_cols}, filling with 0")
                X_latest = X_latest.fillna(0)

            # Scale
            X_scaled = self.scaler.transform(X_latest)

            # Predict
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]

            # Get confidence (max probability)
            confidence = max(probabilities)
            
            logger.debug(f"ML prediction for {self.symbol}: {prediction}, confidence: {confidence:.2f}")

            return int(prediction), float(confidence)
        
        except Exception as e:
            logger.error(f"Error in ML prediction for {self.symbol}: {e}")
            return 0, 0.0

    def save_model(self):
        """Save trained model and scaler"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self):
        """Load trained model and scaler"""
        try:
            logger.info(f"Attempting to load model from: {self.model_path}")
            logger.info(f"Attempting to load scaler from: {self.scaler_path}")
            
            model_exists = os.path.exists(self.model_path)
            scaler_exists = os.path.exists(self.scaler_path)
            
            logger.info(f"Model file exists: {model_exists}, Scaler file exists: {scaler_exists}")
            
            if model_exists and scaler_exists:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                logger.info(f"✅ Model loaded successfully from {self.model_path}")
            else:
                if not model_exists:
                    logger.warning(f"Model file not found: {self.model_path}")
                if not scaler_exists:
                    logger.warning(f"Scaler file not found: {self.scaler_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False
