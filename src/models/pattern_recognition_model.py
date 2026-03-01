"""
Multi-Timeframe Pattern Recognition ML Model
Recognizes high-probability SMC pattern combinations across multiple timeframes

ARCHITECTURE:
- Input: Multi-TF features (1D + 4H + 1H + 15M combined)
- Output: Pattern quality score and confidence
- NOT price prediction - pure pattern effectiveness classification

FLOW:
1. Receives analyzed data from all 4 timeframes
2. Creates unified feature vector from multi-TF cascade
3. Predicts: Will this pattern combination be profitable?
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
from src.utils.logger import get_logger

logger = get_logger()


class PatternRecognitionModel:
    """
    Multi-Timeframe Pattern Recognition Model
    
    Learns: "When 1D bias + 4H structure + 1H pattern + 15M entry = profitable?"
    
    Each asset gets its own model to learn asset-specific behavior.
    """

    def __init__(self, config: Dict, symbol: str = None):
        self.config = config
        self.symbol = symbol
        self.model = None
        self.scaler = StandardScaler()
        
        # Model paths - per symbol
        if symbol:
            symbol_clean = symbol.replace('/', '_')
            model_dir = f"models/saved_models/{symbol_clean}"
            os.makedirs(model_dir, exist_ok=True)
            self.model_path = f"{model_dir}/mtf_pattern_model.pkl"
            self.scaler_path = f"{model_dir}/mtf_pattern_scaler.pkl"
            self.stats_path = f"{model_dir}/mtf_pattern_stats.pkl"
        else:
            self.model_path = "models/saved_models/mtf_pattern_model.pkl"
            self.scaler_path = "models/saved_models/mtf_pattern_scaler.pkl"
            self.stats_path = "models/saved_models/mtf_pattern_stats.pkl"
        
        self.is_trained = False
        self.pattern_stats = {}
        self.feature_columns = []
        
        logger.info(f"PatternRecognitionModel (Multi-TF) initialized for {symbol}")
        self.load_model()

    def train(self, features_df: pd.DataFrame, labels: pd.Series, pattern_stats: Dict = None) -> Dict:
        """
        Train multi-TF pattern recognition model.
        
        Args:
            features_df: Multi-TF features from MTFPatternLabeler
            labels: Outcome labels (+1 winner, -1 loser, 0 neutral)
            pattern_stats: Optional stats about pattern effectiveness
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Starting Multi-TF pattern model training for {self.symbol}...")
        
        if pattern_stats:
            self.pattern_stats = pattern_stats
        
        # Store feature columns for prediction
        self.feature_columns = features_df.columns.tolist()
        
        # Remove neutral labels (nothing to learn from)
        valid_idx = labels != 0
        X = features_df[valid_idx]
        y = labels[valid_idx]
        
        # Handle NaN values (early candles may lack indicator data like ATR/EMA)
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            logger.info(f"Filling {nan_count} NaN values with 0 (missing indicator data)")
            X = X.fillna(0)
        
        if len(X) < 10:
            logger.error(f"Insufficient labeled patterns for training: {len(X)}")
            return {'success': False, 'error': 'Insufficient labeled data'}
        
        # Log class distribution
        win_count = (y == 1).sum()
        loss_count = (y == -1).sum()
        logger.info(f"Training on {len(X)} labeled patterns (wins: {win_count}, losses: {loss_count})")
        
        # ===== FEATURE SELECTION (remove noise features) =====
        # Quick pre-fit to identify important features
        pre_model = GradientBoostingClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, random_state=42
        )
        X_pre_scaled = self.scaler.fit_transform(X)
        pre_model.fit(X_pre_scaled, y)
        
        # Keep top K features to maintain a healthy feature-to-sample ratio (~3:1)
        importances = pre_model.feature_importances_
        max_features = min(len(X) // 3, len(X.columns))  # At most 1 feature per 3 samples
        max_features = max(max_features, 5)  # Keep at least 5
        
        # Sort by importance and keep top K
        feature_importance_pairs = sorted(
            zip(X.columns, importances), key=lambda x: x[1], reverse=True
        )
        selected_features = [col for col, _ in feature_importance_pairs[:max_features]]
        
        if len(selected_features) >= len(X.columns):
            logger.info(f"Feature selection: keeping all {len(selected_features)} features")
        else:
            logger.info(f"Feature selection: {len(X.columns)} → {len(selected_features)} features "
                        f"(kept top {len(selected_features)} by importance)")
        
        X = X[selected_features]
        self.feature_columns = selected_features
        
        # ===== TIME-SERIES CROSS-VALIDATION =====
        # Use TimeSeriesSplit (respects temporal order, no data leakage)
        n_splits = min(5, max(2, len(X) // 20))  # Adaptive: need ~20 samples per fold
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_train_scores = []
        cv_test_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_fold_train = self.scaler.fit_transform(X.iloc[train_idx])
            X_fold_test = self.scaler.transform(X.iloc[test_idx])
            y_fold_train = y.iloc[train_idx]
            y_fold_test = y.iloc[test_idx]
            
            # Adapt complexity to dataset size
            _n_est = 100 if len(X) >= 200 else 50
            _min_split = max(5, min(20, len(X) // 10))
            _min_leaf = max(3, min(10, len(X) // 15))
            fold_model = GradientBoostingClassifier(
                n_estimators=_n_est,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=_min_split,
                min_samples_leaf=_min_leaf,
                subsample=0.75,
                max_features='sqrt',
                random_state=42
            )
            fold_model.fit(X_fold_train, y_fold_train)
            
            train_acc = fold_model.score(X_fold_train, y_fold_train)
            test_acc = fold_model.score(X_fold_test, y_fold_test)
            cv_train_scores.append(train_acc)
            cv_test_scores.append(test_acc)
            logger.info(f"   CV Fold {fold + 1}/{n_splits}: train={train_acc:.3f}, test={test_acc:.3f}")
        
        avg_cv_train = np.mean(cv_train_scores)
        avg_cv_test = np.mean(cv_test_scores)
        overfit_gap = avg_cv_train - avg_cv_test
        logger.info(f"   CV Average: train={avg_cv_train:.3f}, test={avg_cv_test:.3f}, "
                     f"overfit_gap={overfit_gap:.3f}")
        
        if overfit_gap > 0.20:
            logger.warning(f"⚠️  High overfitting detected (gap={overfit_gap:.3f}). "
                           f"Model predictions may be unreliable.")
        
        # ===== FINAL MODEL TRAINING =====
        # Train on all data with the regularized hyperparameters
        # Use time-series split for final train/test
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Adapt complexity to dataset size
        _n_est = 100 if len(X) >= 200 else 50
        _min_split = max(5, min(20, len(X) // 10))
        _min_leaf = max(3, min(10, len(X) // 15))
        self.model = GradientBoostingClassifier(
            n_estimators=_n_est,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=_min_split,
            min_samples_leaf=_min_leaf,
            subsample=0.75,
            max_features='sqrt',
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Feature importance (from final model)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        self.save_model()
        
        logger.info(f"✅ Multi-TF Pattern Model trained!")
        logger.info(f"   Final train: {train_score:.3f}, Final test: {test_score:.3f}")
        logger.info(f"   CV test (avg): {avg_cv_test:.3f} ± {np.std(cv_test_scores):.3f}")
        logger.info(f"   Features used: {len(self.feature_columns)}")
        logger.info(f"   Top 10 features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"      {row['feature']}: {row['importance']:.4f}")
        
        return {
            'success': True,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_test_accuracy': avg_cv_test,
            'cv_test_std': float(np.std(cv_test_scores)),
            'cv_folds': n_splits,
            'overfit_gap': overfit_gap,
            'feature_importance': feature_importance.to_dict('records'),
            'features_selected': len(self.feature_columns),
            'features_original': len(features_df.columns),
            'samples_trained': len(X_train),
            'wins': int(win_count),
            'losses': int(loss_count),
            'pattern_stats': self.pattern_stats
        }

    def predict_from_features(self, features: Dict) -> Tuple[float, float]:
        """
        Predict pattern quality from pre-computed multi-TF features.
        
        Args:
            features: Dict of multi-TF features from MTFPatternLabeler.create_multi_tf_features()
            
        Returns:
            Tuple of (pattern_score, confidence)
            pattern_score: -1 (strong bearish) to +1 (strong bullish)
            confidence: 0-1 how confident is ML
        """
        if not self.is_trained or self.model is None:
            logger.warning(f"Pattern model for {self.symbol} not trained")
            return 0.0, 0.0
        
        try:
            # Create DataFrame from features dict
            X = pd.DataFrame([features])
            
            # Ensure all expected columns are present
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            
            # Keep only expected columns in correct order
            X = X[self.feature_columns]
            
            # Handle NaN values (same as training)
            X = X.fillna(0)
            
            # Scale
            X_scaled = self.scaler.transform(X)
            
            # Get prediction
            prediction = self.model.predict(X_scaled)[0]  # -1 or 1
            probabilities = self.model.predict_proba(X_scaled)[0]  # [P(-1), P(1)]
            
            # Convert to score
            classes = self.model.classes_
            if prediction == 1:
                # Bullish prediction
                idx = list(classes).index(1)
                confidence = probabilities[idx]
                pattern_score = float(confidence)
            elif prediction == -1:
                # Bearish prediction
                idx = list(classes).index(-1)
                confidence = probabilities[idx]
                pattern_score = -float(confidence)
            else:
                confidence = 0.5
                pattern_score = 0.0
            
            logger.debug(
                f"MTF Pattern prediction for {self.symbol}: score={pattern_score:.2f}, "
                f"confidence={confidence:.2f}, classes={classes}"
            )
            
            return float(pattern_score), float(confidence)
        
        except Exception as e:
            logger.error(f"Error in MTF pattern prediction: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0

    def predict(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame, 
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        mtf_labeler
    ) -> Tuple[float, float]:
        """
        Full prediction from raw analyzed DataFrames.
        
        Args:
            df_1d, df_4h, df_1h, df_15m: Analyzed DataFrames for each TF
            mtf_labeler: MTFPatternLabeler instance for feature creation
            
        Returns:
            Tuple of (pattern_score, confidence)
        """
        if not self.is_trained or self.model is None:
            return 0.0, 0.0
        
        try:
            # Get latest 15M timestamp
            if len(df_15m) == 0:
                return 0.0, 0.0
            
            latest_timestamp = df_15m.index[-1]
            
            # Create multi-TF features
            analyzed_data = {
                '1d': df_1d,
                '4h': df_4h,
                '1h': df_1h,
                '15m': df_15m
            }
            features = mtf_labeler.create_multi_tf_features(analyzed_data, latest_timestamp)
            
            return self.predict_from_features(features)
        
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            return 0.0, 0.0

    def save_model(self):
        """Save trained model, scaler, and metadata."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # Save stats and feature columns
            metadata = {
                'pattern_stats': self.pattern_stats,
                'feature_columns': self.feature_columns
            }
            joblib.dump(metadata, self.stats_path)
            
            logger.info(f"✅ MTF Pattern model saved for {self.symbol}")
        except Exception as e:
            logger.error(f"Error saving MTF pattern model: {e}")

    def load_model(self):
        """Load trained model, scaler, and metadata."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                if os.path.exists(self.stats_path):
                    metadata = joblib.load(self.stats_path)
                    self.pattern_stats = metadata.get('pattern_stats', {})
                    self.feature_columns = metadata.get('feature_columns', [])
                
                # Only mark as trained if feature_columns were loaded
                if self.feature_columns:
                    self.is_trained = True
                    logger.info(f"✅ MTF Pattern model loaded for {self.symbol} ({len(self.feature_columns)} features)")
                else:
                    self.is_trained = False
                    logger.warning(f"⚠️ MTF Pattern model for {self.symbol} has no feature columns — needs retraining")
            else:
                logger.info(f"No saved MTF pattern model found for {self.symbol}")
        except Exception as e:
            logger.error(f"Error loading MTF pattern model: {e}")
            self.is_trained = False

    def get_model_summary(self) -> Dict:
        """Get summary of model status and stats."""
        return {
            'symbol': self.symbol,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_columns),
            'pattern_stats': self.pattern_stats
        }
