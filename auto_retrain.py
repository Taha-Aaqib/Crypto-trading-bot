"""
Automatic Model Retraining System
Monitors model performance and retrains when needed.
Includes out-of-sample walk-forward validation to reject bad models.
"""

import os
import shutil
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from train_mtf_models import train_symbol_model, load_config_file, fetch_multi_tf_data
from src.data.data_fetcher import DataFetcher
from src.models.mtf_pattern_labeler import MultiTimeframePatternLabeler
from src.models.pattern_recognition_model import PatternRecognitionModel
from src.utils.logger import get_logger
from src.utils.db_manager import DatabaseManager

logger = get_logger()


class AutoRetrainer:
    """
    Automatically retrain ML model based on:
    1. Time-based: Every N days (configured)
    2. Performance-based: When win rate drops below threshold
    3. Data-based: When enough new data accumulated
    """

    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.db_manager = DatabaseManager(self.config)
        # Check for per-symbol pattern models (the actual model files the bot uses)
        symbols = self.config.get('trading', {}).get('symbols', ['BTC/USDT'])
        first_symbol = symbols[0].replace('/', '_')
        self.model_path = Path(f"models/saved_models/{first_symbol}/mtf_pattern_model.pkl")
        self.metadata_path = Path("models/saved_models/model_metadata.yaml")

        # Retraining parameters
        ml_config = self.config.get('ml_model', {})
        self.retrain_interval_days = ml_config.get('retrain_interval_days', 7)
        self.min_win_rate = 0.60  # Retrain if win rate drops below 60%
        self.min_new_trades = 50  # Retrain after 50 new trades
        self.min_oos_accuracy = 0.55  # Reject model if out-of-sample accuracy < 55%

        logger.info(
            f"AutoRetrainer initialized - Interval: {self.retrain_interval_days} days")

    def should_retrain(self) -> tuple[bool, str]:
        """
        Check if model should be retrained

        Returns:
            (should_retrain: bool, reason: str)
        """

        # Check 1: Does model exist?
        if not self.model_path.exists():
            return True, "No trained model found"

        # Check 2: Time-based retraining
        last_train_time = self._get_last_training_time()
        if last_train_time:
            days_since_training = (datetime.now() - last_train_time).days
            if days_since_training >= self.retrain_interval_days:
                return True, f"Time-based: {days_since_training} days since last training"

        # Check 3: Performance-based retraining
        recent_win_rate = self._get_recent_win_rate()
        if recent_win_rate is not None and recent_win_rate < self.min_win_rate:
            return True, f"Performance-based: Win rate {recent_win_rate:.1%} < {self.min_win_rate:.1%}"

        # Check 4: New data accumulated
        trades_since_training = self._get_trades_since_last_training()
        if trades_since_training >= self.min_new_trades:
            return True, f"Data-based: {trades_since_training} new trades accumulated"

        return False, "Model is up-to-date"

    def retrain_if_needed(self) -> dict:
        """
        Check and retrain model if criteria met

        Returns:
            Dictionary with retraining results
        """
        should_retrain, reason = self.should_retrain()

        if not should_retrain:
            logger.info(f"Model check passed: {reason}")
            return {
                'retrained': False,
                'reason': reason,
                'timestamp': datetime.now()
            }

        logger.info(f"Retraining triggered: {reason}")

        # Perform retraining (trains PatternRecognitionModel — the model the bot actually uses)
        try:
            symbols = self.config.get('trading', {}).get('symbols', ['BTC/USDT'])
            results = {}
            all_validated = True

            for symbol in symbols:
                # Backup current model before retraining
                symbol_dir = Path(f"models/saved_models/{symbol.replace('/', '_')}")
                backup_dir = Path(f"models/saved_models/{symbol.replace('/', '_')}_backup")
                model_existed = symbol_dir.exists() and (symbol_dir / 'mtf_pattern_model.pkl').exists()

                if model_existed:
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                    shutil.copytree(symbol_dir, backup_dir)

                # Train new model
                result = train_symbol_model(self.config, symbol, days=90)
                results[symbol] = result

                if result.get('success'):
                    # Walk-forward validation: test on unseen recent data
                    oos_accuracy = self._validate_out_of_sample(symbol)

                    if oos_accuracy is not None and oos_accuracy < self.min_oos_accuracy:
                        logger.warning(
                            f"⚠️ {symbol} model REJECTED: OOS accuracy {oos_accuracy:.1%} < {self.min_oos_accuracy:.1%}")
                        result['validated'] = False
                        result['oos_accuracy'] = oos_accuracy
                        all_validated = False

                        # Restore backup
                        if model_existed and backup_dir.exists():
                            shutil.rmtree(symbol_dir)
                            shutil.copytree(backup_dir, symbol_dir)
                            logger.info(f"  Restored previous model for {symbol}")
                    else:
                        result['validated'] = True
                        result['oos_accuracy'] = oos_accuracy
                        logger.info(
                            f"✅ {symbol} model VALIDATED: OOS accuracy {oos_accuracy:.1%}" if oos_accuracy else
                            f"✅ {symbol} model trained (OOS validation skipped — insufficient data)")

                # Clean up backup
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)

            # Save metadata
            self._save_training_metadata(results)

            logger.info("✅ Automatic retraining completed successfully")

            return {
                'retrained': True,
                'reason': reason,
                'results': results,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"❌ Automatic retraining failed: {e}")
            return {
                'retrained': False,
                'reason': f"Training failed: {e}",
                'timestamp': datetime.now()
            }

    def _get_last_training_time(self) -> datetime:
        """Get timestamp of last training from metadata"""
        if not self.metadata_path.exists():
            return None

        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
            return datetime.fromisoformat(metadata.get('last_training_time'))
        except Exception:
            return None

    def _get_recent_win_rate(self, last_n_trades: int = 50) -> float:
        """Calculate win rate from recent trades"""
        try:
            trades_df = self.db_manager.get_trade_history(limit=last_n_trades)

            if len(trades_df) < 10:  # Need at least 10 trades
                return None

            # Filter closed trades only
            closed_trades = trades_df[trades_df['exit_time'].notna()]

            if len(closed_trades) == 0:
                return None

            winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
            win_rate = winning_trades / len(closed_trades)

            logger.info(
                f"Recent win rate: {win_rate:.1%} ({winning_trades}/{len(closed_trades)})")
            return win_rate

        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return None

    def _get_trades_since_last_training(self) -> int:
        """Count trades executed since last training"""
        last_train_time = self._get_last_training_time()

        if not last_train_time:
            return 0

        try:
            trades_df = self.db_manager.get_trade_history(limit=1000)

            # Convert entry_time to datetime
            if not trades_df.empty:
                trades_df['entry_time'] = pd.to_datetime(
                    trades_df['entry_time'])
                new_trades = trades_df[trades_df['entry_time']
                                       > last_train_time]
                return len(new_trades)

            return 0

        except Exception as e:
            logger.error(f"Error counting new trades: {e}")
            return 0

    def _validate_out_of_sample(self, symbol: str) -> float:
        """
        Walk-forward out-of-sample validation:
        1. Fetch 90 days of data (same as training used)
        2. Use the last 20% as out-of-sample test set
        3. Generate features/labels on that OOS window
        4. Predict with the newly trained model
        5. Return accuracy on OOS data

        Returns None if insufficient data for validation.
        """
        try:
            fetcher = DataFetcher(self.config)
            labeler = MultiTimeframePatternLabeler(self.config)

            # Load the just-trained model
            model = PatternRecognitionModel(self.config, symbol=symbol)
            model.load_model()
            if not model.is_trained:
                logger.warning(f"Cannot validate {symbol}: model not loaded")
                return None

            # Fetch data (same 90-day window used in training)
            df_1d, df_4h, df_1h, df_15m = fetch_multi_tf_data(fetcher, symbol, days=90)

            if len(df_15m) < 500:
                logger.warning(f"Cannot validate {symbol}: insufficient 15M data ({len(df_15m)} bars)")
                return None

            # Analyze patterns
            analyzed_data = labeler.analyze_all_timeframes(df_1d, df_4h, df_1h, df_15m)
            features_df, labels = labeler.generate_labels(analyzed_data)

            if len(features_df) < 20:
                logger.warning(f"Cannot validate {symbol}: insufficient labeled samples ({len(features_df)})")
                return None

            # Use last 20% as OOS (this data was the most recent — less likely to be in training set)
            oos_start = int(len(features_df) * 0.8)
            oos_features = features_df.iloc[oos_start:]
            oos_labels = labels.iloc[oos_start:] if hasattr(labels, 'iloc') else labels[oos_start:]

            if len(oos_features) < 5:
                logger.warning(f"Cannot validate {symbol}: OOS set too small ({len(oos_features)})")
                return None

            # Predict on OOS data (skip neutral labels — model only predicts win/loss)
            correct = 0
            total = 0
            for i in range(len(oos_features)):
                actual = oos_labels.iloc[i] if hasattr(oos_labels, 'iloc') else oos_labels[i]
                if actual == 0:  # Skip neutral labels
                    continue

                row_dict = oos_features.iloc[i].to_dict()
                pred_direction, pred_confidence = model.predict_from_features(row_dict)

                if pred_confidence < 0.1:  # Skip very low confidence (model abstains)
                    continue

                # Map: actual label 1 = win, -1 = loss; pred_direction > 0 = win, < 0 = loss
                pred_label = 1 if pred_direction > 0 else -1
                if pred_label == actual:
                    correct += 1
                total += 1

            if total == 0:
                logger.warning(f"Cannot validate {symbol}: no OOS predictions made")
                return None

            accuracy = correct / total
            logger.info(f"[OOS Validation] {symbol}: {correct}/{total} correct = {accuracy:.1%}")
            return accuracy

        except Exception as e:
            logger.error(f"Error in OOS validation for {symbol}: {e}")
            return None

    def _save_training_metadata(self, results: dict):
        """Save training metadata for tracking"""
        metadata = {
            'last_training_time': datetime.now().isoformat(),
            'training_results': {
                symbol: {
                    'success': result.get('success', False),
                    'test_accuracy': result.get('test_accuracy', 0.0),
                    'train_accuracy': result.get('train_accuracy', 0.0),
                    'samples_trained': result.get('samples_trained', 0),
                    'oos_accuracy': result.get('oos_accuracy'),
                    'validated': result.get('validated', False)
                }
                for symbol, result in results.items()
            }
        }

        os.makedirs(self.metadata_path.parent, exist_ok=True)
        with open(self.metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        logger.info(f"Training metadata saved to {self.metadata_path}")


def check_and_retrain():
    """
    Standalone function to check and retrain if needed
    Can be called from main bot or run as a cron job
    """
    logger.info("=" * 60)
    logger.info("AUTOMATIC RETRAINING CHECK")
    logger.info("=" * 60)

    retrainer = AutoRetrainer()
    result = retrainer.retrain_if_needed()

    if result['retrained']:
        logger.info("✅ Model retrained successfully")
        logger.info(f"   Reason: {result['reason']}")
    else:
        logger.info(f"ℹ️  No retraining needed: {result['reason']}")

    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Automatic model retraining')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retrain regardless of criteria'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check if retraining needed, do not retrain'
    )

    args = parser.parse_args()

    retrainer = AutoRetrainer()

    if args.force:
        logger.info("🔄 Force retraining with OOS validation...")
        config = load_config_file()
        symbols = config.get('trading', {}).get('symbols', ['BTC/USDT'])
        results = {}
        for symbol in symbols:
            symbol_dir = Path(f"models/saved_models/{symbol.replace('/', '_')}")
            backup_dir = Path(f"models/saved_models/{symbol.replace('/', '_')}_backup")
            model_existed = symbol_dir.exists() and (symbol_dir / 'mtf_pattern_model.pkl').exists()

            # Backup current model before retraining
            if model_existed:
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(symbol_dir, backup_dir)

            result = train_symbol_model(config, symbol, days=90)
            results[symbol] = result
            if result.get('success'):
                oos_accuracy = retrainer._validate_out_of_sample(symbol)
                result['oos_accuracy'] = oos_accuracy
                result['validated'] = oos_accuracy is None or oos_accuracy >= retrainer.min_oos_accuracy
                if result['validated']:
                    logger.info(f"✅ {symbol} VALIDATED: OOS accuracy {oos_accuracy:.1%}" if oos_accuracy else
                                f"✅ {symbol} trained (OOS validation skipped — insufficient data)")
                else:
                    logger.warning(f"⚠️ {symbol} REJECTED: OOS accuracy {oos_accuracy:.1%} < {retrainer.min_oos_accuracy:.1%}")
                    # Restore backup if OOS failed
                    if model_existed and backup_dir.exists():
                        shutil.rmtree(symbol_dir)
                        shutil.copytree(backup_dir, symbol_dir)
                        logger.info(f"  Restored previous model for {symbol}")

            # Clean up backup
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

        retrainer._save_training_metadata(results)
        logger.info("✅ Force retrain complete")

    elif args.check_only:
        should_retrain, reason = retrainer.should_retrain()
        print(f"\nShould retrain: {should_retrain}")
        print(f"Reason: {reason}\n")

    else:
        check_and_retrain()
