"""
Automatic Model Retraining System
Monitors model performance and retrains when needed
"""

import os
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from train_model import train_all_models, train_model_for_symbol
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
        self.model_path = Path("models/saved_models/trading_model.pkl")
        self.metadata_path = Path("models/saved_models/model_metadata.yaml")

        # Retraining parameters
        ml_config = self.config.get('ml_model', {})
        self.retrain_interval_days = ml_config.get('retrain_interval_days', 7)
        self.min_win_rate = 0.60  # Retrain if win rate drops below 60%
        self.min_new_trades = 50  # Retrain after 50 new trades

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

        # Perform retraining
        try:
            results = train_all_models(days_back=365)

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
        except:
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

    def _save_training_metadata(self, results: dict):
        """Save training metadata for tracking"""
        metadata = {
            'last_training_time': datetime.now().isoformat(),
            'training_results': {
                symbol: {
                    'success': result.get('success', False),
                    'test_accuracy': result.get('test_accuracy', 0.0),
                    'train_accuracy': result.get('train_accuracy', 0.0),
                    'samples_trained': result.get('samples_trained', 0)
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
        logger.info("🔄 Force retraining...")
        results = train_all_models(days_back=365)
        retrainer._save_training_metadata(results)
        logger.info("✅ Force retrain complete")

    elif args.check_only:
        should_retrain, reason = retrainer.should_retrain()
        print(f"\nShould retrain: {should_retrain}")
        print(f"Reason: {reason}\n")

    else:
        check_and_retrain()
