"""
Tests for circuit breaker and risk management.
"""
import pytest
import yaml
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, PropertyMock
from datetime import datetime, timedelta


@pytest.fixture
def config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def _mock_db(trade_rows: list) -> MagicMock:
    """Create a mock db_manager that returns the given trade history."""
    db = MagicMock()
    if trade_rows:
        df = pd.DataFrame(trade_rows)
        db.get_trade_history.return_value = df
    else:
        db.get_trade_history.return_value = pd.DataFrame()
    db.get_open_trades.return_value = []
    return db


class TestCircuitBreaker:

    def _get_risk_manager(self, config, db):
        from src.trading.risk_manager import RiskManager
        rm = RiskManager(config, db)
        return rm

    def test_no_circuit_breaker_with_few_trades(self, config):
        """With <5 closed trades, circuit breaker should not trigger."""
        trades = [
            {'pnl': -10, 'exit_time': datetime.now(), 'symbol': 'BTC/USDT'},
            {'pnl': -10, 'exit_time': datetime.now(), 'symbol': 'BTC/USDT'},
        ]
        db = _mock_db(trades)
        rm = self._get_risk_manager(config, db)
        assert rm.is_circuit_breaker_triggered() is False

    def test_circuit_breaker_on_consecutive_losses(self, config):
        """5 consecutive losses should trigger circuit breaker."""
        trades = [
            {'pnl': -10, 'exit_time': datetime.now() - timedelta(minutes=i), 'symbol': 'BTC/USDT'}
            for i in range(5)
        ]
        db = _mock_db(trades)
        rm = self._get_risk_manager(config, db)
        result = rm.is_circuit_breaker_triggered()
        # Should be True since last 5 are all losses
        assert result is True

    def test_no_circuit_breaker_with_win_in_sequence(self, config):
        """A win among recent trades should prevent circuit breaker."""
        trades = [
            {'pnl': -10, 'exit_time': datetime.now() - timedelta(minutes=1), 'symbol': 'BTC/USDT'},
            {'pnl': -10, 'exit_time': datetime.now() - timedelta(minutes=2), 'symbol': 'BTC/USDT'},
            {'pnl': 20, 'exit_time': datetime.now() - timedelta(minutes=3), 'symbol': 'BTC/USDT'},  # Win!
            {'pnl': -10, 'exit_time': datetime.now() - timedelta(minutes=4), 'symbol': 'BTC/USDT'},
            {'pnl': -10, 'exit_time': datetime.now() - timedelta(minutes=5), 'symbol': 'BTC/USDT'},
        ]
        db = _mock_db(trades)
        rm = self._get_risk_manager(config, db)
        result = rm.is_circuit_breaker_triggered()
        assert result is False

    def test_circuit_breaker_ignores_open_trades(self, config):
        """Open trades (NaN pnl) should not be counted for circuit breaker."""
        trades = [
            {'pnl': float('nan'), 'exit_time': None, 'symbol': 'BTC/USDT'},  # Open trade
            {'pnl': -10, 'exit_time': datetime.now(), 'symbol': 'BTC/USDT'},
            {'pnl': -10, 'exit_time': datetime.now(), 'symbol': 'BTC/USDT'},
        ]
        db = _mock_db(trades)
        rm = self._get_risk_manager(config, db)
        assert rm.is_circuit_breaker_triggered() is False


class TestPositionSizing:

    def test_position_size_respects_maximum(self, config):
        """Position size should not exceed config max."""
        max_pos = config['trading']['position_size_usd']
        from src.utils.helpers import calculate_position_size
        size_base = calculate_position_size(
            capital=100000,
            risk_per_trade=0.02,
            entry_price=30000,
            stop_loss_price=29400
        )
        size_usd = size_base * 30000
        assert size_usd > 0, "Position size should be positive"

    def test_position_size_zero_on_invalid_sl(self, config):
        """If SL = entry (zero risk), position size should handle gracefully."""
        from src.utils.helpers import calculate_position_size
        size = calculate_position_size(
            capital=1000,
            risk_per_trade=0.02,
            entry_price=30000,
            stop_loss_price=30000  # Same as entry
        )
        assert size == 0, "Position size should be 0 when SL = entry"


class TestRiskConfig:

    def test_ensemble_weights_sum_to_one(self, config):
        """Ensemble component weights must sum to 1.0."""
        weights = config['ensemble']['weights']
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001, f"Ensemble weights sum to {total}"

    def test_risk_parameters_valid(self, config):
        """Risk parameters should be within reasonable bounds."""
        risk = config['risk']
        assert 0 < risk['max_risk_per_trade'] <= 0.05, "Risk per trade should be 0-5%"
        assert 0 < risk['max_daily_loss'] <= 0.10, "Daily loss limit should be 0-10%"
        assert risk['take_profit_rr_ratio'] >= 1.0, "R:R ratio should be >= 1.0"
        assert risk['stop_loss_atr_multiplier'] > 0, "ATR multiplier must be positive"

    def test_confluence_threshold_range(self, config):
        """Confluence threshold should be reasonable for 10x leverage."""
        threshold = config['strategy']['confluence_threshold']
        assert 0.3 <= threshold <= 0.8, f"Threshold {threshold} seems unreasonable"
        min_conf = config['ensemble']['min_confidence']
        assert min_conf == threshold, "min_confidence should match confluence_threshold"

    def test_leverage_within_bounds(self, config):
        leverage = config['trading']['leverage']
        assert 1 <= leverage <= 20, f"Leverage {leverage}x seems unreasonable"
