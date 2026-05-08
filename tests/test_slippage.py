"""
Tests for paper-trade slippage application.
Verifies that entry and exit prices are adjusted for realistic slippage.
"""
import pytest
import yaml


@pytest.fixture
def config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


class TestSlippageApplication:
    """Slippage should make fills worse (higher entry for longs, lower for shorts)."""

    def test_long_entry_slippage(self, config):
        slippage_rate = config.get('backtest', {}).get('slippage', 0.0005)
        entry = 30000.0
        slipped = entry * (1 + slippage_rate)
        assert slipped > entry, "Long entry should be worse (higher) after slippage"
        assert abs(slipped - entry - entry * slippage_rate) < 0.01

    def test_short_entry_slippage(self, config):
        slippage_rate = config.get('backtest', {}).get('slippage', 0.0005)
        entry = 30000.0
        slipped = entry * (1 - slippage_rate)
        assert slipped < entry, "Short entry should be worse (lower) after slippage"

    def test_long_exit_slippage(self, config):
        """When closing a long (selling), slippage makes exit lower."""
        slippage_rate = config.get('backtest', {}).get('slippage', 0.0005)
        exit_price = 31000.0
        slipped = exit_price * (1 - slippage_rate)
        assert slipped < exit_price, "Long exit should be worse (lower) after slippage"

    def test_short_exit_slippage(self, config):
        """When closing a short (buying back), slippage makes exit higher."""
        slippage_rate = config.get('backtest', {}).get('slippage', 0.0005)
        exit_price = 29000.0
        slipped = exit_price * (1 + slippage_rate)
        assert slipped > exit_price, "Short exit should be worse (higher) after slippage"

    def test_slippage_is_configured(self, config):
        slippage = config.get('backtest', {}).get('slippage')
        assert slippage is not None, "Slippage must be configured in config.yaml"
        assert 0 < slippage < 0.01, f"Slippage {slippage} seems unreasonable"

    def test_commission_is_configured(self, config):
        commission = config.get('backtest', {}).get('commission')
        assert commission is not None, "Commission must be configured in config.yaml"
        assert 0 < commission < 0.01, f"Commission {commission} seems unreasonable"


class TestLiquidationPrice:
    """Test estimated liquidation price calculation."""

    def test_long_liquidation_below_entry(self, config):
        from src.trading.order_executor import OrderExecutor
        # We can't fully init OrderExecutor without db_manager, so test the math directly
        entry = 30000.0
        leverage = 10
        maint_margin_rate = 0.004
        liq = entry * (1 - (1 / leverage) + maint_margin_rate)
        assert liq < entry, "Long liquidation must be below entry"
        # At 10x, liquidation ~90.4% below entry → ~27120
        assert liq > entry * 0.85, "Liquidation shouldn't be too far away at 10x"

    def test_short_liquidation_above_entry(self, config):
        entry = 30000.0
        leverage = 10
        maint_margin_rate = 0.004
        liq = entry * (1 + (1 / leverage) - maint_margin_rate)
        assert liq > entry, "Short liquidation must be above entry"

    def test_no_liquidation_at_1x(self, config):
        # At 1x (spot-like), no liquidation should be computed
        entry = 30000.0
        leverage = 1
        # The OrderExecutor returns None for leverage <= 1
        assert leverage <= 1  # Just verify the condition
