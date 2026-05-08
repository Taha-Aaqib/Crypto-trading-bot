"""
Tests for stop loss calculation, take profit, and risk validation.
Tests the 3-tier SL system: swing-point → ATR → percentage fallback.
"""
import pytest
import pandas as pd
import numpy as np
import yaml


@pytest.fixture
def config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helper: build a minimal SMC DataFrame with swing highs/lows
# ---------------------------------------------------------------------------
def _make_smc_df(swings: dict, n_rows: int = 100) -> pd.DataFrame:
    """
    Create a DataFrame with swing_high / swing_low columns.
    `swings` is e.g. {'swing_low': {80: 29500}, 'swing_high': {80: 31000}}
    """
    df = pd.DataFrame(index=range(n_rows))
    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    for col, mapping in swings.items():
        for idx, val in mapping.items():
            df.loc[idx, col] = val
    return df


# ===========================================================================
# Stop-Loss Tests
# ===========================================================================
class TestStopLossCalculation:

    def _get_strategy(self, config):
        """Lazy-import to keep test discovery fast."""
        from src.trading.strategy import TradingStrategy
        return TradingStrategy(config)

    # --- Swing-point SL ---
    def test_long_sl_uses_swing_low(self, config):
        strategy = self._get_strategy(config)
        entry = 30000.0
        atr = 200.0
        # Place a swing low at 29600 (1.33% below entry → within 0.3–5%)
        df = _make_smc_df({'swing_low': {80: 29600}})
        sl = strategy._calculate_stop_loss(entry, 'long', atr, df_smc=df)
        # SL = swing_low - buffer, so should be < 29600 and < entry
        assert sl < 29600, f"SL {sl} should be below swing low 29600"
        assert sl < entry, f"SL {sl} should be below entry {entry}"

    def test_short_sl_uses_swing_high(self, config):
        strategy = self._get_strategy(config)
        entry = 30000.0
        atr = 200.0
        # Place a swing high at 30400 (1.33% above entry)
        df = _make_smc_df({'swing_high': {80: 30400}})
        sl = strategy._calculate_stop_loss(entry, 'short', atr, df_smc=df)
        assert sl > 30400, f"SL {sl} should be above swing high 30400"
        assert sl > entry, f"SL {sl} should be above entry {entry}"

    # --- ATR fallback ---
    def test_atr_fallback_when_no_swings(self, config):
        strategy = self._get_strategy(config)
        entry = 30000.0
        atr = 500.0  # Must be enough: atr*mult >= entry*sl_pct (500*1.5=750 > 600)
        multiplier = config['risk']['stop_loss_atr_multiplier']
        sl = strategy._calculate_stop_loss(entry, 'long', atr, df_smc=None)
        expected = entry - atr * multiplier
        assert abs(sl - expected) < 1.0, f"ATR-based SL {sl} ≠ expected {expected}"

    def test_atr_fallback_short(self, config):
        strategy = self._get_strategy(config)
        entry = 30000.0
        atr = 500.0  # Must be enough for ATR fallback to activate
        multiplier = config['risk']['stop_loss_atr_multiplier']
        sl = strategy._calculate_stop_loss(entry, 'short', atr, df_smc=None)
        expected = entry + atr * multiplier
        assert abs(sl - expected) < 1.0, f"ATR-based SL {sl} ≠ expected {expected}"

    # --- Percentage fallback ---
    def test_percent_fallback_when_atr_zero(self, config):
        strategy = self._get_strategy(config)
        entry = 30000.0
        atr = 0.0  # Invalid ATR
        sl_pct = config['risk'].get('stop_loss_percent', 0.02)
        sl = strategy._calculate_stop_loss(entry, 'long', atr, df_smc=None)
        # Should use the derived fallback ATR approach; result must still be below entry
        assert sl < entry, f"SL {sl} should be below entry {entry}"
        assert sl > 0, "SL must be positive"

    def test_percent_fallback_nan_atr(self, config):
        strategy = self._get_strategy(config)
        entry = 30000.0
        sl = strategy._calculate_stop_loss(entry, 'long', float('nan'), df_smc=None)
        assert sl < entry and sl > 0, f"SL {sl} invalid for NaN ATR"

    # --- SL always on correct side ---
    def test_long_sl_always_below_entry(self, config):
        strategy = self._get_strategy(config)
        for atr in [100, 500, 1000]:
            sl = strategy._calculate_stop_loss(30000, 'long', atr, df_smc=None)
            assert sl < 30000, f"Long SL={sl} must be below 30000 (ATR={atr})"

    def test_short_sl_always_above_entry(self, config):
        strategy = self._get_strategy(config)
        for atr in [100, 500, 1000]:
            sl = strategy._calculate_stop_loss(30000, 'short', atr, df_smc=None)
            assert sl > 30000, f"Short SL={sl} must be above 30000 (ATR={atr})"


# ===========================================================================
# Take-Profit Tests
# ===========================================================================
class TestTakeProfitCalculation:

    def _get_strategy(self, config):
        from src.trading.strategy import TradingStrategy
        return TradingStrategy(config)

    def test_tp_uses_risk_reward_ratio(self, config):
        strategy = self._get_strategy(config)
        rr = config['risk']['take_profit_rr_ratio']
        entry, sl = 30000.0, 29400.0
        tp = strategy._calculate_take_profit(entry, sl, 'long')
        risk = entry - sl
        expected = entry + risk * rr
        assert abs(tp - expected) < 0.01, f"TP {tp} ≠ expected {expected}"

    def test_tp_short(self, config):
        strategy = self._get_strategy(config)
        rr = config['risk']['take_profit_rr_ratio']
        entry, sl = 30000.0, 30600.0
        tp = strategy._calculate_take_profit(entry, sl, 'short')
        risk = sl - entry
        expected = entry - risk * rr
        assert abs(tp - expected) < 0.01, f"Short TP {tp} ≠ expected {expected}"

    def test_tp_long_above_entry(self, config):
        strategy = self._get_strategy(config)
        tp = strategy._calculate_take_profit(30000, 29400, 'long')
        assert tp > 30000, "Long TP must be above entry"

    def test_tp_short_below_entry(self, config):
        strategy = self._get_strategy(config)
        tp = strategy._calculate_take_profit(30000, 30600, 'short')
        assert tp < 30000, "Short TP must be below entry"
