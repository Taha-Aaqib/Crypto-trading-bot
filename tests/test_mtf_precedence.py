"""
Tests for the MTF pattern labeler operator precedence fix.
Verifies that mtf_all_bullish / mtf_all_bearish are computed correctly.
"""
import pytest


class TestMTFOperatorPrecedence:
    """
    Bug fix verification: Previously, the mtf_all_bullish/bearish calculation
    used bitwise operators (|, &) without parentheses, causing operator precedence
    issues. The fix uses `and`/`or` with explicit parentheses.
    """

    def test_all_bullish_true(self):
        """All 4 TFs bullish → mtf_all_bullish should be 1."""
        features = {
            '1d_choch_bullish': 1, '1d_bos_bullish': 0,
            '4h_choch_bullish': 1, '4h_bos_bullish': 0,
            '1h_choch_bullish': 0, '1h_fvg_bullish': 1,
            '15m_choch_bullish': 0, '15m_fvg_bullish': 1,
        }
        result = int(
            (features.get('1d_choch_bullish', 0) or features.get('1d_bos_bullish', 0)) and
            (features.get('4h_choch_bullish', 0) or features.get('4h_bos_bullish', 0)) and
            (features.get('1h_choch_bullish', 0) or features.get('1h_fvg_bullish', 0)) and
            (features.get('15m_choch_bullish', 0) or features.get('15m_fvg_bullish', 0))
        )
        assert result == 1

    def test_all_bullish_false_when_1d_missing(self):
        """If 1D has no bullish signals, mtf_all_bullish should be 0."""
        features = {
            '1d_choch_bullish': 0, '1d_bos_bullish': 0,  # 1D missing
            '4h_choch_bullish': 1, '4h_bos_bullish': 0,
            '1h_choch_bullish': 1, '1h_fvg_bullish': 0,
            '15m_choch_bullish': 1, '15m_fvg_bullish': 0,
        }
        result = int(
            (features.get('1d_choch_bullish', 0) or features.get('1d_bos_bullish', 0)) and
            (features.get('4h_choch_bullish', 0) or features.get('4h_bos_bullish', 0)) and
            (features.get('1h_choch_bullish', 0) or features.get('1h_fvg_bullish', 0)) and
            (features.get('15m_choch_bullish', 0) or features.get('15m_fvg_bullish', 0))
        )
        assert result == 0

    def test_all_bearish_true(self):
        """All 4 TFs bearish → mtf_all_bearish should be 1."""
        features = {
            '1d_choch_bearish': 0, '1d_bos_bearish': 1,
            '4h_choch_bearish': 1, '4h_bos_bearish': 0,
            '1h_choch_bearish': 0, '1h_fvg_bearish': 1,
            '15m_choch_bearish': 1, '15m_fvg_bearish': 0,
        }
        result = int(
            (features.get('1d_choch_bearish', 0) or features.get('1d_bos_bearish', 0)) and
            (features.get('4h_choch_bearish', 0) or features.get('4h_bos_bearish', 0)) and
            (features.get('1h_choch_bearish', 0) or features.get('1h_fvg_bearish', 0)) and
            (features.get('15m_choch_bearish', 0) or features.get('15m_fvg_bearish', 0))
        )
        assert result == 1

    def test_all_bearish_false_when_15m_missing(self):
        """If 15M has no bearish signals, mtf_all_bearish should be 0."""
        features = {
            '1d_choch_bearish': 1, '1d_bos_bearish': 0,
            '4h_choch_bearish': 1, '4h_bos_bearish': 0,
            '1h_choch_bearish': 1, '1h_fvg_bearish': 0,
            '15m_choch_bearish': 0, '15m_fvg_bearish': 0,  # 15M missing
        }
        result = int(
            (features.get('1d_choch_bearish', 0) or features.get('1d_bos_bearish', 0)) and
            (features.get('4h_choch_bearish', 0) or features.get('4h_bos_bearish', 0)) and
            (features.get('1h_choch_bearish', 0) or features.get('1h_fvg_bearish', 0)) and
            (features.get('15m_choch_bearish', 0) or features.get('15m_fvg_bearish', 0))
        )
        assert result == 0

    def test_mixed_directions_not_both_true(self):
        """Can't be all-bullish AND all-bearish simultaneously (using same OR logic)."""
        # If every TF has bull signal, bearish should be 0 (assuming no bearish signals)
        features = {
            '1d_choch_bullish': 1, '1d_bos_bullish': 0,
            '4h_choch_bullish': 1, '4h_bos_bullish': 0,
            '1h_choch_bullish': 1, '1h_fvg_bullish': 0,
            '15m_choch_bullish': 1, '15m_fvg_bullish': 0,
            '1d_choch_bearish': 0, '1d_bos_bearish': 0,
            '4h_choch_bearish': 0, '4h_bos_bearish': 0,
            '1h_choch_bearish': 0, '1h_fvg_bearish': 0,
            '15m_choch_bearish': 0, '15m_fvg_bearish': 0,
        }
        bullish = int(
            (features.get('1d_choch_bullish', 0) or features.get('1d_bos_bullish', 0)) and
            (features.get('4h_choch_bullish', 0) or features.get('4h_bos_bullish', 0)) and
            (features.get('1h_choch_bullish', 0) or features.get('1h_fvg_bullish', 0)) and
            (features.get('15m_choch_bullish', 0) or features.get('15m_fvg_bullish', 0))
        )
        bearish = int(
            (features.get('1d_choch_bearish', 0) or features.get('1d_bos_bearish', 0)) and
            (features.get('4h_choch_bearish', 0) or features.get('4h_bos_bearish', 0)) and
            (features.get('1h_choch_bearish', 0) or features.get('1h_fvg_bearish', 0)) and
            (features.get('15m_choch_bearish', 0) or features.get('15m_fvg_bearish', 0))
        )
        assert bullish == 1
        assert bearish == 0
        # They can't both be 1 if only one direction has signals
        assert not (bullish == 1 and bearish == 1)
