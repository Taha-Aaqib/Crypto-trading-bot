"""
Helper functions for the trading bot
Common utilities used across modules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
from typing import Dict, List, Optional, Tuple


def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """Load configuration from YAML file and expand environment variables"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Expand environment variables in config
    config = _expand_env_vars(config)
    return config


def _expand_env_vars(obj):
    """Recursively expand environment variables in config dict"""
    if isinstance(obj, dict):
        return {key: _expand_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Expand ${VAR_NAME} or $VAR_NAME patterns
        import re
        pattern = r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)'

        def replace(match):
            var_name = match.group(1) or match.group(2)
            return os.getenv(var_name, match.group(0))
        return re.sub(pattern, replace, obj)
    else:
        return obj


def save_config(config: Dict, config_path: str = 'config/config.yaml'):
    """Save configuration to YAML file"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config_fields(updates: Dict, config_path: str = 'config/config.yaml'):
    """
    Update specific fields in config YAML without expanding env vars.
    Reads the raw file, modifies only the given keys, writes back.
    updates: dict like {'trading.market_type': 'futures', 'trading.leverage': 10}
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    for dotted_key, value in updates.items():
        keys = dotted_key.split('.')
        obj = config
        for k in keys[:-1]:
            obj = obj.setdefault(k, {})
        obj[keys[-1]] = value

    # Read original file to preserve env var placeholders
    with open(config_path, 'r', encoding='utf-8') as f:
        original_text = f.read()

    # For fields we updated, do targeted text replacements
    # Fall back to full rewrite if text replacement fails
    try:
        import re
        new_text = original_text
        for dotted_key, value in updates.items():
            keys = dotted_key.split('.')
            yaml_key = keys[-1]
            # Match the line with the key and replace its value
            if isinstance(value, str):
                val_str = f'"{value}"'
            elif isinstance(value, bool):
                val_str = 'true' if value else 'false'
            else:
                val_str = str(value)

            pattern = rf'(^\s*{re.escape(yaml_key)}\s*:)\s*.*$'
            # Find and replace only the first match
            match = re.search(pattern, new_text, re.MULTILINE)
            if match:
                # Preserve any inline comment
                old_line = match.group(0)
                comment = ''
                comment_match = re.search(r'\s+#.*$', old_line)
                if comment_match:
                    comment = comment_match.group(0)
                new_line = f"{match.group(1)} {val_str}{comment}"
                new_text = new_text[:match.start()] + new_line + new_text[match.end():]

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_text)
    except Exception:
        # Fallback: full dump (will lose env var placeholders)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)


def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float
) -> float:
    """
    Calculate position size based on risk management

    Args:
        capital: Total available capital
        risk_per_trade: Risk percentage per trade (e.g., 0.02 for 2%)
        entry_price: Entry price for the trade
        stop_loss_price: Stop loss price

    Returns:
        Position size in base currency
    """
    risk_amount = capital * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)

    if price_risk == 0:
        return 0

    position_size = risk_amount / price_risk
    return position_size


def format_timestamp(timestamp: int) -> str:
    """Convert timestamp to readable format"""
    return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes"""
    multiplier = int(timeframe[:-1]) if timeframe[:-1].isdigit() else 1
    unit = timeframe[-1]

    if unit == 'm':
        return multiplier
    elif unit == 'h':
        return multiplier * 60
    elif unit == 'd':
        return multiplier * 1440
    elif unit == 'w':
        return multiplier * 10080
    else:
        return 15  # Default to 15 minutes


def calculate_drawdown(equity_curve: pd.Series) -> Tuple[float, float]:
    """
    Calculate maximum drawdown and current drawdown

    Args:
        equity_curve: Series of equity values over time

    Returns:
        Tuple of (max_drawdown, current_drawdown)
    """
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax

    max_drawdown = drawdown.min()
    current_drawdown = drawdown.iloc[-1]

    return max_drawdown, current_drawdown


def validate_api_keys(config: Dict) -> bool:
    """Validate that API keys are configured"""
    exchange_config = config.get('exchange', {})

    if 'YOUR_' in exchange_config.get('api_key', ''):
        return False
    if 'YOUR_' in exchange_config.get('api_secret', ''):
        return False

    return True


def is_market_open(current_time: datetime = None) -> bool:
    """Check if crypto market is effectively open (24/7 but can add logic)"""
    # Crypto markets are 24/7, but you can add custom logic
    # For example, avoid weekends or specific hours
    return True


def round_to_tick_size(price: float, tick_size: float = 0.01) -> float:
    """Round price to exchange tick size"""
    return round(price / tick_size) * tick_size


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float
) -> float:
    """Calculate risk-reward ratio for a trade"""
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)

    if risk == 0:
        return 0

    return reward / risk


def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'data/backtest',
        'logs',
        'models/saved_models'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_data_path(data_type: str, symbol: str, timeframe: str) -> str:
    """Get standardized data file path"""
    symbol_clean = symbol.replace('/', '_')

    if data_type == 'raw':
        return f'data/raw/{symbol_clean}_{timeframe}_raw.csv'
    elif data_type == 'processed':
        return f'data/processed/{symbol_clean}_{timeframe}_processed.csv'
    elif data_type == 'backtest':
        return f'data/backtest/{symbol_clean}_{timeframe}_backtest.csv'

    return None


def merge_signals(*signal_series: pd.Series) -> pd.Series:
    """
    Merge multiple signal series using logical AND
    All signals must agree for final signal
    """
    if not signal_series:
        return pd.Series(dtype=int)

    combined = signal_series[0].copy()
    for signal in signal_series[1:]:
        combined = combined & signal

    return combined
