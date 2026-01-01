"""
FAST Backtest Script - Optimized for Quick Testing
Uses ccxt directly without the slow data fetcher
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import argparse
import logging

# Suppress unnecessary logs for cleaner backtest output
logging.getLogger('TradingBot').setLevel(logging.WARNING)

print("=" * 60)
print("FAST BACKTEST - AI Trading Bot")
print("=" * 60)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fast backtest for trading bot')
parser.add_argument('--symbol', type=str, default='BTC/USDT',
                    help='Trading symbol (default: BTC/USDT)')
parser.add_argument('--days', type=int, default=7,
                    help='Days of historical data (default: 7)')
parser.add_argument('--timeframe', type=str, default='15m',
                    help='Timeframe (default: 15m)')
parser.add_argument('--start-date', type=str, default=None,
                    help='Start date (YYYY-MM-DD) - overrides --days')
parser.add_argument('--end-date', type=str, default=None,
                    help='End date (YYYY-MM-DD, default: today)')
args = parser.parse_args()

SYMBOL = args.symbol
TIMEFRAME = args.timeframe
INITIAL_BALANCE = 10000

# Determine date range
if args.start_date:
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(
        args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    DAYS = (end_date - start_date).days
    print(f"Symbol: {SYMBOL}")
    print(f"Period: {args.start_date} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Duration: {DAYS} days")
else:
    DAYS = args.days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS)
    print(f"Symbol: {SYMBOL}")
    print(f"Period: Last {DAYS} days of data")

print(f"Timeframe: {TIMEFRAME}")
print()

# Initialize exchange (no API key needed for public data)
print("Connecting to Binance...")
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# Calculate how many candles we need
candles_per_day = 96  # 96 x 15min = 24 hours
total_candles = DAYS * candles_per_day

# Convert start_date to timestamp (milliseconds)
since_timestamp = int(start_date.timestamp() * 1000)

print(
    f"Fetching {total_candles} candles from {start_date.strftime('%Y-%m-%d')}...")

try:
    # Fetch OHLCV data using since parameter for historical data
    ohlcv = exchange.fetch_ohlcv(
        SYMBOL, TIMEFRAME, since=since_timestamp, limit=total_candles)

    # Convert to DataFrame
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    print(f"[OK] Fetched {len(df)} candles")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print()

except Exception as e:
    print(f"ERROR: Failed to fetch data: {e}")
    exit(1)

# Calculate simple indicators
print("Calculating indicators...")

# EMA
df['ema_20'] = df['close'].ewm(span=20).mean()
df['ema_50'] = df['close'].ewm(span=50).mean()

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# ATR
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = np.max(ranges, axis=1)
df['atr'] = true_range.rolling(14).mean()

# Simple trend
df['trend'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)

print("[OK] Indicators calculated")
print()

# Simple trading strategy
print("Simulating trades...")
print("-" * 60)

balance = INITIAL_BALANCE
position = None
trades = []

for i in range(100, len(df)):  # Start after 100 candles for indicator warmup
    row = df.iloc[i]

    # ENTRY CONDITIONS
    if position is None:
        # LONG: Bullish trend + oversold RSI
        if row['trend'] == 1 and row['rsi'] < 40:
            position = {
                'side': 'LONG',
                'entry_price': row['close'],
                'entry_time': row.name,
                'stop_loss': row['close'] - (row['atr'] * 1.5),
                'take_profit': row['close'] + (row['atr'] * 3),
                'size': balance * 0.02  # Risk 2%
            }

        # SHORT: Bearish trend + overbought RSI
        elif row['trend'] == -1 and row['rsi'] > 60:
            position = {
                'side': 'SHORT',
                'entry_price': row['close'],
                'entry_time': row.name,
                'stop_loss': row['close'] + (row['atr'] * 1.5),
                'take_profit': row['close'] - (row['atr'] * 3),
                'size': balance * 0.02
            }

    # EXIT CONDITIONS
    elif position is not None:
        exit_trade = False
        exit_reason = ''

        if position['side'] == 'LONG':
            if row['close'] <= position['stop_loss']:
                exit_trade = True
                exit_reason = 'STOP_LOSS'
            elif row['close'] >= position['take_profit']:
                exit_trade = True
                exit_reason = 'TAKE_PROFIT'
            elif row['trend'] == -1:  # Trend reversal
                exit_trade = True
                exit_reason = 'TREND_REVERSAL'

        elif position['side'] == 'SHORT':
            if row['close'] >= position['stop_loss']:
                exit_trade = True
                exit_reason = 'STOP_LOSS'
            elif row['close'] <= position['take_profit']:
                exit_trade = True
                exit_reason = 'TAKE_PROFIT'
            elif row['trend'] == 1:  # Trend reversal
                exit_trade = True
                exit_reason = 'TREND_REVERSAL'

        if exit_trade:
            # Calculate P&L
            if position['side'] == 'LONG':
                pnl_pct = (row['close'] - position['entry_price']
                           ) / position['entry_price']
            else:  # SHORT
                pnl_pct = (position['entry_price'] -
                           row['close']) / position['entry_price']

            pnl_usd = position['size'] * pnl_pct
            balance += pnl_usd

            # Record trade
            trade = {
                'entry': position['entry_price'],
                'exit': row['close'],
                'side': position['side'],
                'pnl_pct': pnl_pct * 100,
                'pnl_usd': pnl_usd,
                'reason': exit_reason,
                'duration': (row.name - position['entry_time']).total_seconds() / 3600
            }
            trades.append(trade)

            # Print trade
            emoji = "[WIN]" if pnl_usd > 0 else "[LOSS]"
            print(f"{emoji} Trade #{len(trades)} ({position['side']}): "
                  f"${position['entry_price']:.2f} → ${row['close']:.2f} | "
                  f"P&L: {pnl_pct*100:+.2f}% (${pnl_usd:+.2f}) | "
                  f"Exit: {exit_reason}")

            position = None

print()
print("=" * 60)
print("BACKTEST RESULTS")
print("=" * 60)

if not trades:
    print("[NONE] No trades executed during backtest period")
    print("\nPossible reasons:")
    print("- Conditions too strict (try adjusting RSI thresholds)")
    print("- Not enough data (try increasing DAYS)")
    print("- Market was in consolidation")
else:
    # Calculate statistics
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    losing_trades = [t for t in trades if t['pnl_usd'] <= 0]

    total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0

    avg_win = np.mean([t['pnl_pct']
                      for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl_pct']
                       for t in losing_trades]) if losing_trades else 0
    avg_duration = np.mean([t['duration'] for t in trades])

    print(f"Period: {DAYS} days ({len(df)} candles)")
    print(f"Strategy: EMA Trend + RSI")
    print()
    print(f"Total Trades: {len(trades)}")
    print(f"  Winning: {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"  Losing: {len(losing_trades)}")
    print()
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Final Balance: ${balance:,.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print()
    print(f"Average Win: {avg_win:+.2f}%")
    print(f"Average Loss: {avg_loss:+.2f}%")

    if avg_loss != 0:
        profit_factor = abs(avg_win / avg_loss)
        print(f"Profit Factor: {profit_factor:.2f}")

    print(f"Average Trade Duration: {avg_duration:.1f} hours")
    print()

    # Win/Loss breakdown by exit reason
    print("Exit Reasons:")
    for reason in ['TAKE_PROFIT', 'STOP_LOSS', 'TREND_REVERSAL']:
        count = sum(1 for t in trades if t['reason'] == reason)
        if count > 0:
            avg_pnl = np.mean([t['pnl_pct']
                              for t in trades if t['reason'] == reason])
            print(f"  {reason}: {count} trades (avg: {avg_pnl:+.2f}%)")

print("=" * 60)
print("\n[DONE] Backtest complete!")
print()
print("📌 NOTE: Backtest uses the MOST RECENT data, so results are the same each time.")
print("   This is normal - it's testing the last 7 days repeatedly.")
print()
print("🔄 To test DIFFERENT periods:")
print(f"   python quick_backtest.py BTC/USDT 14  # Test last 14 days")
print(f"   python quick_backtest.py BTC/USDT 30  # Test last 30 days")
print(f"   python quick_backtest.py ETH/USDT 7   # Test ETH instead")
print()
print("⏰ To see LIVE trading decisions (different each time):")
print(f"   python main.py  # Runs every 15 min with current market data")
print()
print("=" * 60)
