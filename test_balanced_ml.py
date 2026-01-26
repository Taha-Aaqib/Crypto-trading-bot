"""Test the NEW balanced ML model on historical data"""
import pickle
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from src.data.data_fetcher import DataFetcher

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Load new model
model_path = "models/saved_models/BTC_USDT/lstm_model.pkl"
scaler_path = "models/saved_models/BTC_USDT/scaler.pkl"
features_path = "models/saved_models/BTC_USDT/features.pkl"

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(features_path, 'rb') as f:
    feature_columns = pickle.load(f)

print("✅ Loaded balanced ML model")
print(f"   Features: {len(feature_columns)}")

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML features from OHLCV data"""
    df = df.copy()
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']
    
    # Trend indicators
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    
    # EMA differences (trend strength)
    df['ema_9_21_diff'] = (df['ema_9'] - df['ema_21']) / df['close']
    df['ema_21_50_diff'] = (df['ema_21'] - df['ema_50']) / df['close']
    df['ema_50_200_diff'] = (df['ema_50'] - df['ema_200']) / df['close']
    df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['close']
    df['price_vs_ema200'] = (df['close'] - df['ema_200']) / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_normalized'] = df['rsi'] / 100
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_normalized'] = df['macd'] / df['close']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Momentum
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    
    # Candle patterns
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    
    # Higher highs / Lower lows (structure)
    df['hh'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['ll'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['hl'] = (df['low'] > df['low'].shift(1)).astype(int)
    df['lh'] = (df['high'] < df['high'].shift(1)).astype(int)
    
    return df

# Fetch November 2025 data
print("\n📊 Fetching November 2025 historical data...")
fetcher = DataFetcher(config)
start_date = datetime(2025, 11, 1)
end_date = datetime(2025, 11, 30)
df = fetcher.fetch_historical_data('BTC/USDT', '15m', start_date, end_date)
print(f"   Fetched {len(df)} candles")

# Create features
df = create_features(df)
df_clean = df.dropna(subset=feature_columns)

print("\n" + "=" * 70)
print("ML MODEL PREDICTIONS ON NOVEMBER 2025 DATA (BALANCED MODEL)")
print("=" * 70)

# Test at different points
test_dates = [
    '2025-11-06',  # Early Nov
    '2025-11-10',  # Price rising
    '2025-11-15',  # After peak
    '2025-11-19',  # Dropping
    '2025-11-22',  # Near bottom
    '2025-11-27',  # Recovery
]

buy_count = 0
sell_count = 0
hold_count = 0

for test_date in test_dates:
    # Get data up to this date
    mask = df_clean.index <= test_date + ' 23:59:00'
    df_slice = df_clean[mask].tail(1)
    
    if len(df_slice) == 0:
        continue
    
    # Get features and predict
    X = df_slice[feature_columns].values
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    # Get price
    price = df_slice['close'].iloc[-1]
    
    # Get confidence
    confidence = max(probabilities) * 100
    
    # Signal label
    signal_label = {-1: 'SELL 📉', 0: 'HOLD ⏸️', 1: 'BUY 📈'}.get(prediction, prediction)
    
    # Count
    if prediction == 1:
        buy_count += 1
    elif prediction == -1:
        sell_count += 1
    else:
        hold_count += 1
    
    # Get class probabilities
    classes = model.classes_
    sell_idx = list(classes).index(-1)
    hold_idx = list(classes).index(0)
    buy_idx = list(classes).index(1)
    
    print(f"\n{test_date} | BTC: ${price:,.0f}")
    print(f"  Prediction: {signal_label} (confidence: {confidence:.1f}%)")
    print(f"  Probabilities: SELL={probabilities[sell_idx]:.1%}, HOLD={probabilities[hold_idx]:.1%}, BUY={probabilities[buy_idx]:.1%}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"BUY signals:  {buy_count}")
print(f"SELL signals: {sell_count}")
print(f"HOLD signals: {hold_count}")
print(f"\n✅ Total active signals: {buy_count + sell_count} / {len(test_dates)}")

# Test on CURRENT data
print("\n" + "=" * 70)
print("🔴 CURRENT MARKET TEST")
print("=" * 70)

current_df = fetcher.fetch_ohlcv('BTC/USDT', '15m', limit=250)
current_df = create_features(current_df)
current_df = current_df.dropna(subset=feature_columns)

X_current = current_df[feature_columns].tail(1).values
X_current_scaled = scaler.transform(X_current)

pred = model.predict(X_current_scaled)[0]
probs = model.predict_proba(X_current_scaled)[0]
price = current_df['close'].iloc[-1]

signal = {-1: 'SELL 📉', 0: 'HOLD ⏸️', 1: 'BUY 📈'}.get(pred, pred)
print(f"\nCurrent BTC Price: ${price:,.0f}")
print(f"ML Prediction: {signal}")
print(f"Confidence: {max(probs)*100:.1f}%")
print(f"Probabilities: SELL={probs[sell_idx]:.1%}, HOLD={probs[hold_idx]:.1%}, BUY={probs[buy_idx]:.1%}")
