#!/usr/bin/env python3
"""
Test both BTC and ETH balanced ML models on older historical data
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import yaml
from datetime import datetime, timedelta
import ccxt

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_fetcher import DataFetcher
from src.indicators.ta_indicators import TechnicalIndicators

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def create_features(df):
    """Create same 21 features used in training (direct calculation)"""
    features = pd.DataFrame(index=df.index)
    
    # Returns
    features['returns'] = df['close'].pct_change()
    
    # EMAs
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    
    # EMA differences (normalized by price)
    features['ema_9_21_diff'] = (df['ema_9'] - df['ema_21']) / df['close']
    features['ema_21_50_diff'] = (df['ema_21'] - df['ema_50']) / df['close']
    features['ema_50_200_diff'] = (df['ema_50'] - df['ema_200']) / df['close']
    
    # Price vs EMAs
    features['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['close']
    features['price_vs_ema200'] = (df['close'] - df['ema_200']) / df['close']
    
    # Momentum (different periods)
    features['momentum_5'] = df['close'].pct_change(5)
    features['momentum_10'] = df['close'].pct_change(10)
    features['momentum_20'] = df['close'].pct_change(20)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    features['rsi_normalized'] = (rsi - 50) / 50  # -1 to 1
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    features['macd_normalized'] = (macd_line - signal_line) / df['close']
    
    # Bollinger Bands
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # Volatility (ATR)
    high_low = df['high'] - df['low']
    features['atr_pct'] = high_low.rolling(14).mean() / df['close']
    features['volatility'] = df['close'].pct_change().rolling(20).std()
    
    # Volume
    volume_sma = df['volume'].rolling(20).mean()
    features['volume_ratio'] = df['volume'] / (volume_sma + 1e-10)
    
    # Candle patterns
    features['body_size'] = abs(df['close'] - df['open']) / df['close']
    features['is_bullish'] = (df['close'] > df['open']).astype(int)
    
    # Market structure (HH, LL, HL, LH)
    lookback = 20
    features['hh'] = (df['high'] == df['high'].rolling(lookback).max()).astype(int)
    features['ll'] = (df['low'] == df['low'].rolling(lookback).min()).astype(int)
    features['hl'] = ((df['low'] > df['low'].shift(lookback)) & (df['high'] < df['high'].shift(lookback))).astype(int)
    features['lh'] = ((df['high'] < df['high'].shift(lookback)) & (df['low'] > df['low'].shift(lookback))).astype(int)
    
    # Add features back to df
    for col in features.columns:
        df[col] = features[col]
    
    return df

def test_model(symbol, start_date, end_date):
    """Test a specific model on historical data"""
    print(f"\n{'='*70}")
    print(f"📊 TESTING {symbol} MODEL")
    print(f"{'='*70}")
    
    # Load model
    model_dir = f"models/saved_models/{symbol.replace('/', '_')}"
    
    try:
        with open(f"{model_dir}/lstm_model.pkl", 'rb') as f:
            model = pickle.load(f)
        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open(f"{model_dir}/features.pkl", 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✅ Loaded model from {model_dir}/")
    except Exception as e:
        print(f"❌ Failed to load {symbol} model: {e}")
        return
    
    # Fetch historical data
    print(f"\n📅 Fetching data from {start_date} to {end_date}...")
    fetcher = DataFetcher(config)
    
    try:
        df = fetcher.fetch_historical_data(
            symbol=symbol,
            timeframe='15m',
            start_date=start_date,
            end_date=end_date
        )
        print(f"✅ Fetched {len(df)} candles")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return
    
    # Create features
    print(f"\n🔨 Creating features...")
    df = create_features(df)
    df = df.dropna()
    print(f"✅ {len(df)} samples after feature engineering")
    
    # Make predictions
    X = df[feature_names].values
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Model uses classes [-1, 0, 1] = [SELL, HOLD, BUY]
    # Add to dataframe
    df['prediction'] = predictions
    df['prob_sell'] = probabilities[:, 0]  # -1
    df['prob_hold'] = probabilities[:, 1]  # 0
    df['prob_buy'] = probabilities[:, 2]   # 1
    
    # Calculate statistics
    pred_counts = pd.Series(predictions).value_counts()
    
    print(f"\n{'='*70}")
    print(f"📊 PREDICTION DISTRIBUTION")
    print(f"{'='*70}")
    print(f"   SELL: {pred_counts.get(-1, 0):4d} ({pred_counts.get(-1, 0)/len(df)*100:5.1f}%)")
    print(f"   HOLD: {pred_counts.get(0, 0):4d} ({pred_counts.get(0, 0)/len(df)*100:5.1f}%)")
    print(f"   BUY:  {pred_counts.get(1, 0):4d} ({pred_counts.get(1, 0)/len(df)*100:5.1f}%)")
    
    # Show active signals (BUY/SELL with >60% confidence)
    active_signals = df[
        ((df['prediction'] == 1) & (df['prob_buy'] > 0.6)) |   # BUY
        ((df['prediction'] == -1) & (df['prob_sell'] > 0.6))   # SELL
    ].copy()
    
    print(f"\n{'='*70}")
    print(f"🎯 ACTIVE SIGNALS (>60% confidence)")
    print(f"{'='*70}")
    
    if len(active_signals) > 0:
        active_signals['signal'] = active_signals.apply(
            lambda row: f"BUY ({row['prob_buy']*100:.1f}%)" if row['prediction'] == 2 
                       else f"SELL ({row['prob_sell']*100:.1f}%)", 
            axis=1
        )
        active_signals['price'] = active_signals['close']
        
        print(f"\nFound {len(active_signals)} active signals:\n")
        for idx, row in active_signals.iterrows():
            print(f"   {idx.strftime('%Y-%m-%d %H:%M')} | {row['signal']:15s} | Price: ${row['price']:,.2f}")
    else:
        print("\n   No active signals found (all predictions below 60% confidence)")
    
    # Show weekly summary
    print(f"\n{'='*70}")
    print(f"📅 WEEKLY BREAKDOWN")
    print(f"{'='*70}")
    
    df_weekly = df.copy()
    df_weekly['week'] = df_weekly.index.to_period('W')
    
    weekly_stats = df_weekly.groupby('week').agg({
        'prediction': lambda x: pd.Series(x).value_counts().to_dict()
    })
    
    for week, row in weekly_stats.iterrows():
        counts = row['prediction']
        sell = counts.get(-1, 0)  # -1 = SELL
        hold = counts.get(0, 0)   # 0 = HOLD
        buy = counts.get(1, 0)    # 1 = BUY
        total = sell + hold + buy
        
        print(f"\n   Week of {week}:")
        print(f"      SELL: {sell:3d} ({sell/total*100:5.1f}%) | HOLD: {hold:3d} ({hold/total*100:5.1f}%) | BUY: {buy:3d} ({buy/total*100:5.1f}%)")

if __name__ == "__main__":
    print("="*70)
    print("🧪 TESTING BOTH BTC AND ETH BALANCED MODELS")
    print("="*70)
    
    # Test on Nov 2025 (known bullish period)
    start_date = datetime(2025, 11, 1)
    end_date = datetime(2025, 11, 30)
    
    print(f"\n📅 Testing period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("   (November 2025 - bullish period)")
    
    # Test BTC model
    test_model("BTC/USDT", start_date, end_date)
    
    # Test ETH model
    test_model("ETH/USDT", start_date, end_date)
    
    print(f"\n{'='*70}")
    print("✅ Testing complete!")
    print(f"{'='*70}")
