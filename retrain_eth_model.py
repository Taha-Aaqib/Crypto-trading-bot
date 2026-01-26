"""
Retrain ETH ML Model with Balanced Classes
Same approach as BTC - fixes the 91% HOLD bias
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_fetcher import DataFetcher


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


def create_labels_percentile(df: pd.DataFrame, lookahead: int = 10, 
                             buy_percentile: float = 70, 
                             sell_percentile: float = 30) -> pd.Series:
    """
    Create labels using PERCENTILE-based approach
    This guarantees balanced classes!
    """
    # Calculate future returns
    future_returns = df['close'].shift(-lookahead) / df['close'] - 1
    
    # Calculate rolling percentiles
    rolling_window = 200
    upper_threshold = future_returns.rolling(rolling_window).quantile(buy_percentile/100)
    lower_threshold = future_returns.rolling(rolling_window).quantile(sell_percentile/100)
    
    labels = pd.Series(0, index=df.index)  # Default HOLD
    labels[future_returns > upper_threshold] = 1   # BUY (top 30%)
    labels[future_returns < lower_threshold] = -1  # SELL (bottom 30%)
    
    return labels


def main():
    print("=" * 70)
    print("🔧 RETRAINING ETH/USDT ML MODEL WITH BALANCED CLASSES")
    print("=" * 70)
    
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    fetcher = DataFetcher(config)
    
    # Fetch historical data for ETH
    print("\n📊 Fetching ETH/USDT historical data...")
    
    # Get October-December 2025 data for training
    start_date = datetime(2025, 10, 1)
    end_date = datetime(2025, 12, 31)
    
    df = fetcher.fetch_historical_data('ETH/USDT', '15m', start_date, end_date)
    
    print(f"✅ Fetched {len(df)} candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create features
    print("\n🔨 Creating features...")
    df = create_features(df)
    
    # Create balanced labels
    print("\n📊 Creating PERCENTILE-based labels...")
    df['label'] = create_labels_percentile(df, lookahead=10, 
                                           buy_percentile=70, 
                                           sell_percentile=30)
    
    # Compare distributions
    print("\n" + "=" * 50)
    print("LABEL DISTRIBUTION COMPARISON")
    print("=" * 50)
    
    print("\n🔴 OLD METHOD (1% fixed threshold):")
    old_labels = pd.Series(0, index=df.index)
    future_ret = df['close'].shift(-5) / df['close'] - 1
    old_labels[future_ret > 0.01] = 1
    old_labels[future_ret < -0.01] = -1
    old_dist = old_labels.value_counts().sort_index()
    for label, count in old_dist.items():
        name = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[label]
        pct = count / len(old_labels) * 100
        print(f"   {name}: {count} ({pct:.1f}%)")
    
    print("\n🟢 NEW METHOD (Percentile-based):")
    new_dist = df['label'].value_counts().sort_index()
    for label, count in new_dist.items():
        name = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[label]
        pct = count / len(df) * 100
        print(f"   {name}: {count} ({pct:.1f}%)")
    
    # Prepare features
    feature_columns = [
        'returns', 'volatility', 'atr_pct',
        'ema_9_21_diff', 'ema_21_50_diff', 'ema_50_200_diff',
        'price_vs_ema50', 'price_vs_ema200',
        'rsi_normalized', 'macd_normalized', 'bb_position',
        'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20',
        'body_size', 'is_bullish', 'hh', 'll', 'hl', 'lh'
    ]
    
    # Drop NaN rows
    df_clean = df.dropna(subset=feature_columns + ['label'])
    print(f"\n📊 Training samples after cleanup: {len(df_clean)}")
    
    X = df_clean[feature_columns].values
    y = df_clean['label'].values
    
    # Split data (time-series style - no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try SMOTE for additional balancing
    print("\n🔄 Attempting SMOTE for class balancing...")
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"   ✅ SMOTE applied: {len(X_train_balanced)} samples")
        
        balanced_dist = pd.Series(y_train_balanced).value_counts().sort_index()
        for label, count in balanced_dist.items():
            name = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[label]
            print(f"      {name}: {count}")
    except ImportError:
        print("   ⚠️ imbalanced-learn not installed, using class_weight only")
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train
    except Exception as e:
        print(f"   ⚠️ SMOTE failed: {e}, using class_weight only")
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train
    
    # Train model with class weights
    print("\n🤖 Training Random Forest with balanced settings...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Auto-balance classes
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    print("\n" + "=" * 50)
    print("📈 MODEL EVALUATION")
    print("=" * 50)
    
    y_pred = model.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['SELL', 'HOLD', 'BUY']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"           SELL  HOLD  BUY")
    print(f"  SELL  [{cm[0][0]:5d} {cm[0][1]:5d} {cm[0][2]:5d}]")
    print(f"  HOLD  [{cm[1][0]:5d} {cm[1][1]:5d} {cm[1][2]:5d}]")
    print(f"  BUY   [{cm[2][0]:5d} {cm[2][1]:5d} {cm[2][2]:5d}]")
    
    # Test predictions distribution
    print("\n" + "=" * 50)
    print("🧪 PREDICTION DISTRIBUTION TEST")
    print("=" * 50)
    
    # Get all test predictions
    all_preds = model.predict(X_test_scaled)
    pred_dist = pd.Series(all_preds).value_counts().sort_index()
    print("\nTest set predictions:")
    for label, count in pred_dist.items():
        name = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[label]
        pct = count / len(all_preds) * 100
        print(f"   {name}: {count} ({pct:.1f}%)")
    
    # Feature importance
    print("\n📊 Top 10 Important Features:")
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    print("\n💾 Saving new balanced ETH model...")
    
    model_dir = "models/saved_models/ETH_USDT"
    os.makedirs(model_dir, exist_ok=True)
    
    # Backup old model
    old_model_path = os.path.join(model_dir, "lstm_model.pkl")
    if os.path.exists(old_model_path):
        backup_path = os.path.join(model_dir, "lstm_model_backup_imbalanced.pkl")
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy(old_model_path, backup_path)
            print(f"   Backed up old model to: {backup_path}")
    
    # Save new model
    with open(old_model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✅ Saved new model to: {old_model_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Saved scaler to: {scaler_path}")
    
    # Save feature list
    features_path = os.path.join(model_dir, "features.pkl")
    with open(features_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"   ✅ Saved features to: {features_path}")
    
    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'samples': len(df_clean),
        'features': feature_columns,
        'label_method': 'percentile',
        'lookahead': 10,
        'buy_percentile': 70,
        'sell_percentile': 30,
        'model_type': 'RandomForestClassifier',
        'class_balance': 'class_weight=balanced'
    }
    metadata_path = os.path.join(model_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n" + "=" * 70)
    print("✅ ETH/USDT MODEL RETRAINED SUCCESSFULLY!")
    print("=" * 70)
    print("\nKey improvements:")
    print("  1. Percentile-based labels (~30% SELL, ~40% HOLD, ~30% BUY)")
    print("  2. class_weight='balanced' in Random Forest")
    print("  3. More features (21 total)")
    print("  4. Longer lookahead (10 candles = 2.5 hours)")
    print("\n🚀 Both BTC and ETH models are now trained!")
    print("   main.py will automatically use the new balanced models")


if __name__ == "__main__":
    main()
