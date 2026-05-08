"""Debug why ML model returns neutral - Test on HISTORICAL DATA"""
from src.models.trading_model import TradingModel
from src.data.data_fetcher import DataFetcher
from src.data.data_preprocessor import DataPreprocessor
from src.indicators.ta_indicators import TechnicalIndicators
from src.indicators.smc_detector import SMCDetector
import yaml
import os
import pandas as pd
from datetime import datetime

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Load model
model = TradingModel(config, symbol='BTC/USDT')
print(f'Model loaded: {model.is_trained}')
print(f'Model path: {model.model_path}')

if model.is_trained:
    # Get HISTORICAL data - November 2025
    fetcher = DataFetcher(config)

    # Fetch historical data using the proper method
    print("\nFetching November 2025 historical data...")
    start_date = datetime(2025, 11, 1)
    end_date = datetime(2025, 11, 30)

    df_15m = fetcher.fetch_historical_data(
        'BTC/USDT', '15m', start_date, end_date)

    print(f"November 2025 data: {len(df_15m)} candles")
    print(f"Date range: {df_15m.index[0]} to {df_15m.index[-1]}")

    # Preprocess
    preprocessor = DataPreprocessor(config)
    ta = TechnicalIndicators(config)
    smc = SMCDetector(config)

    # Test at different points in November
    test_dates = [
        '2025-11-06',  # Early Nov - price dropping
        '2025-11-10',  # Mid downtrend
        '2025-11-15',  # Strong downtrend
        '2025-11-19',  # Near bottom
        '2025-11-22',  # Recovery start
        '2025-11-27',  # Recovery
    ]

    print("\n" + "="*70)
    print("ML MODEL PREDICTIONS ON NOVEMBER 2025 DATA")
    print("="*70)

    buy_count = 0
    sell_count = 0
    hold_count = 0

    for test_date in test_dates:
        # Get data up to this date
        mask = df_15m.index <= test_date + ' 23:59:00'
        df = df_15m[mask].tail(200).copy()

        if len(df) < 50:
            print(f"\n{test_date}: Not enough data")
            continue

        # Process
        df = preprocessor.process_pipeline(df)
        df = ta.add_all_indicators(df)
        df = smc.analyze_smc(df)

        # Get prediction
        prediction, confidence = model.predict(df)

        # Get probabilities
        X = model.prepare_features(df)
        X_latest = X.iloc[[-1]]
        X_scaled = model.scaler.transform(X_latest)
        probs = model.model.predict_proba(X_scaled)[0]
        classes = model.model.classes_

        # Get price
        price = df['close'].iloc[-1]

        # Signal label
        signal_label = {-1: 'SELL 📉', 0: 'HOLD ⏸️',
                        1: 'BUY 📈'}.get(prediction, prediction)

        # Count
        if prediction == 1:
            buy_count += 1
        elif prediction == -1:
            sell_count += 1
        else:
            hold_count += 1

        print(f"\n{test_date} | BTC: ${price:,.0f}")
        print(f"  Prediction: {signal_label} (confidence: {confidence:.1%})")
        print(
            f"  Probabilities: SELL={probs[list(classes).index(-1)]:.1%}, HOLD={probs[list(classes).index(0)]:.1%}, BUY={probs[list(classes).index(1)]:.1%}")

        # Show key features
        features = model.prepare_features(df).iloc[-1]
        print(
            f"  Key features: RSI={features['rsi']*100:.0f}%, EMA_trend={features['ema_trend']:.0f}, Structure={features['market_structure_score']:.0f}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"BUY signals:  {buy_count}")
    print(f"SELL signals: {sell_count}")
    print(f"HOLD signals: {hold_count}")
    print(
        f"\nTotal active signals: {buy_count + sell_count} / {len(test_dates)}")
else:
    print('Model not trained!')
