"""
Test what signals the REAL trading system is generating RIGHT NOW
"""
from src.trading.strategy import TradingStrategy
from src.indicators.ta_indicators import TechnicalIndicators
from src.indicators.smc_detector import SMCDetector
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_fetcher import DataFetcher
import sys
import yaml
from datetime import datetime, timedelta

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=" * 70)
print("REAL SYSTEM SIGNAL TEST")
print("=" * 70)
print(f"Time: {datetime.now()}")
print(f"Confluence Threshold: {config['strategy']['confluence_threshold']}")
print("=" * 70)

# Initialize components

data_fetcher = DataFetcher(config)
preprocessor = DataPreprocessor(config)
smc_detector = SMCDetector(config)
ta_indicators = TechnicalIndicators(config)
strategy = TradingStrategy(config)

symbol = "BTC/USDT"
print(f"\n📊 Fetching data for {symbol}...")

# Fetch multi-timeframe data
df_1d = data_fetcher.fetch_ohlcv(symbol, '1d', limit=100)
df_4h = data_fetcher.fetch_ohlcv(symbol, '4h', limit=200)
df_15m = data_fetcher.fetch_ohlcv(symbol, '15m', limit=100)

print(f"1D candles: {len(df_1d)}")
print(f"4H candles: {len(df_4h)}")
print(f"15M candles: {len(df_15m)}")

# Process data
df_1d = preprocessor.process_pipeline(df_1d)
df_4h = preprocessor.process_pipeline(df_4h)
df_15m = preprocessor.process_pipeline(df_15m)

# Add indicators
df_15m = ta_indicators.add_all_indicators(df_15m)

# Analyze SMC
print("\n🔍 Analyzing SMC patterns...")
smc_1d = smc_detector.analyze_smc(df_1d)
smc_4h = smc_detector.analyze_smc(df_4h)
smc_15m = smc_detector.analyze_smc(df_15m)

# Current price
current_price = df_15m.iloc[-1]['close']
print(f"\n💰 Current BTC Price: ${current_price:,.2f}")

# Check 1D bias
print("\n📈 1D DAILY BIAS:")
ema_20_1d = df_1d['close'].ewm(span=20).mean().iloc[-1]
ema_50_1d = df_1d['close'].ewm(span=50).mean().iloc[-1]
print(f"  EMA20: ${ema_20_1d:,.2f}")
print(f"  EMA50: ${ema_50_1d:,.2f}")
print(f"  Price: ${df_1d.iloc[-1]['close']:,.2f}")

if current_price > ema_20_1d > ema_50_1d:
    bias_1d = "BULLISH ✅"
elif current_price < ema_20_1d < ema_50_1d:
    bias_1d = "BEARISH 🔴"
else:
    bias_1d = "NEUTRAL ⚪"
print(f"  Bias: {bias_1d}")

# Check 4H structure
print("\n📊 4H STRUCTURE:")
ema_20_4h = df_4h['close'].ewm(span=20).mean().iloc[-1]
ema_50_4h = df_4h['close'].ewm(span=50).mean().iloc[-1]
print(f"  EMA20: ${ema_20_4h:,.2f}")
print(f"  EMA50: ${ema_50_4h:,.2f}")

if current_price > ema_20_4h > ema_50_4h:
    structure_4h = "BULLISH ✅"
elif current_price < ema_20_4h < ema_50_4h:
    structure_4h = "BEARISH 🔴"
else:
    structure_4h = "NEUTRAL ⚪"
print(f"  Structure: {structure_4h}")

# Check 15M signals
print("\n⚡ 15M ENTRY SIGNALS:")
ema_9 = df_15m['close'].ewm(span=9).mean().iloc[-1]
ema_21 = df_15m['close'].ewm(span=21).mean().iloc[-1]
print(f"  EMA9: ${ema_9:,.2f}")
print(f"  EMA21: ${ema_21:,.2f}")

if ema_9 > ema_21:
    ema_trend = "BULLISH ✅"
elif ema_9 < ema_21:
    ema_trend = "BEARISH 🔴"
else:
    ema_trend = "NEUTRAL ⚪"
print(f"  EMA Trend: {ema_trend}")

# Check SMC patterns in last 5 candles
print("\n🎯 RECENT SMC PATTERNS (last 5 candles):")
recent_choch_bull = smc_15m['bullish_choch'].iloc[-5:
                                                  ].any() if 'bullish_choch' in smc_15m else False
recent_choch_bear = smc_15m['bearish_choch'].iloc[-5:
                                                  ].any() if 'bearish_choch' in smc_15m else False
recent_bos_bull = smc_15m['bullish_bos'].iloc[-5:
                                              ].any() if 'bullish_bos' in smc_15m else False
recent_bos_bear = smc_15m['bearish_bos'].iloc[-5:
                                              ].any() if 'bearish_bos' in smc_15m else False
recent_fvg_bull = smc_15m['bullish_fvg'].iloc[-5:
                                              ].any() if 'bullish_fvg' in smc_15m else False
recent_fvg_bear = smc_15m['bearish_fvg'].iloc[-5:
                                              ].any() if 'bearish_fvg' in smc_15m else False

print(f"  Bullish CHOCH: {'✅ YES' if recent_choch_bull else '❌ NO'}")
print(f"  Bearish CHOCH: {'✅ YES' if recent_choch_bear else '❌ NO'}")
print(f"  Bullish BOS: {'✅ YES' if recent_bos_bull else '❌ NO'}")
print(f"  Bearish BOS: {'✅ YES' if recent_bos_bear else '❌ NO'}")
print(f"  Bullish FVG: {'✅ YES' if recent_fvg_bull else '❌ NO'}")
print(f"  Bearish FVG: {'✅ YES' if recent_fvg_bear else '❌ NO'}")

# Calculate confluence score manually
print("\n📊 CONFLUENCE SCORE CALCULATION:")
score = 0.0

# 1D Bias (25%)
if "BULLISH" in bias_1d:
    score += 0.25
    print(f"  1D Bias: +0.25 (bullish)")
elif "BEARISH" in bias_1d:
    score -= 0.25
    print(f"  1D Bias: -0.25 (bearish)")
else:
    print(f"  1D Bias: +0.00 (neutral)")

# 4H Structure (20%)
if "BULLISH" in structure_4h:
    score += 0.20
    print(f"  4H Structure: +0.20 (bullish)")
elif "BEARISH" in structure_4h:
    score -= 0.20
    print(f"  4H Structure: -0.20 (bearish)")
else:
    print(f"  4H Structure: +0.00 (neutral)")

# EMA Trend (15%)
if "BULLISH" in ema_trend:
    score += 0.15
    print(f"  15M EMA: +0.15 (bullish)")
elif "BEARISH" in ema_trend:
    score -= 0.15
    print(f"  15M EMA: -0.15 (bearish)")
else:
    print(f"  15M EMA: +0.00 (neutral)")

# CHOCH (20%)
if recent_choch_bull:
    score += 0.20
    print(f"  CHOCH: +0.20 (bullish)")
elif recent_choch_bear:
    score -= 0.20
    print(f"  CHOCH: -0.20 (bearish)")
else:
    print(f"  CHOCH: +0.00 (none)")

# BOS (12%)
if recent_bos_bull:
    score += 0.12
    print(f"  BOS: +0.12 (bullish)")
elif recent_bos_bear:
    score -= 0.12
    print(f"  BOS: -0.12 (bearish)")
else:
    print(f"  BOS: +0.00 (none)")

# FVG (8%)
if recent_fvg_bull:
    score += 0.08
    print(f"  FVG: +0.08 (bullish)")
elif recent_fvg_bear:
    score -= 0.08
    print(f"  FVG: -0.08 (bearish)")
else:
    print(f"  FVG: +0.00 (none)")

threshold = config['strategy']['confluence_threshold']

print(f"\n" + "=" * 70)
print(f"📊 FINAL CONFLUENCE SCORE: {score:+.2f}")
print(f"📊 THRESHOLD REQUIRED: ±{threshold}")
print("=" * 70)

if score >= threshold:
    print(f"🟢 SIGNAL: LONG (score {score:.2f} >= {threshold})")
elif score <= -threshold:
    print(f"🔴 SIGNAL: SHORT (score {score:.2f} <= -{threshold})")
else:
    print(
        f"⚪ SIGNAL: NO TRADE (score {abs(score):.2f} < {threshold} threshold)")
    print(f"\n💡 WHY NO TRADE:")
    if abs(score) < 0.20:
        print("   - Market is CHOPPY/RANGING - no clear direction")
    else:
        print(
            f"   - Need {threshold - abs(score):.2f} more confluence to trigger")
        print(
            f"   - Missing: {'bullish' if score > 0 else 'bearish'} confirmations")

print("\n" + "=" * 70)
