# 🎮 DEMO MODE - Quick Reference

## What is Demo Mode?

A version of the main bot with **relaxed trading rules** to show it in action faster.

---

## 🔄 Differences: Demo vs Real Mode

| Feature                  | Real Mode (main.py)   | Demo Mode (main_demo.py) |
| ------------------------ | --------------------- | ------------------------ |
| **Conditions Needed**    | 4/4 (all must align)  | 2/4 (only half needed)   |
| **Confidence Threshold** | 60%                   | 40%                      |
| **Expected Trades**      | 0-3 per day           | 3-10 per day             |
| **Time to See Trade**    | 4-12 hours            | 1-2 hours                |
| **Realism**              | High (FYP final demo) | Medium (Quick testing)   |
| **Safety**               | Paper trading ✅      | Paper trading ✅         |

---

## 🚀 How to Run Demo Mode

### Step 1: Start Demo Mode

```powershell
python main_demo.py
```

You'll see:

```
======================================================================
🎮 DEMO MODE - AI Trading Bot
======================================================================

⚠️  WARNING: This is DEMO MODE with relaxed trading rules!

Differences from normal mode:
  • Only 2/4 conditions needed (vs 4/4)
  • Lower confidence threshold (40% vs 60%)
  • MORE FREQUENT TRADES expected

Purpose: Demonstrate the bot working within 1-2 hours

Still using PAPER TRADING - No real money at risk! ✅
======================================================================

Press ENTER to start demo mode...
```

### Step 2: Watch the Logs (Optional)

Open another terminal:

```powershell
Get-Content logs\trading_bot.log -Tail 50 -Wait
```

### Step 3: Wait for Trades

- **First cycle**: Immediate analysis
- **15 minutes later**: 2nd analysis
- **30 minutes later**: 3rd analysis
- **Within 1-2 hours**: Likely to see trades!

### Step 4: Stop Anytime

Press `Ctrl+C` in the demo mode window

---

## 📊 What You'll See

### Normal Analysis (No Trade Yet):

```
2025-10-25 10:00:00 - INFO - 🎮 Running DEMO MODE analysis...
2025-10-25 10:00:15 - INFO - 🎮 DEMO - BTC/USDT: Bullish 1/4, Bearish 0/4
2025-10-25 10:00:16 - INFO - No signal for BTC/USDT - even with relaxed rules
2025-10-25 10:00:20 - INFO - 🎮 DEMO - ETH/USDT: Bullish 2/4, Bearish 0/4
2025-10-25 10:00:21 - INFO - 🎮 DEMO LONG signal for ETH/USDT!
2025-10-25 10:00:22 - INFO - ✅ DEMO TRADE executed successfully: 1
2025-10-25 10:00:22 - INFO -    LONG ETH/USDT @ $3942.41
```

### Trade Execution:

```
✅ PAPER TRADE executed - ID: 1
Entry: $3942.41, SL: $3910.18, TP: $4006.87
```

### Trade Monitoring:

```
2025-10-25 10:15:00 - INFO - Monitoring open trades...
2025-10-25 10:15:01 - INFO - Trade 1: ETH/USDT LONG at $3942, current: $3955 (+0.33%)
```

### Trade Closed:

```
2025-10-25 10:45:00 - INFO - Trade 1 hit take-profit at $4006.87
2025-10-25 10:45:01 - INFO - Trade 1 closed - Reason: take_profit
```

---

## ⏰ Expected Timeline

| Time          | What Happens                                      |
| ------------- | ------------------------------------------------- |
| **0:00**      | Bot starts, immediate analysis                    |
| **0:00-0:15** | First cycle completes, might see trade            |
| **0:15**      | 2nd analysis cycle                                |
| **0:30**      | 3rd analysis cycle                                |
| **0:45**      | 4th analysis cycle                                |
| **1:00**      | 5th cycle - very likely to have seen 1-2 trades   |
| **2:00**      | 9th cycle - definitely should see multiple trades |

**Probability:**

- 30 minutes: 40% chance of trade
- 1 hour: 70% chance of trade
- 2 hours: 95% chance of trade

---

## 🎯 When to Use Demo Mode vs Real Mode

### Use DEMO MODE when:

- ✅ Quick demonstration for FYP presentation
- ✅ Want to see bot working in 1-2 hours
- ✅ Testing if everything is set up correctly
- ✅ Showing multiple trade examples quickly

### Use REAL MODE when:

- ✅ Final FYP evaluation/testing
- ✅ Want realistic performance metrics
- ✅ Evaluating actual strategy effectiveness
- ✅ Long-term testing (days/weeks)

---

## 📈 Checking Results

### View Trades in Database:

```powershell
python -c "import sqlite3; conn = sqlite3.connect('data/trading_bot.db'); cursor = conn.cursor(); trades = cursor.execute('SELECT * FROM trades ORDER BY id DESC LIMIT 10').fetchall(); [print(f'Trade {t[0]}: {t[2]} {t[1]} @ ${t[3]:.2f}') for t in trades]"
```

### View Logs:

```powershell
# Show all trades
Select-String -Path logs\trading_bot.log -Pattern "DEMO TRADE executed"

# Show trade results
Select-String -Path logs\trading_bot.log -Pattern "closed - Reason"

# Count trades
(Select-String -Path logs\trading_bot.log -Pattern "DEMO TRADE executed").Count
```

---

## ⚠️ Important Notes

1. **Still Paper Trading**: Even in demo mode, NO REAL MONEY is used
2. **More Trades ≠ Better**: Demo mode trades more, but real mode is more selective (better for live trading)
3. **For Demonstration Only**: Don't use demo mode rules for actual live trading
4. **Same Components**: Still uses SMC, TA, ML, Sentiment - just more lenient

---

## 🎓 For Your FYP

### Demo Mode is Perfect For:

- ✅ Class presentation (show it working in 1 class period)
- ✅ Quick demonstration to advisor
- ✅ Showing all components working together
- ✅ Proving bot can make decisions

### Real Mode is Better For:

- ✅ Final evaluation results
- ✅ Performance statistics
- ✅ Showing realistic win rates
- ✅ Proving strategy viability

**Tip:** Use BOTH in your FYP!

- Demo mode: "Here's the bot in action"
- Real mode: "Here's the realistic performance over 2 weeks"

---

## 🚀 Quick Start Command

```powershell
# Just run this!
python main_demo.py
```

**That's it!** Wait 1-2 hours and you'll see trades. 🎉

---

## ❓ FAQ

**Q: Is demo mode safe?**  
A: YES! Still paper trading, no real money.

**Q: How long should I run it?**  
A: 1-2 hours minimum to see trades.

**Q: Can I use demo mode results in my FYP?**  
A: Yes, but label it clearly as "demo mode with relaxed rules for demonstration."

**Q: Which is better for final FYP evaluation?**  
A: Real mode (`main.py`) - more realistic and selective.

**Q: Can I stop it anytime?**  
A: YES! Just press Ctrl+C.

---

## 🎯 Bottom Line

**Demo Mode = Quick demonstration of bot working**

Perfect for showing your FYP committee/advisor the bot in action WITHOUT waiting hours/days for trades!

Ready to try it? Just run:

```powershell
python main_demo.py
```

🎮 Happy Demo Trading! 🚀
