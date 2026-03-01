# 🎓 MODEL RETRAINING GUIDE

## 🤔 **DO I NEED TO RETRAIN?**

### ✅ **YES, Retrain When:**

1. **Adding New Features to ML Model**

   - Example: Adding volume profile, new indicators
   - Reason: Model doesn't know about new features
   - Command: `python train_model.py --days 365`

2. **Weekly/Monthly Maintenance** (Automated ✓)

   - Frequency: Every 7 days (configurable in config.yaml)
   - Reason: Markets evolve, patterns change
   - **Automatic:** Bot checks every 6 hours and retrains if needed

3. **After Major Market Events**

   - Example: Bitcoin halving, major crash, regulatory changes
   - Reason: Market regime has shifted
   - Command: `python auto_retrain.py --force`

4. **Performance Degradation** (Automated ✓)

   - Trigger: Win rate drops below 60%
   - **Automatic:** Bot retrains automatically when detected

5. **After Accumulating New Data** (Automated ✓)
   - Trigger: After 50 new trades
   - **Automatic:** Bot retrains to learn from recent trades

### ❌ **NO, Don't Need to Retrain When:**

1. **Changing Ensemble Weights**

   - Just configuration, model unchanged
   - No retraining needed

2. **Adding Non-ML Improvements**

   - Examples: Time filters, correlation checks, stop-loss logic
   - These are rule-based, not learned patterns

3. **Adjusting Risk Parameters**

   - Examples: position size, max trades, circuit breaker
   - Risk management is independent of ML

4. **Changing Trading Symbols**
   - Can use same model for different crypto pairs
   - BTC model works reasonably well for ETH

---

## 🤖 **AUTOMATIC RETRAINING (Already Implemented!)**

Your bot now has **intelligent automatic retraining**:

### **How It Works:**

```
Bot Running
    ↓
Every 6 hours → Check if retraining needed
    ↓
    ├─ Time-based: Last trained > 7 days ago?
    ├─ Performance: Win rate < 60%?
    └─ Data-based: 50+ new trades since last training?
    ↓
If YES → Automatically retrain in background
    ↓
Reload model with new weights
    ↓
Continue trading with improved model
```

### **Configuration:**

Edit `config/config.yaml`:

```yaml
ml_model:
  enabled: true
  retrain_interval_days: 7 # ← Change this (3, 7, 14, 30)
  min_training_samples: 1000
```

### **What You See in Logs:**

```
✅ Model check passed: Model is up-to-date
```

or

```
🔄 Retraining triggered: Performance-based: Win rate 58.3% < 60.0%
   MODEL TRAINED: Model trained successfully for BTC/USDT
   Train accuracy: 0.832
   Test accuracy: 0.817
✅ Automatic retraining completed successfully
```

---

## 🛠️ **MANUAL RETRAINING COMMANDS**

### **1. Check If Retraining Needed (No Action)**

```powershell
python auto_retrain.py --check-only
```

**Output:**

```
Should retrain: True
Reason: Time-based: 9 days since last training
```

### **2. Automatic Retrain (If Needed)**

```powershell
python auto_retrain.py
```

**What it does:**

- Checks all criteria (time, performance, data)
- Only retrains if needed
- Saves metadata for tracking

### **3. Force Retrain (Regardless of Criteria)**

```powershell
python auto_retrain.py --force
```

**When to use:** After adding new features or major market changes

### **4. Train Specific Symbol**

```powershell
python train_model.py --symbol BTC/USDT --days 365
```

### **5. Train All Symbols**

```powershell
python train_model.py --days 365
```

**How long it takes:**

- BTC/USDT: ~2-3 minutes
- ETH/USDT: ~2-3 minutes
- Total for 2 symbols: ~5 minutes

---

## 📊 **RETRAINING TRIGGERS EXPLAINED**

### **1. Time-Based (Every 7 Days)**

```
Why: Markets constantly evolve
Example:
  - Week 1: Bull market patterns
  - Week 2: Bear market patterns
  → Model needs to adapt
```

**Recommended:** 7 days for active markets, 14 days for stable

### **2. Performance-Based (Win Rate < 60%)**

```
Why: Model is underperforming
Example:
  - Last 50 trades: 28 wins, 22 losses = 56% win rate
  - Threshold: 60%
  → Trigger retrain to improve
```

**Configurable in:** `auto_retrain.py` → `self.min_win_rate = 0.60`

### **3. Data-Based (50+ New Trades)**

```
Why: New data = new patterns to learn
Example:
  - Last training: October 1
  - New trades since: 62 trades
  → Enough data to retrain and improve
```

**Configurable in:** `auto_retrain.py` → `self.min_new_trades = 50`

---

## 🎯 **RECOMMENDED RETRAINING SCHEDULE**

### **For Active Trading (Daily):**

```yaml
ml_model:
  retrain_interval_days: 7 # Weekly retraining
```

### **For Development/Testing:**

```yaml
ml_model:
  retrain_interval_days: 3 # More frequent for testing
```

### **For Long-Term Production:**

```yaml
ml_model:
  retrain_interval_days: 14 # Bi-weekly (more stable)
```

---

## 📈 **WHAT HAPPENS DURING RETRAINING?**

### **Step-by-Step:**

1. **Data Collection (2 min)**

   ```
   → Fetch 365 days of historical data
   → ~35,000 candles downloaded
   ```

2. **Feature Engineering (1 min)**

   ```
   → Calculate 21 features per candle
   → Add SMC signals (CHOCH, BOS, FVG)
   → Add technical indicators (RSI, EMA, MACD)
   ```

3. **Model Training (2 min)**

   ```
   → Train Random Forest with 100 trees
   → Test on 20% holdout data
   → Achieve 80-85% accuracy
   ```

4. **Save Model (1 sec)**

   ```
   → Save to models/saved_models/trading_model.pkl
   → Save metadata with timestamp
   ```

5. **Reload in Bot (1 sec)**
   ```
   → Bot detects new model
   → Reloads ML predictions
   → Continues trading with improved model
   ```

**Total Time: ~5-6 minutes**

**Impact on Trading:** NONE (bot continues running, retraining in background)

---

## 🔍 **MONITORING RETRAINING**

### **Check Last Training Time:**

```powershell
# View metadata file
cat models/saved_models/model_metadata.yaml
```

**Output:**

```yaml
last_training_time: "2025-11-04T14:30:00"
training_results:
  BTC/USDT:
    success: true
    test_accuracy: 0.817
    train_accuracy: 0.832
    samples_trained: 28000
```

### **Check Bot Logs:**

```powershell
cat logs/trading_bot.log | Select-String "retrain"
```

**Look for:**

```
✅ Model retrained successfully!
   Reason: Time-based: 8 days since last training
```

---

## ⚙️ **ADVANCED: CUSTOMIZING RETRAINING**

### **Change Retraining Criteria:**

Edit `auto_retrain.py`:

```python
class AutoRetrainer:
    def __init__(self, config_path: str = 'config/config.yaml'):
        # ... existing code ...

        # CUSTOMIZE THESE:
        self.retrain_interval_days = 7      # ← Change to 3, 14, 30
        self.min_win_rate = 0.60            # ← Change to 0.55, 0.65
        self.min_new_trades = 50            # ← Change to 30, 100
```

### **Disable Automatic Retraining:**

Remove from `main.py`:

```python
# Comment out this line:
# schedule.every(6).hours.do(self.check_model_retraining)
```

### **Change Check Frequency:**

Edit `main.py`:

```python
# Instead of every 6 hours:
schedule.every(6).hours.do(self.check_model_retraining)

# Use:
schedule.every(3).hours.do(self.check_model_retraining)  # More frequent
schedule.every(12).hours.do(self.check_model_retraining)  # Less frequent
schedule.every().day.at("02:00").do(self.check_model_retraining)  # Once daily at 2 AM
```

---

## 🎓 **WHEN IMPROVING THE BOT (Strategy #1-10)**

### **Improvements That NEED Retraining:**

| Strategy                  | Retrain? | Why                             |
| ------------------------- | -------- | ------------------------------- |
| #5: Better ML Features    | ✅ YES   | New features added to model     |
| #9: Adaptive Learning     | ✅ YES   | Changes training algorithm      |
| #10: Performance Tracking | ✅ YES   | Adjusts weights during training |

### **Improvements That DON'T Need Retraining:**

| Strategy               | Retrain? | Why                                                  |
| ---------------------- | -------- | ---------------------------------------------------- |
| #1: Dynamic Weights    | ❌ NO    | Only changes ensemble weights                        |
| #2: Confluence Scoring | ❌ NO    | Rule-based filtering                                 |
| #3: Time Filters       | ❌ NO    | Rule-based filtering                                 |
| #4: Volume Profile     | ✅ MAYBE | If added as ML feature → YES, if used as filter → NO |
| #6: Smart Stop-Loss    | ❌ NO    | Risk management logic                                |
| #7: Trailing Stop      | ❌ NO    | Risk management logic                                |
| #8: Correlation Filter | ❌ NO    | Rule-based filtering                                 |

---

## 💡 **BEST PRACTICES**

### **✅ DO:**

1. ✅ Let automatic retraining handle weekly updates
2. ✅ Force retrain after adding new ML features
3. ✅ Force retrain after major market events
4. ✅ Monitor retraining logs weekly
5. ✅ Keep 3-6 months of historical data

### **❌ DON'T:**

1. ❌ Retrain more than once per day (overfitting risk)
2. ❌ Train with < 30 days data (insufficient patterns)
3. ❌ Panic retrain after one bad day
4. ❌ Retrain for every small configuration change
5. ❌ Ignore retraining completely (model becomes stale)

---

## 🚀 **QUICK REFERENCE**

```powershell
# Check if retraining needed
python auto_retrain.py --check-only

# Automatic retrain (if needed)
python auto_retrain.py

# Force retrain
python auto_retrain.py --force

# Train specific symbol
python train_model.py --symbol BTC/USDT

# View last training time
cat models/saved_models/model_metadata.yaml

# Check retraining logs
cat logs/trading_bot.log | Select-String "retrain"
```

---

## ❓ **FAQ**

**Q: Will retraining pause my trading?**  
A: NO. Retraining runs in background, bot continues trading.

**Q: How long does retraining take?**  
A: ~5-6 minutes for 2 symbols (BTC/USDT, ETH/USDT).

**Q: Can I trade during retraining?**  
A: YES. Old model is used until new model finishes training.

**Q: Will I lose my old model?**  
A: Old model is overwritten. If concerned, backup `models/saved_models/` folder.

**Q: What if retraining fails?**  
A: Bot continues with old model. Check logs for error details.

**Q: How often should I retrain?**  
A: Weekly (7 days) is optimal. Let automatic system handle it.

**Q: Does retraining use GPU?**  
A: NO. Uses CPU (Random Forest, not deep learning).

---

## 🎯 **SUMMARY**

### **Your Bot is SMART:**

1. ✅ **Automatic retraining** every 7 days
2. ✅ **Performance monitoring** (retrains if win rate drops)
3. ✅ **Data accumulation** (retrains after 50 new trades)
4. ✅ **Zero interruption** (trains in background)
5. ✅ **No manual work** (unless adding new features)

### **You Only Need to Manually Retrain When:**

1. Adding new ML features (Strategy #5)
2. Major market event (Bitcoin halving, crash)
3. Testing new training algorithm (Strategy #9)

### **Everything Else is AUTOMATIC! 🎉**

---

**Last Updated:** November 4, 2025  
**Next Review:** After implementing improvement strategies
