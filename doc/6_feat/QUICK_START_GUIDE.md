# SGRADT 7.1 - Quick Start Implementation Guide

## 🚀 5-Minute Overview

**SGRADT 7.0 → 7.1 Upgrade Summary:**
```
❌ 5 Features: [body, range, stoch_k, stoch_d, adx]
✅ 6 Features: [body, range, stoch_k, stoch_d, adx, DI+−DI−]

New Feature Behavior:
  • Positive values = Bullish (DI+ > DI−)
  • Negative values = Bearish (DI− > DI+)
  • Adds trend direction to complement ADX trend strength
```

---

## ⚙️ Implementation Checklist (Copy & Paste)

### Step 1: Get Updated Files ✅
```bash
✓ train_sgradt70_strategy.py     (updated)
✓ EA_SGRADT70_ONNX_SL_TP_ATR.mq5 (updated)
✓ UPGRADE_TO_6_FEATURES.md       (documentation)
```

### Step 2: Retrain Models (5 minutes)
```bash
python train_sgradt70_strategy.py \
  --csv data/EURUSD_M5.csv \
  --window 20 \
  --min_profit_points 20.0 \
  --future 50 \
  --stoch_k 7 \
  --stoch_d 3 \
  --adx_period 8 \
  --adx_limit 25.0 \
  --n_iter 7 \
  --output ./onnx

# Output:
# → EURUSD_M5_SGRADT70.onnx (6-feature model)
```

**Expected Output:**
```
[INFO] Signal counts - BUY: 245, SELL: 198, TOTAL: 443
[INFO] Prepared features with window size: 20
[INFO] Prepared training data: X.shape=(3920, 120), y.shape=(3920,)
[INFO] Training completed. Best score: 0.6547
[INFO] Writing ONNX model to EURUSD_M5_SGRADT70.onnx
[INFO] Model saved: EURUSD_M5_SGRADT70.onnx | Accuracy: 0.6547
```

**✅ Success if:**
- X.shape contains **120** (6 features × 20 window)
- No errors about missing indicators
- .onnx file created (2-5 MB)

### Step 3: Update EA in MetaEditor (2 minutes)
```
1. Open MetaEditor
2. File → Open → EA_SGRADT70_ONNX_SL_TP_ATR.mq5
3. Press F5 (Compile)
4. Look for: ✓ 0 errors, 0 warnings (warnings OK)
5. File → Save
```

**Compile Output Should Show:**
```
Compilation successful
No errors
Warnings OK
```

### Step 4: Copy Files to MT5 (1 minute)
```
File locations:
C:\Users\[YourName]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\

Copy files:
→ Experts\         : EA_SGRADT70_ONNX_SL_TP_ATR.mq5
→ Files\           : EURUSD_M5_SGRADT70.onnx (new 6-feature model)
```

### Step 5: Backtest (30 minutes)
```
MT5 Strategy Tester:
  • Symbol: EURUSD (or your trained symbol)
  • Timeframe: M5 (or your trained timeframe)
  • Period: 3 months minimum
  • Model: EURUSD_M5_SGRADT70.onnx
  
Settings in EA:
  • InpModelName = "EURUSD_M5_SGRADT70.onnx"
  • InpWindowSize = 20 (must match --window in training)
  • InpFeaturesPerBar = 6 (auto-set, verify it's 6)
  
Success Metrics:
  ✓ Win Rate > 50%
  ✓ Profit Factor > 1.5
  ✓ Max Drawdown < 20%
  ✓ Sharpe Ratio > 0.8
```

### Step 6: Demo Trade (2-4 weeks)
```
Before going live:
  ✓ Run on demo account for 2-4 weeks
  ✓ Check signal generation daily
  ✓ Monitor P&L
  ✓ Look for consistent patterns
  ✓ Verify ONNX inference speed
```

### Step 7: Go Live ✅
```
When ready:
  ✓ Start with 0.01 lot (minimum)
  ✓ Trade 1-2 weeks at small size
  ✓ Monitor daily
  ✓ Gradually increase lot size
```

---

## 🔍 Quick Diagnostics

### Q1: Model training fails with "PLUSDI/MINUSDI not found"
```python
# Problem: ta-lib missing indicators
# Solution:
pip install --upgrade ta

# or if using ta-lib C library:
pip install TA-Lib==0.4.28
```

### Q2: EA says "Input buffer size incorrect"
```
❌ PROBLEM:
   Using old 5-feature ONNX model with new 6-feature EA
   
✅ SOLUTION:
   Retrain with new Python script to get 6-feature ONNX
```

### Q3: ONNX model inference returns all zeros
```
❌ PROBLEM:
   Input buffer not filled with 6 features
   
✅ SOLUTION:
   Check BuildInputBuffer loop fills positions 0-5
   Verify: for(int i = 0; i < window; i++)
           has 6 assignments: offset+0 through offset+5
```

### Q4: Training takes longer than before
```
✅ NORMAL:
   5-feature: 40-50s
   6-feature: 45-60s (+10-20% expected)
```

### Q5: Which .onnx file to use?
```
❌ WRONG:
   Old model from previous training (5 features)
   
✅ CORRECT:
   New model from new training script (6 features)
```

---

## 📊 Feature Order Reference (CRITICAL)

**Python → ONNX → EA must all match this order:**

```
Position | Feature Name  | Calculation         | Range      | Notes
─────────┼───────────────┼─────────────────────┼────────────┼──────────────
   0     | body          | close - open        | any        | Candle size
   1     | range         | high - low          | 0 to +∞    | Volatility
   2     | stoch_k       | Stoch %K            | 0-100      | Fast momentum
   3     | stoch_d       | Stoch %D            | 0-100      | Slow momentum
   4     | adx           | ADX(8)              | 0-100      | Trend strength
   5     | di_diff       | DI+(8) - DI-(8)     | -100 to +100 | Trend direction ← NEW
```

**Mnemonic:** "**B**ody **R**ange **K** **D** **A**DX **D**I"

---

## 💡 Key Differences (Quick Reference)

| Aspect | v7.0 (5-feat) | v7.1 (6-feat) |
|--------|---------------|---------------|
| ONNX Shape | [1, 100] | [1, 120] |
| Features | 5 | **6** |
| InpFeaturesPerBar | 5 | **6** |
| DI+ / DI- | ❌ No | ✅ Yes |
| Trend Direction | ADX only | ADX + DI |
| Training Time | 40-50s | 45-60s |
| Inference Speed | <0.5ms | <0.5ms |
| Backward Compat | ❌ No | N/A |

---

## 🧪 Validation Tests

### Test 1: Verify Feature Count
```python
# After training, check this output:
# Should show 6 features
python train_sgradt70_strategy.py --csv test.csv --output ./test_onnx
# Look for: "X.shape=(XXXX, 120)" ← 120 = 6 features × 20 window
```

### Test 2: Verify EA Compilation
```
In MetaEditor:
  F5 to compile
  Should show 0 errors
  InpFeaturesPerBar should be visible as 6
```

### Test 3: Verify ONNX Input Size
```
Small backtest:
  Run 1 bar of backtest
  Check EA journal:
  "Input shape set: [1, 120]" ← Should be 120
```

### Test 4: Verify DI Calculation
```mql5
// In MT5 journal, during inference, you should see:
// CopyBuffer PLUS_DI returned 20 values
// CopyBuffer MINUS_DI returned 20 values
// No "CopyBuffer PLUS_DI failed" errors
```

---

## 📋 Troubleshooting Decision Tree

```
ERROR: "Input buffer size incorrect"
├─ ❌ Are you using old 5-feature ONNX?
│  └─ YES → Retrain with new Python script
│  └─ NO → Go to next check
└─ ❌ Is InpFeaturesPerBar = 6 in EA?
   └─ NO → Change to 6 and recompile
   └─ YES → Check BuildInputBuffer loop

ERROR: "Model expects 120 inputs, got 100"
├─ ❌ Did you update to new Python script?
│  └─ NO → Update and retrain
│  └─ YES → Go to next check
└─ ❌ Is training actually using 6 features?
   └─ Check: X.shape should have 120 in 2nd dimension

ERROR: "CopyBuffer PLUS_DI failed"
├─ ❌ Is ADX indicator initialized?
│  └─ NO → Increase sleep time in OnInit
│  └─ YES → Go to next check
└─ ❌ Is g_adx_handle valid?
   └─ Check: g_adx_handle != INVALID_HANDLE in OnInit
```

---

## ✨ Success Indicators

### Training Output ✅
```
✓ "X.shape=(XXXX, 120)" shows 120 inputs
✓ "Label distribution in training data: [count HOLD, count BUY, count SELL]"
✓ "Training completed. Best score: 0.55-0.70" (typical range)
✓ "Writing ONNX model to EURUSD_M5_SGRADT70.onnx"
✓ File size 2-5 MB
```

### EA Compilation ✅
```
✓ "Compilation successful"
✓ "0 errors"
✓ Warnings OK (green checkmark)
```

### Backtest ✅
```
✓ Multiple trades generated
✓ Mix of BUY and SELL signals
✓ P&L shows both wins and losses
✓ No runtime errors in journal
✓ "Input shape set: [1, 120]" in first bar
```

### Demo Trade ✅
```
✓ Signals generated on new bars
✓ Panel shows prediction confidence
✓ P&L tracking reasonable
✓ No EA errors in daily monitoring
✓ Ready for live trading
```

---

## 🎯 Common Questions

**Q: Do I need to retrain?**  
A: **YES**. Old 5-feature models won't work with new 6-feature EA.

**Q: Will profitability improve?**  
A: Maybe. DI feature adds direction clarity. Retest and compare.

**Q: Can I keep using the old version?**  
A: Yes, keep old Python/EA/ONNX together if you prefer 5-feature version.

**Q: How long to retrain 3 years of data?**  
A: 1-3 minutes depending on CSV size and n_iter.

**Q: What's the best value for --window?**  
A: 20 (default) works well. Can try 15-25 for optimization.

**Q: Do I need to update --adx_period?**  
A: Keep it at 8 (same as before). This matches the DI+ DI- periods.

**Q: Is the new feature autocorrelated with ADX?**  
A: Somewhat, but DI adds **direction** while ADX only shows **strength**. Good complement.

---

## 📈 Next Level: Optimization

After going live, you can try:

```bash
# Experiment with different parameters
python train_sgradt70_strategy.py \
  --csv data.csv \
  --window 15 \                    # Try shorter window
  --stoch_k 5 \                    # Try faster stochastic
  --stoch_d 2 \
  --min_profit_points 15 \         # Try lower target
  --future 60 \                    # Try longer horizon
  --adx_period 10 \                # Try longer ADX/DI period
  --n_iter 15 \                    # More search iterations
  --output ./onnx_optimized
```

Then backtest each variant and compare.

---

## 🎓 Learning Path

1. **Understand the 6 features** (read VISUAL_COMPARISON.md)
2. **Retrain your models** (10 minutes)
3. **Backtest new version** (30 minutes)
4. **Demo trade** (2-4 weeks)
5. **Go live with confidence** (when ready)
6. **Monitor and optimize** (ongoing)

---

## ☎️ Support Checklist

Before asking for help, verify:
- [ ] Running new Python script (with PLUSDI, MINUSDI imports)
- [ ] ONNX file has 120 inputs (check X.shape output)
- [ ] EA parameter InpFeaturesPerBar = 6
- [ ] BuildInputBuffer has offset+5 assignment
- [ ] Compiled with 0 errors
- [ ] Copied new ONNX to MQL5/Files/
- [ ] Not mixing old ONNX with new EA

---

**Version:** 7.1.0  
**Last Updated:** March 25, 2025  
**Status:** Ready to Deploy  

**Good luck! 🚀**
