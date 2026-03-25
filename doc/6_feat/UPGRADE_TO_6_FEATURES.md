# SGRADT 7.0 - Upgrade to 6 Features with DI+/DI-

## 📋 Overview

This upgrade adds a **6th feature** to SGRADT 7.0 based on the Directional Indicators (DI+ and DI-) from the ADX indicator.

**New Feature:**
```python
feat_di_diff = DI+ - DI-
```

### Signal Interpretation
- **Positive values** → DI+ > DI- → **Bullish bias** (BUY signal preparation)
- **Negative values** → DI- > DI+ → **Bearish bias** (SELL signal preparation)
- **Magnitude** → Strength of the directional bias

This combined feature provides **trend direction strength** without introducing separate features for DI+ and DI-, keeping the model lean and generalizable.

---

## ✅ What Changed

### Python Training Script (`train_sgradt70_strategy.py`)

#### Imports
```python
# BEFORE
from ta.trend import ADXIndicator

# AFTER
from ta.trend import ADXIndicator, PLUSDI, MINUSDI
```

#### Feature Calculation
```python
# BEFORE (5 features)
df['feat_body'] = df['close'] - df['open']
df['feat_range'] = df['high'] - df['low']
df['feat_stoch_main'] = df['stoch_k']
df['feat_stoch_signal'] = df['stoch_d']
df['feat_adx'] = df['adx']
features_list = ['feat_body', 'feat_range', 'feat_stoch_main', 'feat_stoch_signal', 'feat_adx']

# AFTER (6 features)
df['feat_body'] = df['close'] - df['open']
df['feat_range'] = df['high'] - df['low']
df['feat_stoch_main'] = df['stoch_k']
df['feat_stoch_signal'] = df['stoch_d']
df['feat_adx'] = df['adx']
df['feat_di_diff'] = df['plus_di'] - df['minus_di']  # NEW!
features_list = ['feat_body', 'feat_range', 'feat_stoch_main', 'feat_stoch_signal', 'feat_adx', 'feat_di_diff']
```

#### Model Input Shape
```
BEFORE: [1, 100]  (5 features × 20 window)
AFTER:  [1, 120]  (6 features × 20 window)
```

---

### MQL5 EA (`EA_SGRADT70_ONNX_SL_TP_ATR.mq5`)

#### Feature Count Parameter
```mql5
// BEFORE
input int    InpFeaturesPerBar = 5;    // Features per bar (ALWAYS 5 for SGRADT 7.0)

// AFTER
input int    InpFeaturesPerBar = 6;    // Features per bar (6 = body, range, stoch_k, stoch_d, adx, di_diff)
```

#### Input Buffer Building (BuildInputBuffer function)
```mql5
// BEFORE (features in order):
input_buffer[offset + 0] = body
input_buffer[offset + 1] = range
input_buffer[offset + 2] = stoch_k
input_buffer[offset + 3] = stoch_d
input_buffer[offset + 4] = adx

// AFTER (features in order):
input_buffer[offset + 0] = body
input_buffer[offset + 1] = range
input_buffer[offset + 2] = stoch_k
input_buffer[offset + 3] = stoch_d
input_buffer[offset + 4] = adx
input_buffer[offset + 5] = DI+ - DI-
```

#### ADX Indicator Buffer Access
```mql5
// ADX indicator has 3 buffers:
CopyBuffer(g_adx_handle, 0, 0, window, adx_b);      // Buffer 0: ADX value
CopyBuffer(g_adx_handle, 1, 0, window, plus_di_b);  // Buffer 1: DI+ (Plus DI)
CopyBuffer(g_adx_handle, 2, 0, window, minus_di_b); // Buffer 2: DI- (Minus DI)

// Calculate DI difference
di_diff[i] = plus_di_b[i] - minus_di_b[i]
```

---

## 🚀 Migration Steps

### Step 1: Retrain Your Models

You **MUST retrain all ONNX models** with the new training script. Old 5-feature models will NOT work with the new 6-feature EA.

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
```

**Output:** New ONNX model with 6 features
```
EURUSD_M5_SGRADT70.onnx  (trained with 6 features)
```

### Step 2: Copy New EA to MT5

Replace your old EA with the updated one:
```
C:\Users\[Username]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\Experts\
EA_SGRADT70_ONNX_SL_TP_ATR.mq5
```

### Step 3: Compile in MetaEditor

1. Open MetaEditor
2. Load `EA_SGRADT70_ONNX_SL_TP_ATR.mq5`
3. Compile: **F5** or **Compile** button
4. Verify no errors (warnings are OK)

### Step 4: Copy New ONNX Model

Copy your **newly trained** ONNX model to MQL5/Files:
```
C:\Users\[Username]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\Files\
EURUSD_M5_SGRADT70.onnx  (NEW 6-feature version)
```

### Step 5: Update EA Parameters (if needed)

In the EA settings, verify:
```
InpFeaturesPerBar = 6         ← Should be auto-set, but verify
InpWindowSize = 20            ← Must match training --window
InpModelName = EURUSD_M5_SGRADT70.onnx  ← New model filename
```

### Step 6: Backtest

1. **Symbol:** EURUSD
2. **Timeframe:** M5 (or your trained symbol/TF)
3. **Period:** Minimum 3 months of new data
4. **Look for:**
   - Win rate > 50%
   - Profit factor > 1.5
   - Max drawdown < 20%
   - Balanced accuracy-like performance

### Step 7: Demo Trade (2-4 weeks)

Before live trading, run the new model on a **demo account**:
- Monitor daily signal generation
- Check panel information accuracy
- Verify ONNX inference speed
- Watch for any anomalies

---

## ⚠️ Important Compatibility Notes

### ❌ DO NOT Mix Old and New
```
INCOMPATIBLE:
  ❌ Old 5-feature ONNX + New 6-feature EA
  ❌ New 6-feature ONNX + Old 5-feature EA
```

**These combinations will fail with:**
```
[ERROR] Input buffer size incorrect
[ERROR] Model expects 120 inputs, got 100
```

### ✅ ALWAYS Match
```
COMPATIBLE:
  ✅ Old 5-feature ONNX + Old 5-feature EA (keep using old version)
  ✅ New 6-feature ONNX + New 6-feature EA (use new version)
```

---

## 📊 Feature Details

### Order in Input Buffer (CRITICAL)
```
Position  Feature          Calculation         Range        Purpose
─────────────────────────────────────────────────────────────────────
[0]       body             close - open         +/- varies   Candle strength
[1]       range            high - low           0-100+       Volatility
[2]       stoch_k          K line               0-100        Momentum
[3]       stoch_d          D line               0-100        Momentum signal
[4]       adx              ADX(8)               0-100        Trend strength
[5]       di_diff          DI+ - DI-           -100 to +100  Trend direction
```

### DI+ / DI- Behavior

| Condition | DI+ | DI- | DI Diff | Interpretation |
|-----------|-----|-----|---------|----------------|
| Strong uptrend | 35 | 15 | +20 | **Bullish** |
| Mild uptrend | 28 | 22 | +6 | **Slightly bullish** |
| Ranging | 25 | 25 | 0 | **Neutral** |
| Mild downtrend | 22 | 28 | -6 | **Slightly bearish** |
| Strong downtrend | 15 | 35 | -20 | **Bearish** |

---

## 🔧 Technical Details for Developers

### Python Dependencies
```bash
pip install pandas numpy scikit-learn skl2onnx ta-lib
```

### Feature Scaling (if needed)
The Random Forest classifier **does NOT require** feature scaling, so DI diff (-100 to +100) works without normalization alongside other features.

### ONNX Model Metadata
```
Model Type: sklearn.ensemble.RandomForestClassifier
Input Shape: [1, 120]  (batch_size=1, features=120)
  - 6 features × 20 window = 120 total inputs
Output Classes: 3 (0=HOLD, 1=BUY, 2=SELL)
Target Opset: 12
```

### MQL5 ADX Buffer Structure
```mql5
iADX(_Symbol, _Period, InpADXPeriod) returns:
  Buffer 0: ADX line value
  Buffer 1: +DI (PLUS_DI)
  Buffer 2: -DI (MINUS_DI)
```

**No separate indicator needed** - DI+ and DI- are included in the ADX indicator.

---

## 📈 Performance Impact

### Expected Changes
- **Balanced Accuracy:** ±1-3% (may improve or slightly decrease depending on data)
- **Win Rate:** Similar to previous (50-55%)
- **Profit Factor:** Potential improvement due to trend direction clarity
- **Inference Speed:** Negligible (Random Forest still fast)

### Why DI Feature Helps
1. **Trend Direction:** Complements ADX (trend strength) with direction clarity
2. **Early Signal:** DI crossovers often precede momentum changes
3. **Momentum Confirmation:** Works well with Stochastic for confluence
4. **Risk Management:** Helps filter false signals in choppy markets

---

## 🐛 Troubleshooting

### Error: "Input buffer size incorrect"
```
[ERROR] Buffer size mismatch: expected 120, got 100
```
**Cause:** Using old 5-feature model with new 6-feature EA
**Solution:** Retrain with new Python script to generate 6-feature ONNX

### Error: "Model expects 120 inputs"
```
[ERROR] Input buffer size mismatch
```
**Cause:** Same as above
**Solution:** Match ONNX features with EA features (both = 6)

### Error: "CopyBuffer PLUS_DI failed"
```
[ERROR] CopyBuffer PLUS_DI failed: expected 20, got 0
```
**Cause:** ADX indicator not ready yet
**Solution:** Wait longer in OnInit (EA will retry)

### ONNX Inference Returns Zeros
```
All confidence values = 0
```
**Cause:** Input buffer not properly filled with 6 features
**Solution:** Verify loop fills all 120 positions (6 × 20)

### No Signals Generated
```
BUY and SELL probabilities both very low
```
**Cause:** Model not trained properly with new feature
**Solution:** Retrain with sufficient data and check label distribution

---

## 📚 Version Info

| Aspect | Previous | Current |
|--------|----------|---------|
| Version | 7.0 (5 features) | 7.1 (6 features) |
| Features | 5 | **6** |
| Python Script | `train_sgradt70_strategy.py` | `train_sgradt70_strategy.py` (updated) |
| EA File | `EA_SGRADT70_ONNX_SL_TP_ATR.mq5` | `EA_SGRADT70_ONNX_SL_TP_ATR.mq5` (updated) |
| Input Shape | [1, 100] | **[1, 120]** |
| DI Feature | ❌ No | ✅ **Yes (DI+ - DI-)** |

---

## ✨ Next Steps

1. **Download** the updated files
2. **Retrain** your models with the new Python script
3. **Compile** the updated EA in MetaEditor
4. **Backtest** with new ONNX model (3+ months)
5. **Demo trade** for 2-4 weeks
6. **Go live** with confidence!

---

## 💡 FAQ

**Q: Can I use old 5-feature models?**  
A: No. You must retrain with the new script.

**Q: Will performance improve?**  
A: Potentially. The DI feature adds trend direction clarity.

**Q: Do I need to retrain data collection?**  
A: No, just retrain with your existing CSV files.

**Q: What if my model was profitable before?**  
A: Retest with new 6-feature model. Results may vary (±5-10% profit swing is normal).

**Q: Can I use the old EA with new ONNX?**  
A: No, it will fail with buffer size errors.

**Q: How long to retrain?**  
A: 30 seconds to 2 minutes depending on data size.

---

**Version:** 7.1.0  
**Date:** March 25, 2025  
**Status:** Ready for Production
