# EA_SGRADT50_ONNX - Fixed Expert Advisor Documentation

## 🔧 What Was Fixed

### Original EA Issues

The original `EA_PriceAction_ADX_Stoch_ONNX.mq5` had several mismatches with the new ONNX model:

#### ❌ Problem 1: Wrong Feature Order
```cpp
// OLD (original EA)
input_buffer[offset + 0] = body;
input_buffer[offset + 1] = range;
input_buffer[offset + 2] = adx;
input_buffer[offset + 3] = pdi;
input_buffer[offset + 4] = mdi;
input_buffer[offset + 5] = stoch_k;
input_buffer[offset + 6] = stoch_d;
```

**New training script uses:**
```python
features_list = [
    'feat_body',        # 0
    'feat_range',       # 1
    'feat_stoch_main',  # 2 ← CHANGED
    'feat_stoch_signal',# 3 ← CHANGED
    'feat_adx',         # 4 ← CHANGED
    'feat_pdi',         # 5 ← CHANGED
    'feat_mdi'          # 6 ← CHANGED
]
```

#### ❌ Problem 2: Wrong Default Parameters
```cpp
// OLD
input int InpStochK = 5;      // Should be 7 for SGRADT 5.0
input int InpStochD = 3;      // Correct
input double InpStochOversold = 30.0;  // Should be 20.0
input double InpStochOverbought = 70.0; // Should be 80.0
// No ADX period input (hardcoded to 14)
```

#### ❌ Problem 3: Missing Indicator Handles Management
Original EA recreated indicators on every tick, causing performance issues.

#### ❌ Problem 4: Hardcoded Values
```cpp
int adx_h = iADX(_Symbol, _Period, 14);  // Hardcoded, should match training
```

### ✅ Fixed EA Changes

#### ✅ Fix 1: Correct Feature Order
```cpp
// NEW (fixed EA)
input_buffer[offset + 0] = (float)(close[i] - open[i]);     // body
input_buffer[offset + 1] = (float)(high[i] - low[i]);       // range
input_buffer[offset + 2] = (float)stoch_k_b[i];             // stoch_main ← FIXED
input_buffer[offset + 3] = (float)stoch_d_b[i];             // stoch_signal ← FIXED
input_buffer[offset + 4] = (float)adx_b[i];                 // adx ← FIXED
input_buffer[offset + 5] = (float)pdi_b[i];                 // pdi ← FIXED
input_buffer[offset + 6] = (float)mdi_b[i];                 // mdi ← FIXED
```

**This matches the training script exactly.**

#### ✅ Fix 2: SGRADT 5.0 Default Parameters
```cpp
// NEW defaults match training script
input int    InpStochK          = 7;      // ✓ SGRADT 5.0 default
input int    InpStochD          = 3;      // ✓ Correct
input int    InpStochSlowing    = 3;      // ✓ Correct
input double InpStochOversold   = 20.0;   // ✓ SGRADT 5.0 default
input double InpStochOverbought = 80.0;   // ✓ SGRADT 5.0 default
input int    InpADXPeriod       = 8;      // ✓ SGRADT 5.0 default
input double InpADXLimit        = 32.0;   // ✓ SGRADT 5.0 default
```

#### ✅ Fix 3: Proper Indicator Handles
```cpp
// NEW: Create handles once in OnInit
int g_adx_handle   = INVALID_HANDLE;
int g_stoch_handle = INVALID_HANDLE;

int OnInit()
{
   g_adx_handle = iADX(_Symbol, _Period, InpADXPeriod);
   g_stoch_handle = iStochastic(_Symbol, _Period,
                                InpStochK, InpStochD, InpStochSlowing,
                                MODE_SMA, STO_LOWHIGH);
   // ...
}

void OnDeinit(const int reason)
{
   if(g_adx_handle != INVALID_HANDLE)
      IndicatorRelease(g_adx_handle);
   if(g_stoch_handle != INVALID_HANDLE)
      IndicatorRelease(g_stoch_handle);
}
```

#### ✅ Fix 4: Configurable Parameters
```cpp
// NEW: All indicator parameters are configurable inputs
input int InpADXPeriod = 8;  // Can be changed by user
```

## 📊 Feature Order Comparison

### Training Script (Python)
```python
features_list = [
    'feat_body',         # 0: close - open
    'feat_range',        # 1: high - low
    'feat_stoch_main',   # 2: Stochastic %K
    'feat_stoch_signal', # 3: Stochastic %D
    'feat_adx',          # 4: ADX value
    'feat_pdi',          # 5: +DI
    'feat_mdi'           # 6: -DI
]
```

### Fixed EA (MQL5)
```cpp
// Feature order matches training exactly
input_buffer[offset + 0] = body;          // 0: close - open
input_buffer[offset + 1] = range;         // 1: high - low
input_buffer[offset + 2] = stoch_k;       // 2: Stochastic %K
input_buffer[offset + 3] = stoch_d;       // 3: Stochastic %D
input_buffer[offset + 4] = adx;           // 4: ADX value
input_buffer[offset + 5] = pdi;           // 5: +DI
input_buffer[offset + 6] = mdi;           // 6: -DI
```

**✅ Perfect match!**

## 🚀 How to Use

### Step 1: Train Your Model

```bash
python train_sgradt_strategy.py \
    --csv EUR_USD_H1.csv \
    --strategy combined \
    --window 20 \
    --output ./models
```

This generates:
- `EUR_USD_H1_SGRADT50_combined.onnx`
- `EUR_USD_H1_SGRADT50_combined.meta.json`

### Step 2: Install Files in MetaTrader 5

Copy files to MT5:
```
📁 MQL5/
  ├─ 📁 Experts/
  │   └─ EA_SGRADT50_ONNX.mq5
  │
  └─ 📁 Files/
      ├─ EUR_USD_H1_SGRADT50_combined.onnx
      └─ EUR_USD_H1_SGRADT50_combined.meta.json (optional)
```

### Step 3: Configure EA Parameters

Open the EA settings and configure:

#### Model Configuration
```
InpModelName = "EUR_USD_H1_SGRADT50_combined.onnx"
InpWindowSize = 20        // Must match training
InpFeaturesPerBar = 7     // Must match training
InpMinConf = 0.55         // Minimum confidence (55%)
```

#### Indicator Parameters (must match training)
```
InpStochK = 7             // From --stoch_k
InpStochD = 3             // From --stoch_d
InpStochSlowing = 3       // From --stoch_slowing
InpStochOversold = 20.0   // From --stoch_oversold
InpStochOverbought = 80.0 // From --stoch_overbought
InpADXPeriod = 8          // From --adx_period
InpADXLimit = 32.0        // From --adx_limit
```

#### Risk Management
```
InpLot = 0.1              // Your lot size
InpStopPoints = 50.0      // SL in points
InpTakePoints = 100.0     // TP in points
```

### Step 4: Verify Configuration

Check the Experts tab log when EA starts:

```
✓ ONNX model loaded: EUR_USD_H1_SGRADT50_combined.onnx
✓ Input shape set: [1, 140] (20 bars × 7 features)
✓ Indicators created: ADX(8) + Stochastic(7,3,3)

=== SYMBOL INFORMATION ===
Symbol: EURUSD
Timeframe: PERIOD_H1
Digits: 5
Point: 0.000010
SL: 50 points = 0.000500 price
TP: 100 points = 0.001000 price

✓ Inference mode: New bar only

=== EA INITIALIZED SUCCESSFULLY ===
```

**If you see errors**, check:
1. ONNX file is in `MQL5/Files/` folder
2. `InpWindowSize` matches training (default: 20)
3. `InpFeaturesPerBar` is correct (always 7)

## 🎯 Input Parameters Reference

### AI Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `InpModelName` | `EUR_USD_H1_SGRADT50_combined.onnx` | ONNX model filename |
| `InpMetaFile` | `EUR_USD_H1_SGRADT50_combined.meta.json` | Metadata file (optional) |
| `InpMinConf` | `0.55` | Minimum confidence (0.0-1.0) |
| `InpWindowSize` | `20` | Window size (must match training) |
| `InpFeaturesPerBar` | `7` | Features per bar (always 7) |

### Inference Timing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `InpInferSeconds` | `15` | Inference frequency (0 = new bar only) |
| `InpOneTradePerBar` | `true` | Limit to 1 trade per bar |

### Trading Session

| Parameter | Default | Description |
|-----------|---------|-------------|
| `InpStartHour` | `0` | Session start hour (0-23) |
| `InpEndHour` | `24` | Session end hour (0-24) |

### Indicator Parameters (SGRADT 5.0 Defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `InpStochK` | `7` | Stochastic %K period |
| `InpStochD` | `3` | Stochastic %D smoothing |
| `InpStochSlowing` | `3` | Stochastic slowing |
| `InpStochOversold` | `20.0` | Oversold level |
| `InpStochOverbought` | `80.0` | Overbought level |
| `InpADXPeriod` | `8` | ADX period |
| `InpADXLimit` | `32.0` | ADX trend threshold |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `InpLot` | `1.0` | Lot size |
| `InpMagic` | `5050` | Magic number |
| `InpStopPoints` | `50.0` | Stop Loss in POINTS |
| `InpTakePoints` | `100.0` | Take Profit in POINTS |

### Display Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `InpShowPanel` | `true` | Show information panel |

## 📈 Understanding the Panel

The EA displays a comprehensive panel showing:

```
╔════════════════════════════════════════════╗
║      SGRADT 5.0 - AI TRADING SYSTEM      ║
╚════════════════════════════════════════════╝

📊 SYMBOL: EURUSD [PERIOD_H1]
⏰ SESSION: 00:00-24:00 [✓ ACTIVE]
🔄 MODE: NEW BAR | Inferences: 42
🕐 Last Run: 14:23:15

────────────────────────────────────────────
📈 ADX INDICATOR (Period: 8)
────────────────────────────────────────────
   ADX: 35.42 [TRENDING]
   +DI: 28.15
   -DI: 18.73

────────────────────────────────────────────
📊 STOCHASTIC (7,3,3)
────────────────────────────────────────────
   %K: 75.23
   %D: 72.18
   Zone: NEUTRAL ─
   Cross: %K ABOVE %D ↑

════════════════════════════════════════════
🤖 AI PREDICTION
════════════════════════════════════════════
   Signal: 🟢 BUY

   Confidence Levels:
   ├─ HOLD:  12.34%
   ├─ BUY:   67.89%
   └─ SELL:  19.77%

   Minimum Required: 55.0%

────────────────────────────────────────────
💰 RISK SETTINGS
────────────────────────────────────────────
   Lot Size: 0.10
   Stop Loss:   50 pts (0.00050)
   Take Profit: 100 pts (0.00100)

════════════════════════════════════════════
💼 ACTIVE POSITION: BUY 📈
   P&L: +12.50 USD
════════════════════════════════════════════
```

## 🔍 Troubleshooting

### Error: "Cannot load ONNX model"

**Solution:**
1. Ensure ONNX file is in `MQL5/Files/` folder
2. Check filename matches `InpModelName` exactly
3. Restart MetaTrader 5

### Error: "Not enough data"

**Solution:**
1. Ensure you have at least `InpWindowSize` bars of history loaded
2. For window=20, you need at least 20 completed bars
3. Wait for more data or load more history

### Wrong Predictions / Low Performance

**Checklist:**
- ✅ `InpWindowSize` matches training (`--window` parameter)
- ✅ `InpStochK`, `InpStochD`, `InpStochSlowing` match training
- ✅ `InpADXPeriod` matches training
- ✅ Same symbol as training data
- ✅ Same timeframe as training data

**Feature order is critical!** The fixed EA now matches exactly.

### Confidence Always Low

**Possible causes:**
1. Model not well-trained (low balanced_accuracy)
2. Market conditions different from training data
3. Wrong timeframe or symbol

**Solutions:**
- Retrain with more data
- Lower `InpMinConf` (e.g., from 0.55 to 0.45)
- Use same timeframe as training

## 📝 Important Notes

1. **Feature Order**: The EA now uses the **exact** feature order from the training script:
   - `body, range, stoch_k, stoch_d, adx, pdi, mdi`

2. **Window Size**: Must match training. Default is 20 bars.

3. **Parameters**: Use the same indicator parameters as training:
   - Stochastic: (7, 3, 3, 20, 80)
   - ADX: (8, 32)

4. **Timeframe**: Use the EA on the same timeframe you trained on.

5. **Symbol**: Preferably use the same symbol, or at least similar characteristics.

## 🆚 Old vs New Comparison

| Aspect | Old EA | New EA |
|--------|--------|--------|
| Feature order | ❌ Wrong | ✅ Correct |
| Stochastic defaults | ❌ (5,3,3,30,70) | ✅ (7,3,3,20,80) |
| ADX period | ❌ Hardcoded 14 | ✅ Configurable, default 8 |
| Indicator handles | ❌ Recreated every tick | ✅ Created once in OnInit |
| Parameters match training | ❌ No | ✅ Yes |
| SGRADT 5.0 compatible | ❌ No | ✅ Yes |

## 🎓 Best Practices

1. **Backtesting**: Always backtest before live trading
2. **Demo Account**: Test on demo first
3. **Start Small**: Use small lot sizes initially
4. **Monitor**: Watch the panel to understand predictions
5. **Retrain**: Retrain model periodically with fresh data

## 🔗 Related Files

- `train_sgradt_strategy.py` - Training script
- `compare_strategies.py` - Strategy comparison
- `DOCUMENTATION.md` - Full training documentation
- `README.md` - Quick start guide

---

**Version**: 1.0 (Fixed)  
**Compatible with**: SGRADT 5.0 training script  
**Date**: March 2026
