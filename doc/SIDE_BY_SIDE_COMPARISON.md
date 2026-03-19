# Side-by-Side Comparison: Original vs Fixed EA

## 🔴 Critical Fix: Feature Order

### ❌ ORIGINAL EA (WRONG)
```cpp
// Line 163-180 in original EA
for(int i = 0; i < WINDOW_SIZE; i++)
{
   int idx    = WINDOW_SIZE - 1 - i;
   int offset = i * FEATURES;
   
   input_buffer[offset + 0] = (float)(close[idx] - open[idx]);   // body
   input_buffer[offset + 1] = (float)(high[idx]  - low[idx]);    // range
   input_buffer[offset + 2] = (float)(adx_b[idx]);               // adx      ← WRONG POSITION
   input_buffer[offset + 3] = (float)(pdi_b[idx]);               // pdi      ← WRONG POSITION
   input_buffer[offset + 4] = (float)(mdi_b[idx]);               // mdi      ← WRONG POSITION
   input_buffer[offset + 5] = (float)(stoch_k_b[idx]);           // stoch_k  ← WRONG POSITION
   input_buffer[offset + 6] = (float)(stoch_d_b[idx]);           // stoch_d  ← WRONG POSITION
}
```

### ✅ FIXED EA (CORRECT)
```cpp
// Lines 288-310 in fixed EA
for(int i = 0; i < InpWindowSize; i++)
{
   int idx = InpWindowSize - 1 - i;
   int offset = idx * InpFeaturesPerBar;
   
   // Feature 0: body (close - open)
   input_buffer[offset + 0] = (float)(close[i] - open[i]);
   
   // Feature 1: range (high - low)
   input_buffer[offset + 1] = (float)(high[i] - low[i]);
   
   // Feature 2: stoch_main (%K)                             ← CORRECT POSITION
   input_buffer[offset + 2] = (float)stoch_k_b[i];
   
   // Feature 3: stoch_signal (%D)                           ← CORRECT POSITION
   input_buffer[offset + 3] = (float)stoch_d_b[i];
   
   // Feature 4: ADX                                         ← CORRECT POSITION
   input_buffer[offset + 4] = (float)adx_b[i];
   
   // Feature 5: +DI                                         ← CORRECT POSITION
   input_buffer[offset + 5] = (float)pdi_b[i];
   
   // Feature 6: -DI                                         ← CORRECT POSITION
   input_buffer[offset + 6] = (float)mdi_b[i];
}
```

### 🎯 Training Script Order (Python)
```python
features_list = [
    'feat_body',         # 0
    'feat_range',        # 1
    'feat_stoch_main',   # 2  ← Stochastic BEFORE ADX
    'feat_stoch_signal', # 3  ← Stochastic BEFORE ADX
    'feat_adx',          # 4  ← ADX AFTER Stochastic
    'feat_pdi',          # 5
    'feat_mdi'           # 6
]
```

---

## 🔴 Critical Fix: Default Parameters

### ❌ ORIGINAL EA (WRONG)
```cpp
input group "Stochastic"
input int    InpStochK          = 5;      // ❌ Should be 7
input int    InpStochD          = 3;      // ✓ Correct
input int    InpStochSlowing    = 3;      // ✓ Correct
input double InpStochOversold   = 30.0;   // ❌ Should be 20.0
input double InpStochOverbought = 70.0;   // ❌ Should be 80.0

// ADX is HARDCODED
int adx_h = iADX(_Symbol, _Period, 14);   // ❌ Should be 8
```

### ✅ FIXED EA (CORRECT)
```cpp
input group "=== Indicator Parameters (SGRADT 5.0 Defaults) ==="
input int    InpStochK          = 7;      // ✅ SGRADT 5.0 default
input int    InpStochD          = 3;      // ✅ Correct
input int    InpStochSlowing    = 3;      // ✅ Correct
input double InpStochOversold   = 20.0;   // ✅ SGRADT 5.0 default
input double InpStochOverbought = 80.0;   // ✅ SGRADT 5.0 default
input int    InpADXPeriod       = 8;      // ✅ SGRADT 5.0 default (configurable)
input double InpADXLimit        = 32.0;   // ✅ SGRADT 5.0 default

// ADX is now CONFIGURABLE
g_adx_handle = iADX(_Symbol, _Period, InpADXPeriod);  // ✅ Uses input parameter
```

---

## 🟡 Performance Fix: Indicator Handles

### ❌ ORIGINAL EA (INEFFICIENT)
```cpp
void OnTick()
{
   // ❌ Creates NEW handles on EVERY tick!
   int adx_h   = iADX(_Symbol, _Period, 14);
   int stoch_h = iStochastic(_Symbol, _Period, 
                             InpStochK, InpStochD, InpStochSlowing,
                             MODE_SMA, STO_LOWHIGH);
   
   UpdateIndicatorGlobals(adx_h, stoch_h);
   // ...
}

void RunInference()
{
   // ❌ Creates NEW handles AGAIN!
   int adx_h   = iADX(_Symbol, _Period, 14);
   int stoch_h = iStochastic(_Symbol, _Period,
                             InpStochK, InpStochD, InpStochSlowing,
                             MODE_SMA, STO_LOWHIGH);
   // ...
}
```

**Problem**: Creates indicator handles hundreds of times per minute, wasting resources.

### ✅ FIXED EA (EFFICIENT)
```cpp
// Global handles (created ONCE)
int g_adx_handle   = INVALID_HANDLE;
int g_stoch_handle = INVALID_HANDLE;

int OnInit()
{
   // ✅ Create handles ONCE during initialization
   g_adx_handle = iADX(_Symbol, _Period, InpADXPeriod);
   g_stoch_handle = iStochastic(_Symbol, _Period,
                                InpStochK, InpStochD, InpStochSlowing,
                                MODE_SMA, STO_LOWHIGH);
   // ...
}

void OnDeinit(const int reason)
{
   // ✅ Release handles properly
   if(g_adx_handle != INVALID_HANDLE)
      IndicatorRelease(g_adx_handle);
   if(g_stoch_handle != INVALID_HANDLE)
      IndicatorRelease(g_stoch_handle);
}

void OnTick()
{
   // ✅ Reuse existing handles
   UpdateIndicatorGlobals();  // Uses g_adx_handle, g_stoch_handle
   // ...
}
```

---

## 🟢 Enhancement: Better Error Handling

### ❌ ORIGINAL EA
```cpp
if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, out_l, out_p))
   return;  // ❌ Silent failure, no error message
```

### ✅ FIXED EA
```cpp
if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, out_label, out_probs))
{
   Print("❌ ERROR: ONNX inference failed. Error: ", GetLastError());
   return;  // ✅ Clear error message in log
}
```

---

## 🟢 Enhancement: Better Logging

### ❌ ORIGINAL EA
```cpp
// No inference logging
if(g_prediction > 0 && ...)
{
   // Trade execution without logging
   if(m_trade.Buy(...))
      g_last_traded_bar = current_bar;
}
```

### ✅ FIXED EA
```cpp
// Detailed inference logging
if(g_prediction > 0)
{
   PrintFormat("🔍 Inference #%d: %s (conf: %.2f%%) | ADX: %.1f | Stoch: %.1f/%.1f",
               g_infer_count,
               (g_prediction == 1 ? "BUY" : "SELL"),
               active_conf * 100,
               g_curr_adx,
               g_stoch_k,
               g_stoch_d);
}

// Trade execution with detailed logging
if(g_prediction == 1)
{
   PrintFormat("📈 Opening BUY: Price=%.5f, SL=%.5f, TP=%.5f, Lot=%.2f",
               ask, sl, tp, InpLot);
   
   if(m_trade.Buy(...))
   {
      Print("✅ BUY order executed successfully");
   }
   else
   {
      Print("❌ BUY order failed. Error: ", m_trade.ResultRetcode());
   }
}
```

---

## 📊 Complete Feature Mapping

### Training Script → Fixed EA

| Position | Training Script | Fixed EA | Original EA (❌) |
|----------|----------------|----------|------------------|
| 0 | `feat_body` | `close[i] - open[i]` | `close[idx] - open[idx]` ✓ |
| 1 | `feat_range` | `high[i] - low[i]` | `high[idx] - low[idx]` ✓ |
| 2 | `feat_stoch_main` | `stoch_k_b[i]` | `adx_b[idx]` ❌ |
| 3 | `feat_stoch_signal` | `stoch_d_b[i]` | `pdi_b[idx]` ❌ |
| 4 | `feat_adx` | `adx_b[i]` | `mdi_b[idx]` ❌ |
| 5 | `feat_pdi` | `pdi_b[i]` | `stoch_k_b[idx]` ❌ |
| 6 | `feat_mdi` | `mdi_b[i]` | `stoch_d_b[idx]` ❌ |

**Red flags (❌)**: Positions 2-6 are completely shuffled in original EA!

---

## 🎯 Input Parameters Comparison

| Parameter | Original | Fixed | Training Default |
|-----------|----------|-------|------------------|
| **Model** |
| Model name | `modelo_selectivo_puntos.onnx` | `EUR_USD_H1_SGRADT50_combined.onnx` | (generated) |
| Window size | `25` ❌ | `20` ✅ | `--window 20` |
| Features | `7` ✓ | `7` ✓ | Always 7 |
| **Stochastic** |
| K period | `5` ❌ | `7` ✅ | `--stoch_k 7` |
| D period | `3` ✓ | `3` ✓ | `--stoch_d 3` |
| Slowing | `3` ✓ | `3` ✓ | `--stoch_slowing 3` |
| Oversold | `30.0` ❌ | `20.0` ✅ | `--stoch_oversold 20` |
| Overbought | `70.0` ❌ | `80.0` ✅ | `--stoch_overbought 80` |
| **ADX** |
| Period | `14` (hardcoded) ❌ | `8` ✅ | `--adx_period 8` |
| Limit | `24.0` ❌ | `32.0` ✅ | `--adx_limit 32` |

---

## 🔄 Migration Steps

### From Original EA to Fixed EA

1. **Backup** your current EA settings
2. **Remove** old EA from chart
3. **Install** fixed EA: `EA_SGRADT50_ONNX.mq5`
4. **Copy** new ONNX model to `MQL5/Files/`
5. **Configure** parameters:
   ```
   InpWindowSize = 20      (was 25)
   InpStochK = 7           (was 5)
   InpStochOversold = 20   (was 30)
   InpStochOverbought = 80 (was 70)
   InpADXPeriod = 8        (was hardcoded 14)
   InpADXLimit = 32        (was 24)
   ```
6. **Test** on demo account first

---

## ⚠️ Why This Matters

### Original EA Issues Impact

1. **Wrong Feature Order** → Model receives scrambled data → **Random predictions**
2. **Wrong Parameters** → Indicators don't match training → **Invalid features**
3. **Wrong Window Size** → Array size mismatch → **Potential crashes**

### Example of Data Corruption

**Training expects:**
```
[body, range, stoch_k, stoch_d, adx, pdi, mdi]
[0.02, 0.05,   75.2,    72.1,   35.4, 28.1, 18.7]
```

**Original EA sends:**
```
[body, range,  adx,  pdi,  mdi, stoch_k, stoch_d]
[0.02, 0.05,  35.4, 28.1, 18.7,  75.2,    72.1  ]
```

**Result**: Model thinks ADX=75.2 (extreme high), Stochastic=18.7 (oversold)  
**Reality**: ADX=35.4 (trending), Stochastic=75.2 (near overbought)

**Outcome**: Completely wrong predictions! 🔴

---

## ✅ Verification Checklist

After installing fixed EA, verify in log:

```
✓ ONNX model loaded: EUR_USD_H1_SGRADT50_combined.onnx
✓ Input shape set: [1, 140] (20 bars × 7 features)
✓ Indicators created: ADX(8) + Stochastic(7,3,3)
```

If you see:
- ❌ ADX(14) → Wrong, should be ADX(8)
- ❌ [1, 175] → Wrong window size (25 instead of 20)
- ❌ Stochastic(5,3,3) → Wrong K period

Then check your input parameters!

---

## 📞 Support

If you have issues:

1. Check **EA_FIXES_DOCUMENTATION.md** for detailed troubleshooting
2. Verify **feature order** matches training script
3. Confirm **parameters** match your `.meta.json` file
4. Test with **demo account** first

---

**Remember**: The feature order is CRITICAL. Even one position wrong will make the model useless! 🎯
