# SGRADT 7.0 - Quick Reference Guide

## 🎯 What is SGRADT 7.0?

**Pure Machine Learning Strategy with ADX + Stochastic**
- 5 essential features
- Random Forest classifier
- ONNX model for MT5
- No EMA, no manual gates, no complexity
- Maximum simplicity = maximum generalization

---

## 📊 Features (5 Total)

| # | Feature | Calculation | Purpose |
|---|---------|-------------|---------|
| 1 | **body** | close - open | Candle structure |
| 2 | **range** | high - low | Price volatility |
| 3 | **stoch_k** | Stochastic %K | Momentum |
| 4 | **stoch_d** | Stochastic %D | Momentum signal |
| 5 | **adx** | ADX(8) | Trend strength |

---

## 🔧 Key Parameters

### Training Script
```bash
python train_sgradt70_strategy.py \
  --csv data/EURUSD_M5.csv \
  --window 20                      # Lookback window
  --min_profit_points 20.0         # Target profit in points
  --future 50                      # Bars to search for target
  --stoch_k 7                      # Stochastic K period
  --stoch_d 3                      # Stochastic D period
  --adx_period 8                   # ADX period
  --adx_limit 25.0                 # Minimum ADX for entry
  --n_iter 7                       # RandomSearch iterations
  --output ./onnx
```

### MT5 EA
```
InpFeaturesPerBar = 5          ← CRITICAL: 5 features
InpWindow = 20                 ← Must match training
InpMagic = 7070               ← Unique magic number
InpADXPeriod = 8              ← Must match training
InpStochK = 7                 ← Must match training
InpStochD = 3                 ← Must match training
InpADXLimit = 25.0            ← Minimum ADX threshold
```

---

## 📈 Model Specifications

| Parameter | Value |
|-----------|-------|
| **Algorithm** | Random Forest Classifier |
| **Input Dimension** | [1, 100] (5 features × 20 window) |
| **Output Classes** | 3 (HOLD=0, BUY=1, SELL=2) |
| **Cross-Validation** | TimeSeriesSplit(n_splits=3) |
| **Scoring Metric** | Balanced Accuracy |

---

## 🎓 Entry Logic

**BUY Condition:**
```python
adx_strong = ADX > 25
stoch_oversold_cross = (stoch_k[-2] < stoch_d[-2]) and (stoch_k[-1] > stoch_d[-1]) and (stoch_k[-1] <= 20)
strong_momentum = stoch_k[-1] > stoch_k[-2] + 7

buy_signal = adx_strong and (stoch_oversold_cross or strong_momentum)
```

**SELL Condition:**
```python
adx_strong = ADX > 25
stoch_overbought_cross = (stoch_k[-2] > stoch_d[-2]) and (stoch_k[-1] < stoch_d[-1]) and (stoch_k[-1] >= 80)
weak_momentum = stoch_k[-1] < stoch_k[-2] - 7

sell_signal = adx_strong and (stoch_overbought_cross or weak_momentum)
```

---

## 📊 Exit Logic

**Fixed Bar Horizon Labeling:**
```python
# Entry at open of next bar
entry_price = open[i+1]

# Check if profit_target reached in next 50 bars
for j in range(i+1, i+51):
    if high[j] - entry_price >= 20 points:
        label = 1  # BUY success
        break
    if entry_price - low[j] >= 20 points:
        label = 2  # SELL success
        break
```

---

## 📋 Files Generated

After training:
```
onnx/
├── EURUSD_M5_SGRADT70.onnx          ← Model (ready for MT5)
└── (metadata if needed)
```

**File Size:** Typically 2-5 MB

**Copy to MT5:** 
```
C:\Users\[Username]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\Files\
EURUSD_M5_SGRADT70.onnx
```

---

## ✅ Validation Metrics

### Balanced Accuracy Interpretation
| Score | Quality | Action |
|-------|---------|--------|
| < 0.55 | Poor | Adjust parameters |
| 0.55-0.60 | Weak | Consider alternatives |
| 0.60-0.70 | **Acceptable** | Good for trading |
| 0.70-0.80 | Good | Excellent model |
| > 0.80 | Excellent | **Watch for overfitting** |

### Backtest Metrics
- **Profit Factor** > 1.5
- **Sharpe Ratio** > 1.0
- **Max Drawdown** < 20%
- **Win Rate** > 50%
- **Recovery Factor** > 2.0

---

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Input buffer size incorrect" | Check `InpFeaturesPerBar = 5` |
| "Model expects 100 inputs" | Window mismatch: verify `--window` = `InpWindow` |
| No signals generated | Reduce `--min_profit_points` or increase `--future` |
| Excellent backtest, poor live | Overfitting: use more training data |
| Model file not found | Copy .onnx to MQL5/Files/ folder |

---

## 📈 Recommended Parameters by Instrument

| Instrument | min_profit | future | adx_limit | Notes |
|------------|-----------|--------|-----------|-------|
| NAS100 | 40-60 | 20-30 | 25 | Volatile |
| EURUSD | 15-25 | 40-50 | 25 | Mixed |
| GBPUSD | 20-30 | 40-60 | 25 | Very volatile |
| FTSE100 | 25-40 | 25-40 | 25 | Medium volatile |
| BTC/ETH | 60-100 | 15-25 | 20 | Highly volatile |

**Start with:** `--min_profit_points 20 --future 50 --adx_limit 25`

---

## 🔄 Workflow

### 1. Data Preparation
```bash
# CSV format required:
# open,high,low,close,tick_volume
```

### 2. Training
```bash
python train_sgradt70_strategy.py \
  --csv data.csv \
  --output ./onnx
```

### 3. Validation
```
Check: Balanced Accuracy >= 0.60
```

### 4. MT5 Setup
```
1. Copy .onnx file
2. Configure EA parameters (5 features, window, magic)
3. Compile without errors
4. Backtest minimum 3 months
```

### 5. Live Deployment
```
1. Demo test 2-4 weeks
2. Small lot size (0.01)
3. Monitor daily
4. Gradually increase size
```

---

## 🎯 Performance Expectations

**Balanced Accuracy:** 0.60-0.70 (typical)
**Win Rate:** 50-55% (realistic)
**Profit Factor:** 1.5-2.5 (good)
**Sharpe Ratio:** 0.8-1.5 (acceptable)
**Max Drawdown:** 10-20% (manageable)

---

## 📚 Documentation Structure

| Section | Purpose |
|---------|---------|
| Overview | What is SGRADT 7.0 |
| Features | The 5 inputs |
| Architecture | ML model design |
| Training | How to train |
| MT5 Configuration | How to setup EA |
| Labeling | How labels are created |
| Troubleshooting | Common problems |
| FAQ | Q&A |
| Migration | Upgrading from v2/v3 |

---

## 🔗 Key Differences from Previous Versions

| Aspect | v2 | v3 | v7.0 |
|--------|----|----|------|
| Features | 7 | 6 | **5** |
| EMA | Yes | No | **No** |
| PDI/MDI | Yes | Yes | **No** |
| Volume Gate | Yes | Yes | **No** |
| Indicators | 4+ | 3+ | **2** |
| Complexity | High | Medium | **Low** |
| Generalization | Medium | Good | **Excellent** |

---

## 💡 Philosophy

> **"Everything should be made as simple as possible, but not simpler." - Albert Einstein**

SGRADT 7.0 embodies this philosophy:
- ✅ Minimum viable features
- ✅ Maximum generalization
- ✅ Purest indicators (ADX + Stochastic)
- ✅ Direct learning (no filters)
- ✅ Fast training & inference

---

## 📞 Support

**Before Asking:**
1. Read the full README_SGRADT70.md
2. Check the FAQ section
3. Review Troubleshooting
4. Check parameter recommendations

**Common Checks:**
- Is `InpFeaturesPerBar = 5`?
- Does `InpWindow` match training `--window`?
- Is the .onnx file in MQL5/Files/?
- Are indicator parameters matching?

---

## ⚖️ Disclaimer

- **Educational purposes only**
- **No guarantee of profitability**
- **Trading involves significant risk**
- **Always test thoroughly before live trading**
- **Risk management is essential**

---

**Version:** 7.0.0  
**Created:** March 25, 2025  
**Last Updated:** March 25, 2025

*For detailed documentation, see README_SGRADT70.md*
