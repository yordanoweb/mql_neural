# Technical Documentation: ONNX 3-Feature Strategy (Body, Range, RSI)

This document provides a detailed technical overview of the machine learning training pipeline (`train_onnx_3_feat.py`) and its corresponding MetaTrader 5 Expert Advisor implementation (`SimpleONNX_3_Feat.mq5`).

## Table of Contents
1. [Introduction](#introduction)
2. [Feature Engineering](#feature-engineering)
3. [Training Pipeline (`train_onnx_3_feat.py`)](#training-pipeline)
4. [Expert Advisor (`SimpleONNX_3_Feat.mq5`)](#expert-advisor)
5. [Model Compatibility & Export](#model-compatibility)
6. [Usage Guide](#usage-guide)

---

## 1. Introduction <a name="introduction"></a>
The **ONNX 3-Feature Strategy** is a supervised learning system designed to predict short-term price movements based on bar geometry and momentum. It uses a **Random Forest Classifier** to analyze historical data and identifies patterns that lead to a specific profit target within a defined window.

The final model is exported to **ONNX (Open Neural Network Exchange)** format, allowing it to run natively within MetaTrader 5 without requiring an external Python environment during live trading or backtesting.

---

## 2. Feature Engineering <a name="feature-engineering"></a>
Both the Python trainer and the MQL5 EA share a synchronized feature engineering logic to ensure consistency between training and inference.

Three primary features are calculated for each bar in a sliding window of size `N` (default: 20):

| Feature Name | Logic | Description |
| :--- | :--- | :--- |
| **feat_body** | `(Close - Open) / pip_unit` | Measures the magnitude and direction of the candle body in pips/points. |
| **feat_range** | `(High - Low) / pip_unit` | Measures total candle volatility (High to Low) in pips/points. |
| **feat_rsi** | `RSI(Close, Period) / 100.0` | Normalized Relative Strength Index (0.0 to 1.0) for momentum. |

### Normalization
- **Pip Unit Logic**: The system automatically detects the pip unit (0.0001 for 5-digit brokers, 0.01 for 3-digit brokers) to keep feature values consistent across different symbols/brokers.
- **RSI**: Scaled between 0 and 1 to facilitate faster convergence and stable predictions.

---

## 3. Training Pipeline (`train_onnx_3_feat.py`) <a name="training-pipeline"></a>
The Python script is responsible for data preprocessing, target labeling, model optimization, and ONNX conversion.

### 3.1 Target Labeling (The "Future" Logic)
The script creates binary labels based on a look-ahead window.
- **Label 1 (Positive)**: If the high price of any of the next `F` bars (default: 5) is at least `P` points (default: 10) above the current close.
- **Label 0 (Negative)**: If the target is not reached within the look-ahead window.

### 3.2 Model Architecture
- **Classifier**: `RandomForestClassifier` from `scikit-learn`.
- **Optimization**: Uses `RandomizedSearchCV` with a `TimeSeriesSplit` cross-validation strategy to prevent look-ahead bias during training.
- **Balanced Weights**: Uses `class_weight='balanced'` to handle datasets where one class (e.g., non-profitable moves) might be more frequent.

### 3.3 CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input_csv` | (Required) | Path to historical rates exported from MT5. |
| `--rsi_period` | 14 | Period for the RSI calculation. |
| `--window` | 20 | Number of bars to include in the input vector. |
| `--future` | 5 | Number of bars to look ahead for target labeling. |
| `--min_profit_points` | 10.0 | Minimum profit (in pips/points) for a positive label. |
| `--n_iter` | 5 | Iterations for hyperparameter search. |

---

## 4. Expert Advisor (`SimpleONNX_3_Feat.mq5`) <a name="expert-advisor"></a>
The MQL5 Expert Advisor implements the inference engine and trade execution logic.

### 4.1 Key Components
- **ONNX Runtime**: Loads the `.onnx` file using `OnnxCreate`.
- **Input Preparation**: Constructs the input vector of size $20 \times 3 = 60$ floats representing the last 20 candles.
- **Probability Extraction**: Unlike simple classifiers, this EA retrieves class probabilities (`output_probs`) to filter trades by confidence.

### 4.2 Trade Logic
Trades are opened at the start of a new bar if:
1. **Time Filter**: The current hour is within the `InpStartHour` and `InpEndHour` range.
2. **Confidence**: The model's prediction confidence is greater than or equal to `InpMinConf` (default: 0.55).
3. **Capacity**: There is no open position for the current symbol.

### 4.3 Risk Management
- **SL/TP**: Calculated dynamically using the **Average True Range (ATR)**.
  - **Stop Loss**: $ATR(6) \times Multiplier$.
  - **Take Profit**: $Stop Loss \times 1.5$ (Fixed R:R ratio).

### 4.4 Backtesting
Best parameters were:
- InpMinConf: 0.77
- InpStartHour: 2
- InpEndHour: 18
- InpATR: 8
- InpMultiplier: 1.5

---

## 5. Model Compatibility & Export <a name="model-compatibility"></a>
The exporter in `train_onnx_3_feat.py` uses specific settings required for MetaTrader 5:
- **Opset Version**: 12 (Ensures compatibility with MT5's ONNX runtime).
- **ZipMap Disabled**: MetaTrader expects raw probability arrays, not the dictionary-like ZipMap output common in scikit-learn.
- **Input Type**: `FloatTensorType([None, 60])`.

---

## 6. Usage Guide <a name="usage-guide"></a>

### Training
```bash
python train_onnx_3_feat.py --input_csv data/eurusd_m15.csv --output_dir models --min_profit_points 15
```

### Deployment
1. Copy the generated `.onnx` file to the MT5 `/MQL5/Files/` directory.
2. Launch the `SimpleONNX_3_Feat` EA.
3. Set the `InpModelFile` parameter to the name of your ONNX file.
4. Adjust `InpMinConf` to control the trade frequency and quality.

---
*Documentation generated for mql_neural project.*
