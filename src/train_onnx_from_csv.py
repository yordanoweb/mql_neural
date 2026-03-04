import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from indicators import calculate_rsi  as rsi
import onnx

# --- CONFIGURATION ---
if len(sys.argv) < 2:
    print("Usage: python train_onnx_from_csv.py <csv_file>")
    print("Example: python train_onnx_from_csv.py eurusd_m15_2024.csv")
    sys.exit(1)

csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found")
    sys.exit(1)

# Generate output filename: same basename as CSV but with .onnx extension
output_filename = Path(csv_file).stem + ".onnx"
print(f"--- QUICK TRAINING ---")
print(f"Loading rates from: {csv_file}")
print(f"Output ONNX will be: {output_filename}")

def calculate_rsi(series, period=14):
    """Calculate RSI using indicators module, compatible with pandas Series"""
    rsi_list = rsi(series.values.tolist(), period)
    return pd.Series(rsi_list, index=series.index)

# 1. LOAD DATA FROM CSV
df = pd.read_csv(csv_file)
print(f"Records loaded: {len(df)}")

# Infer pip unit from data (optional, or set based on symbol detection if available)
# If symbol info is not available, we'll use a reasonable default
pip_unit = 0.0001  # Default for most pairs; could be refined if symbol is known

df['feat_body'] = (df['close'] - df['open']) / pip_unit
df['feat_range'] = (df['high'] - df['low']) / pip_unit
df['feat_rsi'] = calculate_rsi(df['close'], 14) / 100.0
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

# 2. PREPARE WINDOWS (60 inputs)
window = 20
X, y = [], []
features = ['feat_body', 'feat_range', 'feat_rsi']

for i in range(window, len(df) - 1):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

# 3. QUICK OPTIMIZATION (Only 10 iterations)
print("Searching for efficient configuration (Random Search)...")
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 8, 12],
    'min_samples_leaf': [1, 5]
}

# TimeSeriesSplit with 2 folds for speed
tscv = TimeSeriesSplit(n_splits=2)

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=5, # Only tests 5 random combinations (very fast)
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1
)

search.fit(X, y)
model = search.best_estimator_
print(f"Best configuration: {search.best_params_}")

# 4. EXPORT WITH NAME BASED ON CSV
initial_type = [('float_input', FloatTensorType([None, 60]))]
# Use target_opset=12 for MetaTrader 5 compatibility (MT5 supports opset 1-21, but lower is safer)
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12, options={type(model): {'zipmap': False}})

# Validate ONNX model before saving
onnx.checker.check_model(onx)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Model saved to: {output_filename}")
print(f"Opset version: 12 (MT5 compatible)")
print(f"--- PROCESS COMPLETED ---")
