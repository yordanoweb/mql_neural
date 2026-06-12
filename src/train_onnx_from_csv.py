import pandas as pd
import numpy as np
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from indicators import calculate_rsi  as rsi
import onnx

def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest and export ONNX from OHLC CSV.")
    parser.add_argument("--input-csv", required=True, help="Input CSV file path")
    parser.add_argument("--output-filename", required=True, help="Output ONNX filename/path")
    parser.add_argument("--window", type=int, default=20, help="Feature window size in bars (default: 20)")
    parser.add_argument("--pip-unit", type=float, default=0.0001, help="Pip unit for body/range normalization (default: 0.0001)")
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI period (default: 14)")
    parser.add_argument("--n-iter", type=int, default=5, help="RandomizedSearchCV iterations (default: 5)")
    parser.add_argument("--n-splits", type=int, default=2, help="TimeSeriesSplit folds (default: 2)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for search (default: -1)")
    return parser.parse_args()

args = parse_args()
csv_file = args.input_csv
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Input CSV not found: {csv_file}")

if args.window <= 0:
    raise ValueError("--window must be greater than 0")
if args.pip_unit <= 0:
    raise ValueError("--pip-unit must be greater than 0")
if args.rsi_period <= 0:
    raise ValueError("--rsi-period must be greater than 0")
if args.n_iter <= 0:
    raise ValueError("--n-iter must be greater than 0")
if args.n_splits < 2:
    raise ValueError("--n-splits must be at least 2")

output_filename = args.output_filename
print(f"--- FAST TRAINING ---")
print(f"Loading rates from: {csv_file}")
print(f"Output ONNX will be: {output_filename}")

def calculate_rsi(series, period=14):
    """Calculate RSI using indicators module, compatible with pandas Series"""
    rsi_list = rsi(series.values.tolist(), period)
    return pd.Series(rsi_list, index=series.index)

# 1. LOADING DATA FROM CSV
df = pd.read_csv(csv_file)
print(f"Records loaded: {len(df)}")

pip_unit = args.pip_unit

df['feat_body'] = (df['close'] - df['open']) / pip_unit
df['feat_range'] = (df['high'] - df['low']) / pip_unit
df['feat_rsi'] = calculate_rsi(df['close'], args.rsi_period) / 100.0
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

# 2. PREPARE WINDOWS
window = args.window
X, y = [], []
features = ['feat_body', 'feat_range', 'feat_rsi']

for i in range(window, len(df) - 1):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

# 3. FAST OPTIMIZATION
print("Searching for efficient configuration (Random Search)...")
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 8, 12],
    'min_samples_leaf': [1, 5]
}

tscv = TimeSeriesSplit(n_splits=args.n_splits)

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=args.n_iter,
    cv=tscv,
    scoring='accuracy',
    n_jobs=args.n_jobs
)

search.fit(X, y)
model = search.best_estimator_
print(f"Best configuration: {search.best_params_}")

# 4. EXPORT TO ONNX
initial_type = [('float_input', FloatTensorType([None, window * len(features)]))]
# Use target_opset=12 for MetaTrader 5 compatibility (MT5 supports opset 1-21, but lower is safer)
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12, options={type(model): {'zipmap': False}})

# Validate ONNX model before saving
onnx.checker.check_model(onx)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Model saved to: {output_filename}")
print(f"Opset version: 12 (MT5 compatible)")
print(f"--- PROCESS COMPLETED ---")
