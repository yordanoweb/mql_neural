import pandas as pd
import numpy as np
import argparse
import os
import re
import json
from collections import Counter
from datetime import datetime, timezone
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
    parser.add_argument("--symbol", default="auto", help="Trading symbol for metadata (default: auto/infer)")
    parser.add_argument("--timeframe", default="auto", help="Timeframe for metadata (default: auto/infer)")
    return parser.parse_args()

def infer_symbol_timeframe_from_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    tf_regex = re.compile(r"^(M1|M2|M3|M4|M5|M6|M10|M12|M15|M20|M30|H1|H2|H3|H4|H6|H8|H12|D1|W1|MN1)$")
    inferred_symbol = parts[0] if len(parts) > 0 else "unknown"
    inferred_timeframe = "unknown"
    for part in parts[1:]:
        if tf_regex.match(part):
            inferred_timeframe = part
            break
    return inferred_symbol, inferred_timeframe

def set_onnx_metadata(model, metadata):
    for key, value in metadata.items():
        value_str = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        updated = False
        for prop in model.metadata_props:
            if prop.key == key:
                prop.value = value_str
                updated = True
                break
        if not updated:
            prop = model.metadata_props.add()
            prop.key = key
            prop.value = value_str

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
inferred_symbol, inferred_timeframe = infer_symbol_timeframe_from_filename(csv_file)
resolved_symbol = inferred_symbol if args.symbol == "auto" else args.symbol
resolved_timeframe = inferred_timeframe if args.timeframe == "auto" else args.timeframe
print(f"--- FAST TRAINING ---")
print(f"Loading rates from: {csv_file}")
print(f"Output ONNX will be: {output_filename}")
print(f"Symbol (metadata): {resolved_symbol}")
print(f"Timeframe (metadata): {resolved_timeframe}")

def calculate_rsi(series, period=14):
    """Calculate RSI using indicators module, compatible with pandas Series"""
    rsi_list = rsi(series.values.tolist(), period)
    return pd.Series(rsi_list, index=series.index)

# 1. LOADING DATA FROM CSV
df = pd.read_csv(csv_file)
records_loaded = len(df)
print(f"Records loaded: {records_loaded}")

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
class_counts = Counter(y.tolist())
print(f"Target class distribution: {dict(class_counts)}")
if len(class_counts) < 2:
    raise ValueError(f"Training target has only one class: {dict(class_counts)}")

# 3. FAST OPTIMIZATION
print("Searching for efficient configuration (Random Search)...")
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 8, 12],
    'min_samples_leaf': [1, 5],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

tscv = TimeSeriesSplit(n_splits=args.n_splits)

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=args.n_iter,
    cv=tscv,
    scoring='balanced_accuracy',
    n_jobs=args.n_jobs
)

search.fit(X, y)
model = search.best_estimator_
print(f"Best configuration: {search.best_params_}")
y_pred = model.predict(X)
pred_counts = Counter(y_pred.tolist())
print(f"Train prediction distribution: {dict(pred_counts)}")
if len(pred_counts) < 2:
    print("WARNING: Model predicts only one class on training windows.")

# 4. EXPORT TO ONNX
initial_type = [('float_input', FloatTensorType([None, window * len(features)]))]
# Use target_opset=12 for MetaTrader 5 compatibility (MT5 supports opset 1-21, but lower is safer)
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12, options={type(model): {'zipmap': False}})

# Validate ONNX model before saving
onnx.checker.check_model(onx)

metadata = {
    "training.created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "training.input_csv_path": os.path.abspath(csv_file),
    "training.input_csv_name": os.path.basename(csv_file),
    "training.output_filename": output_filename,
    "training.symbol": resolved_symbol,
    "training.timeframe": resolved_timeframe,
    "training.window": args.window,
    "training.pip_unit": args.pip_unit,
    "training.rsi_period": args.rsi_period,
    "training.features": features,
    "training.feature_count": len(features),
    "training.input_size": window * len(features),
    "training.n_iter": args.n_iter,
    "training.n_splits": args.n_splits,
    "training.n_jobs": args.n_jobs,
    "training.records_loaded": int(records_loaded),
    "training.records_after_dropna": int(len(df)),
    "training.samples_used": int(len(X)),
    "training.target": "close[t+1] > close[t]",
    "training.target_class_distribution": dict(class_counts),
    "training.model_type": "RandomForestClassifier",
    "training.random_state": 42,
    "training.param_dist": param_dist,
    "training.best_params": search.best_params_,
    "training.best_cv_score": float(search.best_score_),
    "training.scoring": "balanced_accuracy",
    "training.train_prediction_distribution": dict(pred_counts),
    "training.cli_args": vars(args)
}
set_onnx_metadata(onx, metadata)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Model saved to: {output_filename}")
print(f"Opset version: 12 (MT5 compatible)")
print(f"--- PROCESS COMPLETED ---")
