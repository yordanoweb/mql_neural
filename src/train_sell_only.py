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
from indicators import calculate_rsi as rsi
import onnx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a sell-only RandomForest classifier and export ONNX from OHLC CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV file path")
    parser.add_argument("--output-filename", required=True, help="Output ONNX filename/path")
    parser.add_argument("--window", type=int, default=20, help="Feature window size in bars")
    parser.add_argument("--forward", type=int, default=10, help="Forward bars used for sell label")
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI period")
    parser.add_argument("--n-iter", type=int, default=5, help="RandomizedSearchCV iterations")
    parser.add_argument("--n-splits", type=int, default=2, help="TimeSeriesSplit folds")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for search")
    parser.add_argument("--symbol", default="auto", help="Trading symbol for metadata")
    parser.add_argument("--timeframe", default="auto", help="Timeframe for metadata")
    parser.add_argument("--date-col", default="date", help="Column name for date")
    parser.add_argument("--time-col", default="time", help="Column name for time")
    parser.add_argument("--open-col", default="open", help="Column name for open price")
    parser.add_argument("--high-col", default="high", help="Column name for high price")
    parser.add_argument("--low-col", default="low", help="Column name for low price")
    parser.add_argument("--close-col", default="close", help="Column name for close price")
    parser.add_argument("--volume-col", default="tick_volume", help="Column name for volume")
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


def calculate_rsi(series, period=14):
    """Calculate RSI using indicators module, compatible with pandas Series"""
    rsi_list = rsi(series.values.tolist(), period)
    return pd.Series(rsi_list, index=series.index)


def resolve_column(df, preferred, fallback):
    """Return preferred column if present, otherwise fallback if present, else None."""
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    return None


args = parse_args()
csv_file = args.input_csv
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Input CSV not found: {csv_file}")

if args.window <= 0:
    raise ValueError("--window must be greater than 0")
if args.forward <= 0:
    raise ValueError("--forward must be greater than 0")
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

print("--- SELL-ONLY FAST TRAINING ---")
print(f"Loading rates from: {csv_file}")
print(f"Output ONNX will be: {output_filename}")
print(f"Symbol (metadata): {resolved_symbol}")
print(f"Timeframe (metadata): {resolved_timeframe}")

# 1. LOADING DATA FROM CSV
df = pd.read_csv(csv_file)
records_loaded = len(df)
print(f"Records loaded: {records_loaded}")

# Resolve column names
open_col = resolve_column(df, args.open_col, "open")
high_col = resolve_column(df, args.high_col, "high")
low_col = resolve_column(df, args.low_col, "low")
close_col = resolve_column(df, args.close_col, "close")

required_cols = [open_col, high_col, low_col, close_col]
missing = [c for c in required_cols if c is None or c not in df.columns]
if missing:
    raise ValueError(f"Required OHLC columns missing or not mapped: {missing}. Available columns: {list(df.columns)}")

# 2. FEATURE ENGINEERING (no pip-unit normalization)
df['feat_body'] = df[close_col] - df[open_col]
df['feat_range'] = df[high_col] - df[low_col]
df['feat_rsi'] = calculate_rsi(df[close_col], args.rsi_period) / 100.0

# Target: sell if close[t + forward] < close[t]
forward = args.forward
df['target'] = (df[close_col].shift(-forward) < df[close_col]).astype(int)
df.dropna(inplace=True)

# 3. PREPARE WINDOWS
window = args.window
X, y = [], []
features = ['feat_body', 'feat_range', 'feat_rsi']

for i in range(window, len(df) - forward):
    window_data = df[features].iloc[i - window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)
class_counts = Counter(y.tolist())
print(f"Target class distribution: {dict(class_counts)}")
if len(class_counts) < 2:
    raise ValueError(f"Training target has only one class: {dict(class_counts)}")

# 4. FAST OPTIMIZATION
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

# 5. EXPORT TO ONNX
initial_type = [('float_input', FloatTensorType([None, window * len(features)]))]
# Use target_opset=12 for MetaTrader 5 compatibility (MT5 supports opset 1-21, but lower is safer)
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12, options={type(model): {'zipmap': False}})

# Validate ONNX model before saving
onnx.checker.check_model(onx)

# Verify probabilities output shape
prob_output = next((o for o in onx.graph.output if o.name == 'probabilities'), None)
if prob_output is None:
    raise RuntimeError("ONNX export missing 'probabilities' output")
prob_shape = [d.dim_value if d.dim_value else d.dim_param for d in prob_output.type.tensor_type.shape.dim]
print(f"ONNX probabilities output shape: {prob_shape}")
if len(prob_shape) != 2 or prob_shape[1] != 2:
    raise RuntimeError(f"Expected probabilities shape [*, 2], got {prob_shape}")

metadata = {
    "training.created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "training.input_csv_path": os.path.abspath(csv_file),
    "training.input_csv_name": os.path.basename(csv_file),
    "training.output_filename": output_filename,
    "training.symbol": resolved_symbol,
    "training.timeframe": resolved_timeframe,
    "training.window": args.window,
    "training.forward": args.forward,
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
    "training.target": f"close[t+{forward}] < close[t]",
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
print("--- PROCESS COMPLETED ---")
