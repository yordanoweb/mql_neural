import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import argparse

parser = argparse.ArgumentParser(description="Train ONNX model (body, range, effort/volume)")
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--window", type=int, default=20)
parser.add_argument("--future", type=int, default=5)
parser.add_argument("--n_iter", type=int, default=5)
parser.add_argument("--min_profit_points", type=float, default=10.0)
parser.add_argument("--pip_unit", type=float, default=0.01)
parser.add_argument("--vol_window", type=int, default=20)

args = parser.parse_args()

df = pd.read_csv(args.input_csv)

# --- FEATURES ---
df['feat_body'] = (df['close'] - df['open']) / args.pip_unit
df['feat_range'] = (df['high'] - df['low']) / args.pip_unit

# --- Volume Effort/Result (replaces RSI) ---
df['vol_mean'] = df['tick_volume'].rolling(args.vol_window).mean()
df['vol_mean'] = df['vol_mean'].replace(0, np.nan)

rel_vol = df['tick_volume'] / df['vol_mean']

range_ = (df['high'] - df['low']).replace(0, np.nan)
efficiency = (df['close'] - df['open']).abs() / range_

df['feat_vol'] = (rel_vol * efficiency).clip(0, 5)

# --- TARGET ---
labels = np.zeros(len(df))
for i in range(len(df) - args.future):
    entry = df['close'].iloc[i]
    future_high = df['high'].iloc[i+1:i+args.future+1].max()
    profit = (future_high - entry) / args.pip_unit
    if profit >= args.min_profit_points:
        labels[i] = 1

df['target'] = labels
df.dropna(inplace=True)

features = ['feat_body', 'feat_range', 'feat_vol']

X, y = [], []
for i in range(args.window, len(df) - args.future):
    X.append(df[features].iloc[i-args.window:i].values.flatten())
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

param_dist = {
    'n_estimators': [100,150,200],
    'max_depth': [5,8,12],
    'min_samples_leaf': [1,5]
}

tscv = TimeSeriesSplit(n_splits=2)

search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_distributions=param_dist,
    n_iter=args.n_iter,
    cv=tscv,
    scoring='balanced_accuracy',
    n_jobs=-1
)

search.fit(X, y)
model = search.best_estimator_

output_path = os.path.join(args.output_dir, Path(args.input_csv).stem + "_3_feat_vol.onnx")

onx = convert_sklearn(
    model,
    initial_types=[('float_input', FloatTensorType([None, args.window * 3]))],
    target_opset=12,
    options={type(model): {'zipmap': False}}
)

onnx.checker.check_model(onx)

with open(output_path, "wb") as f:
    f.write(onx.SerializeToString())

print("Model saved:", output_path)
