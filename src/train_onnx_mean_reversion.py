import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from indicators import calculate_rsi, calculate_stochastic, calculate_adx
import onnx

# --- CONFIGURATION ---
if len(sys.argv) < 2:
    print("Usage: python train_onnx_mean_reversion.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found")
    sys.exit(1)

output_filename = Path(csv_file).stem + "_meanrev.onnx"
print(f"--- MEAN-REVERSION TRAINING ---")
print(f"Loading rates from: {csv_file}")
print(f"Output ONNX will be: {output_filename}")

df = pd.read_csv(csv_file)
print(f"Records loaded: {len(df)}")

pip_unit = 0.0001
window = 20
adx_period = 14
k_period = 14
d_period = 3
slowing = 3

plus_di, minus_di, dx, adx = calculate_adx(
    df['high'].tolist(), df['low'].tolist(), df['close'].tolist(), period=adx_period)
k_fast, d_fast, k_slow, d_slow = calculate_stochastic(
    df['high'].tolist(), df['low'].tolist(), df['close'].tolist(), k_period=k_period, d_period=d_period, slowing=slowing)

# Feature engineering
df['feat_body'] = (df['close'] - df['open']) / pip_unit
df['feat_range'] = (df['high'] - df['low']) / pip_unit
df['feat_rsi'] = pd.Series(calculate_rsi(df['close'], 14)) / 100.0
df['feat_adx'] = pd.Series(adx)
df['feat_k_slow'] = pd.Series(k_slow)
df['feat_d_slow'] = pd.Series(d_slow)

# Label: mean-reversion entry
adx_thresh = 20
move_pips = 5 * pip_unit
future_window = 5
labels = []
for i in range(window, len(df) - future_window):
    # Mean-reversion filter: ADX < threshold and Stochastic oversold
    if adx[i] is not None and k_slow[i] is not None and d_slow[i] is not None:
        if adx[i] < adx_thresh and k_slow[i] < 20 and k_slow[i] > d_slow[i]:
            entry_price = df['close'].iloc[i]
            future_max = df['close'].iloc[i+1:i+future_window+1].max()
            labels.append(int(future_max > entry_price + move_pips))
        elif adx[i] < adx_thresh and k_slow[i] > 80 and k_slow[i] < d_slow[i]:
            entry_price = df['close'].iloc[i]
            future_min = df['close'].iloc[i+1:i+future_window+1].min()
            labels.append(int(future_min < entry_price - move_pips))
        else:
            labels.append(0)
    else:
        labels.append(0)

df = df.iloc[window:len(df)-future_window].copy()
df['target'] = labels

features = ['feat_body', 'feat_range', 'feat_rsi', 'feat_adx', 'feat_k_slow', 'feat_d_slow']
X = []
y = []
for i in range(window, len(df)):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

print(f"Training samples: {len(X)}")

# Model training
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 8, 12],
    'min_samples_leaf': [1, 5]
}
tscv = TimeSeriesSplit(n_splits=2)
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=5,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1
)
search.fit(X, y)
model = search.best_estimator_
print(f"Best configuration: {search.best_params_}")

initial_type = [('float_input', FloatTensorType([None, len(features)*window]))]
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12, options={type(model): {'zipmap': False}})
onnx.checker.check_model(onx)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Model saved to: {output_filename}")
print(f"Opset version: 12 (MT5 compatible)")
print(f"--- PROCESS COMPLETED ---")
