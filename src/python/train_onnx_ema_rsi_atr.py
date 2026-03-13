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

# Importar indicadores desde la librería local
from indicators import calculate_rsi, calculate_ema, calculate_atr

# --- CONFIGURATION ---
if len(sys.argv) < 3:
    print("Usage: python train_onnx_ema_rsi_atr.py <csv_file> <onnx_output_dir>")
    sys.exit(1)

csv_file = sys.argv[1]
onnx_output_dir = sys.argv[2]

if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found")
    sys.exit(1)

if not os.path.exists(onnx_output_dir):
    os.makedirs(onnx_output_dir)

output_filename = Path(onnx_output_dir) / (Path(csv_file).stem + "_ema_rsi_atr.onnx")
print("--- EMA+RSI+ATR TRAINING ---")
print(f"Loading rates from: {csv_file}")
print(f"Output ONNX will be: {output_filename}")

df = pd.read_csv(csv_file)
print(f"Records loaded: {len(df)}")

pip_unit = 0.0001
window = 20

# Feature engineering usando la librería de indicadores
df['feat_body'] = (df['close'] - df['open']) / pip_unit
df['feat_range'] = (df['high'] - df['low']) / pip_unit
df['feat_ema'] = pd.Series(calculate_ema(df['close'].tolist(), 9))
df['feat_rsi'] = pd.Series(calculate_rsi(df['close'].tolist(), 7)) / 100.0
df['feat_atr'] = pd.Series(calculate_atr(
    df['high'].tolist(), df['low'].tolist(), df['close'].tolist(), 14, method='ema'
))

# Label: entry based on EMA + RSI + ATR
future_window = 5
labels = []
for i in range(window, len(df) - future_window):
    entry_price = df['close'].iloc[i]
    future_max = df['close'].iloc[i+1:i+future_window+1].max()
    # Filtro: precio > EMA y RSI < 0.7
    if df['close'].iloc[i] > df['feat_ema'].iloc[i] and df['feat_rsi'].iloc[i] < 0.7:
        labels.append(int(future_max > entry_price + df['feat_atr'].iloc[i]))
    else:
        labels.append(0)

df = df.iloc[window:len(df)-future_window].copy()
df['target'] = labels

features = ['feat_body', 'feat_range', 'feat_ema', 'feat_rsi', 'feat_atr']
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

