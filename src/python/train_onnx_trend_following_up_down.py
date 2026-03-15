import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from indicators import calculate_rsi, calculate_adx
import onnx

# --- CONFIGURATION ---
if len(sys.argv) < 3:
    print("Usage: python train_onnx_trend_following.py <csv_file> <onnx_output_dir>")
    sys.exit(1)

csv_file = sys.argv[1]
onnx_output_dir = sys.argv[2]

if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found")
    sys.exit(1)

if not os.path.exists(onnx_output_dir):
    os.makedirs(onnx_output_dir)

output_filename = Path(onnx_output_dir) / (Path(csv_file).stem + "_trend.onnx")
print(f"--- TREND-FOLLOWING TRAINING (BUY + SELL) ---")
print(f"Loading rates from: {csv_file}")
print(f"Output ONNX will be: {output_filename}")

df = pd.read_csv(csv_file)
print(f"Records loaded: {len(df)}")

pip_unit = 0.0001

# Feature engineering
window = 20
adx_period = 14
plus_di, minus_di, dx, adx = calculate_adx(
    df['high'].tolist(), df['low'].tolist(), df['close'].tolist(), period=adx_period)

df['feat_body']     = (df['close'] - df['open']) / pip_unit
df['feat_range']    = (df['high'] - df['low']) / pip_unit
df['feat_rsi']      = pd.Series(calculate_rsi(df['close'], 14)) / 100.0
df['feat_adx']      = pd.Series(adx)
df['feat_plus_di']  = pd.Series(plus_di)
df['feat_minus_di'] = pd.Series(minus_di)

# --- LABELS: multiclass trend-following ---
# 0 = no signal
# 1 = BUY  (trend alcista: ADX > thresh, +DI > -DI, precio sube >= move_pips)
# 2 = SELL (trend bajista: ADX > thresh, -DI > +DI, precio baja >= move_pips)
trend_thresh  = 25
move_pips     = 5 * pip_unit
future_window = 5

labels = []
for i in range(window, len(df) - future_window):
    adx_val   = adx[i]
    pdi_val   = plus_di[i]
    mdi_val   = minus_di[i]

    if adx_val is None or pdi_val is None or mdi_val is None:
        labels.append(0)
        continue

    entry_price = df['close'].iloc[i]
    future_prices = df['close'].iloc[i + 1 : i + future_window + 1]

    if adx_val > trend_thresh:
        if pdi_val > mdi_val:
            # Tendencia alcista → buscar señal BUY
            future_max = future_prices.max()
            labels.append(1 if future_max > entry_price + move_pips else 0)
        elif mdi_val > pdi_val:
            # Tendencia bajista → buscar señal SELL
            future_min = future_prices.min()
            labels.append(2 if future_min < entry_price - move_pips else 0)
        else:
            labels.append(0)
    else:
        labels.append(0)

df = df.iloc[window : len(df) - future_window].copy()
df['target'] = labels

# Distribución de clases para diagnóstico
counts = pd.Series(labels).value_counts().sort_index()
print(f"Label distribution → 0 (no signal): {counts.get(0,0)}, "
      f"1 (buy): {counts.get(1,0)}, 2 (sell): {counts.get(2,0)}")

# --- FEATURE MATRIX ---
features = ['feat_body', 'feat_range', 'feat_rsi', 'feat_adx', 'feat_plus_di', 'feat_minus_di']
X = []
y = []
for i in range(window, len(df)):
    window_data = df[features].iloc[i - window : i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

print(f"Training samples: {len(X)}")

# --- MODEL TRAINING ---
param_dist = {
    'n_estimators':    [100, 150, 200],
    'max_depth':       [5, 8, 12],
    'min_samples_leaf':[1, 5],
    # class_weight balanced compensa el desbalance 0 >> 1,2
    'class_weight':    ['balanced', None],
}
tscv = TimeSeriesSplit(n_splits=2)
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=8,
    cv=tscv,
    scoring='balanced_accuracy',   # mejor métrica para clases desbalanceadas
    n_jobs=-1,
    random_state=42,
)
search.fit(X, y)
model = search.best_estimator_
print(f"Best configuration: {search.best_params_}")
print(f"Best balanced_accuracy: {search.best_score_:.4f}")

# --- EXPORT ONNX ---
# El modelo produce 3 clases: 0, 1, 2
# En MT5 leer output_label (int64) → 0=nada, 1=buy, 2=sell
initial_type = [('float_input', FloatTensorType([None, len(features) * window]))]
onx = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=12,
    options={type(model): {'zipmap': False}},
)
onnx.checker.check_model(onx)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Model saved to: {output_filename}")
print(f"Opset version: 12 (MT5 compatible)")
print(f"Output classes → 0: no signal | 1: BUY | 2: SELL")
print(f"--- PROCESS COMPLETED ---")