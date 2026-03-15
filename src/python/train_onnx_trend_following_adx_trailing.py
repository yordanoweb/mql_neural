# PYTHON - Improved version for 3-class trend following

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, f1_score
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

output_filename = Path(onnx_output_dir) / (Path(csv_file).stem + "_trend_3class.onnx")
print(f"--- TREND-FOLLOWING TRAINING (3-CLASS) ---")
print(f"Loading rates from: {csv_file}")
print(f"Output ONNX will be: {output_filename}")

df = pd.read_csv(csv_file)
print(f"Records loaded: {len(df)}")

pip_unit = 0.0001  # adjust for JPY or other pairs if needed; you can derive from data

# Feature engineering
window = 20
adx_period = 14
plus_di, minus_di, dx, adx = calculate_adx(
    df['high'].tolist(), df['low'].tolist(), df['close'].tolist(), period=adx_period)

df['feat_body'] = (df['close'] - df['open']) / pip_unit
df['feat_range'] = (df['high'] - df['low']) / pip_unit
df['feat_rsi'] = pd.Series(calculate_rsi(df['close'], 14)) / 100.0
df['feat_adx'] = pd.Series(adx)
df['feat_plus_di'] = pd.Series(plus_di)
df['feat_minus_di'] = pd.Series(minus_di)

# Label: 0 = no trade, 1 = buy, 2 = sell
trend_thresh = 25
move_pips = 5 * pip_unit   # threshold for price movement (5 pips)
future_window = 5

labels = []
for i in range(window, len(df) - future_window):
    # Get current values
    if adx[i] is None or plus_di[i] is None or minus_di[i] is None:
        labels.append(0)
        continue
    entry_price = df['close'].iloc[i]
    future_max = df['close'].iloc[i+1:i+future_window+1].max()
    future_min = df['close'].iloc[i+1:i+future_window+1].min()
    
    # Buy condition: strong uptrend and price moves up
    if (adx[i] > trend_thresh and plus_di[i] > minus_di[i] and 
        future_max > entry_price + move_pips):
        labels.append(1)
    # Sell condition: strong downtrend and price moves down
    elif (adx[i] > trend_thresh and minus_di[i] > plus_di[i] and 
          future_min < entry_price - move_pips):
        labels.append(2)
    else:
        labels.append(0)

# Align dataframe
df = df.iloc[window:len(df)-future_window].copy()
df['target'] = labels

# Check class balance
print("Class distribution:")
print(df['target'].value_counts())

features = ['feat_body', 'feat_range', 'feat_rsi', 'feat_adx', 'feat_plus_di', 'feat_minus_di']
X = []
y = []
for i in range(window, len(df)):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

print(f"Training samples: {len(X)}")

# Model training with class balancing and better hyperparameter search
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [5, 8, 12, 15, None],
    'min_samples_leaf': [1, 2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

tscv = TimeSeriesSplit(n_splits=3)
# Use f1_macro to account for multi-class imbalance
scorer = make_scorer(f1_score, average='macro')

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=30,              # more iterations for better tuning
    cv=tscv,
    scoring=scorer,
    n_jobs=-1,
    random_state=42
)
search.fit(X, y)
model = search.best_estimator_
print(f"Best configuration: {search.best_params_}")
print(f"Best cross-validation F1-macro: {search.best_score_:.4f}")

# Convert to ONNX
# Input shape: (None, window * number_of_features)
initial_type = [('float_input', FloatTensorType([None, len(features)*window]))]
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12,
                      options={type(model): {'zipmap': False}})
onnx.checker.check_model(onx)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Model saved to: {output_filename}")
print(f"Opset version: 12 (MT5 compatible)")
print(f"--- PROCESS COMPLETED ---")
