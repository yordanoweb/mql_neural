import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import argparse
import ta

# ---------------- CONFIG ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--window", type=int, default=20)
parser.add_argument("--n_iter", type=int, default=10)
parser.add_argument("--start_hour", type=int, default=20)
parser.add_argument("--end_hour", type=int, default=21)
parser.add_argument("--atr_period", type=int, default=14)

args = parser.parse_args()

# Thresholds (ATR-based)
ATR_THRESHOLD = 0.8
DOMINANCE_THRESHOLD = 0.3
CLOSE_CONFIRM = 0.1

# ---------------- LOAD ----------------
df = pd.read_csv(args.input_csv)

if 'time' in df.columns:
    df['datetime'] = pd.to_datetime(df['time'])
elif 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
else:
    print("No datetime column found")
    sys.exit(1)

df.sort_values('datetime', inplace=True)

# ---------------- ATR ----------------
atr_indicator = ta.volatility.AverageTrueRange(
    high=df['high'], low=df['low'], close=df['close'], window=args.atr_period
)
df['atr'] = atr_indicator.average_true_range()

# ---------------- FEATURES ----------------
df['feat_body'] = (df['close'] - df['open']) / df['atr']
df['feat_range'] = (df['high'] - df['low']) / df['atr']

print("ATR NaNs:", df['atr'].isna().sum())
print("ATR zeros:", (df['atr'] == 0).sum())
print("Rows before dropna:", len(df))

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Rows after dropna:", len(df))

# ---------------- BUILD DAILY SAMPLES ----------------
X, y = [], []

df['date'] = df['datetime'].dt.date

for date, day_df in df.groupby('date'):

    # ventana objetivo
    window_df = day_df[
        (day_df['datetime'].dt.hour >= args.start_hour) &
        (day_df['datetime'].dt.hour < args.end_hour)
    ]

    if len(window_df) < 2:
        continue

    # punto de entrada = última vela antes de la ventana
    window_start_time = window_df['datetime'].iloc[0]
    pre_window = df[df['datetime'] < window_start_time].tail(args.window)

    if len(pre_window) < args.window:
        continue

    entry_idx = pre_window.index[-1]
    entry_pos = df.index.get_loc(entry_idx)

    # protección por si no hay suficientes datos hacia atrás
    if entry_pos < args.window:
        continue

    entry_price = df.iloc[entry_pos]['close']
    atr = df.iloc[entry_pos]['atr']

    if atr == 0 or np.isnan(atr):
        continue

    # ---------------- TARGET ----------------
    max_high = window_df['high'].max()
    min_low = window_df['low'].min()
    close_price = window_df['close'].iloc[-1]

    max_up = (max_high - entry_price) / atr
    max_down = (entry_price - min_low) / atr
    close_move = (close_price - entry_price) / atr
    print(f"up={max_up:.2f}, down={max_down:.2f}, close={close_move:.2f}")

    label = 0

    # LONG
    if (
        max_up >= ATR_THRESHOLD and
        max_up > max_down and
        abs(max_up - max_down) >= DOMINANCE_THRESHOLD and
        close_move >= CLOSE_CONFIRM
    ):
        label = 1

    # SHORT
    elif (
        max_down >= ATR_THRESHOLD and
        max_down > max_up and
        abs(max_up - max_down) >= DOMINANCE_THRESHOLD and
        -close_move >= CLOSE_CONFIRM
    ):
        label = 2

    # FILTRO AMBIGUO
    elif (
        max_up >= ATR_THRESHOLD and
        max_down >= ATR_THRESHOLD and
        abs(max_up - max_down) < DOMINANCE_THRESHOLD
    ):
        continue  # eliminar muestra

    # ---------------- FEATURES WINDOW ----------------
    window_slice = df.iloc[entry_pos - args.window + 1: entry_pos + 1]

    window_data = window_slice[['feat_body', 'feat_range']].values.flatten()

    if len(window_data) != args.window * 2:
        continue

    X.append(window_data)
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Samples:", len(X))
print("Distribution:", dict(zip(*np.unique(y, return_counts=True))))

if len(X) < 50:
    print("Not enough samples")
    sys.exit(1)

# ---------------- TRAIN ----------------
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 8, 12],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

tscv = TimeSeriesSplit(n_splits=3)

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist,
    n_iter=args.n_iter,
    cv=tscv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=2
)

search.fit(X, y)
model = search.best_estimator_

print("Best params:", search.best_params_)
print("Best score:", search.best_score_)

# ---------------- EXPORT ONNX ----------------
initial_type = [('float_input', FloatTensorType([None, args.window * 2]))]

onx = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=12,
    options={type(model): {'zipmap': False}}
)

onnx.checker.check_model(onx)

<<<<<<< Updated upstream
with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())
=======
output_path = os.path.join(
    args.output_dir,
    Path(args.input_csv).stem + "_time_window_atr.onnx"
)
>>>>>>> Stashed changes

onnx.save(onx, output_path)

print("Model saved at:", output_path)
