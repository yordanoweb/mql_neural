import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import ta
import onnx
import argparse
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# ---------- Color setup ----------
class Colors:
    RESET   = '\033[0m'
    RED     = '\033[91m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN    = '\033[96m'
    WHITE   = '\033[97m'

def colorize(text, color):
    return f"{color}{text}{Colors.RESET}"

# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description="Train ONNX model using ATR-normalized features and multi-class labeling")
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--rsi_period", type=int, default=14)
parser.add_argument("--atr_period", type=int, default=14)
parser.add_argument("--window", type=int, default=20)
parser.add_argument("--future", type=int, default=5)
parser.add_argument("--n_iter", type=int, default=5)
parser.add_argument("--min_profit_atr", type=float, default=1.5)
parser.add_argument("--dominance_ratio", type=float, default=1.2, help="Required dominance ratio between directions")

args = parser.parse_args()

csv_file = args.input_csv
output_dir = args.output_dir
rsi_period = args.rsi_period
atr_period = args.atr_period
window = args.window
future = args.future
n_iter = args.n_iter
min_profit_atr = args.min_profit_atr
dominance_ratio = args.dominance_ratio

if not os.path.exists(csv_file):
    print(colorize(f"Error: File '{csv_file}' not found", Colors.RED))
    sys.exit(1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_filename = os.path.join(
    output_dir,
    Path(csv_file).stem + f"_atr_w{window}_f{future}_t{min_profit_atr}_rsi{rsi_period}_atr{atr_period}.onnx"
).replace("_rates", "")

print(colorize("--- ATR TRAINING PIPELINE ---", Colors.CYAN))
print(f"CSV: {colorize(csv_file, Colors.WHITE)}")
print(f"Output: {colorize(output_filename, Colors.YELLOW)}")

# 1. LOAD DATA
print(colorize("[1] Loading CSV...", Colors.BLUE))
df = pd.read_csv(csv_file)
print(f"Rows loaded: {colorize(str(len(df)), Colors.GREEN)}")

# 2. INDICATORS
print(colorize("[2] Calculating ATR & RSI...", Colors.BLUE))
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period).average_true_range()

# --- FEATURE ENGINEERING V2 ---
# 1. Dirección/momentum real (MUY IMPORTANTE)
df['feat_momentum'] = (df['close'] - df['close'].shift(window)) / df['atr']
# 2. Posición dentro del rango reciente (contexto)
rolling_high = df['high'].rolling(window).max()
rolling_low = df['low'].rolling(window).min()
df['feat_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low)
# 3. RSI (se mantiene)
df['feat_rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi() / 100.0

# --- OLD FEATURES ---
# df['feat_body'] = (df['close'] - df['open']) / df['atr']
# df['feat_range'] = (df['high'] - df['low']) / df['atr']

# 3. LABELING
print(colorize("[3] Generating labels (ATR-based)...", Colors.BLUE))
labels = np.zeros(len(df))

# 🔥 NUEVO TARGET (regresión)
future_close = df['close'].shift(-future)

df['target_reg'] = (future_close - df['close']) / df['atr']

buy_count = 0
sell_count = 0
neutral_count = 0

for i in range(len(df) - future):
    entry = df['close'].iloc[i]
    current_atr = df['atr'].iloc[i]

    if current_atr == 0 or np.isnan(current_atr):
        continue

    future_high = df['high'].iloc[i+1:i+future+1].max()
    future_low = df['low'].iloc[i+1:i+future+1].min()

    max_up = future_high - entry
    max_down = entry - future_low

    up_atr = max_up / current_atr
    down_atr = max_down / current_atr

    if up_atr >= min_profit_atr and up_atr > down_atr * dominance_ratio:
        labels[i] = 1
        buy_count += 1
    elif down_atr >= min_profit_atr and down_atr > up_atr * dominance_ratio:
        labels[i] = -1
        sell_count += 1
    else:
        labels[i] = 0
        neutral_count += 1

# Debug stats
print(colorize("Label distribution:", Colors.MAGENTA))
total_labeled = buy_count + sell_count + neutral_count
print(f"BUY: {buy_count} ({buy_count/total_labeled:.2%}) | SELL: {sell_count} ({sell_count/total_labeled:.2%}) | NEUTRAL: {neutral_count} ({neutral_count/total_labeled:.2%})")

# --- QUICK LABEL VALIDATION ---
print(colorize("[VALIDATION] Sampling labeled cases...", Colors.YELLOW))

sample_indices = np.linspace(0, len(df) - future - 1, num=10, dtype=int)

for idx in sample_indices:
    entry = df['close'].iloc[idx]
    atr_val = df['atr'].iloc[idx]

    if np.isnan(atr_val) or atr_val == 0:
        continue

    future_high = df['high'].iloc[idx+1:idx+future+1].max()
    future_low = df['low'].iloc[idx+1:idx+future+1].min()

    up = (future_high - entry) / atr_val
    down = (entry - future_low) / atr_val
    label = labels[idx]

    print(f"i={idx} | label={label} | up_atr={up:.2f} | down_atr={down:.2f}")

print(colorize("[VALIDATION DONE]", Colors.YELLOW))

df['target'] = labels

# Clean NaNs
before_drop = len(df)
df.dropna(inplace=True)
after_drop = len(df)
print(f"Dropped NaNs: {before_drop - after_drop}")

# 4. FEATURES (NO WINDOW, DIRECT)
print(colorize("[4] Building features (no window)...", Colors.BLUE))

features = ['feat_momentum', 'feat_position', 'feat_rsi']

# Limpiar infinitos
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Forzar tipo eficiente
df[features] = df[features].astype(np.float32)

# Construcción vectorizada (SIN loop)
start = window
end = len(df) - future

close = df["close"].values
returns = np.diff(close, prepend=close[0])

momentum_3 = close - np.roll(close, 3)
momentum_5 = close - np.roll(close, 5)

atr = df["atr"].values
volatility = atr / close

# --- RSI ---
delta = df["close"].diff()

gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()

rs = gain / (loss + 1e-9)
df["rsi"] = 100 - (100 / (1 + rs))
rsi = df["rsi"].values

X = np.column_stack([
    returns,
    rsi,
    atr,
    momentum_3,
    momentum_5,
    volatility
]).astype(np.float32)

y = (df["close"].shift(-future) - df["close"]) / df["atr"]

# --- DESPUÉS (fix) ---
min_len = min(len(X), len(y))
X = X[:min_len]
y = y[:min_len]

# 🔥 FIX: Eliminar NaNs/Inf que quedan por el shift(-future)
y = y.replace([np.inf, -np.inf], np.nan)
valid_mask = ~np.isnan(y)
X = X[valid_mask]
y = y[valid_mask].astype(np.float32)

print(f"X dtype: {X.dtype}")
print(f"X shape: {X.shape}")
print(f"Samples: {len(X)} | Features per sample: {X.shape[1]}")
print(f"Valid samples after NaN removal: {len(y)}")

# 5. TRAINING
print(colorize("[5] Hyperparameter search...", Colors.BLUE))
from xgboost import XGBRFRegressor

model = XGBRFRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42
)

# =========================
# TRAIN / TEST SPLIT (TIME SERIES)
# =========================
print(colorize("\n[5] Splitting train/test (time-based)...", Colors.BLUE))

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print(colorize("\n[6] Training model (train set only)...", Colors.BLUE))
model.fit(X_train, y_train)

# =========================
# TEST SET EVALUATION
# =========================
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print(colorize("\n[7] Evaluating on TEST set (out-of-sample)...", Colors.CYAN))

from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = model.predict(X_test)

print("\n[DEBUG] Prediction stats:")
print(f"min: {y_pred.min():.4f}")
print(f"max: {y_pred.max():.4f}")
print(f"mean: {y_pred.mean():.4f}")
print(f"std: {y_pred.std():.4f}")

print("\n[TEST] Regression Evaluation")

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

# 🔥 convertir predicción en señales
threshold = 1.0  # 1 ATR

signals = np.zeros_like(y_pred)
signals[y_pred > threshold] = 1
signals[y_pred < -threshold] = -1

# 🔥 ground truth en formato señales
real = np.zeros_like(y_test)
real[y_test > threshold] = 1
real[y_test < -threshold] = -1

mask = signals != 0

print("\n[Trading Evaluation]")
print(f"Signals issued: {mask.sum()} ({mask.sum()/len(mask)*100:.2f}%)")

if mask.sum() > 0:
    accuracy = (signals[mask] == real[mask]).mean()
    print(f"Signal accuracy: {accuracy*100:.2f}%")
else:
    print("No signals issued")

# 6. EXPORT
print(colorize("[6] Exporting to ONNX...", Colors.BLUE))
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 3]))]
onx = convert_xgboost(model, initial_types=initial_type)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(colorize(f"Model saved at: {output_filename}", Colors.GREEN))
print(colorize("--- DONE ---", Colors.CYAN))
