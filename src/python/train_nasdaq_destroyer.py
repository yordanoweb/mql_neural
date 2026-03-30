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
parser = argparse.ArgumentParser(description="Train ONNX model from CSV data with 3 features (body, range, RSI)")
parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the ONNX model (default: current directory)")
parser.add_argument("--rsi_period", type=int, default=14, help="Period for RSI calculation (default: 14)")
parser.add_argument("--window", type=int, default=20, help="Window size (number of bars) for features")
parser.add_argument("--future", type=int, default=5, help="Number of bars to look into the future for target labeling")
parser.add_argument("--n_iter", type=int, default=5, help="Number of iterations for RandomizedSearchCV")
parser.add_argument("--min_profit_points", type=float, default=10.0, help="Minimum profit points for a positive target")

args = parser.parse_args()

csv_file = args.input_csv
output_dir = args.output_dir
rsi_period = args.rsi_period
window = args.window
future = args.future
n_iter = args.n_iter
min_profit_points = args.min_profit_points

if not os.path.exists(csv_file):
    print(colorize(f"Error: File '{csv_file}' not found", Colors.RED))
    sys.exit(1)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate output filename: same basename as CSV but with .onnx extension, in output_dir
output_filename = os.path.join(output_dir, Path(csv_file).stem + f"_oc_hl_w{window}_f{future}_minp{min_profit_points}_rsi{rsi_period}.onnx")

print(colorize("--- FAST TRAINING ---", Colors.CYAN))
print(f"Loading rates from: {colorize(csv_file, Colors.WHITE)}")
print(f"Output ONNX will be: {colorize(output_filename, Colors.YELLOW)}")

# 1. LOAD DATA FROM CSV
df = pd.read_csv(csv_file)
print(f"Rows loaded: {colorize(str(len(df)), Colors.GREEN)}")

# Infer pip unit from data (optional, or set based on symbol detection if available)
# If symbol info is not available, we'll use a reasonable default
pip_unit = 0.0001  # Default for most pairs; could be refined if symbol is known

df['feat_body'] = (df['close'] - df['open']) / pip_unit
df['feat_range'] = (df['high'] - df['low']) / pip_unit
df['feat_rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi() / 100.0

# 2. GENERATE TARGET (Labeling based on future profit)
print(f"Generating target with future={colorize(str(future), Colors.MAGENTA)} and min_profit_points={colorize(str(min_profit_points), Colors.MAGENTA)}...")
labels = np.zeros(len(df))
for i in range(len(df) - future):
    entry_price = df['close'].iloc[i]
    # Check if price reaches target profit point in the next 'future' candles
    future_prices = df['high'].iloc[i+1 : i+future+1]
    profit = (future_prices.max() - entry_price) / pip_unit
    if profit >= min_profit_points:
        labels[i] = 1

df['target'] = labels
df.dropna(inplace=True)

# 3. PREPARE WINDOWS
X, y = [], []
features = ['feat_body', 'feat_range', 'feat_rsi']

for i in range(window, len(df) - future):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

# 4. FAST OPTIMIZATION (Random Search)
print(colorize("Searching for efficient configuration (Random Search)...", Colors.YELLOW))
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 8, 12],
    'min_samples_leaf': [1, 5]
}

# TimeSeriesSplit con 2 pliegues para velocidad
tscv = TimeSeriesSplit(n_splits=2)

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist,
    n_iter=n_iter,
    cv=tscv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=2
)

print(colorize(f"Starting training with {X.shape[0]} samples...", Colors.CYAN))
search.fit(X, y)
model = search.best_estimator_
print(colorize(f"Best configuration: {search.best_params_}", Colors.GREEN))

# 5. EXPORT WITH CSV-BASED NAME
initial_type = [('float_input', FloatTensorType([None, window * 3]))]
# Use target_opset=12 for MetaTrader 5 compatibility (MT5 supports opset 1-21, but lower is safer)
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12, options={type(model): {'zipmap': False}})

# Validate ONNX model before saving
onnx.checker.check_model(onx)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(colorize(f"Model saved at: {output_filename}", Colors.GREEN))
print(colorize("--- PROCESS COMPLETED ---", Colors.CYAN))