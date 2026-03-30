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
parser = argparse.ArgumentParser(description="Train ONNX model from CSV data with features (body, range, Stoch, Vol) using ATR normalization")
parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the ONNX model (default: current directory)")
parser.add_argument("--atr_period", type=int, default=14, help="Period for ATR calculation (default: 14)")
parser.add_argument("--window", type=int, default=20, help="Window size (number of bars) for features")
parser.add_argument("--future", type=int, default=5, help="Number of bars to look into the future for target labeling")
parser.add_argument("--n_iter", type=int, default=5, help="Number of iterations for RandomizedSearchCV")
parser.add_argument("--min_profit_atr", type=float, default=1.5, help="Minimum profit in ATR multiples for a positive target (default: 1.5)")

args = parser.parse_args()

csv_file = args.input_csv
output_dir = args.output_dir
atr_period = args.atr_period
window = args.window
future = args.future
n_iter = args.n_iter
min_profit_atr = args.min_profit_atr

if not os.path.exists(csv_file):
    print(colorize(f"Error: File '{csv_file}' not found", Colors.RED))
    sys.exit(1)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate output filename: same basename as CSV but with .onnx extension, in output_dir
output_filename = os.path.join(output_dir, Path(csv_file).stem + f"_w{window}_f{future}_atr{atr_period}_minp{min_profit_atr}.onnx")

print(colorize("--- FAST TRAINING WITH ATR NORMALIZATION ---", Colors.CYAN))
print(f"Loading rates from: {colorize(csv_file, Colors.WHITE)}")
print(f"Output ONNX will be: {colorize(output_filename, Colors.YELLOW)}")

# 1. LOAD DATA FROM CSV
df = pd.read_csv(csv_file)
print(f"Rows loaded: {colorize(str(len(df)), Colors.GREEN)}")

# Calculate ATR for normalization
atr_indicator = ta.volatility.AverageTrueRange(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    window=atr_period
)
df['atr'] = atr_indicator.average_true_range()

stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)

# Calculate features normalized by ATR
df['feat_body'] = (df['close'] - df['open']) / df['atr']
df['feat_range'] = (df['high'] - df['low']) / df['atr']
df['feat_stoch'] = (stoch.stoch() - stoch.stoch_signal()) / 100.0
df['feat_vol'] = df['tick_volume'] / df['tick_volume'].rolling(window=20).mean()

# 2. GENERATE TARGET (Labeling based on future profit in ATR multiples)
print(f"Generating target with future={colorize(str(future), Colors.MAGENTA)} and min_profit_atr={colorize(str(min_profit_atr), Colors.MAGENTA)}...")
labels = np.zeros(len(df))
for i in range(len(df) - future):
    if pd.isna(df['atr'].iloc[i]) or df['atr'].iloc[i] == 0:
        continue
    
    entry_price = df['close'].iloc[i]
    current_atr = df['atr'].iloc[i]
    
    # Check if price reaches target profit in ATR multiples in the next 'future' candles
    future_prices = df['high'].iloc[i+1 : i+future+1]
    profit = (future_prices.max() - entry_price) / current_atr
    
    if profit >= min_profit_atr:
        labels[i] = 1

df['target'] = labels
df.dropna(inplace=True)

# 3. PREPARE WINDOWS
X, y = [], []
features = ['feat_body', 'feat_range', 'feat_stoch', 'feat_vol']

for i in range(window, len(df) - future):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

print(f"Training samples: {colorize(str(len(X)), Colors.GREEN)}")
print(f"Positive samples: {colorize(str(int(y.sum())), Colors.GREEN)} ({colorize(f'{(y.sum()/len(y)*100):.2f}%', Colors.YELLOW)})")

# 4. FAST OPTIMIZATION (Random Search)
print(colorize("Searching for efficient configuration (Random Search)...", Colors.YELLOW))
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 8, 12],
    'min_samples_leaf': [1, 5]
}

# TimeSeriesSplit with 2 folds for speed
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
print(colorize(f"Best score: {search.best_score_:.4f}", Colors.GREEN))

# 5. EXPORT WITH CSV-BASED NAME
initial_type = [('float_input', FloatTensorType([None, window * len(features)]))]
# Use target_opset=12 for MetaTrader 5 compatibility (MT5 supports opset 1-21, but lower is safer)
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12, options={type(model): {'zipmap': False}})

# Validate ONNX model before saving
onnx.checker.check_model(onx)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(colorize(f"Model saved at: {output_filename}", Colors.GREEN))
print(colorize("--- PROCESS COMPLETED ---", Colors.CYAN))
